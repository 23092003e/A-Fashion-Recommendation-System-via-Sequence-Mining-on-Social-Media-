"""
Contrastive Learning + Ensemble Approach
Target: HR@10 >= 0.4, MRR >= 0.3

Key ideas:
1. InfoNCE contrastive loss for better representations
2. Multiple model ensemble
3. Temperature scaling
4. Hard negative mining
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import ast
import random
import os
from collections import Counter

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ============================================================
# 1. DATA LOADING
# ============================================================

df_seq = pd.read_csv('input/user_behavior.csv')
if isinstance(df_seq['posts_sequence'].iloc[0], str):
    df_seq['posts_sequence_list'] = df_seq['posts_sequence'].apply(ast.literal_eval)
else:
    df_seq['posts_sequence_list'] = df_seq['posts_sequence']

all_post_ids = set()
for seq in df_seq['posts_sequence_list']:
    all_post_ids.update(seq)

sorted_ids = sorted(list(all_post_ids))
post2idx = {pid: i+1 for i, pid in enumerate(sorted_ids)}
idx2post = {i+1: pid for i, pid in enumerate(sorted_ids)}
VOCAB_SIZE = len(post2idx) + 1

pretrained_weights = np.load('post_embeddings_multimodal.npy')
print(f'Vocab: {VOCAB_SIZE}')

# Prepare samples
train_samples = []
test_samples = []

for idx, row in df_seq.iterrows():
    raw_seq = row['posts_sequence_list']
    seq = [post2idx.get(pid, 0) for pid in raw_seq]
    if len(seq) < 2:
        continue
    test_samples.append((seq[:-1], seq[-1]))
    train_seq = seq[:-1]
    for i in range(1, len(train_seq)):
        start = max(0, i - 20)
        train_samples.append((train_seq[start:i], train_seq[i]))

print(f'Train: {len(train_samples)}, Test: {len(test_samples)}')

class SimpleDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        seq, target = self.samples[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

def collate_fn(batch):
    seqs, targets = zip(*batch)
    seqs_padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
    return seqs_padded, torch.stack(targets)

BATCH_SIZE = 32
train_loader = DataLoader(SimpleDataset(train_samples), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(SimpleDataset(test_samples), batch_size=BATCH_SIZE, collate_fn=collate_fn)

# ============================================================
# 2. MODEL WITH CONTRASTIVE LEARNING
# ============================================================

class ContrastiveModel(nn.Module):
    def __init__(self, pretrained_weights, hidden_dim=256, num_heads=4, dropout=0.5):
        super(ContrastiveModel, self).__init__()

        weights_tensor = torch.FloatTensor(pretrained_weights)
        self.vocab_size = weights_tensor.shape[0]
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding.from_pretrained(weights_tensor, freeze=True, padding_idx=0)

        # Item embedding (learnable) for contrastive learning
        self.item_embed = nn.Embedding(self.vocab_size, hidden_dim, padding_idx=0)
        nn.init.normal_(self.item_embed.weight, std=0.02)

        self.text_proj = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True, bidirectional=True)

        self.mha = nn.MultiheadAttention(hidden_dim*2, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim * 2)
        self.norm2 = nn.LayerNorm(hidden_dim * 2)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2)
        )

        self.dropout = nn.Dropout(dropout)

        # Output projection to match item embedding dimension
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Temperature for contrastive learning
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def encode_sequence(self, x):
        """Encode sequence to get user representation"""
        embedded = self.embedding(x)
        text_emb = embedded[:, :, :768]
        image_emb = embedded[:, :, 768:]

        text_proj = self.text_proj(text_emb)
        image_proj = self.image_proj(image_emb)

        concat = torch.cat([text_proj, image_proj], dim=-1)
        gate = self.gate(concat)
        fused = gate * text_proj + (1 - gate) * image_proj

        lstm_out, _ = self.lstm(fused)

        attn_out, _ = self.mha(lstm_out, lstm_out, lstm_out)
        out = self.norm1(attn_out + lstm_out)

        ffn_out = self.ffn(out)
        out = self.norm2(ffn_out + out)

        mask = (x != 0).unsqueeze(-1).float()
        out = out * mask
        context = torch.sum(out, dim=1) / (torch.sum(mask, dim=1) + 1e-9)

        # Project to item embedding space
        user_repr = self.out_proj(self.dropout(context))  # [B, hidden_dim]

        return user_repr

    def forward(self, x, targets=None):
        user_repr = self.encode_sequence(x)  # [B, H]

        # Get all item embeddings
        item_embeds = self.item_embed.weight  # [V, H]

        # Compute logits via dot product
        logits = torch.matmul(user_repr, item_embeds.T)  # [B, V]
        logits = logits / self.temperature.clamp(min=0.01)

        return logits

    def contrastive_loss(self, user_repr, targets, negatives=None):
        """InfoNCE contrastive loss"""
        B = user_repr.shape[0]

        # Positive item embeddings
        pos_embeds = self.item_embed(targets)  # [B, H]

        # Compute positive scores
        pos_scores = (user_repr * pos_embeds).sum(dim=-1)  # [B]

        # In-batch negatives + random negatives
        all_item_embeds = self.item_embed.weight[1:]  # Exclude padding

        # Compute all scores
        all_scores = torch.matmul(user_repr, all_item_embeds.T)  # [B, V-1]

        # InfoNCE loss
        logits = all_scores / self.temperature.clamp(min=0.01)

        # Create labels (index of positive in all items)
        labels = targets - 1  # Adjust for padding removal

        loss = F.cross_entropy(logits, labels)

        return loss

# ============================================================
# 3. TRAINING
# ============================================================

def train_contrastive(model, train_loader, epochs=100, lr=0.0003):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    print('Training with Contrastive Learning...')

    for epoch in range(epochs):
        total_loss = 0
        for seqs, targets in train_loader:
            seqs, targets = seqs.to(device), targets.to(device)

            optimizer.zero_grad()

            user_repr = model.encode_sequence(seqs)
            loss = model.contrastive_loss(user_repr, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f'   Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}')

    return model

def evaluate(model, test_loader, k_list=[5, 10, 20]):
    model.eval()
    hits = {k: 0 for k in k_list}
    mrr_sum = 0
    total = 0

    with torch.no_grad():
        for seqs, targets in test_loader:
            seqs, targets = seqs.to(device), targets.to(device)

            logits = model(seqs)

            _, topk = torch.topk(logits, max(k_list), dim=1)
            topk = topk.cpu().numpy()
            targets_np = targets.cpu().numpy()

            for i, target in enumerate(targets_np):
                pred_ids = topk[i]

                for k in k_list:
                    if target in pred_ids[:k]:
                        hits[k] += 1

                if target in pred_ids:
                    rank = np.where(pred_ids == target)[0][0] + 1
                    mrr_sum += 1.0 / rank

                total += 1

    return {f'HR@{k}': v/total for k, v in hits.items()}, mrr_sum/total

# ============================================================
# 4. ENSEMBLE OF MODELS
# ============================================================

print('\n' + '='*70)
print('TRAINING ENSEMBLE OF MODELS')
print('='*70)

models = []
all_metrics = []

for i, seed in enumerate([42, 123, 456]):
    print(f'\n--- Model {i+1}/3 (seed={seed}) ---')
    set_seed(seed)

    model = ContrastiveModel(pretrained_weights, hidden_dim=256, num_heads=4, dropout=0.5).to(device)
    model = train_contrastive(model, train_loader, epochs=100, lr=0.0003)

    metrics, mrr = evaluate(model, test_loader)
    print(f'   HR@10={metrics["HR@10"]:.4f}, MRR={mrr:.4f}')

    models.append(model)
    all_metrics.append((metrics, mrr))

# Ensemble prediction
print('\n--- Ensemble Prediction ---')

def ensemble_evaluate(models, test_loader, k_list=[5, 10, 20]):
    for m in models:
        m.eval()

    hits = {k: 0 for k in k_list}
    mrr_sum = 0
    total = 0

    with torch.no_grad():
        for seqs, targets in test_loader:
            seqs, targets = seqs.to(device), targets.to(device)

            # Average logits from all models
            all_logits = []
            for m in models:
                logits = m(seqs)
                all_logits.append(logits)

            avg_logits = torch.stack(all_logits).mean(dim=0)

            _, topk = torch.topk(avg_logits, max(k_list), dim=1)
            topk = topk.cpu().numpy()
            targets_np = targets.cpu().numpy()

            for i, target in enumerate(targets_np):
                pred_ids = topk[i]

                for k in k_list:
                    if target in pred_ids[:k]:
                        hits[k] += 1

                if target in pred_ids:
                    rank = np.where(pred_ids == target)[0][0] + 1
                    mrr_sum += 1.0 / rank

                total += 1

    return {f'HR@{k}': v/total for k, v in hits.items()}, mrr_sum/total

ensemble_metrics, ensemble_mrr = ensemble_evaluate(models, test_loader)

print('\n' + '='*70)
print('RESULTS')
print('='*70)

print('\nIndividual Models:')
for i, (m, mrr) in enumerate(all_metrics):
    print(f'   Model {i+1}: HR@10={m["HR@10"]:.4f}, MRR={mrr:.4f}')

print(f'\nEnsemble:')
print(f"   HR@5:  {ensemble_metrics['HR@5']:.4f}")
print(f"   HR@10: {ensemble_metrics['HR@10']:.4f}")
print(f"   HR@20: {ensemble_metrics['HR@20']:.4f}")
print(f"   MRR:   {ensemble_mrr:.4f}")

print('\nComparison:')
print(f"   Previous best HR@10: 0.3506 | MRR: 0.1972")
print(f"   Contrastive+Ensemble HR@10: {ensemble_metrics['HR@10']:.4f} | MRR: {ensemble_mrr:.4f}")

# Save best model
os.makedirs('models', exist_ok=True)
torch.save({
    'models': [m.state_dict() for m in models],
    'ensemble_metrics': ensemble_metrics,
    'ensemble_mrr': ensemble_mrr
}, 'models/contrastive_ensemble.pth')
print('\nModels saved!')
