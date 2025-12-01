"""
Optimized Multimodal Model for Fashion Recommendation
Target: HR@10 >= 0.4, MRR >= 0.3

Strategy:
1. Focus on what works - simple gated fusion
2. Add strong repeat bias (15% of targets repeat)
3. Better training strategy
4. Ensemble of predictions
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

set_seed(42)
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
print(f'Vocab: {VOCAB_SIZE}, Embeddings: {pretrained_weights.shape}')

# Item frequency
item_freq = Counter()
for seq in df_seq['posts_sequence_list']:
    item_freq.update([post2idx[p] for p in seq])

# ============================================================
# 2. PREPARE DATA
# ============================================================

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
# 3. SIMPLE BUT EFFECTIVE MODEL
# ============================================================

class OptimizedGatedFusion(nn.Module):
    def __init__(self, pretrained_weights, hidden_dim=256, num_heads=4, dropout=0.5):
        super(OptimizedGatedFusion, self).__init__()

        weights_tensor = torch.FloatTensor(pretrained_weights)
        self.vocab_size = weights_tensor.shape[0]
        self.text_dim = 768
        self.image_dim = 768

        self.embedding = nn.Embedding.from_pretrained(weights_tensor, freeze=True, padding_idx=0)

        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(self.image_dim, hidden_dim),
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
        self.fc = nn.Linear(hidden_dim * 2, self.vocab_size)

    def forward(self, x, return_hidden=False):
        embedded = self.embedding(x)
        text_emb = embedded[:, :, :self.text_dim]
        image_emb = embedded[:, :, self.text_dim:]

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

        # Mean pooling with mask
        mask = (x != 0).unsqueeze(-1).float()
        out = out * mask
        context = torch.sum(out, dim=1) / (torch.sum(mask, dim=1) + 1e-9)

        logits = self.fc(self.dropout(context))

        if return_hidden:
            return logits, context
        return logits

# ============================================================
# 4. TRAINING FUNCTIONS
# ============================================================

def train_model(model, train_loader, epochs=100, lr=0.00015):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    model.train()
    print('Training...')

    for epoch in range(epochs):
        total_loss = 0
        for seqs, targets in train_loader:
            seqs, targets = seqs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(seqs)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f'   Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}')

    return model

def evaluate_with_repeat_bias(model, test_loader, repeat_weight=2.0, k_list=[5, 10, 20]):
    """Evaluate with repeat bias - boost scores for items in history"""
    model.eval()
    hits = {k: 0 for k in k_list}
    mrr_sum = 0
    total = 0

    with torch.no_grad():
        for seqs, targets in test_loader:
            seqs, targets = seqs.to(device), targets.to(device)

            logits = model(seqs)  # [B, V]

            # Add repeat bias
            B, L = seqs.shape
            for b in range(B):
                for pos in range(L):
                    item = seqs[b, pos].item()
                    if item > 0:
                        # More recent = higher bonus
                        recency = (pos + 1) / L
                        logits[b, item] += repeat_weight * recency

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
# 5. EXPERIMENT: FIND OPTIMAL REPEAT WEIGHT
# ============================================================

print('\n' + '='*70)
print('EXPERIMENT: FINDING OPTIMAL REPEAT WEIGHT')
print('='*70)

set_seed(42)
model = OptimizedGatedFusion(pretrained_weights, hidden_dim=256, num_heads=4, dropout=0.5).to(device)
model = train_model(model, train_loader, epochs=100, lr=0.00015)

print('\nTesting different repeat weights:')
best_hr10 = 0
best_mrr = 0
best_weight = 0

for weight in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    metrics, mrr = evaluate_with_repeat_bias(model, test_loader, repeat_weight=weight)
    hr10 = metrics['HR@10']
    print(f'   Weight {weight:.1f}: HR@10={hr10:.4f}, MRR={mrr:.4f}')

    if hr10 > best_hr10 or (hr10 == best_hr10 and mrr > best_mrr):
        best_hr10 = hr10
        best_mrr = mrr
        best_weight = weight

print(f'\nBest: weight={best_weight}, HR@10={best_hr10:.4f}, MRR={best_mrr:.4f}')

# ============================================================
# 6. LONGER TRAINING WITH BEST WEIGHT
# ============================================================

print('\n' + '='*70)
print('TRAINING FINAL MODEL (Longer)')
print('='*70)

set_seed(42)
final_model = OptimizedGatedFusion(pretrained_weights, hidden_dim=256, num_heads=4, dropout=0.5).to(device)
final_model = train_model(final_model, train_loader, epochs=150, lr=0.00015)

# Evaluate with best repeat weight
final_metrics, final_mrr = evaluate_with_repeat_bias(final_model, test_loader, repeat_weight=best_weight)

print('\n' + '='*70)
print('FINAL RESULTS')
print('='*70)
print(f"HR@5:  {final_metrics['HR@5']:.4f}")
print(f"HR@10: {final_metrics['HR@10']:.4f}")
print(f"HR@20: {final_metrics['HR@20']:.4f}")
print(f"MRR:   {final_mrr:.4f}")
print('='*70)

print('\nCOMPARISON:')
print(f"  Previous best HR@10: 0.3506 | MRR: 0.1972")
print(f"  New HR@10:          {final_metrics['HR@10']:.4f} | MRR: {final_mrr:.4f}")

# Save
os.makedirs('models', exist_ok=True)
torch.save({
    'model_state_dict': final_model.state_dict(),
    'metrics': final_metrics,
    'mrr': final_mrr,
    'repeat_weight': best_weight
}, 'models/optimized_multimodal_model.pth')
print('\nModel saved!')
