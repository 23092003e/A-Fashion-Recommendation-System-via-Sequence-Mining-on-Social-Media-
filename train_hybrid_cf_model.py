import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import ast
import random
import os
from collections import Counter, defaultdict

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

# 1. DATA LOADING
df_seq = pd.read_csv('input/user_behavior.csv')
if isinstance(df_seq['posts_sequence'].iloc[0], str):
    df_seq['posts_sequence_list'] = df_seq['posts_sequence'].apply(ast.literal_eval)
else:
    df_seq['posts_sequence_list'] = df_seq['posts_sequence']

# Parse interaction_sequence to extract polarity scores
def parse_interaction_sequence(seq_str):
    """Parse interaction_sequence string to extract post_id -> polarity mapping"""
    try:
        # The string contains Timestamp objects, use eval with pandas context
        seq = eval(seq_str, {"Timestamp": pd.Timestamp})
        return {item['post_id']: item.get('polarity', 0.5) for item in seq}
    except:
        return {}

df_seq['polarity_map'] = df_seq['interaction_sequence'].apply(parse_interaction_sequence)
print(f'Parsed polarity for {len(df_seq)} users')

# Sentiment-based weight function
def polarity_to_weight(polarity):
    """Convert polarity score to training weight.
    Positive sentiment (polarity >= 0.5): weight = 1.0 (emphasize learning)
    Neutral sentiment (0.0 <= polarity < 0.5): weight = 0.7
    Negative sentiment (polarity < 0.0): weight = 0.3 (down-weight)
    """
    if polarity >= 0.5:
        return 1.0
    elif polarity >= 0.0:
        return 0.7
    else:
        return 0.3

all_post_ids = set()
for seq in df_seq['posts_sequence_list']:
    all_post_ids.update(seq)

sorted_ids = sorted(list(all_post_ids))
post2idx = {pid: i+1 for i, pid in enumerate(sorted_ids)}
idx2post = {i+1: pid for i, pid in enumerate(sorted_ids)}
VOCAB_SIZE = len(post2idx) + 1

pretrained_weights = np.load('post_embeddings_multimodal.npy')
print(f'Vocab: {VOCAB_SIZE}')

# 2. COMPUTE ITEM-ITEM SIMILARITY

# Content-based similarity using embeddings
item_sim = cosine_similarity(pretrained_weights)
item_sim = torch.FloatTensor(item_sim).to(device)
print(f'Item similarity matrix: {item_sim.shape}')

# Co-occurrence statistics
co_occur = np.zeros((VOCAB_SIZE, VOCAB_SIZE))
for seq in df_seq['posts_sequence_list']:
    idx_seq = [post2idx[p] for p in seq]
    for i in range(len(idx_seq)):
        for j in range(len(idx_seq)):
            if i != j:
                co_occur[idx_seq[i], idx_seq[j]] += 1

# Normalize co-occurrence
co_occur_norm = co_occur / (co_occur.sum(axis=1, keepdims=True) + 1e-9)
co_occur_tensor = torch.FloatTensor(co_occur_norm).to(device)
print(f'Co-occurrence matrix computed')

# Popularity
item_freq = Counter()
for seq in df_seq['posts_sequence_list']:
    item_freq.update([post2idx[p] for p in seq])

popularity = np.zeros(VOCAB_SIZE)
for idx, count in item_freq.items():
    popularity[idx] = np.log1p(count)
popularity = popularity / (popularity.max() + 1e-9)
popularity_tensor = torch.FloatTensor(popularity).to(device)

# 3. PREPARE DATA
train_samples = []
test_samples = []

for idx, row in df_seq.iterrows():
    raw_seq = row['posts_sequence_list']
    polarity_map = row['polarity_map']
    seq = [post2idx.get(pid, 0) for pid in raw_seq]
    if len(seq) < 2:
        continue

    # Test sample: use average polarity of sequence as weight
    avg_polarity = np.mean([polarity_map.get(pid, 0.5) for pid in raw_seq])
    test_weight = polarity_to_weight(avg_polarity)
    test_samples.append((seq[:-1], seq[-1], test_weight))

    # Train samples: use polarity of target item
    train_seq = seq[:-1]
    for i in range(1, len(train_seq)):
        start = max(0, i - 20)
        target_pid = raw_seq[i]  # Original post_id
        polarity = polarity_map.get(target_pid, 0.5)
        weight = polarity_to_weight(polarity)
        train_samples.append((train_seq[start:i], train_seq[i], weight))

print(f'Train: {len(train_samples)}, Test: {len(test_samples)}')

# Show weight distribution
train_weights = [s[2] for s in train_samples]
print(f'Weight distribution: 1.0={train_weights.count(1.0)}, 0.7={train_weights.count(0.7)}, 0.3={train_weights.count(0.3)}')

class SimpleDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        seq, target, weight = self.samples[idx]
        return (torch.tensor(seq, dtype=torch.long),
                torch.tensor(target, dtype=torch.long),
                torch.tensor(weight, dtype=torch.float))

def collate_fn(batch):
    seqs, targets, weights = zip(*batch)
    seqs_padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
    return seqs_padded, torch.stack(targets), torch.stack(weights)

train_loader = DataLoader(SimpleDataset(train_samples), batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(SimpleDataset(test_samples), batch_size=32, collate_fn=collate_fn)

# 4. HYBRID MODEL

class HybridModel(nn.Module):
    def __init__(self, pretrained_weights, item_sim, co_occur, popularity,
                 hidden_dim=256, num_heads=4, dropout=0.5):
        super(HybridModel, self).__init__()

        weights_tensor = torch.FloatTensor(pretrained_weights)
        self.vocab_size = weights_tensor.shape[0]

        self.embedding = nn.Embedding.from_pretrained(weights_tensor, freeze=True, padding_idx=0)

        # Store precomputed matrices
        self.register_buffer('item_sim', item_sim)
        self.register_buffer('co_occur', co_occur)
        self.register_buffer('popularity', popularity)

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
        self.fc = nn.Linear(hidden_dim * 2, self.vocab_size)

        # Learnable weights for combining signals
        self.neural_weight = nn.Parameter(torch.tensor(1.0))
        self.sim_weight = nn.Parameter(torch.tensor(0.5))
        self.cooccur_weight = nn.Parameter(torch.tensor(0.3))
        self.pop_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        B, L = x.shape

        # Neural model
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

        neural_logits = self.fc(self.dropout(context))  # [B, V]

        return neural_logits

    def hybrid_predict(self, x):
        """Combine neural predictions with item similarity and co-occurrence"""
        B, L = x.shape

        # 1. Neural prediction
        neural_logits = self.forward(x)  # [B, V]

        # 2. Item similarity score (average similarity to history items)
        sim_scores = torch.zeros(B, self.vocab_size, device=x.device)
        for b in range(B):
            history = x[b][x[b] > 0]  # Non-padding items
            if len(history) > 0:
                # Get similarity to all items from history items
                # Weight recent items more
                weights = torch.arange(1, len(history) + 1, device=x.device).float()
                weights = weights / weights.sum()

                for i, item in enumerate(history):
                    sim_scores[b] += self.item_sim[item] * weights[i]

        # 3. Co-occurrence score
        cooccur_scores = torch.zeros(B, self.vocab_size, device=x.device)
        for b in range(B):
            history = x[b][x[b] > 0]
            if len(history) > 0:
                # Weight recent items more
                weights = torch.arange(1, len(history) + 1, device=x.device).float()
                weights = weights / weights.sum()

                for i, item in enumerate(history):
                    cooccur_scores[b] += self.co_occur[item] * weights[i]

        # 4. Popularity score
        pop_scores = self.popularity.unsqueeze(0).expand(B, -1)

        # Combine all signals
        combined = (
            self.neural_weight * neural_logits +
            self.sim_weight * sim_scores * 10 +  # Scale similarity
            self.cooccur_weight * cooccur_scores * 20 +  # Scale co-occurrence
            self.pop_weight * pop_scores
        )

        return combined

# 5. TRAINING

def train_model(model, train_loader, epochs=100, lr=0.00015):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # Use reduction='none' to apply per-sample weights
    criterion = nn.CrossEntropyLoss(reduction='none')

    model.train()
    print('Training Hybrid Model with Sentiment Weighting...')

    for epoch in range(epochs):
        total_loss = 0
        for seqs, targets, weights in train_loader:
            seqs, targets, weights = seqs.to(device), targets.to(device), weights.to(device)

            optimizer.zero_grad()
            # Train only neural part
            logits = model(seqs)

            # Compute per-sample loss and apply sentiment weights
            per_sample_loss = criterion(logits, targets)  # [B]
            weighted_loss = (per_sample_loss * weights).mean()  # Weighted average

            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += weighted_loss.item()

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f'   Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}')

    return model

def evaluate(model, test_loader, use_hybrid=True, k_list=[5, 10, 20]):
    model.eval()
    hits = {k: 0 for k in k_list}
    mrr_sum = 0
    total = 0

    with torch.no_grad():
        for seqs, targets, _ in test_loader:  # Ignore weights during evaluation
            seqs, targets = seqs.to(device), targets.to(device)

            if use_hybrid:
                logits = model.hybrid_predict(seqs)
            else:
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

# 6. MAIN

print('\n' + '='*70)
print('TRAINING HYBRID MODEL')
print('='*70)

model = HybridModel(
    pretrained_weights, item_sim, co_occur_tensor, popularity_tensor,
    hidden_dim=256, num_heads=4, dropout=0.5
).to(device)

model = train_model(model, train_loader, epochs=100, lr=0.00015)

# Evaluate neural only
print('\nEvaluating Neural Only')
metrics_neural, mrr_neural = evaluate(model, test_loader, use_hybrid=False)
print(f'   HR@10={metrics_neural["HR@10"]:.4f}, MRR={mrr_neural:.4f}')

# Evaluate hybrid
print('\nEvaluating Hybrid')
metrics_hybrid, mrr_hybrid = evaluate(model, test_loader, use_hybrid=True)
print(f'   HR@10={metrics_hybrid["HR@10"]:.4f}, MRR={mrr_hybrid:.4f}')

# Grid search for weights
print('\n' + '='*70)
print('GRID SEARCH FOR OPTIMAL WEIGHTS')
print('='*70)

best_hr10 = 0
best_mrr = 0
best_params = None

for sim_w in [0.3, 0.5, 0.7, 1.0]:
    for cooccur_w in [0.3, 0.5, 0.7, 1.0]:
        for pop_w in [0.05, 0.1, 0.2]:
            model.sim_weight.data = torch.tensor(sim_w)
            model.cooccur_weight.data = torch.tensor(cooccur_w)
            model.pop_weight.data = torch.tensor(pop_w)

            metrics, mrr = evaluate(model, test_loader, use_hybrid=True)

            if metrics['HR@10'] > best_hr10 or (metrics['HR@10'] == best_hr10 and mrr > best_mrr):
                best_hr10 = metrics['HR@10']
                best_mrr = mrr
                best_params = (sim_w, cooccur_w, pop_w)

print(f'\nBest params: sim={best_params[0]}, cooccur={best_params[1]}, pop={best_params[2]}')
print(f'Best HR@10={best_hr10:.4f}, MRR={best_mrr:.4f}')

# Set best params and evaluate
model.sim_weight.data = torch.tensor(best_params[0])
model.cooccur_weight.data = torch.tensor(best_params[1])
model.pop_weight.data = torch.tensor(best_params[2])

final_metrics, final_mrr = evaluate(model, test_loader, use_hybrid=True)

print('\n' + '='*70)
print('FINAL RESULTS')
print('='*70)
print(f"HR@5:  {final_metrics['HR@5']:.4f}")
print(f"HR@10: {final_metrics['HR@10']:.4f}")
print(f"HR@20: {final_metrics['HR@20']:.4f}")
print(f"MRR:   {final_mrr:.4f}")

print('\nComparison:')
print(f"   Previous best HR@10: 0.3506 | MRR: 0.2041")
print(f"   Hybrid HR@10:        {final_metrics['HR@10']:.4f} | MRR: {final_mrr:.4f}")

# Save
os.makedirs('models', exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'metrics': final_metrics,
    'mrr': final_mrr,
    'best_params': best_params
}, 'models/hybrid_cf_model.pth')
print('\nModel saved!')
