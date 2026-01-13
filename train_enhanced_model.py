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
import re
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

# ============================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================

df_seq = pd.read_csv('input/user_behavior.csv')
df_content = pd.read_csv('input/content.csv')
df_images = pd.read_csv('input/image_descriptions_fashion_structured.csv')

if isinstance(df_seq['posts_sequence'].iloc[0], str):
    df_seq['posts_sequence_list'] = df_seq['posts_sequence'].apply(ast.literal_eval)
else:
    df_seq['posts_sequence_list'] = df_seq['posts_sequence']

# Build vocabulary
all_post_ids = set()
for seq in df_seq['posts_sequence_list']:
    all_post_ids.update(seq)

sorted_ids = sorted(list(all_post_ids))
post2idx = {pid: i+1 for i, pid in enumerate(sorted_ids)}
idx2post = {i+1: pid for i, pid in enumerate(sorted_ids)}
VOCAB_SIZE = len(post2idx) + 1
print(f'Vocab Size: {VOCAB_SIZE}')

# Load multimodal embeddings
pretrained_weights = np.load('post_embeddings_multimodal.npy')
print(f'Embeddings shape: {pretrained_weights.shape}')

# ============================================================
# 2. COMPUTE ADDITIONAL FEATURES
# ============================================================

# Item popularity
item_counts = Counter()
for seq in df_seq['posts_sequence_list']:
    item_counts.update(seq)

popularity = np.zeros(VOCAB_SIZE)
max_count = max(item_counts.values())
for pid, count in item_counts.items():
    idx = post2idx[pid]
    popularity[idx] = count / max_count

print(f'Popularity computed for {len(item_counts)} items')

# Co-occurrence matrix (sparse representation)
co_occur = defaultdict(Counter)
for seq in df_seq['posts_sequence_list']:
    for i in range(len(seq)):
        for j in range(len(seq)):
            if i != j:
                idx_i = post2idx[seq[i]]
                idx_j = post2idx[seq[j]]
                co_occur[idx_i][idx_j] += 1

print(f'Co-occurrence computed')

# Fashion categories
def extract_categories(desc):
    cats = set()
    if pd.isna(desc):
        return cats
    desc = str(desc).lower()
    if 'outfit:' in desc:
        outfit = desc.split('outfit:')[1].split('|')[0]
        for item in outfit.split(','):
            item = item.strip()
            if item and len(item) > 2:
                cats.add(item)
    if 'accessories:' in desc:
        acc = desc.split('accessories:')[1].split('|')[0]
        for item in acc.split(','):
            item = item.strip()
            if item and len(item) > 2:
                cats.add(item)
    return cats

item_categories = defaultdict(set)
for _, row in df_images.iterrows():
    img_id = row['image_id']
    if img_id in post2idx:
        idx = post2idx[img_id]
        cats = extract_categories(row.get('fashion_description', ''))
        item_categories[idx] = cats

# Category to index
all_cats = set()
for cats in item_categories.values():
    all_cats.update(cats)
cat2idx = {cat: i for i, cat in enumerate(sorted(all_cats))}
NUM_CATS = len(cat2idx)
print(f'Fashion categories: {NUM_CATS}')

# Category features for each item
cat_features = np.zeros((VOCAB_SIZE, NUM_CATS))
for idx, cats in item_categories.items():
    for cat in cats:
        if cat in cat2idx:
            cat_features[idx, cat2idx[cat]] = 1

# ============================================================
# 3. DATA AUGMENTATION
# ============================================================

train_samples = []
test_samples = []

for idx, row in df_seq.iterrows():
    raw_seq = row['posts_sequence_list']
    seq = [post2idx.get(pid, 0) for pid in raw_seq]

    if len(seq) < 2:
        continue

    # Test: predict last item
    test_samples.append((seq[:-1], seq[-1]))

    # Training with augmentation
    train_seq_full = seq[:-1]

    # Original sliding window
    for i in range(1, len(train_seq_full)):
        start = max(0, i - 20)
        train_samples.append((train_seq_full[start:i], train_seq_full[i]))

    # Additional: random subsequences for longer sequences
    if len(train_seq_full) >= 4:
        for _ in range(2):
            # Random start and end
            start = random.randint(0, len(train_seq_full) - 3)
            end = random.randint(start + 2, len(train_seq_full))
            sub_seq = train_seq_full[start:end]
            if len(sub_seq) >= 2:
                train_samples.append((sub_seq[:-1], sub_seq[-1]))

print(f'Train samples: {len(train_samples)} (augmented)')
print(f'Test samples: {len(test_samples)}')

# ============================================================
# 4. DATASET WITH FEATURES
# ============================================================

class EnhancedDataset(Dataset):
    def __init__(self, samples, popularity, co_occur, is_train=True):
        self.samples = samples
        self.popularity = popularity
        self.co_occur = co_occur
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, target = self.samples[idx]

        # Compute repeat mask (which items in history match target)
        repeat_mask = [1 if item == target else 0 for item in seq]

        # Compute recency weights (more recent = higher weight)
        positions = list(range(1, len(seq) + 1))

        # Co-occurrence scores with target
        co_scores = [self.co_occur[item].get(target, 0) for item in seq]

        return {
            'seq': torch.tensor(seq, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long),
            'repeat_mask': torch.tensor(repeat_mask, dtype=torch.float),
            'positions': torch.tensor(positions, dtype=torch.float),
            'co_scores': torch.tensor(co_scores, dtype=torch.float)
        }

def collate_fn(batch):
    max_len = max(len(b['seq']) for b in batch)

    seqs = torch.zeros(len(batch), max_len, dtype=torch.long)
    repeat_masks = torch.zeros(len(batch), max_len)
    positions = torch.zeros(len(batch), max_len)
    co_scores = torch.zeros(len(batch), max_len)
    targets = torch.stack([b['target'] for b in batch])

    for i, b in enumerate(batch):
        length = len(b['seq'])
        seqs[i, :length] = b['seq']
        repeat_masks[i, :length] = b['repeat_mask']
        positions[i, :length] = b['positions']
        co_scores[i, :length] = b['co_scores']

    return {
        'seq': seqs,
        'target': targets,
        'repeat_mask': repeat_masks,
        'positions': positions,
        'co_scores': co_scores
    }

BATCH_SIZE = 32
train_loader = DataLoader(
    EnhancedDataset(train_samples, popularity, co_occur, is_train=True),
    batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
test_loader = DataLoader(
    EnhancedDataset(test_samples, popularity, co_occur, is_train=False),
    batch_size=BATCH_SIZE, collate_fn=collate_fn
)

# ============================================================
# 5. ENHANCED MODEL ARCHITECTURE
# ============================================================

class EnhancedMultimodalModel(nn.Module):
    """
    Enhanced model with:
    - Position-aware attention
    - Repeat/recency bias
    - Category features
    - Popularity features
    """
    def __init__(self, pretrained_weights, cat_features, popularity,
                 hidden_dim=256, num_heads=4, dropout=0.4):
        super(EnhancedMultimodalModel, self).__init__()

        weights_tensor = torch.FloatTensor(pretrained_weights)
        self.vocab_size = weights_tensor.shape[0]
        self.text_dim = 768
        self.image_dim = 768

        # Frozen embeddings
        self.embedding = nn.Embedding.from_pretrained(weights_tensor, freeze=True, padding_idx=0)

        # Category features (learnable projection)
        cat_tensor = torch.FloatTensor(cat_features)
        self.cat_features = nn.Parameter(cat_tensor, requires_grad=False)
        self.cat_proj = nn.Linear(cat_tensor.shape[1], hidden_dim // 4)

        # Popularity features
        pop_tensor = torch.FloatTensor(popularity).unsqueeze(1)
        self.popularity = nn.Parameter(pop_tensor, requires_grad=False)

        # Projections
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(self.image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15)
        )

        # Gated fusion
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        # Position encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, 50, hidden_dim) * 0.02)

        # BiLSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, 2, batch_first=True,
                           bidirectional=True, dropout=dropout)

        # Multi-Head Attention with position bias
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim*2, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim * 2)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2)
        )
        self.norm2 = nn.LayerNorm(hidden_dim * 2)

        # Attention pooling
        self.pool_attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.vocab_size)

        # Repeat bias (learnable weight for repeat prediction)
        self.repeat_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, batch):
        x = batch['seq']  # [B, L]
        positions = batch['positions']  # [B, L]

        B, L = x.shape

        # Get embeddings
        embedded = self.embedding(x)  # [B, L, 1536]
        text_emb = embedded[:, :, :self.text_dim]
        image_emb = embedded[:, :, self.text_dim:]

        # Project
        text_proj = self.text_proj(text_emb)  # [B, L, H]
        image_proj = self.image_proj(image_emb)

        # Gated fusion
        concat = torch.cat([text_proj, image_proj], dim=-1)
        gate = self.gate(concat)
        fused = gate * text_proj + (1 - gate) * image_proj  # [B, L, H]

        # Add position encoding
        if L <= 50:
            fused = fused + self.pos_encoding[:, :L, :]

        # BiLSTM
        lstm_out, _ = self.lstm(fused)  # [B, L, H*2]

        # MHA with residual
        attn_out, _ = self.mha(lstm_out, lstm_out, lstm_out)
        out = self.norm1(attn_out + lstm_out)

        # FFN with residual
        ffn_out = self.ffn(out)
        out = self.norm2(ffn_out + out)  # [B, L, H*2]

        # Attention pooling with position bias
        mask = (x != 0).float()  # [B, L]

        # Position-weighted attention (more weight to recent items)
        pos_weights = positions / (positions.max(dim=1, keepdim=True)[0] + 1e-9)  # Normalize

        attn_scores = self.pool_attn(out).squeeze(-1)  # [B, L]
        attn_scores = attn_scores + pos_weights * 2  # Add position bias
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # [B, L, 1]

        context = (attn_weights * out).sum(dim=1)  # [B, H*2]

        # Output
        hidden = F.gelu(self.fc1(self.dropout(context)))
        logits = self.fc2(hidden)  # [B, V]

        # Add popularity bias
        pop_bias = self.popularity.squeeze(-1) * 0.5  # [V]
        logits = logits + pop_bias

        return logits

    def predict_with_repeat(self, batch):
        """Predict with repeat bias for items in history"""
        logits = self.forward(batch)  # [B, V]

        x = batch['seq']
        B, L = x.shape

        # Add bonus for items in history (repeat bias)
        repeat_bonus = torch.zeros_like(logits)
        for b in range(B):
            for pos, item in enumerate(x[b]):
                if item > 0:
                    # More recent items get higher bonus
                    recency = (pos + 1) / L
                    repeat_bonus[b, item] += self.repeat_weight * recency * 3

        return logits + repeat_bonus

# ============================================================
# 6. TRAINING WITH IMPROVED LOSS
# ============================================================

def train_model(model, train_loader, epochs=100, lr=0.0002):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    model.train()
    print('Training Enhanced Model...')

    best_loss = float('inf')
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch['target'])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 10 == 0:
            print(f'   Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}')

    return model

def evaluate(model, test_loader, k_list=[5, 10, 20]):
    model.eval()
    hits = {k: 0 for k in k_list}
    mrr_sum = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # Use predict with repeat bias
            logits = model.predict_with_repeat(batch)

            _, topk = torch.topk(logits, max(k_list), dim=1)
            topk = topk.cpu().numpy()
            targets = batch['target'].cpu().numpy()

            for i, target in enumerate(targets):
                pred_ids = topk[i]

                for k in k_list:
                    if target in pred_ids[:k]:
                        hits[k] += 1

                if target in pred_ids:
                    rank = np.where(pred_ids == target)[0][0] + 1
                    mrr_sum += 1.0 / rank

                total += 1

    metrics = {f'HR@{k}': v/total for k, v in hits.items()}
    metrics['MRR'] = mrr_sum / total
    return metrics

# ============================================================
# 7. MAIN TRAINING
# ============================================================

print('\n' + '='*70)
print('TRAINING ENHANCED MULTIMODAL MODEL')
print('='*70)

model = EnhancedMultimodalModel(
    pretrained_weights, cat_features, popularity,
    hidden_dim=256, num_heads=4, dropout=0.4
).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total parameters: {total_params:,}')
print(f'Trainable parameters: {trainable_params:,}')

model = train_model(model, train_loader, epochs=150, lr=0.0002)
metrics = evaluate(model, test_loader)

print('\n' + '='*70)
print('ENHANCED MODEL RESULTS')
print('='*70)
print(f"HR@5:  {metrics['HR@5']:.4f}")
print(f"HR@10: {metrics['HR@10']:.4f}")
print(f"HR@20: {metrics['HR@20']:.4f}")
print(f"MRR:   {metrics['MRR']:.4f}")
print('='*70)

# Baselines
print('\nCOMPARISON:')
print(f"  Sequence-only HR@10:   0.2727 | MRR: 0.2060")
print(f"  Content-only HR@10:    0.3333 | MRR: 0.2009")
print(f"  Multimodal HR@10:      0.3506 | MRR: 0.1972")
print(f"  Enhanced HR@10:        {metrics['HR@10']:.4f} | MRR: {metrics['MRR']:.4f}")

improvement_hr = ((metrics['HR@10'] - 0.3506) / 0.3506) * 100
improvement_mrr = ((metrics['MRR'] - 0.1972) / 0.1972) * 100
print(f"\n  Improvement vs Multimodal:")
print(f"    HR@10: {improvement_hr:+.2f}%")
print(f"    MRR:   {improvement_mrr:+.2f}%")

# Save
os.makedirs('models', exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'metrics': metrics
}, 'models/enhanced_multimodal_model.pth')
print('\nModel saved!')
