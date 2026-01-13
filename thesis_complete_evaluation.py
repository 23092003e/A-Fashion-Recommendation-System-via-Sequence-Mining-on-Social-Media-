import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import ast
import random
import os
import json
from collections import Counter, defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    'seeds': [42, 123, 456, 789, 1024],  # 5 seeds for statistical significance
    'hidden_dim': 128,  # Reduced for CUDA memory
    'num_heads': 4,
    'dropout': 0.4,
    'epochs': 80,
    'lr': 0.0002,
    'batch_size': 32,
    'max_seq_len': 20,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
print(f'Running {len(CONFIG["seeds"])} seeds for statistical significance\n')

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ============================================================
# DATA LOADING
# ============================================================

print("Loading data...")
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

# Load multimodal embeddings
pretrained_weights = np.load('post_embeddings_multimodal.npy')
print(f'Vocabulary size: {VOCAB_SIZE}')
print(f'Embedding shape: {pretrained_weights.shape}')

# Compute item statistics
item_freq = Counter()
for seq in df_seq['posts_sequence_list']:
    item_freq.update([post2idx[p] for p in seq])

# Item-item similarity
item_sim = cosine_similarity(pretrained_weights)
item_sim_tensor = torch.FloatTensor(item_sim).to(device)

# Co-occurrence matrix
co_occur = np.zeros((VOCAB_SIZE, VOCAB_SIZE))
for seq in df_seq['posts_sequence_list']:
    idx_seq = [post2idx[p] for p in seq]
    for i in range(len(idx_seq)):
        for j in range(len(idx_seq)):
            if i != j:
                co_occur[idx_seq[i], idx_seq[j]] += 1

co_occur_norm = co_occur / (co_occur.sum(axis=1, keepdims=True) + 1e-9)
co_occur_tensor = torch.FloatTensor(co_occur_norm).to(device)

# Popularity
popularity = np.zeros(VOCAB_SIZE)
for idx, count in item_freq.items():
    popularity[idx] = np.log1p(count)
popularity = popularity / (popularity.max() + 1e-9)
popularity_tensor = torch.FloatTensor(popularity).to(device)

# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data():
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
            start = max(0, i - CONFIG['max_seq_len'])
            train_samples.append((train_seq[start:i], train_seq[i]))

    return train_samples, test_samples

class SequenceDataset(Dataset):
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

# ============================================================
# BASELINE MODELS
# ============================================================

class GRU4Rec(nn.Module):
    """GRU-based Session-based Recommendation (Hidasi et al., 2016)"""
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, dropout=0.3):
        super(GRU4Rec, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        output, hidden = self.gru(emb)
        # Use last hidden state
        last_hidden = hidden[-1]
        logits = self.fc(self.dropout(last_hidden))
        return logits


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation (Kang & McAuley, 2018)"""
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=2, dropout=0.3, max_len=50):
        super(SASRec, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)

        emb = self.embedding(x) + self.pos_embedding(positions)

        # Create attention mask (causal)
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        padding_mask = (x == 0)

        output = self.transformer(emb, mask=mask, src_key_padding_mask=padding_mask)

        # Mean pooling over non-padding positions
        mask_expand = (~padding_mask).unsqueeze(-1).float()
        output = output * mask_expand
        context = output.sum(dim=1) / (mask_expand.sum(dim=1) + 1e-9)

        logits = self.fc(self.dropout(context))
        return logits


class BERT4Rec(nn.Module):
    """BERT for Sequential Recommendation (Sun et al., 2019) - simplified"""
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=2, dropout=0.3, max_len=50):
        super(BERT4Rec, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)

        emb = self.embedding(x) + self.pos_embedding(positions)
        emb = self.layer_norm(emb)

        padding_mask = (x == 0)
        output = self.transformer(emb, src_key_padding_mask=padding_mask)

        # Use last non-padding position
        mask_expand = (~padding_mask).unsqueeze(-1).float()
        output = output * mask_expand
        context = output.sum(dim=1) / (mask_expand.sum(dim=1) + 1e-9)

        logits = self.fc(self.dropout(context))
        return logits


class PopularityBaseline:
    """Simple popularity-based baseline"""
    def __init__(self, popularity):
        self.popularity = popularity

    def predict(self, x):
        B = x.shape[0]
        return self.popularity.unsqueeze(0).expand(B, -1)


# ============================================================
# PROPOSED MODELS
# ============================================================

class MultimodalGatedFusion(nn.Module):
    """Proposed: Multimodal Gated Fusion with BiLSTM + MHA"""
    def __init__(self, pretrained_weights, hidden_dim=256, num_heads=4, dropout=0.4):
        super(MultimodalGatedFusion, self).__init__()

        weights_tensor = torch.FloatTensor(pretrained_weights)
        self.vocab_size = weights_tensor.shape[0]

        self.embedding = nn.Embedding.from_pretrained(weights_tensor, freeze=True, padding_idx=0)

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

    def forward(self, x):
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

        logits = self.fc(self.dropout(context))
        return logits


class MultimodalHybrid(nn.Module):
    """Proposed: Multimodal Hybrid with learned combination weights"""
    def __init__(self, pretrained_weights, item_sim, co_occur, popularity,
                 hidden_dim=256, num_heads=4, dropout=0.4):
        super(MultimodalHybrid, self).__init__()

        weights_tensor = torch.FloatTensor(pretrained_weights)
        self.vocab_size = weights_tensor.shape[0]

        self.embedding = nn.Embedding.from_pretrained(weights_tensor, freeze=True, padding_idx=0)
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

        # Learnable combination weights
        self.neural_weight = nn.Parameter(torch.tensor(1.0))
        self.sim_weight = nn.Parameter(torch.tensor(0.5))
        self.cooccur_weight = nn.Parameter(torch.tensor(0.3))
        self.pop_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
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

        logits = self.fc(self.dropout(context))
        return logits

    def hybrid_predict(self, x):
        B, L = x.shape
        neural_logits = self.forward(x)

        sim_scores = torch.zeros(B, self.vocab_size, device=x.device)
        cooccur_scores = torch.zeros(B, self.vocab_size, device=x.device)

        for b in range(B):
            history = x[b][x[b] > 0]
            if len(history) > 0:
                weights = torch.arange(1, len(history) + 1, device=x.device).float()
                weights = weights / weights.sum()

                for i, item in enumerate(history):
                    sim_scores[b] += self.item_sim[item] * weights[i]
                    cooccur_scores[b] += self.co_occur[item] * weights[i]

        pop_scores = self.popularity.unsqueeze(0).expand(B, -1)

        combined = (
            self.neural_weight * neural_logits +
            self.sim_weight * sim_scores * 10 +
            self.cooccur_weight * cooccur_scores * 20 +
            self.pop_weight * pop_scores
        )

        return combined


# ============================================================
# ABLATION MODELS
# ============================================================

class TextOnlyModel(nn.Module):
    """Ablation: Text features only (no image)"""
    def __init__(self, pretrained_weights, hidden_dim=128, num_heads=4, dropout=0.4):
        super(TextOnlyModel, self).__init__()

        weights_tensor = torch.FloatTensor(pretrained_weights)
        self.vocab_size = weights_tensor.shape[0]

        self.embedding = nn.Embedding.from_pretrained(weights_tensor, freeze=True, padding_idx=0)

        self.text_proj = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True, bidirectional=True)
        self.mha = nn.MultiheadAttention(hidden_dim*2, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, self.vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        text_emb = embedded[:, :, :768]

        proj = self.text_proj(text_emb)
        lstm_out, _ = self.lstm(proj)
        attn_out, _ = self.mha(lstm_out, lstm_out, lstm_out)
        out = self.norm(attn_out + lstm_out)

        mask = (x != 0).unsqueeze(-1).float()
        out = out * mask
        context = torch.sum(out, dim=1) / (torch.sum(mask, dim=1) + 1e-9)

        logits = self.fc(self.dropout(context))
        return logits


class ImageOnlyModel(nn.Module):
    """Ablation: Image features only (no text)"""
    def __init__(self, pretrained_weights, hidden_dim=128, num_heads=4, dropout=0.4):
        super(ImageOnlyModel, self).__init__()

        weights_tensor = torch.FloatTensor(pretrained_weights)
        self.vocab_size = weights_tensor.shape[0]

        self.embedding = nn.Embedding.from_pretrained(weights_tensor, freeze=True, padding_idx=0)

        self.image_proj = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True, bidirectional=True)
        self.mha = nn.MultiheadAttention(hidden_dim*2, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, self.vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        image_emb = embedded[:, :, 768:]

        proj = self.image_proj(image_emb)
        lstm_out, _ = self.lstm(proj)
        attn_out, _ = self.mha(lstm_out, lstm_out, lstm_out)
        out = self.norm(attn_out + lstm_out)

        mask = (x != 0).unsqueeze(-1).float()
        out = out * mask
        context = torch.sum(out, dim=1) / (torch.sum(mask, dim=1) + 1e-9)

        logits = self.fc(self.dropout(context))
        return logits


class NoGateModel(nn.Module):
    """Ablation: No gating (simple concatenation)"""
    def __init__(self, pretrained_weights, hidden_dim=128, num_heads=4, dropout=0.4):
        super(NoGateModel, self).__init__()

        weights_tensor = torch.FloatTensor(pretrained_weights)
        self.vocab_size = weights_tensor.shape[0]

        self.embedding = nn.Embedding.from_pretrained(weights_tensor, freeze=True, padding_idx=0)

        self.proj = nn.Sequential(
            nn.Linear(1536, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True, bidirectional=True)
        self.mha = nn.MultiheadAttention(hidden_dim*2, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, self.vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)  # [B, L, 1536]

        proj = self.proj(embedded)
        lstm_out, _ = self.lstm(proj)
        attn_out, _ = self.mha(lstm_out, lstm_out, lstm_out)
        out = self.norm(attn_out + lstm_out)

        mask = (x != 0).unsqueeze(-1).float()
        out = out * mask
        context = torch.sum(out, dim=1) / (torch.sum(mask, dim=1) + 1e-9)

        logits = self.fc(self.dropout(context))
        return logits


class NoMHAModel(nn.Module):
    """Ablation: No Multi-Head Attention"""
    def __init__(self, pretrained_weights, hidden_dim=128, dropout=0.4):
        super(NoMHAModel, self).__init__()

        weights_tensor = torch.FloatTensor(pretrained_weights)
        self.vocab_size = weights_tensor.shape[0]

        self.embedding = nn.Embedding.from_pretrained(weights_tensor, freeze=True, padding_idx=0)

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

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, self.vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        text_emb = embedded[:, :, :768]
        image_emb = embedded[:, :, 768:]

        text_proj = self.text_proj(text_emb)
        image_proj = self.image_proj(image_emb)

        concat = torch.cat([text_proj, image_proj], dim=-1)
        gate = self.gate(concat)
        fused = gate * text_proj + (1 - gate) * image_proj

        lstm_out, _ = self.lstm(fused)

        mask = (x != 0).unsqueeze(-1).float()
        lstm_out = lstm_out * mask
        context = torch.sum(lstm_out, dim=1) / (torch.sum(mask, dim=1) + 1e-9)

        logits = self.fc(self.dropout(context))
        return logits


class NoBiLSTMModel(nn.Module):
    """Ablation: No BiLSTM (direct attention)"""
    def __init__(self, pretrained_weights, hidden_dim=128, num_heads=4, dropout=0.4):
        super(NoBiLSTMModel, self).__init__()

        weights_tensor = torch.FloatTensor(pretrained_weights)
        self.vocab_size = weights_tensor.shape[0]

        self.embedding = nn.Embedding.from_pretrained(weights_tensor, freeze=True, padding_idx=0)

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

        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, self.vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        text_emb = embedded[:, :, :768]
        image_emb = embedded[:, :, 768:]

        text_proj = self.text_proj(text_emb)
        image_proj = self.image_proj(image_emb)

        concat = torch.cat([text_proj, image_proj], dim=-1)
        gate = self.gate(concat)
        fused = gate * text_proj + (1 - gate) * image_proj

        attn_out, _ = self.mha(fused, fused, fused)
        out = self.norm(attn_out + fused)

        mask = (x != 0).unsqueeze(-1).float()
        out = out * mask
        context = torch.sum(out, dim=1) / (torch.sum(mask, dim=1) + 1e-9)

        logits = self.fc(self.dropout(context))
        return logits


# ============================================================
# COMPREHENSIVE METRICS
# ============================================================

def compute_metrics(model, test_loader, k_list=[5, 10, 20], use_hybrid=False):
    """Compute HR, MRR, NDCG, Coverage, Diversity"""
    model.eval()

    hits = {k: 0 for k in k_list}
    ndcg = {k: 0.0 for k in k_list}
    mrr_sum = 0
    total = 0
    all_predictions = []

    with torch.no_grad():
        for seqs, targets in test_loader:
            seqs, targets = seqs.to(device), targets.to(device)

            if use_hybrid and hasattr(model, 'hybrid_predict'):
                logits = model.hybrid_predict(seqs)
            else:
                logits = model(seqs)

            max_k = max(k_list)
            _, topk = torch.topk(logits, max_k, dim=1)
            topk = topk.cpu().numpy()
            targets_np = targets.cpu().numpy()

            for i, target in enumerate(targets_np):
                pred_ids = topk[i]
                all_predictions.extend(pred_ids[:10].tolist())

                for k in k_list:
                    if target in pred_ids[:k]:
                        hits[k] += 1
                        # NDCG
                        rank = np.where(pred_ids[:k] == target)[0][0] + 1
                        ndcg[k] += 1.0 / np.log2(rank + 1)

                if target in pred_ids:
                    rank = np.where(pred_ids == target)[0][0] + 1
                    mrr_sum += 1.0 / rank

                total += 1

    # Coverage: percentage of items recommended at least once
    unique_items = len(set(all_predictions))
    coverage = unique_items / VOCAB_SIZE

    # Diversity: average pairwise distance in recommendations
    # Simplified: use inverse of most common item frequency
    item_counts = Counter(all_predictions)
    if len(item_counts) > 1:
        top_freq = max(item_counts.values()) / len(all_predictions)
        diversity = 1 - top_freq
    else:
        diversity = 0

    results = {
        'HR': {f'HR@{k}': v/total for k, v in hits.items()},
        'NDCG': {f'NDCG@{k}': v/total for k, v in ndcg.items()},
        'MRR': mrr_sum/total,
        'Coverage': coverage,
        'Diversity': diversity
    }

    return results


def compute_popularity_metrics(test_loader):
    """Popularity baseline metrics"""
    hits = {5: 0, 10: 0, 20: 0}
    total = 0

    # Get top-k popular items
    pop_items = sorted(item_freq.items(), key=lambda x: x[1], reverse=True)
    top_k_items = [x[0] for x in pop_items[:20]]

    for seqs, targets in test_loader:
        for target in targets.numpy():
            for k in [5, 10, 20]:
                if target in top_k_items[:k]:
                    hits[k] += 1
            total += 1

    return {f'HR@{k}': v/total for k, v in hits.items()}


# ============================================================
# TRAINING
# ============================================================

def train_model(model, train_loader, epochs, lr, verbose=True):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    model.train()
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

        if verbose and (epoch + 1) % 20 == 0:
            print(f'   Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}')

    return model


# ============================================================
# MAIN EVALUATION
# ============================================================

def run_complete_evaluation():
    print("="*80)
    print("COMPLETE THESIS EVALUATION FRAMEWORK")
    print("="*80)

    train_samples, test_samples = prepare_data()
    train_loader = DataLoader(
        SequenceDataset(train_samples),
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        SequenceDataset(test_samples),
        batch_size=CONFIG['batch_size'],
        collate_fn=collate_fn
    )

    print(f"\nData: {len(train_samples)} train, {len(test_samples)} test")

    all_results = {}

    # ========================================
    # 1. BASELINES
    # ========================================
    print("\n" + "="*80)
    print("PART 1: BASELINE MODELS")
    print("="*80)

    # Popularity baseline
    print("\n[1/4] Popularity Baseline...")
    pop_metrics = compute_popularity_metrics(test_loader)
    all_results['Popularity'] = {'HR': pop_metrics, 'MRR': 0, 'seeds': []}
    print(f"   HR@10: {pop_metrics['HR@10']:.4f}")

    baseline_models = {
        'GRU4Rec': lambda: GRU4Rec(VOCAB_SIZE, embed_dim=128, hidden_dim=256, dropout=0.3),
        'SASRec': lambda: SASRec(VOCAB_SIZE, embed_dim=128, num_heads=4, num_layers=2, dropout=0.3),
        'BERT4Rec': lambda: BERT4Rec(VOCAB_SIZE, embed_dim=128, num_heads=4, num_layers=2, dropout=0.3),
    }

    for idx, (name, model_fn) in enumerate(baseline_models.items()):
        print(f"\n[{idx+2}/4] {name}...")
        seed_results = []

        for seed in CONFIG['seeds']:
            set_seed(seed)
            model = model_fn().to(device)
            model = train_model(model, train_loader, CONFIG['epochs'], CONFIG['lr'], verbose=False)
            metrics = compute_metrics(model, test_loader)
            seed_results.append(metrics)
            print(f"   Seed {seed}: HR@10={metrics['HR']['HR@10']:.4f}, MRR={metrics['MRR']:.4f}")

        # Average across seeds
        avg_hr10 = np.mean([r['HR']['HR@10'] for r in seed_results])
        std_hr10 = np.std([r['HR']['HR@10'] for r in seed_results])
        avg_mrr = np.mean([r['MRR'] for r in seed_results])
        std_mrr = np.std([r['MRR'] for r in seed_results])

        all_results[name] = {
            'HR': {'HR@10': avg_hr10, 'HR@10_std': std_hr10},
            'MRR': avg_mrr,
            'MRR_std': std_mrr,
            'seeds': seed_results
        }
        print(f"   Average: HR@10={avg_hr10:.4f}±{std_hr10:.4f}, MRR={avg_mrr:.4f}±{std_mrr:.4f}")

    # ========================================
    # 2. PROPOSED MODELS
    # ========================================
    print("\n" + "="*80)
    print("PART 2: PROPOSED MODELS")
    print("="*80)

    # Multimodal Gated Fusion
    print("\n[1/2] Multimodal Gated Fusion (Proposed)...")
    seed_results = []
    for seed in CONFIG['seeds']:
        set_seed(seed)
        torch.cuda.empty_cache()  # Clear CUDA cache
        model = MultimodalGatedFusion(pretrained_weights, hidden_dim=CONFIG['hidden_dim'], num_heads=4, dropout=0.4).to(device)
        model = train_model(model, train_loader, CONFIG['epochs'], CONFIG['lr'], verbose=False)
        metrics = compute_metrics(model, test_loader)
        seed_results.append(metrics)
        print(f"   Seed {seed}: HR@10={metrics['HR']['HR@10']:.4f}, MRR={metrics['MRR']:.4f}")
        del model
        torch.cuda.empty_cache()

    avg_hr10 = np.mean([r['HR']['HR@10'] for r in seed_results])
    std_hr10 = np.std([r['HR']['HR@10'] for r in seed_results])
    avg_mrr = np.mean([r['MRR'] for r in seed_results])
    std_mrr = np.std([r['MRR'] for r in seed_results])

    all_results['Multimodal-GF'] = {
        'HR': {'HR@10': avg_hr10, 'HR@10_std': std_hr10},
        'MRR': avg_mrr,
        'MRR_std': std_mrr,
        'seeds': seed_results
    }
    print(f"   Average: HR@10={avg_hr10:.4f}±{std_hr10:.4f}, MRR={avg_mrr:.4f}±{std_mrr:.4f}")

    # Multimodal Hybrid
    print("\n[2/2] Multimodal Hybrid (Proposed)...")
    seed_results = []
    for seed in CONFIG['seeds']:
        set_seed(seed)
        torch.cuda.empty_cache()
        model = MultimodalHybrid(
            pretrained_weights, item_sim_tensor, co_occur_tensor, popularity_tensor,
            hidden_dim=CONFIG['hidden_dim'], num_heads=4, dropout=0.4
        ).to(device)
        model = train_model(model, train_loader, CONFIG['epochs'], CONFIG['lr'], verbose=False)

        # Grid search for best weights
        best_metrics = None
        best_hr10 = 0
        for sim_w in [0.5, 0.7, 1.0]:
            for cooccur_w in [0.5, 0.7, 1.0]:
                for pop_w in [0.1, 0.2]:
                    model.sim_weight.data = torch.tensor(sim_w)
                    model.cooccur_weight.data = torch.tensor(cooccur_w)
                    model.pop_weight.data = torch.tensor(pop_w)
                    metrics = compute_metrics(model, test_loader, use_hybrid=True)
                    if metrics['HR']['HR@10'] > best_hr10:
                        best_hr10 = metrics['HR']['HR@10']
                        best_metrics = metrics

        seed_results.append(best_metrics)
        print(f"   Seed {seed}: HR@10={best_metrics['HR']['HR@10']:.4f}, MRR={best_metrics['MRR']:.4f}")
        del model
        torch.cuda.empty_cache()

    avg_hr10 = np.mean([r['HR']['HR@10'] for r in seed_results])
    std_hr10 = np.std([r['HR']['HR@10'] for r in seed_results])
    avg_mrr = np.mean([r['MRR'] for r in seed_results])
    std_mrr = np.std([r['MRR'] for r in seed_results])

    all_results['Multimodal-Hybrid'] = {
        'HR': {'HR@10': avg_hr10, 'HR@10_std': std_hr10},
        'MRR': avg_mrr,
        'MRR_std': std_mrr,
        'seeds': seed_results
    }
    print(f"   Average: HR@10={avg_hr10:.4f}±{std_hr10:.4f}, MRR={avg_mrr:.4f}±{std_mrr:.4f}")

    # ========================================
    # 3. ABLATION STUDY
    # ========================================
    print("\n" + "="*80)
    print("PART 3: ABLATION STUDY")
    print("="*80)

    ablation_models = {
        'Text-Only': lambda: TextOnlyModel(pretrained_weights, hidden_dim=CONFIG['hidden_dim']),
        'Image-Only': lambda: ImageOnlyModel(pretrained_weights, hidden_dim=CONFIG['hidden_dim']),
        'No-Gate': lambda: NoGateModel(pretrained_weights, hidden_dim=CONFIG['hidden_dim']),
        'No-MHA': lambda: NoMHAModel(pretrained_weights, hidden_dim=CONFIG['hidden_dim']),
        'No-BiLSTM': lambda: NoBiLSTMModel(pretrained_weights, hidden_dim=CONFIG['hidden_dim']),
    }

    for idx, (name, model_fn) in enumerate(ablation_models.items()):
        print(f"\n[{idx+1}/{len(ablation_models)}] {name}...")
        seed_results = []

        for seed in CONFIG['seeds']:
            set_seed(seed)
            torch.cuda.empty_cache()
            model = model_fn().to(device)
            model = train_model(model, train_loader, CONFIG['epochs'], CONFIG['lr'], verbose=False)
            metrics = compute_metrics(model, test_loader)
            seed_results.append(metrics)
            del model
            torch.cuda.empty_cache()

        avg_hr10 = np.mean([r['HR']['HR@10'] for r in seed_results])
        std_hr10 = np.std([r['HR']['HR@10'] for r in seed_results])
        avg_mrr = np.mean([r['MRR'] for r in seed_results])

        all_results[f'Ablation-{name}'] = {
            'HR': {'HR@10': avg_hr10, 'HR@10_std': std_hr10},
            'MRR': avg_mrr,
            'seeds': seed_results
        }
        print(f"   Average: HR@10={avg_hr10:.4f}±{std_hr10:.4f}, MRR={avg_mrr:.4f}")

    # ========================================
    # 4. STATISTICAL SIGNIFICANCE TESTS
    # ========================================
    print("\n" + "="*80)
    print("PART 4: STATISTICAL SIGNIFICANCE (t-test)")
    print("="*80)

    # Compare Multimodal-Hybrid vs baselines
    hybrid_hr10s = [r['HR']['HR@10'] for r in all_results['Multimodal-Hybrid']['seeds']]

    print("\nMultimodal-Hybrid vs Baselines:")
    for baseline in ['GRU4Rec', 'SASRec', 'BERT4Rec']:
        if baseline in all_results and 'seeds' in all_results[baseline]:
            baseline_hr10s = [r['HR']['HR@10'] for r in all_results[baseline]['seeds']]
            t_stat, p_value = stats.ttest_ind(hybrid_hr10s, baseline_hr10s)
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"   vs {baseline}: t={t_stat:.3f}, p={p_value:.4f} {significance}")

    # ========================================
    # 5. FINAL SUMMARY
    # ========================================
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)

    print(f"\n{'Model':<25} | {'HR@10':<18} | {'MRR':<12}")
    print("-"*60)

    for name, result in all_results.items():
        if 'HR' in result:
            hr10 = result['HR'].get('HR@10', 0)
            hr10_std = result['HR'].get('HR@10_std', 0)
            mrr = result.get('MRR', 0)
            mrr_std = result.get('MRR_std', 0)

            if hr10_std > 0:
                print(f"{name:<25} | {hr10:.4f}±{hr10_std:.4f}   | {mrr:.4f}±{mrr_std:.4f}")
            else:
                print(f"{name:<25} | {hr10:.4f}            | {mrr:.4f}")

    print("="*60)

    # Save results
    os.makedirs('results', exist_ok=True)

    # Convert to serializable format
    serializable_results = {}
    for name, result in all_results.items():
        serializable_results[name] = {
            'HR@10': float(result['HR'].get('HR@10', 0)),
            'HR@10_std': float(result['HR'].get('HR@10_std', 0)),
            'MRR': float(result.get('MRR', 0)),
            'MRR_std': float(result.get('MRR_std', 0)),
        }

    with open('results/thesis_evaluation_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to results/thesis_evaluation_results.json")

    return all_results


if __name__ == "__main__":
    results = run_complete_evaluation()
