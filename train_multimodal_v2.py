import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import ast
import random
import os

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

# Load data
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
VOCAB_SIZE = len(post2idx) + 1

# Load embeddings
pretrained_weights = np.load('post_embeddings_multimodal.npy')
print(f'Embeddings shape: {pretrained_weights.shape}')

# Prepare samples
train_samples = []
test_samples = []
for idx, row in df_seq.iterrows():
    raw_seq = row['posts_sequence_list']
    seq = [post2idx.get(pid, 0) for pid in raw_seq]
    if len(seq) < 2:
        continue
    test_samples.append((seq[:-1], seq[-1]))
    train_seq_full = seq[:-1]
    for i in range(1, len(train_seq_full)):
        start = max(0, i - 20)
        train_samples.append((train_seq_full[start:i], train_seq_full[i]))

print(f'Train: {len(train_samples)}, Test: {len(test_samples)}')

class InteractionDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return (torch.tensor(self.samples[idx][0], dtype=torch.long),
                torch.tensor(self.samples[idx][1], dtype=torch.long))

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    return inputs_padded, torch.stack(targets)

BATCH_SIZE = 32
train_loader = DataLoader(InteractionDataset(train_samples), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(InteractionDataset(test_samples), batch_size=BATCH_SIZE, collate_fn=collate_fn)


class MultimodalCrossAttention_Model(nn.Module):
    """
    Advanced Multimodal Model with Cross-Attention between modalities
    This allows text and image features to attend to each other
    """
    def __init__(self, pretrained_weights, hidden_dim=128, num_layers=1, num_heads=4, dropout=0.5):
        super(MultimodalCrossAttention_Model, self).__init__()
        weights_tensor = torch.FloatTensor(pretrained_weights)
        vocab_size = weights_tensor.shape[0]
        self.embedding = nn.Embedding.from_pretrained(weights_tensor, freeze=True, padding_idx=0)
        self.text_dim = 768
        self.image_dim = 768

        # Separate projections with larger capacity
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.image_projection = nn.Sequential(
            nn.Linear(self.image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Cross-attention: text attends to image
        self.text_to_image_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        # Cross-attention: image attends to text
        self.image_to_text_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.cross_norm1 = nn.LayerNorm(hidden_dim)
        self.cross_norm2 = nn.LayerNorm(hidden_dim)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        # BiLSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

        # Self-attention on sequence
        self.self_mha = nn.MultiheadAttention(
            embed_dim=hidden_dim*2, num_heads=num_heads, dropout=dropout, batch_first=True
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

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        text_emb = embedded[:, :, :self.text_dim]
        image_emb = embedded[:, :, self.text_dim:]

        # Project
        text_proj = self.text_projection(text_emb)  # [B, L, H]
        image_proj = self.image_projection(image_emb)  # [B, L, H]

        # Cross-attention
        text_attended, _ = self.text_to_image_attn(text_proj, image_proj, image_proj)
        text_proj = self.cross_norm1(text_proj + text_attended)

        image_attended, _ = self.image_to_text_attn(image_proj, text_proj, text_proj)
        image_proj = self.cross_norm2(image_proj + image_attended)

        # Concatenate and fuse
        concat = torch.cat([text_proj, image_proj], dim=-1)
        fused = self.fusion(concat)  # [B, L, H]

        # BiLSTM
        lstm_out, _ = self.lstm(fused)  # [B, L, H*2]

        # Self-attention with residual
        attn_out, _ = self.self_mha(lstm_out, lstm_out, lstm_out)
        out = self.norm1(attn_out + lstm_out)

        # FFN with residual
        ffn_out = self.ffn(out)
        out = self.norm2(ffn_out + out)

        # Masked mean pooling
        mask = (x != 0).unsqueeze(-1).float()
        out = out * mask
        sum_out = torch.sum(out, dim=1)
        count = torch.sum(mask, dim=1)
        context = sum_out / (count + 1e-9)

        logits = self.fc(self.dropout(context))
        return logits


class MultimodalAttentionFusion_Model(nn.Module):
    """
    Multimodal Model with Attention-based Fusion
    Uses attention to dynamically weight modalities
    """
    def __init__(self, pretrained_weights, hidden_dim=128, num_layers=1, num_heads=4, dropout=0.5):
        super(MultimodalAttentionFusion_Model, self).__init__()
        weights_tensor = torch.FloatTensor(pretrained_weights)
        vocab_size = weights_tensor.shape[0]
        self.embedding = nn.Embedding.from_pretrained(weights_tensor, freeze=True, padding_idx=0)
        self.text_dim = 768
        self.image_dim = 768

        self.text_projection = nn.Sequential(
            nn.Linear(self.text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.image_projection = nn.Sequential(
            nn.Linear(self.image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Attention-based fusion
        self.modality_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),  # 2 modalities
            nn.Softmax(dim=-1)
        )

        # BiLSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

        # Multi-Head Attention
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim*2, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim * 2)
        self.norm2 = nn.LayerNorm(hidden_dim * 2)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2)
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        text_emb = embedded[:, :, :self.text_dim]
        image_emb = embedded[:, :, self.text_dim:]

        text_proj = self.text_projection(text_emb)  # [B, L, H]
        image_proj = self.image_projection(image_emb)  # [B, L, H]

        # Attention-based fusion
        concat = torch.cat([text_proj, image_proj], dim=-1)  # [B, L, H*2]
        weights = self.modality_attention(concat)  # [B, L, 2]

        # Weighted sum
        fused = weights[:, :, 0:1] * text_proj + weights[:, :, 1:2] * image_proj  # [B, L, H]

        # BiLSTM
        lstm_out, _ = self.lstm(fused)

        # MHA with residual
        attn_out, _ = self.mha(lstm_out, lstm_out, lstm_out)
        out = self.norm1(attn_out + lstm_out)

        # FFN with residual
        ffn_out = self.ffn(out)
        out = self.norm2(ffn_out + out)

        # Masked mean pooling
        mask = (x != 0).unsqueeze(-1).float()
        out = out * mask
        sum_out = torch.sum(out, dim=1)
        count = torch.sum(mask, dim=1)
        context = sum_out / (count + 1e-9)

        logits = self.fc(self.dropout(context))
        return logits


def train_model(model, train_loader, epochs=50, lr=0.001, name="Model"):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    model.train()

    print(f'Training {name}...')
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, topk = torch.topk(outputs, max(k_list), dim=1)
            topk = topk.cpu().numpy()

            for i, target in enumerate(targets):
                true_id = target.item()
                pred_ids = topk[i]
                for k in k_list:
                    if true_id in pred_ids[:k]:
                        hits[k] += 1
                if true_id in pred_ids:
                    rank = np.where(pred_ids == true_id)[0][0] + 1
                    mrr_sum += 1.0 / rank
                total += 1

    return {f'HR@{k}': v/total for k, v in hits.items()}, mrr_sum/total


# Hyperparameters
HIDDEN_DIM = 256
LAYERS = 1
HEADS = 4
DROPOUT = 0.5
EPOCHS = 100
LR = 0.00012

results = {}

# Model 1: Cross-Attention
print("="*60)
print("Model 1: Multimodal Cross-Attention")
print("="*60)
set_seed(42)
model1 = MultimodalCrossAttention_Model(
    pretrained_weights, hidden_dim=HIDDEN_DIM, num_layers=LAYERS,
    num_heads=HEADS, dropout=DROPOUT
).to(device)
model1 = train_model(model1, train_loader, epochs=EPOCHS, lr=LR, name="Cross-Attention")
metrics1, mrr1 = evaluate(model1, test_loader)
results['Cross-Attention'] = {**metrics1, 'MRR': mrr1}
print(f"\nResults: HR@5={metrics1['HR@5']:.4f}, HR@10={metrics1['HR@10']:.4f}, MRR={mrr1:.4f}")

# Model 2: Attention Fusion
print("\n" + "="*60)
print("Model 2: Multimodal Attention Fusion")
print("="*60)
set_seed(42)
model2 = MultimodalAttentionFusion_Model(
    pretrained_weights, hidden_dim=HIDDEN_DIM, num_layers=LAYERS,
    num_heads=HEADS, dropout=DROPOUT
).to(device)
model2 = train_model(model2, train_loader, epochs=EPOCHS, lr=LR, name="Attention Fusion")
metrics2, mrr2 = evaluate(model2, test_loader)
results['Attention Fusion'] = {**metrics2, 'MRR': mrr2}
print(f"\nResults: HR@5={metrics2['HR@5']:.4f}, HR@10={metrics2['HR@10']:.4f}, MRR={mrr2:.4f}")


# Summary
print("\n" + "="*70)
print("SUMMARY OF ALL MULTIMODAL MODELS")
print("="*70)
print(f"{'Model':<25} | {'HR@5':<10} | {'HR@10':<10} | {'MRR':<10}")
print("-"*70)

# Add baselines
baselines = {
    'Sequence Only (MHA)': {'HR@5': 0.2208, 'HR@10': 0.2727, 'MRR': 0.2060},
    'Content Only (MHA)': {'HR@5': 0.2597, 'HR@10': 0.3333, 'MRR': 0.2009},
    'Gated Fusion': {'HR@5': 0.2727, 'HR@10': 0.3506, 'MRR': 0.1985},
}

all_results = {**baselines, **results}

for name, m in all_results.items():
    print(f"{name:<25} | {m['HR@5']:.4f}     | {m['HR@10']:.4f}     | {m['MRR']:.4f}")
print("="*70)

# Find best model
best_hr10_model = max(all_results.keys(), key=lambda k: all_results[k]['HR@10'])
best_hr10 = all_results[best_hr10_model]['HR@10']
best_mrr = all_results[best_hr10_model]['MRR']

content_baseline = 0.3333
improvement = ((best_hr10 - content_baseline) / content_baseline) * 100

print(f"\nBest Model by HR@10: {best_hr10_model}")
print(f"  HR@10: {best_hr10:.4f}")
print(f"  MRR:   {best_mrr:.4f}")
print(f"\nImprovement vs Content-Only: {improvement:+.2f}%")

# Save best model
os.makedirs('models', exist_ok=True)
if 'Cross' in best_hr10_model:
    best_model = model1
elif 'Attention Fusion' in best_hr10_model:
    best_model = model2
else:
    best_model = None

if best_model is not None:
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'metrics': all_results[best_hr10_model]
    }, 'models/best_multimodal_model.pth')
    print(f'\nBest model saved to models/best_multimodal_model.pth')
