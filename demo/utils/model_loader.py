"""
Model Loader Utility for Fashion Recommendation Demo
Loads trained models and provides inference functionality
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class HybridModel(nn.Module):
    """Hybrid Recommendation Model: Neural + Item Similarity + Co-occurrence + Popularity"""

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

        neural_logits = self.fc(self.dropout(context))

        return neural_logits

    def hybrid_predict(self, x):
        """Combine neural predictions with item similarity and co-occurrence"""
        B, L = x.shape

        # 1. Neural prediction
        neural_logits = self.forward(x)

        # 2. Item similarity score
        sim_scores = torch.zeros(B, self.vocab_size, device=x.device)
        for b in range(B):
            history = x[b][x[b] > 0]
            if len(history) > 0:
                weights = torch.arange(1, len(history) + 1, device=x.device).float()
                weights = weights / weights.sum()

                for i, item in enumerate(history):
                    sim_scores[b] += self.item_sim[item] * weights[i]

        # 3. Co-occurrence score
        cooccur_scores = torch.zeros(B, self.vocab_size, device=x.device)
        for b in range(B):
            history = x[b][x[b] > 0]
            if len(history) > 0:
                weights = torch.arange(1, len(history) + 1, device=x.device).float()
                weights = weights / weights.sum()

                for i, item in enumerate(history):
                    cooccur_scores[b] += self.co_occur[item] * weights[i]

        # 4. Popularity score
        pop_scores = self.popularity.unsqueeze(0).expand(B, -1)

        # Combine all signals
        combined = (
            self.neural_weight * neural_logits +
            self.sim_weight * sim_scores * 10 +
            self.cooccur_weight * cooccur_scores * 20 +
            self.pop_weight * pop_scores
        )

        return combined

    def get_signal_contributions(self, x):
        """Get individual signal contributions for visualization"""
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

        return {
            'neural': neural_logits.detach(),
            'similarity': sim_scores.detach() * 10,
            'cooccurrence': cooccur_scores.detach() * 20,
            'popularity': pop_scores.detach(),
            'weights': {
                'neural': self.neural_weight.item(),
                'similarity': self.sim_weight.item(),
                'cooccurrence': self.cooccur_weight.item(),
                'popularity': self.pop_weight.item()
            }
        }


class ModelLoader:
    """Utility class to load and manage recommendation models"""

    def __init__(self, base_path: str = None):
        if base_path is None:
            base_path = Path(__file__).parent.parent.parent
        self.base_path = Path(base_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.post2idx = None
        self.idx2post = None

    def load_hybrid_model(self, embeddings: np.ndarray, post2idx: dict):
        """Load the hybrid recommendation model"""
        from sklearn.metrics.pairwise import cosine_similarity

        self.post2idx = post2idx
        self.idx2post = {v: k for k, v in post2idx.items()}
        vocab_size = len(post2idx) + 1

        # Compute item similarity
        item_sim = cosine_similarity(embeddings)
        item_sim = torch.FloatTensor(item_sim).to(self.device)

        # Initialize empty co-occurrence (will be loaded from checkpoint if available)
        co_occur = torch.zeros(vocab_size, vocab_size).to(self.device)
        popularity = torch.zeros(vocab_size).to(self.device)

        # Create model
        self.model = HybridModel(
            embeddings, item_sim, co_occur, popularity,
            hidden_dim=256, num_heads=4, dropout=0.5
        ).to(self.device)

        # Load checkpoint
        model_path = self.base_path / 'models' / 'hybrid_cf_model.pth'
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"Model loaded from {model_path}")
            print(f"Model metrics: HR@10={checkpoint.get('metrics', {}).get('HR@10', 'N/A')}")
        else:
            print(f"Warning: Model file not found at {model_path}")

        self.model.eval()
        return self.model

    def predict(self, sequence: list, top_k: int = 10, use_hybrid: bool = True):
        """Generate recommendations for a given sequence"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_hybrid_model first.")

        # Convert post IDs to indices
        idx_seq = [self.post2idx.get(pid, 0) for pid in sequence]

        # Pad sequence
        max_len = 20
        if len(idx_seq) > max_len:
            idx_seq = idx_seq[-max_len:]

        # Create tensor
        seq_tensor = torch.tensor([idx_seq], dtype=torch.long).to(self.device)

        with torch.no_grad():
            if use_hybrid:
                logits = self.model.hybrid_predict(seq_tensor)
            else:
                logits = self.model(seq_tensor)

            # Mask items already in sequence
            for idx in idx_seq:
                logits[0, idx] = float('-inf')

            # Get top-k
            scores, indices = torch.topk(logits[0], top_k)

            recommendations = []
            for score, idx in zip(scores.cpu().numpy(), indices.cpu().numpy()):
                post_id = self.idx2post.get(int(idx), None)
                if post_id is not None:
                    recommendations.append({
                        'post_id': post_id,
                        'score': float(score),
                        'idx': int(idx)
                    })

        return recommendations

    def get_signal_analysis(self, sequence: list, top_k: int = 10):
        """Get detailed signal analysis for recommendations"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_hybrid_model first.")

        idx_seq = [self.post2idx.get(pid, 0) for pid in sequence]
        max_len = 20
        if len(idx_seq) > max_len:
            idx_seq = idx_seq[-max_len:]

        seq_tensor = torch.tensor([idx_seq], dtype=torch.long).to(self.device)

        with torch.no_grad():
            signals = self.model.get_signal_contributions(seq_tensor)

            # Get combined scores
            combined = (
                signals['weights']['neural'] * signals['neural'] +
                signals['weights']['similarity'] * signals['similarity'] +
                signals['weights']['cooccurrence'] * signals['cooccurrence'] +
                signals['weights']['popularity'] * signals['popularity']
            )

            # Mask items in sequence
            for idx in idx_seq:
                combined[0, idx] = float('-inf')

            _, top_indices = torch.topk(combined[0], top_k)

            analysis = []
            for idx in top_indices.cpu().numpy():
                post_id = self.idx2post.get(int(idx), None)
                if post_id is not None:
                    analysis.append({
                        'post_id': post_id,
                        'neural_score': float(signals['neural'][0, idx]),
                        'similarity_score': float(signals['similarity'][0, idx]),
                        'cooccurrence_score': float(signals['cooccurrence'][0, idx]),
                        'popularity_score': float(signals['popularity'][0, idx]),
                        'combined_score': float(combined[0, idx])
                    })

        return analysis, signals['weights']
