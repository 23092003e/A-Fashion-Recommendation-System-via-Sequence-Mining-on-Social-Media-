"""
Data Loader Utility for Fashion Recommendation Demo
Loads user behavior, content metadata, and embeddings
"""

import pandas as pd
import numpy as np
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class DataLoader:
    """Utility class to load and manage recommendation data"""

    def __init__(self, base_path: str = None):
        if base_path is None:
            base_path = Path(__file__).resolve().parent.parent.parent
        self.base_path = Path(base_path).resolve()

        self.user_behavior = None
        self.content = None
        self.embeddings = None
        self.fashion_descriptions = None
        self.post2idx = None
        self.idx2post = None

    def load_all_data(self) -> bool:
        """Load all required data files"""
        try:
            print(f"Base path: {self.base_path}")
            print(f"Base path exists: {self.base_path.exists()}")
            self._load_user_behavior()
            self._load_content()
            self._load_embeddings()
            self._load_fashion_descriptions()
            self._build_vocab()
            return True
        except Exception as e:
            import traceback
            print(f"Error loading data: {e}")
            traceback.print_exc()
            return False

    def _load_user_behavior(self):
        """Load user behavior/interaction sequences"""
        path = self.base_path / 'input' / 'user_behavior.csv'
        self.user_behavior = pd.read_csv(path)

        # Parse posts_sequence
        if isinstance(self.user_behavior['posts_sequence'].iloc[0], str):
            self.user_behavior['posts_sequence_list'] = self.user_behavior['posts_sequence'].apply(ast.literal_eval)
        else:
            self.user_behavior['posts_sequence_list'] = self.user_behavior['posts_sequence']

        # Parse interaction_sequence for polarity
        def parse_interaction(seq_str):
            try:
                seq = eval(seq_str, {"Timestamp": pd.Timestamp})
                return seq
            except:
                return []

        self.user_behavior['interaction_list'] = self.user_behavior['interaction_sequence'].apply(parse_interaction)
        print(f"Loaded {len(self.user_behavior)} users")

    def _load_content(self):
        """Load post content metadata"""
        path = self.base_path / 'input' / 'content.csv'
        self.content = pd.read_csv(path)
        self.content = self.content.set_index('post_id')
        print(f"Loaded {len(self.content)} posts")

    def _load_embeddings(self):
        """Load pre-computed multimodal embeddings"""
        path = self.base_path / 'post_embeddings_multimodal.npy'
        self.embeddings = np.load(path)
        print(f"Loaded embeddings: {self.embeddings.shape}")

    def _load_fashion_descriptions(self):
        """Load structured fashion descriptions"""
        path = self.base_path / 'input' / 'image_descriptions_fashion_structured.csv'
        if path.exists():
            self.fashion_descriptions = pd.read_csv(path)
            # Check for post_id or image_id column
            if 'post_id' in self.fashion_descriptions.columns:
                self.fashion_descriptions = self.fashion_descriptions.set_index('post_id')
            elif 'image_id' in self.fashion_descriptions.columns:
                self.fashion_descriptions = self.fashion_descriptions.set_index('image_id')
            print(f"Loaded {len(self.fashion_descriptions)} fashion descriptions")
        else:
            # Fallback to regular image descriptions
            path = self.base_path / 'input' / 'image_descriptions.csv'
            if path.exists():
                self.fashion_descriptions = pd.read_csv(path)
                if 'post_id' in self.fashion_descriptions.columns:
                    self.fashion_descriptions = self.fashion_descriptions.set_index('post_id')
                elif 'image_id' in self.fashion_descriptions.columns:
                    self.fashion_descriptions = self.fashion_descriptions.set_index('image_id')
                print(f"Loaded {len(self.fashion_descriptions)} image descriptions")

    def _build_vocab(self):
        """Build post ID to index mapping"""
        all_post_ids = set()
        for seq in self.user_behavior['posts_sequence_list']:
            all_post_ids.update(seq)

        sorted_ids = sorted(list(all_post_ids))
        self.post2idx = {pid: i+1 for i, pid in enumerate(sorted_ids)}
        self.idx2post = {i+1: pid for i, pid in enumerate(sorted_ids)}
        print(f"Vocabulary size: {len(self.post2idx) + 1}")

    def get_user_list(self) -> List[str]:
        """Get list of all users"""
        if self.user_behavior is None:
            return []
        return self.user_behavior['commentUser'].tolist()

    def get_user_sequence(self, username: str) -> Tuple[List[int], List[dict]]:
        """Get interaction sequence for a specific user"""
        if self.user_behavior is None:
            return [], []

        user_row = self.user_behavior[self.user_behavior['commentUser'] == username]
        if len(user_row) == 0:
            return [], []

        posts = user_row.iloc[0]['posts_sequence_list']
        interactions = user_row.iloc[0]['interaction_list']

        return posts, interactions

    def get_post_info(self, post_id: int) -> Dict:
        """Get metadata for a specific post"""
        if self.content is None:
            return {}

        try:
            row = self.content.loc[post_id]
            info = {
                'post_id': post_id,
                'caption': row.get('caption', ''),
                'hashtags': row.get('hashtags', ''),
                'postUser': row.get('postUser', ''),
                'likesCount': row.get('likesCount', 0),
                'commentsCount': row.get('commentsCount', 0),
                'timestamp': row.get('timestamp', ''),
                'mentions': row.get('mentions', ''),
                'image_description': row.get('image_description', '')
            }

            # Add fashion description if available
            if self.fashion_descriptions is not None and post_id in self.fashion_descriptions.index:
                fashion_row = self.fashion_descriptions.loc[post_id]
                info['fashion_description'] = fashion_row.get('fashion_description',
                                                               fashion_row.get('description', ''))

            return info
        except KeyError:
            return {'post_id': post_id, 'caption': 'N/A', 'error': 'Post not found'}

    def get_post_embedding(self, post_id: int) -> Optional[np.ndarray]:
        """Get embedding vector for a post"""
        if self.embeddings is None or self.post2idx is None:
            return None

        idx = self.post2idx.get(post_id, None)
        if idx is None or idx >= len(self.embeddings):
            return None

        return self.embeddings[idx]

    def compute_similarity(self, post_id1: int, post_id2: int) -> float:
        """Compute cosine similarity between two posts"""
        emb1 = self.get_post_embedding(post_id1)
        emb2 = self.get_post_embedding(post_id2)

        if emb1 is None or emb2 is None:
            return 0.0

        from numpy.linalg import norm
        return float(np.dot(emb1, emb2) / (norm(emb1) * norm(emb2) + 1e-9))

    def get_similar_posts(self, post_id: int, top_k: int = 5) -> List[Tuple[int, float]]:
        """Find most similar posts to a given post"""
        target_emb = self.get_post_embedding(post_id)
        if target_emb is None:
            return []

        from numpy.linalg import norm

        similarities = []
        for pid, idx in self.post2idx.items():
            if pid == post_id:
                continue
            emb = self.embeddings[idx]
            sim = float(np.dot(target_emb, emb) / (norm(target_emb) * norm(emb) + 1e-9))
            similarities.append((pid, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        stats = {
            'num_users': len(self.user_behavior) if self.user_behavior is not None else 0,
            'num_posts': len(self.content) if self.content is not None else 0,
            'embedding_dim': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'vocab_size': len(self.post2idx) + 1 if self.post2idx is not None else 0,
        }

        if self.user_behavior is not None:
            seq_lengths = self.user_behavior['sequence_length']
            stats['avg_seq_length'] = float(seq_lengths.mean())
            stats['max_seq_length'] = int(seq_lengths.max())
            stats['min_seq_length'] = int(seq_lengths.min())

        if self.content is not None:
            stats['avg_likes'] = float(self.content['likesCount'].mean())
            stats['avg_comments'] = float(self.content['commentsCount'].mean())

        return stats

    def extract_fashion_keywords(self, text: str) -> List[str]:
        """Extract fashion-related keywords from text"""
        if not text or pd.isna(text):
            return []

        text = str(text).lower()

        fashion_keywords = {
            'clothing': ['dress', 'gown', 'shirt', 'blouse', 'top', 'jacket', 'blazer',
                        'coat', 'sweater', 'hoodie', 'pants', 'jeans', 'shorts', 'skirt',
                        'suit', 'tuxedo', 'cardigan', 'vest', 'jumpsuit', 'romper'],
            'footwear': ['shoes', 'boots', 'sneakers', 'heels', 'sandals', 'flats',
                        'loafers', 'pumps', 'wedges', 'oxfords'],
            'accessories': ['bag', 'handbag', 'purse', 'clutch', 'hat', 'cap', 'scarf',
                           'belt', 'watch', 'bracelet', 'necklace', 'earrings', 'ring',
                           'sunglasses', 'glasses'],
            'colors': ['white', 'black', 'red', 'blue', 'green', 'yellow', 'pink',
                      'purple', 'orange', 'brown', 'beige', 'gold', 'silver', 'navy'],
            'styles': ['casual', 'formal', 'elegant', 'chic', 'vintage', 'modern',
                      'minimalist', 'bohemian', 'streetwear', 'luxury', 'sporty'],
            'patterns': ['floral', 'striped', 'plaid', 'checkered', 'polka dot',
                        'leopard', 'geometric', 'solid', 'printed']
        }

        found_keywords = []
        for category, keywords in fashion_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    found_keywords.append(keyword)

        return list(set(found_keywords))
