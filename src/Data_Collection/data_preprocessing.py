"""
Data Preprocessing Module for Fashion Recommender System
Shared utilities for loading and preparing data across all algorithms
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import ast
import random

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Configuration
WINDOW_SIZE = 10
EMBEDDING_DIM = 387
TEST_SIZE = 0.2


def load_raw_data(data_dir=''):
    """
    Load raw CSV files

    Returns:
        comments_df, posts_df, user_profile_df
    """
    comments_df = pd.read_csv(f'{data_dir}comments_df.csv')
    posts_df = pd.read_csv(f'{data_dir}posts_df.csv')
    user_profile_df = pd.read_csv(f'{data_dir}user_profile.csv')

    print(f"✓ Loaded data:")
    print(f"  Comments: {len(comments_df)} rows")
    print(f"  Posts: {len(posts_df)} rows")
    print(f"  Users: {len(user_profile_df)} rows")

    return comments_df, posts_df, user_profile_df


def create_user_sequences(comments_df, posts_df, min_interactions=3):
    """
    Create user interaction sequences

    Returns:
        df_sequence: DataFrame with columns [commentUser, posts_sequence, interaction_sequence]
    """
    # Merge comments with posts to get post features
    user_interactions = comments_df.merge(
        posts_df[['post_id', 'timestamp', 'polarity']],
        on='post_id',
        how='left'
    )

    # Sort by user and timestamp
    user_interactions = user_interactions.sort_values(['commentUser', 'timestamp'])

    def build_interaction_sequence(df):
        """Build sequence of interactions for a user"""
        return df[['post_id', 'timestamp', 'polarity']].to_dict('records')

    # Group by user and create sequences
    df_sequence = user_interactions.groupby('commentUser').apply(
        build_interaction_sequence,
        include_groups=False
    ).reset_index().rename(columns={0: 'interaction_sequence'})

    # Extract post_id sequences
    df_sequence["posts_sequence"] = df_sequence["interaction_sequence"].apply(
        lambda x: [item["post_id"] for item in x]
    )

    # Convert strings to lists if needed
    def safe_parse(x):
        if isinstance(x, list):
            return x
        try:
            return ast.literal_eval(x)
        except Exception:
            return []

    df_sequence['posts_sequence'] = df_sequence['posts_sequence'].apply(safe_parse)

    # Filter users with minimum interactions
    df_sequence = df_sequence[df_sequence['posts_sequence'].apply(lambda x: len(x) > min_interactions)]

    print(f"\n✓ Created sequences:")
    print(f"  Users with >{min_interactions} interactions: {len(df_sequence)}")
    print(f"  Avg sequence length: {df_sequence['posts_sequence'].apply(len).mean():.1f}")

    return df_sequence


def create_embeddings(df_sequence, posts_df):
    """
    Create embeddings for each post and map to sequences

    Returns:
        df_sequence_with_embeddings: DataFrame with embedding_sequence column
        embedding_dict: Dictionary mapping post_id to embedding
        post_ids: List of all unique post IDs
    """
    # Get unique posts from sequences
    all_posts_in_sequences = set()
    for seq in df_sequence['posts_sequence']:
        all_posts_in_sequences.update(seq)

    # Filter posts_df to only posts in sequences
    posts_in_sequences = posts_df[posts_df['post_id'].isin(all_posts_in_sequences)].copy()

    print(f"\n✓ Creating embeddings for {len(posts_in_sequences)} posts...")

    # Create embeddings (simplified version - you should use actual embeddings)
    # Here we'll use a placeholder that combines text and numerical features

    # Text embedding (would normally use sentence transformer)
    # For now, use random embeddings as placeholder
    text_embedding_dim = 384
    posts_in_sequences['text_embedding'] = posts_in_sequences.apply(
        lambda row: np.random.randn(text_embedding_dim).tolist(), axis=1
    )

    # Numerical features
    numerical_cols = ['likes', 'shares', 'views'] if all(col in posts_in_sequences.columns for col in ['likes', 'shares', 'views']) else []

    if numerical_cols:
        # Normalize numerical features
        for col in numerical_cols:
            posts_in_sequences[col] = (posts_in_sequences[col] - posts_in_sequences[col].mean()) / (posts_in_sequences[col].std() + 1e-8)
    else:
        # Create dummy numerical features
        for col in ['likes', 'shares', 'views']:
            posts_in_sequences[col] = 0.0

    # Combine text + numerical embeddings
    posts_in_sequences['full_embedding'] = posts_in_sequences.apply(
        lambda row: row['text_embedding'] + [row['likes'], row['shares'], row['views']],
        axis=1
    )

    # Create embedding dictionary
    embedding_dict = dict(zip(
        posts_in_sequences['post_id'],
        posts_in_sequences['full_embedding']
    ))

    # Map embeddings to sequences
    df_sequence['embedding_sequence'] = df_sequence['posts_sequence'].apply(
        lambda seq: [embedding_dict.get(post_id, [0]*EMBEDDING_DIM) for post_id in seq]
    )

    post_ids = list(posts_in_sequences['post_id'])

    print(f"✓ Created {len(embedding_dict)} embeddings of dimension {EMBEDDING_DIM}")

    return df_sequence, embedding_dict, post_ids


def augment_sequences(df_sequence, num_augments=2, noise_std=0.01, min_crop_len=5):
    """
    Augment sequences with noise and cropping

    Returns:
        df_sequence_augmented: Original + augmented sequences
    """
    augmented_rows = []

    for idx, row in df_sequence.iterrows():
        seq = np.array(row['embedding_sequence'])

        for aug_id in range(num_augments):
            # Method 1: Gaussian noise
            if aug_id == 0:
                noise = np.random.normal(0, noise_std, seq.shape)
                aug_seq = seq + noise
                aug_type = 'noise'

            # Method 2: Random crop
            elif aug_id == 1 and len(seq) > min_crop_len:
                start_idx = random.randint(0, len(seq) - min_crop_len)
                aug_seq = seq[start_idx:]
                aug_type = 'crop'
            else:
                continue

            augmented_rows.append({
                'commentUser': f"{row['commentUser']}_aug{aug_id}_{aug_type}",
                'posts_sequence': row['posts_sequence'],
                'embedding_sequence': aug_seq.tolist(),
                'augmented': True,
                'aug_method': aug_type
            })

    # Create DataFrame from augmented data
    df_aug = pd.DataFrame(augmented_rows)

    # Add columns to original
    df_sequence['augmented'] = False
    df_sequence['aug_method'] = 'original'

    # Combine
    df_sequence_augmented = pd.concat([df_sequence, df_aug], ignore_index=True)

    print(f"\n✓ Data augmentation:")
    print(f"  Original sequences: {len(df_sequence)}")
    print(f"  Augmented sequences: {len(df_aug)}")
    print(f"  Total: {len(df_sequence_augmented)}")
    print(f"  Augmentation ratio: {len(df_sequence_augmented)/len(df_sequence):.2f}x")

    return df_sequence_augmented


def create_training_samples(df_sequence_augmented, window_size=WINDOW_SIZE):
    """
    Create sliding window training samples

    Returns:
        X_all, y_all, y_all_ids: Input sequences, target embeddings, target post IDs
    """
    X_all, y_all, y_all_ids = [], [], []

    for idx, row in df_sequence_augmented.iterrows():
        seq = np.array(row['embedding_sequence'])
        posts = row['posts_sequence']

        # Skip too short sequences
        if len(seq) < 2:
            continue

        # Create sliding windows
        for i in range(1, len(seq)):
            start = max(0, i - window_size)
            X_all.append(seq[start:i])      # Input: previous posts
            y_all.append(seq[i])            # Target: next post embedding
            y_all_ids.append(posts[i])      # Target: next post ID

    print(f"\n✓ Created {len(X_all)} training samples from sequences")

    # Pad sequences to uniform length
    if len(X_all) > 0:
        X_all = pad_sequences(X_all, dtype='float32', padding='pre', maxlen=window_size)
        y_all = np.array(y_all, dtype='float32')
        y_all_ids = np.array(y_all_ids)

        print(f"  X shape: {X_all.shape} (samples, window_size, embedding_dim)")
        print(f"  y shape: {y_all.shape} (samples, embedding_dim)")
        print(f"  y_ids shape: {y_all_ids.shape}")
    else:
        print("  WARNING: No training samples created!")

    return X_all, y_all, y_all_ids


def split_data(X_all, y_all, y_all_ids, test_size=TEST_SIZE, random_state=SEED):
    """
    Split data into train/validation sets

    Returns:
        X_train, X_val, y_train, y_val, y_train_ids, y_val_ids
    """
    X_train, X_val, y_train, y_val, y_train_ids, y_val_ids = train_test_split(
        X_all, y_all, y_all_ids,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    print(f"\n✓ Train/Test split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Split ratio: {(1-test_size)*100:.0f}/{test_size*100:.0f}")

    return X_train, X_val, y_train, y_val, y_train_ids, y_val_ids


def prepare_full_pipeline(data_dir='', min_interactions=3, num_augments=2,
                          window_size=WINDOW_SIZE, test_size=TEST_SIZE):
    """
    Complete data preparation pipeline

    Returns:
        Dictionary with all prepared data
    """
    print("="*80)
    print("DATA PREPARATION PIPELINE")
    print("="*80)

    # Step 1: Load data
    comments_df, posts_df, user_profile_df = load_raw_data(data_dir)

    # Step 2: Create sequences
    df_sequence = create_user_sequences(comments_df, posts_df, min_interactions)

    # Step 3: Create embeddings
    df_sequence, embedding_dict, post_ids = create_embeddings(df_sequence, posts_df)

    # Step 4: Augment data
    df_sequence_augmented = augment_sequences(df_sequence, num_augments)

    # Step 5: Create training samples
    X_all, y_all, y_all_ids = create_training_samples(df_sequence_augmented, window_size)

    # Step 6: Split data
    X_train, X_val, y_train, y_val, y_train_ids, y_val_ids = split_data(
        X_all, y_all, y_all_ids, test_size, SEED
    )

    print("\n" + "="*80)
    print("DATA PREPARATION COMPLETE!")
    print("="*80)

    return {
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val': y_val,
        'y_train_ids': y_train_ids,
        'y_val_ids': y_val_ids,
        'embedding_dict': embedding_dict,
        'post_ids': post_ids,
        'df_sequence': df_sequence,
        'df_sequence_augmented': df_sequence_augmented,
        'posts_df': posts_df,
        'comments_df': comments_df,
        'user_profile_df': user_profile_df
    }


if __name__ == "__main__":
    # Test the pipeline
    data = prepare_full_pipeline()
    print(f"\n✓ Pipeline test successful!")
    print(f"  Keys available: {list(data.keys())}")