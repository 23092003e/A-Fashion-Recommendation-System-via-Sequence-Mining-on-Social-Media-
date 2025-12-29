# Fashion Recommendation System Demo

Interactive Streamlit demo showcasing the Hybrid Neural + Collaborative Filtering recommendation system.

## Features

### 1. 🎯 Personalized Recommendations
- Select any user from the dataset (231 users)
- View interaction history with sentiment analysis
- Get top-K personalized recommendations
- See confidence scores for each recommendation

### 2. 📊 Signal Analysis
- Visualize contribution weights (Neural, Similarity, Co-occurrence, Popularity)
- Stacked bar charts showing score breakdown
- Understand why each item was recommended

### 3. 📈 Model Comparison
- Compare performance across different models:
  - GRU4Rec (baseline)
  - SASRec
  - BERT4Rec
  - Gated Fusion V2
  - Hybrid CF (Our best model)
- View HR@5, HR@10, HR@20, and MRR metrics

### 4. 🔍 Similar Items Explorer
- Select any post to find similar items
- Based on multimodal embeddings (text + image)
- Useful for content-based exploration

### 5. 📉 Dataset Statistics
- User and post counts
- Sequence length distribution
- Top influencers by engagement
- Embedding dimensions

## Quick Start

### 1. Install Dependencies

```bash
cd demo
pip install -r requirements.txt
```

### 2. Run the Demo

```bash
streamlit run app.py
```

### 3. Open in Browser

The app will automatically open at `http://localhost:8501`

## Project Structure

```
demo/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── utils/
    ├── __init__.py
    ├── model_loader.py # Model loading and inference
    └── data_loader.py  # Data loading utilities
```

## Required Data Files

The demo expects these files in the parent directory:

- `input/user_behavior.csv` - User interaction sequences
- `input/content.csv` - Post metadata
- `input/image_descriptions_fashion_structured.csv` - Fashion descriptions
- `post_embeddings_multimodal.npy` - Pre-computed embeddings
- `models/hybrid_cf_model.pth` - Trained model weights

## Model Performance

| Model | HR@10 | MRR |
|-------|-------|-----|
| GRU4Rec | 0.0381 | 0.0165 |
| SASRec | 0.2121 | 0.1129 |
| BERT4Rec | 0.2381 | 0.1529 |
| Gated Fusion V2 | 0.3177 | 0.1875 |
| **Hybrid CF (Ours)** | **0.6416** | **0.3561** |

## Screenshots

The demo includes:
- User profile with interaction timeline
- Sentiment trend visualization
- Recommendation cards with fashion tags
- Signal contribution pie charts
- Interactive model comparison charts

## Customization

### Adjust Number of Recommendations
Use the sidebar slider to change top-K (3-20)

### Toggle Features
- Show/hide post details
- Enable/disable signal analysis
- Switch between hybrid and neural-only mode

## Technical Details

### Architecture
- **BiLSTM**: Sequence encoding
- **Multi-Head Attention**: 4 heads
- **Gated Fusion**: Text + Image modality combination
- **Hybrid Signals**: Neural + Item Similarity + Co-occurrence + Popularity

### Embeddings
- Dimension: 1536 (768 text + 768 image)
- Text: BERT embeddings from captions + hashtags
- Image: BERT embeddings from image descriptions

## Thesis Project

This demo is part of the Fashion Marketing Automation Solutions thesis project, demonstrating state-of-the-art recommendation performance on Instagram fashion content.
