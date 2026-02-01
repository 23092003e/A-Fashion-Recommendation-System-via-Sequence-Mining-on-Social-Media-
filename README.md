# Fashion Recommendation System via Sequence Mining on Social Media

## 📋 Project Description

An intelligent fashion recommendation system that utilizes sequence mining techniques and deep learning to analyze social media data. This project combines advanced methods such as multimodal learning, collaborative filtering, and neural networks to provide personalized fashion recommendations.

## ✨ Key Features

- **Multimodal Learning**: Combines image and text information from social media
- **Sequence Mining**: Analyzes user behavior sequences for accurate recommendations
- **Deep Learning Models**: Utilizes state-of-the-art models (LSTM, Transformer, CNN)
- **Hybrid Collaborative Filtering**: Combines content-based and collaborative filtering approaches
- **Social Influence Analysis**: Analyzes social influence on fashion trends

## 🏗️ Project Structure

```
.
├── data/                          # Raw and processed data
├── demo/                          # Demos and examples
├── figures/                       # Charts and result visualizations
├── images/                        # Product images
├── input/                         # Input data
├── notebooks/                     # Jupyter notebooks for analysis
├── results/                       # Experiment results
├── src/                          # Main source code
├── train_enhanced_model.py       # Enhanced model training
├── train_hybrid_cf_model.py      # Hybrid collaborative filtering training
├── train_multimodal_final.py     # Multimodal model training
├── thesis_complete_evaluation.py # Comprehensive evaluation
├── best_thesis_model.pth         # Best trained model
├── post_embeddings_multimodal.npy    # Multimodal embeddings
└── post_embeddings_text_only.npy     # Text-only embeddings
```

## 🔧 System Requirements

### Main Dependencies

```
Python >= 3.8
PyTorch >= 1.9.0
NumPy >= 1.19.0
Pandas >= 1.2.0
Scikit-learn >= 0.24.0
Transformers >= 4.0.0
torchvision >= 0.10.0
```

### Installation

```bash
# Clone repository
git clone https://github.com/23092003e/A-Fashion-Recommendation-System-via-Sequence-Mining-on-Social-Media-.git
cd A-Fashion-Recommendation-System-via-Sequence-Mining-on-Social-Media-

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage Guide

### 1. Data Preparation

```python
# Place data in the data/ folder
# Format: CSV or JSON with fields:
# - user_id: User ID
# - post_id: Post ID
# - image_url: Image URL
# - text: Text description
# - timestamp: Post timestamp
# - likes, comments, shares: Interactions
```

### 2. Training Models

#### a. Multimodal Model

```bash
python train_multimodal_final.py --epochs 50 --batch_size 32 --learning_rate 0.001
```

#### b. Enhanced Model

```bash
python train_enhanced_model.py --data_path ./data --output_dir ./results
```

#### c. Hybrid Collaborative Filtering

```bash
python train_hybrid_cf_model.py --model_type hybrid --embedding_dim 128
```

### 3. Model Evaluation

```bash
python thesis_complete_evaluation.py --model_path ./best_thesis_model.pth
```

### 4. Using Pre-trained Model

```python
import torch
import numpy as np

# Load model
model = torch.load('best_thesis_model.pth')
model.eval()

# Load embeddings
multimodal_embeddings = np.load('post_embeddings_multimodal.npy')
text_embeddings = np.load('post_embeddings_text_only.npy')

# Inference
with torch.no_grad():
    recommendations = model.recommend(user_id, top_k=10)
```

## 📊 System Architecture

### 1. Data Processing Pipeline

```
Raw Data → Data Cleaning → Feature Extraction → Embedding Generation → Training
```

### 2. Model Architecture

- **Feature Extractors**:
  - Vision Encoder: CNN/ResNet for images
  - Text Encoder: BERT/Transformer for text
  - Sequence Encoder: LSTM/GRU for behavior sequences

- **Fusion Module**:
  - Multimodal Attention
  - Cross-modal Learning
  - Feature Concatenation/Addition

- **Recommendation Module**:
  - Collaborative Filtering
  - Content-based Filtering
  - Hybrid Ranking

### 3. Training Strategy

- Loss Function: Triplet Loss + Cross-Entropy
- Optimizer: Adam with learning rate scheduling
- Regularization: Dropout, L2 regularization
- Data Augmentation: Image transforms, text augmentation

## 📈 Experimental Results

### Evaluation Metrics

| Model | Precision@10 | Recall@10 | NDCG@10 | MRR |
|-------|-------------|-----------|---------|-----|
| Text-only | 0.45 | 0.38 | 0.52 | 0.41 |
| Image-only | 0.48 | 0.41 | 0.55 | 0.44 |
| **Multimodal** | **0.62** | **0.56** | **0.68** | **0.59** |
| Hybrid CF | 0.59 | 0.52 | 0.65 | 0.56 |

### Visualization

Detailed results and charts are available in the `figures/` and `results/` directories

## 🔬 Research Methodology

### 1. Sequence Mining

- Applied Sequential Pattern Mining algorithms to discover shopping patterns
- Used attention mechanisms to learn temporal dependencies
- Incorporated session-based recommendations

### 2. Social Media Analysis

- Crawled and analyzed data from Instagram, Pinterest
- Extracted fashion trends from influencer posts
- Sentiment analysis on comments and reviews

### 3. Personalization

- User profiling based on interaction history
- Style preference learning
- Size and fit recommendations

## 🛠️ Notebooks

The `notebooks/` directory contains Jupyter notebooks for:

- Exploratory Data Analysis (EDA)
- Feature Engineering experiments
- Model visualization
- Error analysis
- Case studies

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{fashion_recommendation_2024,
  title={A Fashion Recommendation System via Sequence Mining on Social Media},
  author={[Hoang Van Manh]},
  year={2025},
  publisher={23092003a},
  url={https://github.com/23092003e/A-Fashion-Recommendation-System-via-Sequence-Mining-on-Social-Media-}
}
```

## 🤝 Contributing

All contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is distributed under the [MIT License](LICENSE).

## 📧 Contact

- GitHub: [@23092003e](https://github.com/23092003e)
- Email: [hoangvanmanh2309@gmail.com]

## 🙏 Acknowledgments

- Thanks to the PyTorch and Hugging Face communities
- Dataset collected from public sources
- Referenced research on Fashion Recommendation Systems
- Used pre-trained models from ImageNet and BERT

## 📚 References

1. Fashion Recommendation Systems using Deep Learning
2. Sequential Pattern Mining for E-commerce
3. Multimodal Learning for Visual Question Answering
4. Social Influence in Fashion Industry
5. Attention Mechanisms in Neural Networks

## 🔄 Updates

- **v1.0.0** (2024): Initial release with multimodal model
- **v1.1.0**: Added hybrid collaborative filtering
- **v1.2.0**: Enhanced model with attention mechanism
- **Latest**: Improved evaluation metrics and visualization

---

**Note**: This project is under active development. Feedback and suggestions are highly appreciated!
