# 3.3 Textual Feature Extraction

## 3.3.1 Overview

Social media fashion posts contain rich textual information that complements visual content. In the context of Instagram fashion influencers, textual data encompasses three distinct modalities: captions, hashtags, and mentions. Each modality carries unique semantic signals that contribute to understanding user preferences and content characteristics.

This section presents our methodology for extracting dense vector representations from textual data using BERT (Bidirectional Encoder Representations from Transformers). The extracted textual features are subsequently combined with visual features to form comprehensive multimodal embeddings.

## 3.3.2 Textual Data Modalities

### Caption Text

Captions represent free-form natural language descriptions written by content creators. In fashion contexts, captions typically contain:

- **Outfit descriptions**: Details about clothing items, brands, and styling choices
- **Context information**: Event type, location, occasion
- **Personal narratives**: Stories, experiences, and lifestyle content
- **Call-to-actions**: Engagement prompts and audience interactions

### Hashtags

Hashtags serve as categorical labels and discovery mechanisms on social media platforms. Fashion-related hashtags encode:

- **Style categories**: #streetstyle, #casualwear, #formalattire
- **Brand associations**: #chanel, #dior, #zara
- **Trend indicators**: #ootd (outfit of the day), #fashionweek
- **Community markers**: #fashionblogger, #styleinspo

### Mentions

Mentions (@username) establish connections between content creators and other entities:

- **Brand tags**: Official brand accounts for clothing items
- **Collaborator credits**: Photographers, stylists, makeup artists
- **Location tags**: Venues, events, boutiques
- **Community engagement**: Other influencers and followers

## 3.3.3 Text Preprocessing Pipeline

Prior to feature extraction, raw textual data undergoes systematic preprocessing to ensure consistency and quality.

### Caption Preprocessing

The caption preprocessing pipeline consists of the following steps:

1. **URL Removal**: Eliminate hyperlinks that do not contribute semantic value
$$\text{caption} = \text{regex\_sub}(r'\text{http}\backslash S+', '', \text{caption})$$

2. **Hashtag Extraction**: Separate hashtags from caption body for independent processing
$$\text{hashtags} = \text{regex\_findall}(r'\#(\backslash w+)', \text{caption})$$

3. **Mention Extraction**: Extract mentioned usernames
$$\text{mentions} = \text{regex\_findall}(r'@(\backslash w+)', \text{caption})$$

4. **Text Normalization**: Convert to lowercase and remove special characters
$$\text{clean\_caption} = \text{normalize}(\text{caption})$$

5. **Emoji Removal**: Remove emoji characters that may interfere with tokenization

### Preprocessing Statistics

Table 3.3 summarizes the textual data characteristics after preprocessing:

**Table 3.3: Textual Data Statistics**

| Modality | Total Count | Avg. Length (tokens) | Unique Values |
|----------|-------------|---------------------|---------------|
| Captions | 822 | 47.3 | 819 |
| Hashtags | 822 | 2.8 | 1,247 |
| Mentions | 822 | 3.1 | 2,156 |

## 3.3.4 BERT Model Architecture

### Model Selection

We employ BERT-base-uncased as our text encoder based on the following considerations:

1. **Bidirectional Context**: Unlike unidirectional models, BERT captures context from both left and right directions, enabling richer semantic understanding.

2. **Pre-trained Knowledge**: BERT was pre-trained on BookCorpus (800M words) and English Wikipedia (2,500M words), providing robust language understanding capabilities.

3. **Transfer Learning**: The pre-trained representations transfer effectively to domain-specific tasks without extensive fine-tuning.

### Architecture Details

BERT-base consists of 12 Transformer encoder layers with the following specifications:

| Component | Specification |
|-----------|---------------|
| Hidden size ($d_h$) | 768 |
| Attention heads | 12 |
| Encoder layers | 12 |
| Total parameters | 110M |
| Vocabulary size | 30,522 |
| Max sequence length | 512 |

### Embedding Extraction

For each input text sequence $X = [x_1, x_2, ..., x_n]$, BERT produces contextualized embeddings for all tokens. We extract the [CLS] token embedding as the sequence-level representation:

$$\mathbf{h} = \text{BERT}(X) \in \mathbb{R}^{(n+2) \times d_h}$$

$$\mathbf{e}_{\text{text}} = \mathbf{h}_{[\text{CLS}]} \in \mathbb{R}^{768}$$

where $\mathbf{h}_{[\text{CLS}]}$ denotes the hidden state corresponding to the [CLS] token, which aggregates sequence-level semantic information through the self-attention mechanism.

The [CLS] token embedding is chosen over alternatives (e.g., mean pooling) because:

1. It is specifically designed during pre-training to capture sequence-level semantics
2. It provides a fixed-dimensional representation regardless of input length
3. It has been empirically validated across numerous downstream tasks

## 3.3.5 Feature Extraction Process

### Encoding Pipeline

The textual feature extraction process is formalized as follows:

**Input**: Raw text string $T$ (caption, hashtag sequence, or mention sequence)

**Output**: Dense embedding vector $\mathbf{e} \in \mathbb{R}^{768}$

**Process**:

1. **Tokenization**: Convert text to subword tokens using WordPiece tokenizer
$$\text{tokens} = \text{Tokenize}(T) = [[\text{CLS}], t_1, t_2, ..., t_n, [\text{SEP}]]$$

2. **Padding/Truncation**: Ensure uniform sequence length
$$\text{tokens}_{\text{padded}} = \text{Pad}(\text{tokens}, \text{max\_length}=128)$$

3. **Encoding**: Pass through BERT encoder
$$\mathbf{H} = \text{BERT\_Encoder}(\text{tokens}_{\text{padded}})$$

4. **Extraction**: Extract [CLS] embedding
$$\mathbf{e} = \mathbf{H}[0, :] \in \mathbb{R}^{768}$$

### Batch Processing

To optimize computational efficiency, texts are processed in batches:

$$\mathbf{E}_{\text{batch}} = \text{BERT\_Batch}([T_1, T_2, ..., T_B]) \in \mathbb{R}^{B \times 768}$$

where $B$ denotes the batch size (set to 32 in our implementation).

### Handling Missing Values

For posts with missing textual data (e.g., no hashtags or mentions), we employ a placeholder strategy:

$$\mathbf{e}_{\text{missing}} = \mathbf{e}_{[\text{PAD}]}$$

where $\mathbf{e}_{[\text{PAD}]}$ represents the embedding of the [PAD] token, providing a neutral representation that does not introduce bias.

## 3.3.6 Multi-Modal Text Encoding

Each fashion post yields three distinct textual embeddings:

### Caption Embedding

$$\mathbf{e}_{\text{caption}} = \text{BERT}_{\text{CLS}}(\text{clean\_caption}) \in \mathbb{R}^{768}$$

Captures the semantic content of the post's narrative, including outfit descriptions, context, and creator's voice.

### Hashtag Embedding

$$\mathbf{e}_{\text{hashtag}} = \text{BERT}_{\text{CLS}}(\text{hashtag\_sequence}) \in \mathbb{R}^{768}$$

where hashtag\_sequence is the space-separated concatenation of all hashtags (without # symbol).

Encodes categorical and trend information associated with the post.

### Mention Embedding

$$\mathbf{e}_{\text{mention}} = \text{BERT}_{\text{CLS}}(\text{mention\_sequence}) \in \mathbb{R}^{768}$$

where mention\_sequence is the space-separated concatenation of all mentions (without @ symbol).

Captures brand associations and collaborative relationships.

## 3.3.7 Encoding Configuration

Table 3.4 presents the hyperparameters used for textual feature extraction:

**Table 3.4: BERT Encoding Hyperparameters**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | bert-base-uncased | Balance between performance and efficiency |
| Max sequence length | 128 | Sufficient for social media text |
| Batch size | 32 | Optimized for GPU memory |
| Pooling strategy | [CLS] token | Standard for sequence classification |
| Output dimension | 768 | BERT-base hidden size |

## 3.3.8 Feature Extraction Pipeline

The complete textual feature extraction pipeline is illustrated in Figure 3.3:

```
┌─────────────────────────────────────────────────────────────────────┐
│                 TEXTUAL FEATURE EXTRACTION PIPELINE                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Raw Caption                                                       │
│   "Cannes 2023 with @kilianparis #KilianCannes Wearing @mrselfport" │
│                              │                                      │
│                              ▼                                      │
│                    ┌─────────────────┐                              │
│                    │  Preprocessing  │                              │
│                    └────────┬────────┘                              │
│                             │                                       │
│              ┌──────────────┼──────────────┐                        │
│              ▼              ▼              ▼                        │
│    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │
│    │   Caption   │  │  Hashtags   │  │  Mentions   │               │
│    │ "cannes     │  │"kiliancannes│  │"kilianparis │               │
│    │  2023 with  │  │            "│  │mrselfport"  │               │
│    │  wearing"   │  │             │  │             │               │
│    └──────┬──────┘  └──────┬──────┘  └──────┬──────┘               │
│           │                │                │                       │
│           ▼                ▼                ▼                       │
│    ┌─────────────────────────────────────────────┐                 │
│    │              BERT Encoder                   │                 │
│    │         (bert-base-uncased)                 │                 │
│    │                                             │                 │
│    │  [CLS] Token Embedding Extraction           │                 │
│    └─────────────────────────────────────────────┘                 │
│           │                │                │                       │
│           ▼                ▼                ▼                       │
│    ┌───────────┐    ┌───────────┐    ┌───────────┐                 │
│    │ e_caption │    │ e_hashtag │    │ e_mention │                 │
│    │  (768-d)  │    │  (768-d)  │    │  (768-d)  │                 │
│    └───────────┘    └───────────┘    └───────────┘                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                Figure 3.3: Textual Feature Extraction Pipeline
```

## 3.3.9 Output Specifications

The textual feature extraction process produces the following outputs:

**Table 3.5: Textual Feature Output Files**

| Output File | Shape | Description |
|-------------|-------|-------------|
| caption_embeddings.npy | (822, 768) | Caption embeddings |
| hashtag_embeddings.npy | (822, 768) | Hashtag embeddings |
| mention_embeddings.npy | (822, 768) | Mention embeddings |

### Embedding Statistics

Table 3.6 presents statistical properties of the extracted embeddings:

**Table 3.6: Embedding Statistics**

| Modality | Mean | Std Dev | Min | Max |
|----------|------|---------|-----|-----|
| Caption | 0.0012 | 0.3847 | -2.156 | 2.089 |
| Hashtag | 0.0008 | 0.4012 | -2.234 | 2.178 |
| Mention | 0.0015 | 0.3956 | -2.189 | 2.145 |

## 3.3.10 Semantic Properties

The BERT-based textual embeddings exhibit desirable semantic properties:

### Semantic Similarity

Posts with similar textual content produce embeddings with high cosine similarity:

$$\text{sim}(\mathbf{e}_i, \mathbf{e}_j) = \frac{\mathbf{e}_i \cdot \mathbf{e}_j}{||\mathbf{e}_i|| \cdot ||\mathbf{e}_j||}$$

### Clustering Behavior

Embeddings naturally cluster according to:
- **Topic**: Fashion events, daily outfits, brand collaborations
- **Style**: Casual, formal, streetwear, haute couture
- **Sentiment**: Positive, neutral, promotional

### Cross-Modal Alignment

Since visual features (Section 3.2) are also encoded through BERT, textual and visual embeddings reside in a shared semantic space, facilitating multimodal fusion.

## 3.3.11 Summary

This section presented our methodology for textual feature extraction from social media fashion posts. The key aspects of our approach include:

1. **Multi-Modal Text Processing**: We separately encode captions, hashtags, and mentions to preserve the unique semantic contributions of each modality.

2. **BERT-Based Encoding**: Leveraging pre-trained BERT representations enables robust semantic understanding without domain-specific training data.

3. **Unified Embedding Space**: By using the same encoder (BERT) for both visual descriptions and textual content, we ensure semantic alignment across modalities.

4. **Scalable Pipeline**: Batch processing and efficient tokenization enable processing of large-scale social media datasets.

The extracted textual features, combined with visual features from Section 3.2, form the foundation for multimodal content representation, which will be integrated in the fusion layer described in Section 3.4.
