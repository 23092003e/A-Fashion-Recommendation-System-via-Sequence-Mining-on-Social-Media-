# 3.2 Visual Feature Extraction

## 3.2.1 Overview

In multimodal fashion recommendation systems, visual information plays a crucial role in understanding user preferences and item characteristics. Unlike traditional collaborative filtering approaches that rely solely on user-item interactions, our methodology incorporates rich visual features extracted from fashion images to capture fine-grained style attributes such as colors, patterns, silhouettes, and overall aesthetics.

This section presents our approach to visual feature extraction using Florence-2, a state-of-the-art vision-language foundation model. The extracted visual features are subsequently integrated with textual features to form comprehensive multimodal representations for each fashion post.

## 3.2.2 Florence-2 Vision-Language Model

### Model Selection Rationale

We selected Microsoft's Florence-2 as our visual feature extractor based on several key considerations:

1. **Unified Vision-Language Architecture**: Florence-2 employs a sequence-to-sequence framework that naturally bridges visual and textual modalities, making it well-suited for generating semantically rich descriptions of fashion images.

2. **Pre-training on Diverse Visual Tasks**: The model was pre-trained on FLD-5B, a dataset containing 5.4 billion annotations across 126 million images, enabling robust zero-shot performance on fashion imagery without domain-specific fine-tuning.

3. **Detailed Caption Generation**: Unlike classification-based approaches that output discrete labels, Florence-2 generates natural language descriptions that capture nuanced visual attributes essential for fashion understanding.

### Architecture

Florence-2 consists of two primary components:

**Vision Encoder**: The model employs DaViT (Dual Attention Vision Transformer) as its vision backbone. DaViT utilizes both spatial and channel attention mechanisms to extract hierarchical visual features from input images. Given an input image $I \in \mathbb{R}^{H \times W \times 3}$, the vision encoder produces a sequence of visual tokens:

$$V = \text{DaViT}(I) \in \mathbb{R}^{N_v \times d}$$

where $N_v$ denotes the number of visual tokens and $d$ represents the hidden dimension.

**Multimodal Encoder-Decoder**: The visual tokens are combined with task-specific text prompts and processed through a standard Transformer encoder-decoder architecture. For image captioning, given a task prompt $P$ (e.g., "\<MORE_DETAILED_CAPTION\>"), the model generates a textual description $T$ by:

$$T = \text{Decoder}(\text{Encoder}(V, P))$$

The overall architecture is illustrated in Figure 3.1.

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│     Input Image              DaViT Vision Encoder               │
│    ┌─────────┐              ┌─────────────────┐                 │
│    │         │              │  Spatial Attention │                │
│    │ Fashion │    ────►     │        +          │    ────►  Visual│
│    │  Image  │              │ Channel Attention │          Tokens │
│    │         │              └─────────────────┘                 │
│    └─────────┘                                                  │
│                                       │                         │
│                                       ▼                         │
│    Task Prompt           ┌─────────────────────────┐            │
│   ┌───────────┐          │                         │            │
│   │<DETAILED  │   ────►  │  Transformer Encoder-   │            │
│   │ _CAPTION> │          │       Decoder           │            │
│   └───────────┘          │                         │            │
│                          └───────────┬─────────────┘            │
│                                      │                          │
│                                      ▼                          │
│                          ┌─────────────────────────┐            │
│                          │  "A woman wearing a     │            │
│                          │   elegant red gown..."  │            │
│                          └─────────────────────────┘            │
│                             Fashion Description                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                    Figure 3.1: Florence-2 Architecture
```

## 3.2.3 Image Description Generation

### Task Formulation

Florence-2 supports multiple caption generation tasks with varying levels of detail. We evaluated three task configurations:

| Task Type | Description | Average Length |
|-----------|-------------|----------------|
| CAPTION | Brief, single-sentence description | ~20 tokens |
| DETAILED_CAPTION | Standard description with key attributes | ~50 tokens |
| MORE_DETAILED_CAPTION | Comprehensive description with fine-grained details | ~100+ tokens |

For fashion recommendation, we selected **MORE_DETAILED_CAPTION** as our primary task. This choice is motivated by the need to capture comprehensive visual information including:

- **Clothing categories**: dresses, blazers, jeans, accessories
- **Color attributes**: primary colors, color combinations, patterns
- **Style characteristics**: formal, casual, elegant, sporty
- **Material textures**: silk, denim, leather, cotton
- **Contextual elements**: occasion, setting, styling choices

### Generation Configuration

The description generation process employs beam search decoding with the following hyperparameters:

| Parameter | Value | Justification |
|-----------|-------|---------------|
| max_new_tokens | 512 | Accommodates detailed fashion descriptions |
| num_beams | 3 | Balances quality and computational cost |
| do_sample | False | Ensures deterministic, reproducible outputs |

### Output Examples

Table 3.1 presents representative examples of generated fashion descriptions:

**Table 3.1: Sample Generated Descriptions**

| Image Type | Generated Description |
|------------|----------------------|
| Formal Event | "The image shows a woman posing on a red carpet at a formal event. She is wearing a long, flowing pink gown with a sweetheart neckline and off-the-shoulder straps. The gown has a fitted bodice and a full skirt that flares out at the bottom. She has her hair styled in loose waves and is wearing statement jewelry." |
| Casual Street | "The image shows a young woman standing on a sidewalk. She is wearing a white t-shirt, light blue jeans, and white sneakers. She has a black blazer draped over her shoulders and is carrying a black crossbody bag. Her hair is styled in loose waves and she is wearing sunglasses." |
| Business Attire | "The image shows a woman standing in a hallway. She is wearing a white blazer over a green blouse and beige trousers. She has blonde hair styled in loose waves and is holding a small purse. The overall look appears professional and polished." |

## 3.2.4 Integration with Text Encoding

The generated visual descriptions serve as a bridge between raw image pixels and semantic text representations. This approach offers several advantages over direct visual feature extraction:

1. **Semantic Alignment**: By converting images to text, visual features become directly comparable with other textual modalities (captions, hashtags, mentions) in a unified semantic space.

2. **Interpretability**: Text descriptions provide human-readable representations of visual content, facilitating model interpretation and debugging.

3. **Efficient Storage**: Text descriptions require significantly less storage than raw image features or high-dimensional visual embeddings.

The visual descriptions are subsequently processed by BERT to generate dense vector representations:

$$\mathbf{v}_{\text{image}} = \text{BERT}_{\text{CLS}}(\text{Florence-2}(I)) \in \mathbb{R}^{768}$$

where $\mathbf{v}_{\text{image}}$ represents the 768-dimensional image embedding obtained from the [CLS] token of BERT's output.

## 3.2.5 Processing Pipeline

The complete visual feature extraction pipeline is illustrated in Figure 3.2:

```
┌────────────────────────────────────────────────────────────────────┐
│                 VISUAL FEATURE EXTRACTION PIPELINE                 │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   Step 1: Image Loading                                            │
│   ┌──────────────┐                                                 │
│   │ Fashion Post │ ──► Load RGB Image ──► Resize/Normalize         │
│   │    Images    │                                                 │
│   └──────────────┘                                                 │
│          │                                                         │
│          ▼                                                         │
│   Step 2: Description Generation                                   │
│   ┌──────────────┐                                                 │
│   │  Florence-2  │ ──► Generate Detailed Caption                   │
│   │    Model     │     (MORE_DETAILED_CAPTION task)                │
│   └──────────────┘                                                 │
│          │                                                         │
│          ▼                                                         │
│   Step 3: Text Encoding                                            │
│   ┌──────────────┐                                                 │
│   │  BERT Model  │ ──► Extract [CLS] Token Embedding               │
│   │              │     (768 dimensions)                            │
│   └──────────────┘                                                 │
│          │                                                         │
│          ▼                                                         │
│   Step 4: Feature Storage                                          │
│   ┌──────────────┐                                                 │
│   │   Output:    │     image_embeddings.npy                        │
│   │  v_image     │     Shape: (N_posts × 768)                      │
│   └──────────────┘                                                 │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
                Figure 3.2: Visual Feature Extraction Pipeline
```

## 3.2.6 Dataset Statistics

Table 3.2 summarizes the visual feature extraction results on our fashion dataset:

**Table 3.2: Visual Feature Extraction Statistics**

| Metric | Value |
|--------|-------|
| Total images processed | 822 |
| Successfully processed | 817 (99.4%) |
| Average description length | 89.3 tokens |
| Output embedding dimension | 768 |
| Total processing time | ~16.4 minutes |

## 3.2.7 Summary

This section presented our methodology for visual feature extraction in the multimodal fashion recommendation system. Key contributions include:

1. **Florence-2 Integration**: We leveraged a state-of-the-art vision-language model to generate rich, detailed descriptions of fashion images without requiring domain-specific fine-tuning.

2. **Text-Mediated Visual Features**: By converting visual content to natural language descriptions, we enabled seamless integration with textual modalities in a unified semantic space.

3. **Comprehensive Fashion Understanding**: The MORE_DETAILED_CAPTION task captures fine-grained fashion attributes essential for accurate style-based recommendations.

The extracted visual features, combined with textual features from captions, hashtags, and mentions, form the foundation of our multimodal content representation, which will be discussed in the following section.
