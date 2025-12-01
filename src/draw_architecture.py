"""
Draw Detailed Architecture Diagram for Multimodal Hybrid Fashion Recommendation Model
Final Thesis Model
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
import os

# Set style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9

def draw_box(ax, x, y, width, height, text, color='lightblue', fontsize=9, text_color='black', bold=False):
    """Draw a rounded box with text"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, weight=weight, wrap=True)

def draw_arrow(ax, start, end, color='black', lw=1.5):
    """Draw arrow between two points"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))

def draw_dimension(ax, x, y, text, fontsize=8):
    """Draw dimension annotation"""
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color='#666666', style='italic')

# Create output directories
os.makedirs('images', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ==================== FINAL MODEL: Multimodal Hybrid ====================
fig, ax = plt.subplots(1, 1, figsize=(18, 26))
ax.set_xlim(0, 18)
ax.set_ylim(-4, 26)
ax.axis('off')
ax.set_title('Multimodal Hybrid Fashion Recommendation Model\n(BiLSTM + Multi-Head Attention + Collaborative Filtering)',
             fontsize=18, fontweight='bold', pad=20)

# Colors
colors = {
    'input': '#E3F2FD',
    'embedding': '#FFF3E0',
    'projection': '#E8F5E9',
    'fusion': '#FCE4EC',
    'sequence': '#F3E5F5',
    'attention': '#E0F7FA',
    'hybrid': '#FFF8E1',
    'output': '#FFEBEE',
}

# ============ SECTION 1: INPUT LAYER ============
ax.text(9, 25.2, '① INPUT LAYER', ha='center', fontsize=14, fontweight='bold', color='#1565C0')

# Input boxes
draw_box(ax, 3, 24, 4, 1.2, 'User Interaction\nSequence\n[item₁, item₂, ..., itemₙ]', colors['input'], 10, bold=True)
draw_box(ax, 8, 24, 4, 1.2, 'Text Content\n(Caption, Hashtags,\nMentions)', colors['input'], 10, bold=True)
draw_box(ax, 13, 24, 4, 1.2, 'Image Description\n(Fashion Attributes,\nColors, Styles)', colors['input'], 10, bold=True)

# ============ SECTION 2: EMBEDDING LAYER ============
ax.text(9, 22.3, '② EMBEDDING LAYER (BERT Pre-trained, Frozen)', ha='center', fontsize=14, fontweight='bold', color='#1565C0')

draw_box(ax, 5.5, 21, 6, 1.5, 'BERT Text Embedding\n(768-dimensional)\nFrozen weights from pre-training', colors['embedding'], 10)
draw_dimension(ax, 9.2, 21, '[B, S, 768]', 9)

draw_box(ax, 12.5, 21, 6, 1.5, 'BERT Image Embedding\n(768-dimensional)\nFashion description vectors', colors['embedding'], 10)
draw_dimension(ax, 16.2, 21, '[B, S, 768]', 9)

# Arrows from input to embedding
draw_arrow(ax, (3, 23.3), (5, 21.8))
draw_arrow(ax, (8, 23.3), (6.5, 21.8))
draw_arrow(ax, (13, 23.3), (12, 21.8))

# Combined embedding notation
ax.text(9, 19.8, 'Concatenated Multimodal Embedding: [B, S, 1536] = [768 text + 768 image]',
        ha='center', fontsize=10, style='italic', color='#666666')

# ============ SECTION 3: PROJECTION LAYER ============
ax.text(9, 19, '③ PROJECTION LAYER', ha='center', fontsize=14, fontweight='bold', color='#1565C0')

draw_box(ax, 5, 17.5, 5.5, 1.8, 'Text Projection\nLinear(768 → 128)\n+ LayerNorm\n+ ReLU + Dropout(0.2)', colors['projection'], 9)
draw_dimension(ax, 8.5, 17.5, '[B, S, 128]', 9)

draw_box(ax, 13, 17.5, 5.5, 1.8, 'Image Projection\nLinear(768 → 128)\n+ LayerNorm\n+ ReLU + Dropout(0.2)', colors['projection'], 9)
draw_dimension(ax, 16.5, 17.5, '[B, S, 128]', 9)

# Arrows
draw_arrow(ax, (5.5, 20.2), (5, 18.5))
draw_arrow(ax, (12.5, 20.2), (13, 18.5))

# ============ SECTION 4: GATED FUSION ============
ax.text(9, 16, '④ GATED MULTIMODAL FUSION', ha='center', fontsize=14, fontweight='bold', color='#1565C0')

# Fusion box
fusion_box = FancyBboxPatch((4, 13.3), 10, 2.2,
                             boxstyle="round,pad=0.05",
                             facecolor=colors['fusion'], edgecolor='#C2185B', linewidth=2)
ax.add_patch(fusion_box)
ax.text(9, 15, 'Gated Fusion Module', ha='center', fontsize=11, fontweight='bold', color='#C2185B')
ax.text(9, 14.3, 'g = σ(W · [text_proj; image_proj])', ha='center', fontsize=10, family='monospace')
ax.text(9, 13.7, 'fused = g ⊙ text_proj + (1-g) ⊙ image_proj', ha='center', fontsize=10, family='monospace')
draw_dimension(ax, 15, 14.3, 'Output: [B, S, 128]', 9)

# Arrows to fusion
draw_arrow(ax, (5, 16.5), (6, 15.5))
draw_arrow(ax, (13, 16.5), (12, 15.5))

# ============ SECTION 5: SEQUENCE MODELING ============
ax.text(9, 12.5, '⑤ SEQUENCE MODELING (BiLSTM)', ha='center', fontsize=14, fontweight='bold', color='#1565C0')

draw_box(ax, 9, 10.8, 10, 2.2, 'Bidirectional LSTM\n\n→ LSTM(128 → 128) → Forward\n← LSTM(128 → 128) ← Backward\n\nOutput: Concatenated [B, S, 256]', colors['sequence'], 10)
draw_dimension(ax, 15, 10.8, '[B, S, 256]', 9)

draw_arrow(ax, (9, 13.3), (9, 12))

# ============ SECTION 6: MULTI-HEAD ATTENTION ============
ax.text(9, 9, '⑥ MULTI-HEAD SELF-ATTENTION', ha='center', fontsize=14, fontweight='bold', color='#1565C0')

# MHA box with residual
mha_box = FancyBboxPatch((3.5, 5.5), 11, 3,
                          boxstyle="round,pad=0.05",
                          facecolor=colors['attention'], edgecolor='#00838F', linewidth=2)
ax.add_patch(mha_box)
ax.text(9, 8, 'Multi-Head Self-Attention (4 heads)', ha='center', fontsize=11, fontweight='bold', color='#00838F')
ax.text(9, 7.3, 'Attention(Q,K,V) = softmax(QKᵀ/√d) · V', ha='center', fontsize=10, family='monospace')
ax.text(9, 6.6, 'Q = K = V = lstm_output', ha='center', fontsize=9, style='italic')
ax.text(9, 6, '+ Residual Connection + LayerNorm', ha='center', fontsize=9)

# Draw 4 attention heads
for i, head_x in enumerate([5.5, 7.5, 9.5, 11.5]):
    head_box = FancyBboxPatch((head_x - 0.6, 5.7), 1.2, 0.8,
                              boxstyle="round,pad=0.02",
                              facecolor='#FFCC80', edgecolor='#E65100', linewidth=1.5)
    ax.add_patch(head_box)
    ax.text(head_x, 6.1, f'Head {i+1}\n(64d)', ha='center', va='center', fontsize=8, fontweight='bold')

draw_arrow(ax, (9, 9.6), (9, 8.5))
draw_dimension(ax, 15.5, 6.5, '[B, S, 256]', 9)

# FFN and Pooling
draw_box(ax, 9, 4.3, 9, 1.3, 'Feed-Forward Network\nLinear(256→1024→256) + Residual + LayerNorm', colors['attention'], 9)
draw_arrow(ax, (9, 5.5), (9, 5))

draw_box(ax, 9, 2.8, 6, 1, 'Mean Pooling (with mask)\nAggregate sequence → context vector', colors['attention'], 9)
draw_dimension(ax, 13, 2.8, '[B, 256]', 9)
draw_arrow(ax, (9, 3.6), (9, 3.3))

# Neural output
draw_box(ax, 9, 1.5, 5, 0.9, 'Neural Logits\nLinear(256 → 530)', colors['output'], 9, bold=True)
draw_dimension(ax, 12.5, 1.5, '[B, 530]', 9)
draw_arrow(ax, (9, 2.2), (9, 2))

# ============ SECTION 7: HYBRID COMBINATION ============
ax.text(9, 0.5, '⑦ HYBRID SCORE COMBINATION', ha='center', fontsize=14, fontweight='bold', color='#1565C0')

# CF Components on the side
cf_box = FancyBboxPatch((14, 5.5), 3.5, 6,
                         boxstyle="round,pad=0.05",
                         facecolor=colors['hybrid'], edgecolor='#F57C00', linewidth=2)
ax.add_patch(cf_box)
ax.text(15.75, 11, 'Collaborative\nFiltering\nSignals', ha='center', fontsize=10, fontweight='bold', color='#F57C00')

draw_box(ax, 15.75, 9.5, 2.8, 1, 'Item-Item\nSimilarity\n(Content-based)', colors['hybrid'], 8)
draw_box(ax, 15.75, 8, 2.8, 1, 'Co-occurrence\nMatrix\n(User patterns)', colors['hybrid'], 8)
draw_box(ax, 15.75, 6.5, 2.8, 1, 'Popularity\nPrior\n(Item frequency)', colors['hybrid'], 8)

# Final combination
final_box = FancyBboxPatch((3, -1.8), 12, 1.8,
                            boxstyle="round,pad=0.05",
                            facecolor='#FFEBEE', edgecolor='#C62828', linewidth=2)
ax.add_patch(final_box)
ax.text(9, -0.5, 'Hybrid Score Combination', ha='center', fontsize=11, fontweight='bold', color='#C62828')
ax.text(9, -1.2, 'Score = α·Neural + β·Similarity + γ·Co-occurrence + δ·Popularity',
        ha='center', fontsize=10, family='monospace')
ax.text(9, -1.8, '(α=1.0, β=0.7, γ=1.0, δ=0.2) - Optimized via Grid Search',
        ha='center', fontsize=9, style='italic', color='#666666')

# Arrows to final combination
draw_arrow(ax, (9, 1), (9, 0.1))
draw_arrow(ax, (15.75, 5.9), (14.5, -0.3), color='#F57C00')

# Output
draw_box(ax, 9, -3.2, 5, 0.9, 'Top-K Recommendations', '#C8E6C9', 11, bold=True)
draw_arrow(ax, (9, -1.9), (9, -2.7))

# ============ LEGEND & INFO ============
# Model dimensions
dim_text = """Model Dimensions:
• Vocab Size: 530 items
• Text Embedding: 768d (BERT)
• Image Embedding: 768d (BERT)
• Projection: 128d
• BiLSTM Output: 256d (128×2)
• Attention Heads: 4 (64d each)
• FFN Hidden: 1024d
• Dropout: 0.4"""

ax.text(0.3, 8, dim_text, fontsize=9, va='top',
        bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='#999999'),
        family='monospace')

# Training config
train_text = """Training Configuration:
• Optimizer: AdamW
• Learning Rate: 0.0002
• Weight Decay: 1e-4
• Scheduler: Cosine Annealing
• Epochs: 80
• Batch Size: 32
• Max Sequence: 20"""

ax.text(0.3, 3.5, train_text, fontsize=9, va='top',
        bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#4CAF50'),
        family='monospace')

# Results
results_text = """Final Results:
• HR@10: 0.6416 ± 0.0114
• MRR: 0.3561 ± 0.0091
• vs BERT4Rec: +169.5%
• p < 0.001 (significant)"""

ax.text(0.3, -0.5, results_text, fontsize=9, va='top',
        bbox=dict(boxstyle='round', facecolor='#FFEBEE', edgecolor='#F44336'),
        family='monospace', fontweight='bold')

plt.tight_layout()
plt.savefig('images/architecture_final_thesis.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('figures/architecture_final_thesis.png', dpi=200, bbox_inches='tight', facecolor='white')
print("Saved: images/architecture_final_thesis.png")
print("Saved: figures/architecture_final_thesis.png")

plt.close()

# ==================== SIMPLIFIED DIAGRAM ====================
fig2, ax2 = plt.subplots(1, 1, figsize=(14, 10))
ax2.set_xlim(0, 14)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('Multimodal Hybrid Model - Simplified Architecture', fontsize=16, fontweight='bold', pad=15)

# Simplified flow
draw_box(ax2, 2, 8.5, 3, 1.2, 'Text Content\n(BERT 768d)', '#E3F2FD', 10, bold=True)
draw_box(ax2, 5.5, 8.5, 3, 1.2, 'Image Features\n(BERT 768d)', '#E3F2FD', 10, bold=True)
draw_box(ax2, 9, 8.5, 3.5, 1.2, 'User Sequence\n[item₁...itemₙ]', '#E3F2FD', 10, bold=True)
draw_box(ax2, 12.5, 8.5, 2, 1.2, 'CF\nSignals', '#FFF8E1', 10, bold=True)

# Gated Fusion
draw_box(ax2, 3.75, 6.5, 5, 1.2, 'Gated Multimodal Fusion\ng·text + (1-g)·image', '#FCE4EC', 9)
draw_arrow(ax2, (2, 7.8), (3, 7.2))
draw_arrow(ax2, (5.5, 7.8), (4.5, 7.2))

# BiLSTM
draw_box(ax2, 6, 4.5, 5, 1.2, 'BiLSTM (Bidirectional)\n256-dim output', '#F3E5F5', 10)
draw_arrow(ax2, (3.75, 5.8), (5, 5.2))
draw_arrow(ax2, (9, 7.8), (7, 5.2))

# MHA
draw_box(ax2, 6, 2.5, 5, 1.2, 'Multi-Head Attention\n(4 heads, residual)', '#E0F7FA', 10)
draw_arrow(ax2, (6, 3.8), (6, 3.2))

# Hybrid Combination
draw_box(ax2, 8, 0.8, 6, 1.2, 'Hybrid: α·Neural + β·Sim + γ·CF + δ·Pop', '#FFEBEE', 9, bold=True)
draw_arrow(ax2, (6, 1.8), (7, 1.5))
draw_arrow(ax2, (12.5, 7.8), (10, 1.5), color='#F57C00')

# Performance
ax2.text(2, 0.5, 'HR@10=0.6416 | MRR=0.3561 | +169% vs baseline',
         fontsize=11, fontweight='bold', color='#C62828',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='#C62828'))

plt.tight_layout()
plt.savefig('images/architecture_simplified.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('figures/architecture_simplified.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: images/architecture_simplified.png")
print("Saved: figures/architecture_simplified.png")

plt.close()

print("\n" + "="*60)
print("Architecture diagrams generated successfully!")
print("="*60)
