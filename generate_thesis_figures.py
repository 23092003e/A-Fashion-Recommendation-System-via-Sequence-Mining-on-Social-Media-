"""
Generate Publication-Quality Figures for Thesis
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150

# Create figures directory
os.makedirs('figures', exist_ok=True)

# Results data (from evaluation)
results = {
    'Popularity': {'HR@10': 0.0519, 'HR@10_std': 0},
    'GRU4Rec': {'HR@10': 0.0381, 'HR@10_std': 0.0117, 'MRR': 0.0133, 'MRR_std': 0.0043},
    'SASRec': {'HR@10': 0.2121, 'HR@10_std': 0.0072, 'MRR': 0.1520, 'MRR_std': 0.0020},
    'BERT4Rec': {'HR@10': 0.2381, 'HR@10_std': 0.0061, 'MRR': 0.1529, 'MRR_std': 0.0028},
    'Multimodal-GF': {'HR@10': 0.3177, 'HR@10_std': 0.0089, 'MRR': 0.1875, 'MRR_std': 0.0055},
    'Multimodal-Hybrid': {'HR@10': 0.6416, 'HR@10_std': 0.0114, 'MRR': 0.3561, 'MRR_std': 0.0091},
}

ablation_results = {
    'Full Model': {'HR@10': 0.3177, 'HR@10_std': 0.0089},
    'Text-Only': {'HR@10': 0.3203, 'HR@10_std': 0.0061},
    'Image-Only': {'HR@10': 0.1983, 'HR@10_std': 0.0050},
    'No-Gate': {'HR@10': 0.3143, 'HR@10_std': 0.0089},
    'No-MHA': {'HR@10': 0.2571, 'HR@10_std': 0.0161},
    'No-BiLSTM': {'HR@10': 0.2684, 'HR@10_std': 0.0082},
}

# ============================================================
# Figure 1: Model Comparison Bar Chart
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

models = ['Popularity', 'GRU4Rec', 'SASRec', 'BERT4Rec', 'MM-GF\n(Ours)', 'MM-Hybrid\n(Ours)']
hr10_values = [0.0519, 0.0381, 0.2121, 0.2381, 0.3177, 0.6416]
hr10_stds = [0, 0.0117, 0.0072, 0.0061, 0.0089, 0.0114]

colors = ['#cccccc', '#99ccff', '#66b3ff', '#3399ff', '#ff9966', '#ff6633']

bars1 = ax1.bar(models, hr10_values, yerr=hr10_stds, capsize=5, color=colors, edgecolor='black', linewidth=0.5)
ax1.set_ylabel('HR@10')
ax1.set_title('(a) Hit Rate @ 10 Comparison')
ax1.set_ylim(0, 0.75)

# Add value labels
for bar, val in zip(bars1, hr10_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}',
             ha='center', va='bottom', fontsize=9)

mrr_values = [0, 0.0133, 0.1520, 0.1529, 0.1875, 0.3561]
mrr_stds = [0, 0.0043, 0.0020, 0.0028, 0.0055, 0.0091]

bars2 = ax2.bar(models, mrr_values, yerr=mrr_stds, capsize=5, color=colors, edgecolor='black', linewidth=0.5)
ax2.set_ylabel('MRR')
ax2.set_title('(b) Mean Reciprocal Rank Comparison')
ax2.set_ylim(0, 0.45)

for bar, val in zip(bars2, mrr_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015, f'{val:.3f}',
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('figures/model_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/model_comparison.pdf', bbox_inches='tight')
print("Saved: figures/model_comparison.png")

# ============================================================
# Figure 2: Ablation Study
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))

ablation_models = ['Full Model\n(Proposed)', 'Text-Only', 'Image-Only', 'No-Gate', 'No-MHA', 'No-BiLSTM']
ablation_hr10 = [0.3177, 0.3203, 0.1983, 0.3143, 0.2571, 0.2684]
ablation_std = [0.0089, 0.0061, 0.0050, 0.0089, 0.0161, 0.0082]

colors_ablation = ['#ff6633'] + ['#99ccff'] * 5
bars = ax.bar(ablation_models, ablation_hr10, yerr=ablation_std, capsize=5,
              color=colors_ablation, edgecolor='black', linewidth=0.5)

ax.set_ylabel('HR@10')
ax.set_title('Ablation Study: Component Contribution Analysis')
ax.axhline(y=0.3177, color='red', linestyle='--', alpha=0.5, label='Full Model Baseline')
ax.set_ylim(0, 0.4)

# Add value labels and delta
for i, (bar, val) in enumerate(zip(bars, ablation_hr10)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015, f'{val:.3f}',
            ha='center', va='bottom', fontsize=9)
    if i > 0:
        delta = val - 0.3177
        color = 'green' if delta >= 0 else 'red'
        ax.text(bar.get_x() + bar.get_width()/2, 0.02, f'{delta:+.3f}',
                ha='center', va='bottom', fontsize=8, color=color, fontweight='bold')

ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('figures/ablation_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/ablation_study.pdf', bbox_inches='tight')
print("Saved: figures/ablation_study.png")

# ============================================================
# Figure 3: Improvement Over Baselines
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

baselines = ['GRU4Rec', 'SASRec', 'BERT4Rec']
baseline_hr10 = [0.0381, 0.2121, 0.2381]
proposed_hr10 = 0.6416

improvements = [(proposed_hr10 - b) / b * 100 for b in baseline_hr10]

colors_imp = ['#66b3ff', '#3399ff', '#0066cc']
bars = ax.bar(baselines, improvements, color=colors_imp, edgecolor='black', linewidth=0.5)

ax.set_ylabel('Improvement (%)')
ax.set_title('Relative Improvement of Multimodal-Hybrid Over Baselines')

for bar, val in zip(bars, improvements):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, f'{val:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/improvement_chart.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/improvement_chart.pdf', bbox_inches='tight')
print("Saved: figures/improvement_chart.png")

# ============================================================
# Figure 4: Architecture Diagram
# ============================================================

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors
input_color = '#e6f3ff'
process_color = '#fff2e6'
fusion_color = '#ffe6e6'
output_color = '#e6ffe6'
text_color = '#000000'

def draw_box(ax, x, y, w, h, label, color, fontsize=10):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=fontsize, wrap=True)

def draw_arrow(ax, start, end, color='black'):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

# Title
ax.text(7, 9.5, 'Multimodal Hybrid Fashion Recommendation Architecture',
        ha='center', fontsize=14, fontweight='bold')

# Input Layer
draw_box(ax, 0.5, 7, 2.5, 1, 'User Interaction\nSequence', input_color)
draw_box(ax, 3.5, 7, 2.5, 1, 'Text Content\n(Caption, Hashtags)', input_color)
draw_box(ax, 6.5, 7, 2.5, 1, 'Image Features\n(Fashion Desc.)', input_color)
draw_box(ax, 9.5, 7, 2.5, 1, 'Co-occurrence\nMatrix', input_color)

# Embedding Layer
draw_box(ax, 0.5, 5, 2.5, 1, 'Position\nEmbedding', process_color)
draw_box(ax, 3.5, 5, 2.5, 1, 'BERT Text\nEmbedding (768d)', process_color)
draw_box(ax, 6.5, 5, 2.5, 1, 'BERT Image\nEmbedding (768d)', process_color)
draw_box(ax, 9.5, 5, 2.5, 1, 'Item-Item\nSimilarity', process_color)

# Fusion Layer
draw_box(ax, 4, 3, 4, 1, 'Gated Multimodal Fusion\ng(·) = σ(W[text; image])', fusion_color)

# Sequence Modeling
draw_box(ax, 1, 1.5, 3, 1, 'BiLSTM\nSequence Encoder', process_color)
draw_box(ax, 5, 1.5, 3, 1, 'Multi-Head Attention\n(4 heads)', process_color)
draw_box(ax, 9, 1.5, 3, 1, 'Collaborative\nFiltering Module', process_color)

# Output
draw_box(ax, 4.5, 0, 5, 0.8, 'Hybrid Score Combination\nα·Neural + β·Sim + γ·CF + δ·Pop', output_color, fontsize=9)

# Arrows
draw_arrow(ax, (1.75, 7), (1.75, 6))
draw_arrow(ax, (4.75, 7), (4.75, 6))
draw_arrow(ax, (7.75, 7), (7.75, 6))
draw_arrow(ax, (10.75, 7), (10.75, 6))

draw_arrow(ax, (4.75, 5), (5, 4))
draw_arrow(ax, (7.75, 5), (7, 4))

draw_arrow(ax, (6, 3), (2.5, 2.5))
draw_arrow(ax, (6, 3), (6.5, 2.5))
draw_arrow(ax, (10.75, 5), (10.5, 2.5))

draw_arrow(ax, (2.5, 1.5), (5.5, 0.8))
draw_arrow(ax, (6.5, 1.5), (7, 0.8))
draw_arrow(ax, (10.5, 1.5), (8.5, 0.8))

plt.tight_layout()
plt.savefig('figures/architecture_diagram.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/architecture_diagram.pdf', bbox_inches='tight')
print("Saved: figures/architecture_diagram.png")

# ============================================================
# Figure 5: Statistical Significance
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

# t-test results
comparisons = ['vs GRU4Rec', 'vs SASRec', 'vs BERT4Rec']
t_stats = [73.675, 63.506, 62.272]
p_values = [0.0000, 0.0000, 0.0000]

y_pos = np.arange(len(comparisons))
bars = ax.barh(y_pos, t_stats, color='#3399ff', edgecolor='black', linewidth=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(comparisons)
ax.set_xlabel('t-statistic')
ax.set_title('Statistical Significance Tests (Two-Sample t-test)\np < 0.001 for all comparisons (***)')

for i, (bar, t, p) in enumerate(zip(bars, t_stats, p_values)):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            f't={t:.2f}, p<0.001***', va='center', fontsize=10)

ax.axvline(x=2.776, color='red', linestyle='--', alpha=0.7, label='Critical value (α=0.05, df=8)')
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('figures/statistical_significance.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/statistical_significance.pdf', bbox_inches='tight')
print("Saved: figures/statistical_significance.png")

# ============================================================
# Generate LaTeX Table
# ============================================================

latex_table = r"""
\begin{table}[htbp]
\centering
\caption{Comparison of Recommendation Models on Fashion Dataset}
\label{tab:model_comparison}
\begin{tabular}{l|cc|cc}
\toprule
\textbf{Model} & \textbf{HR@10} & \textbf{HR@10 Std} & \textbf{MRR} & \textbf{MRR Std} \\
\midrule
\multicolumn{5}{l}{\textit{Baselines}} \\
Popularity & 0.0519 & - & - & - \\
GRU4Rec & 0.0381 & 0.0117 & 0.0133 & 0.0043 \\
SASRec & 0.2121 & 0.0072 & 0.1520 & 0.0020 \\
BERT4Rec & 0.2381 & 0.0061 & 0.1529 & 0.0028 \\
\midrule
\multicolumn{5}{l}{\textit{Proposed Models}} \\
Multimodal-GF & 0.3177 & 0.0089 & 0.1875 & 0.0055 \\
\textbf{Multimodal-Hybrid} & \textbf{0.6416} & 0.0114 & \textbf{0.3561} & 0.0091 \\
\midrule
\multicolumn{5}{l}{\textit{Improvement over best baseline (BERT4Rec)}} \\
Multimodal-Hybrid & \multicolumn{2}{c|}{+169.5\%} & \multicolumn{2}{c}{+132.9\%} \\
\bottomrule
\end{tabular}
\end{table}
"""

with open('figures/results_table.tex', 'w') as f:
    f.write(latex_table)
print("Saved: figures/results_table.tex")

# Ablation table
ablation_table = r"""
\begin{table}[htbp]
\centering
\caption{Ablation Study Results}
\label{tab:ablation}
\begin{tabular}{l|cc|c}
\toprule
\textbf{Configuration} & \textbf{HR@10} & \textbf{Std} & \textbf{$\Delta$ HR@10} \\
\midrule
Full Model (Proposed) & 0.3177 & 0.0089 & - \\
\midrule
Text-Only & 0.3203 & 0.0061 & +0.0026 \\
Image-Only & 0.1983 & 0.0050 & -0.1194 \\
No-Gate (Concat) & 0.3143 & 0.0089 & -0.0034 \\
No-MHA & 0.2571 & 0.0161 & -0.0606 \\
No-BiLSTM & 0.2684 & 0.0082 & -0.0493 \\
\bottomrule
\end{tabular}
\end{table}
"""

with open('figures/ablation_table.tex', 'w') as f:
    f.write(ablation_table)
print("Saved: figures/ablation_table.tex")

print("\n" + "="*60)
print("All figures generated successfully!")
print("="*60)
