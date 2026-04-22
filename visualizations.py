"""
Visualization Script for TOPSIS Text Classification Analysis
=============================================================
Generates comprehensive visualizations including bar charts, radar charts, and heatmaps.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Model data
models = ['BERT-base', 'RoBERTa-base', 'DistilBERT', 'ALBERT-base', 'XLNet-base', 'DeBERTa-base']
criteria = ['Accuracy (%)', 'F1-Score (%)', 'Inference Time (ms)', 'Model Size (MB)', 'Training Data (GB)']

data = np.array([
    [92.5, 91.8, 45.2, 438, 16],   # BERT-base
    [93.2, 92.5, 47.8, 498, 160],  # RoBERTa-base
    [90.8, 90.1, 28.5, 255, 16],   # DistilBERT
    [91.2, 90.7, 38.4, 44, 16],    # ALBERT-base
    [93.5, 92.8, 62.1, 535, 126],  # XLNet-base
    [94.1, 93.6, 52.3, 520, 160]   # DeBERTa-base
])

# Read TOPSIS results
results_df = pd.read_csv('results/detailed_results.csv', index_col=0)
topsis_scores = results_df['TOPSIS Score'].values
ranks = results_df['Rank'].values

print("Generating visualizations...")

# 1. TOPSIS Score Comparison Bar Chart
print("1. Creating TOPSIS score comparison chart...")
fig, ax = plt.subplots(figsize=(12, 6))

colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
bars = ax.barh(models, topsis_scores, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for i, (bar, score) in enumerate(zip(bars, topsis_scores)):
    width = bar.get_width()
    ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
            f'{score:.4f}', ha='left', va='center', fontweight='bold', fontsize=10)

ax.set_xlabel('TOPSIS Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Model', fontsize=12, fontweight='bold')
ax.set_title('TOPSIS Score Comparison - Text Classification Models', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(0, max(topsis_scores) * 1.15)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('results/topsis_scores.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Criteria Comparison - Multiple Bar Charts
print("2. Creating criteria comparison charts...")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, criterion in enumerate(criteria):
    ax = axes[idx]
    values = data[:, idx]
    
    colors_crit = plt.cm.Set3(np.linspace(0, 1, len(models)))
    bars = ax.bar(range(len(models)), values, color=colors_crit, edgecolor='black', linewidth=1.2)
    
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(criterion, fontsize=10, fontweight='bold')
    ax.set_title(f'{criterion}', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('Performance Metrics Comparison Across Models', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('results/criteria_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Radar Chart
print("3. Creating radar chart...")
# Normalize data for radar chart (0-1 scale, inverted for cost criteria)
radar_data = np.zeros_like(data, dtype=float)
for i in range(data.shape[1]):
    if i in [2, 3]:  # Cost criteria (lower is better) - invert
        radar_data[:, i] = 1 - (data[:, i] - data[:, i].min()) / (data[:, i].max() - data[:, i].min())
    else:  # Benefit criteria (higher is better)
        radar_data[:, i] = (data[:, i] - data[:, i].min()) / (data[:, i].max() - data[:, i].min())

# Create radar chart
angles = [n / float(len(criteria)) * 2 * pi for n in range(len(criteria))]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))

colors_radar = plt.cm.tab10(np.linspace(0, 1, len(models)))

for idx, model in enumerate(models):
    values = radar_data[idx].tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors_radar[idx])
    ax.fill(angles, values, alpha=0.15, color=colors_radar[idx])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(criteria, fontsize=10)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
ax.grid(True)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
ax.set_title('Model Performance Radar Chart\n(Normalized Scores)', 
             fontsize=14, fontweight='bold', pad=30)

plt.tight_layout()
plt.savefig('results/radar_chart.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Heatmap of Normalized Decision Matrix
print("4. Creating heatmap...")
# Normalize data
norm_divisors = np.sqrt(np.sum(data**2, axis=0))
normalized_data = data / norm_divisors

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(normalized_data, annot=True, fmt='.3f', cmap='YlOrRd', 
            xticklabels=criteria, yticklabels=models, 
            cbar_kws={'label': 'Normalized Value'}, linewidths=0.5, ax=ax)
ax.set_title('Normalized Decision Matrix Heatmap', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Criteria', fontsize=12, fontweight='bold')
ax.set_ylabel('Models', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('results/heatmap_normalized.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Ranking Visualization
print("5. Creating ranking visualization...")
fig, ax = plt.subplots(figsize=(12, 7))

# Sort by rank
sorted_indices = np.argsort(ranks)
sorted_models = [models[i] for i in sorted_indices]
sorted_scores = topsis_scores[sorted_indices]
sorted_ranks = ranks[sorted_indices]

# Create color gradient (best = green, worst = red)
colors_rank = plt.cm.RdYlGn(np.linspace(0.9, 0.3, len(models)))

bars = ax.barh(sorted_models, sorted_scores, color=colors_rank, 
               edgecolor='black', linewidth=2)

# Add rank labels
for i, (bar, rank, score) in enumerate(zip(bars, sorted_ranks, sorted_scores)):
    width = bar.get_width()
    # Rank badge
    ax.text(-0.02, bar.get_y() + bar.get_height()/2, 
            f'#{int(rank)}', ha='right', va='center', 
            fontweight='bold', fontsize=14, 
            bbox=dict(boxstyle='circle', facecolor='gold', edgecolor='black', linewidth=2))
    # Score value
    ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
            f'{score:.4f}', ha='left', va='center', fontweight='bold', fontsize=11)

ax.set_xlabel('TOPSIS Score', fontsize=13, fontweight='bold')
ax.set_ylabel('Model', fontsize=13, fontweight='bold')
ax.set_title('Final Model Ranking - Text Classification\n(Based on TOPSIS Analysis)', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xlim(-0.05, max(sorted_scores) * 1.15)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('results/final_ranking.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Combined Performance Overview
print("6. Creating combined performance overview...")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Subplot 1: Accuracy vs F1-Score
ax1 = fig.add_subplot(gs[0, 0])
scatter = ax1.scatter(data[:, 0], data[:, 1], s=300, c=topsis_scores, 
                      cmap='viridis', edgecolors='black', linewidth=2, alpha=0.8)
for i, model in enumerate(models):
    ax1.annotate(model, (data[i, 0], data[i, 1]), 
                fontsize=9, ha='center', va='bottom', fontweight='bold')
ax1.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_ylabel('F1-Score (%)', fontsize=11, fontweight='bold')
ax1.set_title('Accuracy vs F1-Score', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax1, label='TOPSIS Score')

# Subplot 2: Inference Time vs Model Size
ax2 = fig.add_subplot(gs[0, 1])
scatter2 = ax2.scatter(data[:, 2], data[:, 3], s=300, c=topsis_scores, 
                       cmap='viridis', edgecolors='black', linewidth=2, alpha=0.8)
for i, model in enumerate(models):
    ax2.annotate(model, (data[i, 2], data[i, 3]), 
                fontsize=9, ha='center', va='bottom', fontweight='bold')
ax2.set_xlabel('Inference Time (ms)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Model Size (MB)', fontsize=11, fontweight='bold')
ax2.set_title('Inference Time vs Model Size', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=ax2, label='TOPSIS Score')

# Subplot 3: Overall Score Comparison
ax3 = fig.add_subplot(gs[1, :])
x_pos = np.arange(len(models))
colors_overview = plt.cm.plasma(np.linspace(0.2, 0.9, len(models)))
bars = ax3.bar(x_pos, topsis_scores, color=colors_overview, 
               edgecolor='black', linewidth=1.5, alpha=0.8)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(models, rotation=0, fontsize=11)
ax3.set_ylabel('TOPSIS Score', fontsize=12, fontweight='bold')
ax3.set_title('Overall TOPSIS Score Comparison', fontsize=13, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Add value labels and rank
for i, (bar, score, rank) in enumerate(zip(bars, topsis_scores, ranks)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{score:.4f}\n(Rank #{int(rank)})', 
            ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.suptitle('Text Classification Models - Performance Overview', 
             fontsize=16, fontweight='bold', y=0.98)
plt.savefig('results/performance_overview.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nAll visualizations created successfully!")
print("\nGenerated files:")
print("  - results/topsis_scores.png")
print("  - results/criteria_comparison.png")
print("  - results/radar_chart.png")
print("  - results/heatmap_normalized.png")
print("  - results/final_ranking.png")
print("  - results/performance_overview.png")
