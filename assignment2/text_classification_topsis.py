"""
TOPSIS Analysis for Text Classification Models
================================================
This script implements TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
to evaluate and rank pre-trained models for text classification tasks.
"""

import numpy as np
import pandas as pd
from tabulate import tabulate
import os

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Model data based on literature and benchmarks
# Models: BERT, RoBERTa, DistilBERT, ALBERT, XLNet, DeBERTa
models = ['BERT-base', 'RoBERTa-base', 'DistilBERT', 'ALBERT-base', 'XLNet-base', 'DeBERTa-base']

# Criteria data (example values based on typical performance)
# Criteria: Accuracy (%), F1-Score (%), Inference Time (ms), Model Size (MB), Training Data Size (GB)
data = np.array([
    [92.5, 91.8, 45.2, 438, 16],   # BERT-base
    [93.2, 92.5, 47.8, 498, 160],  # RoBERTa-base
    [90.8, 90.1, 28.5, 255, 16],   # DistilBERT
    [91.2, 90.7, 38.4, 44, 16],    # ALBERT-base
    [93.5, 92.8, 62.1, 535, 126],  # XLNet-base
    [94.1, 93.6, 52.3, 520, 160]   # DeBERTa-base
])

criteria = ['Accuracy (%)', 'F1-Score (%)', 'Inference Time (ms)', 'Model Size (MB)', 'Training Data (GB)']

# Criteria types: True for benefit (higher is better), False for cost (lower is better)
beneficial = [True, True, False, False, True]

# Weights for each criterion (must sum to 1)
weights = np.array([0.30, 0.30, 0.20, 0.10, 0.10])

print("="*80)
print("TOPSIS ANALYSIS FOR TEXT CLASSIFICATION MODELS")
print("="*80)
print()

# Display input data
print("Input Decision Matrix:")
print("-" * 80)
df = pd.DataFrame(data, index=models, columns=criteria)
print(tabulate(df, headers='keys', tablefmt='grid', floatfmt='.2f'))
print()

# Step 1: Normalize the decision matrix
print("Step 1: Normalizing the decision matrix...")
print("-" * 80)

# Calculate the square root of sum of squares for each column
norm_divisors = np.sqrt(np.sum(data**2, axis=0))
normalized_data = data / norm_divisors

df_norm = pd.DataFrame(normalized_data, index=models, columns=criteria)
print(tabulate(df_norm, headers='keys', tablefmt='grid', floatfmt='.4f'))
print()

# Step 2: Calculate weighted normalized decision matrix
print("Step 2: Calculating weighted normalized matrix...")
print("-" * 80)

weighted_normalized = normalized_data * weights

df_weighted = pd.DataFrame(weighted_normalized, index=models, columns=criteria)
print(tabulate(df_weighted, headers='keys', tablefmt='grid', floatfmt='.4f'))
print()

# Step 3: Determine ideal and negative-ideal solutions
print("Step 3: Determining ideal and negative-ideal solutions...")
print("-" * 80)

ideal_best = np.zeros(len(criteria))
ideal_worst = np.zeros(len(criteria))

for i in range(len(criteria)):
    if beneficial[i]:
        ideal_best[i] = np.max(weighted_normalized[:, i])
        ideal_worst[i] = np.min(weighted_normalized[:, i])
    else:
        ideal_best[i] = np.min(weighted_normalized[:, i])
        ideal_worst[i] = np.max(weighted_normalized[:, i])

print("Ideal Best (A+):", ideal_best)
print("Ideal Worst (A-):", ideal_worst)
print()

# Step 4: Calculate separation measures
print("Step 4: Calculating separation measures...")
print("-" * 80)

# Euclidean distance from ideal best
separation_best = np.sqrt(np.sum((weighted_normalized - ideal_best)**2, axis=1))

# Euclidean distance from ideal worst
separation_worst = np.sqrt(np.sum((weighted_normalized - ideal_worst)**2, axis=1))

df_separation = pd.DataFrame({
    'Model': models,
    'Distance from Best (S+)': separation_best,
    'Distance from Worst (S-)': separation_worst
})
print(tabulate(df_separation, headers='keys', tablefmt='grid', floatfmt='.6f', showindex=False))
print()

# Step 5: Calculate TOPSIS score (relative closeness to ideal solution)
print("Step 5: Calculating TOPSIS scores...")
print("-" * 80)

topsis_score = separation_worst / (separation_best + separation_worst)

# Create results dataframe
results_df = pd.DataFrame({
    'Model': models,
    'TOPSIS Score': topsis_score,
    'Rank': np.argsort(-topsis_score) + 1  # Rank in descending order
})

results_df = results_df.sort_values('Rank')

print(tabulate(results_df, headers='keys', tablefmt='grid', floatfmt='.6f', showindex=False))
print()

# Step 6: Display final ranking
print("="*80)
print("FINAL RANKING")
print("="*80)
for idx, row in results_df.iterrows():
    print(f"{int(row['Rank'])}. {row['Model']:<20} - Score: {row['TOPSIS Score']:.6f}")
print()

# Best model recommendation
best_model = results_df.iloc[0]['Model']
best_score = results_df.iloc[0]['TOPSIS Score']
print("="*80)
print(f"RECOMMENDATION: {best_model} (Score: {best_score:.6f})")
print("="*80)
print()

# Export results
# Save detailed results
detailed_results = pd.DataFrame(data, index=models, columns=criteria)
detailed_results['TOPSIS Score'] = topsis_score
detailed_results['Rank'] = np.argsort(-topsis_score) + 1
detailed_results = detailed_results.sort_values('Rank')
detailed_results.to_csv('results/detailed_results.csv')

# Save TOPSIS calculation breakdown
breakdown = pd.DataFrame({
    'Model': models,
    'Distance_from_Best': separation_best,
    'Distance_from_Worst': separation_worst,
    'TOPSIS_Score': topsis_score,
    'Rank': np.argsort(-topsis_score) + 1
})
breakdown.to_csv('results/topsis_breakdown.csv', index=False)

print("Results exported to:")
print("  - results/detailed_results.csv")
print("  - results/topsis_breakdown.csv")
print()
print("Script completed successfully!")
