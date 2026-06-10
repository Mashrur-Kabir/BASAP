# notebooks/05_compare_datasets.py
# Run this AFTER running the full pipeline on all 3 datasets
import pandas as pd
import matplotlib.pyplot as plt
import os

datasets = ['depression', 'sst2', 'hate']
labels = ['Depression\n(Reddit)', 'Sentiment\n(SST-2)', 'Hate Speech\n(Twitter)']
colors_cond = ['#1f4e79','#e74c3c','#27ae60','#2e75b6','#8e44ad','#d35400','#16a085']
conditions = ['original_only (LR)','raw_llm (LR)','basap_filtered (LR)','basap_balanced (LR)','basap_balanced (SVM)','basap_balanced (RF)','basap_balanced (NB)']

os.makedirs('results/comparison', exist_ok=True)

# ── Load results from each dataset ──
all_results = {}
for ds in datasets:
    path = f'results/{ds}/ablation_table.csv'
    if os.path.exists(path):
        all_results[ds] = pd.read_csv(path)
    else:
        print(f'WARNING: {path} not found. Run pipeline on {ds} first.')

if len(all_results) < 2:
    print('Need at least 2 datasets completed. Exiting.')
    exit()

# ── Plot 1: F1 comparison across datasets for key conditions ──
key_conds = [1, 2, 6]  # Baseline, Raw LLM, BASAP RF (best tradeoff)
key_labels = ['Baseline\n(LR)', 'Raw LLM\n(LR)', 'BASAP RF\n(Best)']
x = range(len(all_results))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 7))
for i, (cond_num, cond_label) in enumerate(zip(key_conds, key_labels)):
    f1_vals = []
    for ds in all_results:
        row = all_results[ds][all_results[ds]['condition'] == cond_num]
        f1_vals.append(row['f1_macro'].values[0] if len(row) else 0)
    bars = ax.bar([xi + i*width for xi in x], f1_vals, width,
                  label=cond_label, color=colors_cond[cond_num-1])
    for b, v in zip(bars, f1_vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.002, f'{v:.3f}',
                ha='center', fontsize=10, fontweight='bold')

ax.set_xticks([xi + width for xi in x])
ax.set_xticklabels([labels[i] for i,ds in enumerate(all_results.keys())], fontsize=12)
ax.set_ylabel('Macro F1 Score', fontsize=12)
ax.set_title('BASAP: F1 Score Across Datasets — Baseline vs Raw LLM vs BASAP RF', fontsize=13)
ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3)
yvals = [v for ds in all_results for v in [all_results[ds]['f1_macro'].min(), all_results[ds]['f1_macro'].max()]]
ax.set_ylim(max(0, min(yvals)-0.05), min(1.02, max(yvals)+0.05))
plt.tight_layout()
plt.savefig('results/comparison/f1_cross_dataset.png', dpi=150)
plt.show()

# ── Plot 2: Gender flip rate comparison ──
fig2, ax2 = plt.subplots(figsize=(12, 7))
for i, (cond_num, cond_label) in enumerate(zip(key_conds, key_labels)):
    fair_vals = []
    for ds in all_results:
        path = f'results/{ds}/fairness_results.csv'
        if os.path.exists(path):
            fdf = pd.read_csv(path)
            row = fdf[fdf['condition'] == cond_num]
            fair_vals.append(row['gender_flip'].values[0] if len(row) else 0)
        else:
            fair_vals.append(0)
    bars2 = ax2.bar([xi + i*width for xi in x], fair_vals, width,
                    label=cond_label, color=colors_cond[cond_num-1])
    for b, v in zip(bars2, fair_vals):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.001, f'{v:.4f}',
                 ha='center', fontsize=10, fontweight='bold')

ax2.set_xticks([xi + width for xi in x])
ax2.set_xticklabels([labels[i] for i,ds in enumerate(all_results.keys())], fontsize=12)
ax2.set_ylabel('Gender Flip Rate (lower = fairer)', fontsize=12)
ax2.set_title('BASAP: Fairness Across Datasets — Baseline vs Raw LLM vs BASAP RF', fontsize=13)
ax2.legend(fontsize=11); ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/comparison/fairness_cross_dataset.png', dpi=150)
plt.show()

# ── Table: Summary ──
print('\n=== CROSS-DATASET SUMMARY ===')
for ds in all_results:
    df = all_results[ds]
    baseline = df[df['condition']==1]['f1_macro'].values[0]
    raw_llm = df[df['condition']==2]['f1_macro'].values[0]
    basap_rf = df[df['condition']==6]['f1_macro'].values[0]
    print(f'\n{ds.upper()}:')
    print(f'  Baseline F1:  {baseline:.4f}')
    print(f'  Raw LLM F1:   {raw_llm:.4f} ({"+" if raw_llm>baseline else ""}{raw_llm-baseline:.4f} vs baseline)')
    print(f'  BASAP RF F1:  {basap_rf:.4f} ({"+" if basap_rf>baseline else ""}{basap_rf-baseline:.4f} vs baseline)')

print('\nCross-dataset charts saved to results/comparison/')