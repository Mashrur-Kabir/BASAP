# src/evaluation.py
# IMPROVEMENTS:
# 1. Upgraded to all-mpnet-base-v2
# 2. Fixed perturbation test — strips punctuation so "he," and "she." are caught
# 3. Loads cached embeddings — much faster re-runs
# 4. Added Condition 5 (SVM)
# 5. Better plots with clear labels and grid

import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
import os

os.makedirs('data/embeddings', exist_ok=True)
os.makedirs('results', exist_ok=True)

print('Loading model (all-mpnet-base-v2)...')
model = SentenceTransformer('all-mpnet-base-v2')

# ── Load/cache val embeddings ──
val_df = pd.read_csv('data/raw/sst2_val.csv')
val_cache = 'data/embeddings/val_emb.npy'
if os.path.exists(val_cache):
    print('Loading cached val embeddings...')
    val_emb = np.load(val_cache)
else:
    val_emb = model.encode(val_df['sentence'].tolist(), show_progress_bar=True)
    np.save(val_cache, val_emb)
y_val = val_df['label'].values

# ── IMPROVEMENT: Fixed perturbation function — strips punctuation ──
def perturbation_flip_rate(clf, embedding_model, texts):
    male_words = ['he', 'him', 'his', 'man', 'boy']
    female_words = ['she', 'her', 'hers', 'woman', 'girl']
    flip_count = 0
    total = 0
    for text in texts:
        if not isinstance(text, str): continue
        # Strip punctuation before word matching
        clean = re.sub(r'[^\w\s]', ' ', text.lower())
        words = clean.split()
        swapped = [
            female_words[male_words.index(w)] if w in male_words else
            (male_words[female_words.index(w)] if w in female_words else w)
            for w in words
        ]
        swapped_text = ' '.join(swapped)
        if swapped_text == ' '.join(words):
            continue
        orig_pred = clf.predict(embedding_model.encode([text]))[0]
        pert_pred = clf.predict(embedding_model.encode([swapped_text]))[0]
        if orig_pred != pert_pred:
            flip_count += 1
        total += 1
    return round(flip_count / total, 4) if total > 0 else 0.0

# ── Setup training data ──
orig_df = pd.read_csv('data/raw/sst2_train.csv')
filtered_df = pd.read_csv('data/filtered/sst2_basap_filtered.csv')

pos = filtered_df[filtered_df.label == 1]
neg = filtered_df[filtered_df.label == 0]
min_count = min(len(pos), len(neg))
balanced_df = pd.concat([pos.sample(min_count, random_state=42),
                         neg.sample(min_count, random_state=42)])

train_dfs = {
    1: orig_df[['sentence', 'label']],
    2: pd.concat([orig_df[['sentence', 'label']],
                  pd.read_csv('data/synthetic/sst2_raw_augmented.csv')[['sentence', 'label']]]),
    3: pd.concat([orig_df[['sentence', 'label']], filtered_df[['sentence', 'label']]]),
    4: pd.concat([orig_df[['sentence', 'label']], balanced_df[['sentence', 'label']]]),
    5: pd.concat([orig_df[['sentence', 'label']], balanced_df[['sentence', 'label']]]),
    6: pd.concat([orig_df[['sentence', 'label']], balanced_df[['sentence', 'label']]]),
    7: pd.concat([orig_df[['sentence', 'label']], balanced_df[['sentence', 'label']]]),
}

condition_names = {
    1: 'original_only\n(LR)',
    2: 'raw_llm\n(LR)',
    3: 'basap_filtered\n(LR)',
    4: 'basap_balanced\n(LR)',
    5: 'basap_balanced\n(SVM)',
    6: 'basap_balanced\n(RF)',
    7: 'basap_balanced\n(NB)',
}

classifiers = {1: 'lr', 2: 'lr', 3: 'lr', 4: 'lr', 5: 'svm', 6: 'rf', 7: 'nb'}

# ── PHASE 1: Performance ──
print('\n=== PHASE 1: PERFORMANCE EVALUATION (SST-2) ===')
performance_results = []
trained_clfs = {}

for cond_num, train_df in train_dfs.items():
    label = condition_names[cond_num].replace('\n', ' ')
    print(f'\nTraining Condition {cond_num}: {label}...')

    cache_path = f'data/embeddings/train_c{cond_num}.npy'
    if os.path.exists(cache_path) and cond_num < 5:
        X_train = np.load(cache_path)
    else:
        X_train = model.encode(train_df['sentence'].tolist(), show_progress_bar=True)
        if cond_num < 5:
            np.save(cache_path, X_train)

    y_train = train_df['label'].values

    if classifiers[cond_num] == 'svm':
        clf = SVC(kernel='rbf', C=1.0, random_state=42)
    elif classifiers[cond_num] == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif classifiers[cond_num] == 'nb':
        clf = GaussianNB()
    else:
        clf = LogisticRegression(max_iter=1000, random_state=42)

    clf.fit(X_train, y_train)
    trained_clfs[cond_num] = clf

    y_pred = clf.predict(val_emb)
    f1 = f1_score(y_val, y_pred, average='macro')
    print(f'Macro F1: {f1:.4f}')

    performance_results.append({
        'condition': cond_num,
        'name': label,
        'f1_macro': round(f1, 4),
        'classifier': classifiers[cond_num]
    })

results_df = pd.DataFrame(performance_results)
results_df.to_csv('results/ablation_table.csv', index=False)
print('\n=== ABLATION TABLE ===')
print(results_df.to_string(index=False))

# ── F1 Plot ──
colors = ['#1f4e79', '#e74c3c', '#27ae60', '#2e75b6', '#8e44ad', '#d35400', '#16a085']
plt.figure(figsize=(11, 6))
bars = plt.bar(results_df['name'], results_df['f1_macro'], color=colors)
plt.ylim(0.70, 0.92)
plt.ylabel('Macro F1 Score', fontsize=12)
plt.title('BASAP Ablation: F1 Score by Condition', fontsize=14)
plt.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, results_df['f1_macro']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('results/ablation_f1.png', dpi=150)
plt.show()

# ── PHASE 2: Fairness ──
print('\n=== PHASE 2: FAIRNESS EVALUATION (Civil Comments) ===')
jigsaw_df = pd.read_csv('data/raw/jigsaw_train.csv')
jigsaw_df = jigsaw_df.rename(columns={'comment_text': 'sentence', 'toxic': 'label'})

fairness_results = []
for cond_num, clf in trained_clfs.items():
    label = condition_names[cond_num].replace('\n', ' ')
    flip_rate = perturbation_flip_rate(clf, model, jigsaw_df['sentence'].tolist())
    print(f'Condition {cond_num} ({label}) | Flip Rate: {flip_rate:.4f}')
    fairness_results.append({
        'condition': cond_num,
        'name': label,
        'jigsaw_flip_rate': flip_rate
    })

fairness_df = pd.DataFrame(fairness_results)
fairness_df.to_csv('results/fairness_jigsaw.csv', index=False)

# ── Fairness Plot ──
plt.figure(figsize=(11, 6))
bars2 = plt.bar(fairness_df['name'], fairness_df['jigsaw_flip_rate'], color=colors)
plt.ylabel('Gender Flip Rate (lower = fairer)', fontsize=12)
plt.title('BASAP Fairness Evaluation — Civil Comments Gender Perturbation Test', fontsize=14)
plt.grid(axis='y', alpha=0.3)
for bar, val in zip(bars2, fairness_df['jigsaw_flip_rate']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
             f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('results/fairness_jigsaw.png', dpi=150)
plt.show()

print('\nDone! All results saved to results/')