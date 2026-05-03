# src/filtering.py
# IMPROVEMENTS:
# 1. Upgraded to all-mpnet-base-v2
# 2. Added within-synthetic diversity filter (removes near-duplicates inside synthetic set)
# 3. Added SVM as Condition 5
# 4. Loads cached embeddings where available

import json
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score
import os

os.makedirs('data/embeddings', exist_ok=True)

print('Loading model...')
model = SentenceTransformer('all-mpnet-base-v2')

# ── Load cached val embeddings if available ──
val_df = pd.read_csv('data/raw/sst2_val.csv')
val_cache = 'data/embeddings/val_emb.npy'
if os.path.exists(val_cache):
    print('Loading cached val embeddings...')
    val_emb = np.load(val_cache)
else:
    print('Encoding val set...')
    val_emb = model.encode(val_df['sentence'].tolist(), show_progress_bar=True)
    np.save(val_cache, val_emb)
y_val = val_df['label'].values

def train_and_evaluate(train_df, condition_num, condition_name, classifier='lr'):
    cache_path = f'data/embeddings/train_c{condition_num}.npy'
    if os.path.exists(cache_path):
        print(f'Loading cached train embeddings for condition {condition_num}...')
        X_train = np.load(cache_path)
    else:
        X_train = model.encode(train_df['sentence'].tolist(), show_progress_bar=True)
        np.save(cache_path, X_train)

    y_train = train_df['label'].values

    if classifier == 'svm':
        clf = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    else:
        clf = LogisticRegression(max_iter=1000, random_state=42)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(val_emb)
    f1 = f1_score(y_val, y_pred, average='macro')

    print(f'\n=== CONDITION {condition_num}: {condition_name} ({classifier.upper()}) ===')
    print(classification_report(y_val, y_pred))
    print(f'Macro F1: {f1:.4f}')

    with open(f'results/condition{condition_num}.json', 'w') as f:
        json.dump({'condition': condition_num, 'name': condition_name,
                   'f1_macro': f1, 'classifier': classifier}, f, indent=2)
    return clf

def filter_synthetic(orig_df, synth_df, label_mismatch_indices=None,
                     keep_per_label=200):
    '''
    Filter synthetic samples by:
    1. Removing label mismatches
    2. Keeping the most diverse samples per label using greedy diversity selection
    '''
    synth_emb = model.encode(synth_df['sentence'].tolist(), show_progress_bar=True)

    # Remove label mismatches first
    keep_mask = np.ones(len(synth_df), dtype=bool)
    if label_mismatch_indices:
        keep_mask[label_mismatch_indices] = False

    clean_df = synth_df[keep_mask].copy().reset_index(drop=True)
    clean_emb = synth_emb[keep_mask]

    # Greedy diversity selection per label
    # Pick samples that are LEAST similar to already-selected samples
    selected_indices = []
    for label in [0, 1]:
        label_mask = clean_df['label'].values == label
        label_indices = np.where(label_mask)[0]
        label_emb = clean_emb[label_indices]

        if len(label_indices) <= keep_per_label:
            selected_indices.extend(label_indices.tolist())
            continue

        # Start with the sample closest to mean (most representative)
        mean_emb = label_emb.mean(axis=0, keepdims=True)
        sims_to_mean = cosine_similarity(label_emb, mean_emb).flatten()
        first = label_indices[np.argmax(sims_to_mean)]
        chosen = [first]
        chosen_emb = [clean_emb[first]]

        # Greedily add samples most dissimilar to already chosen
        for _ in range(keep_per_label - 1):
            chosen_matrix = np.vstack(chosen_emb)
            sims = cosine_similarity(label_emb, chosen_matrix).max(axis=1)
            # Exclude already chosen
            for idx in chosen:
                local_idx = np.where(label_indices == idx)[0]
                if len(local_idx) > 0:
                    sims[local_idx[0]] = 1.0
            next_idx = label_indices[np.argmin(sims)]
            chosen.append(next_idx)
            chosen_emb.append(clean_emb[next_idx])

        selected_indices.extend(chosen)

    filtered = clean_df.iloc[selected_indices].reset_index(drop=True)
    print(f'Kept {len(filtered)} / {len(synth_df)} synthetic samples')
    print(f'Removed {len(synth_df) - len(filtered)} samples')
    return filtered

# ── Load data ──
with open('results/label_mismatches.json') as f:
    mismatches = json.load(f)

orig_df = pd.read_csv('data/raw/sst2_train.csv')
ctrl_df = pd.read_csv('data/synthetic/sst2_controlled.csv')

# ── Filter ──
print('\nFiltering synthetic data...')
filtered_df = filter_synthetic(orig_df, ctrl_df,
                               label_mismatch_indices=mismatches,
                               keep_per_label=200)
filtered_df.to_csv('data/filtered/sst2_basap_filtered.csv', index=False)

# ── Condition 3: Original + BASAP filtered (LR) ──
combined_c3 = pd.concat([orig_df[['sentence', 'label']],
                          filtered_df[['sentence', 'label']]]).reset_index(drop=True)
train_and_evaluate(combined_c3, 3, 'basap_filtered', classifier='lr')

# ── Condition 4: Original + BASAP balanced (LR) ──
pos = filtered_df[filtered_df.label == 1]
neg = filtered_df[filtered_df.label == 0]
min_count = min(len(pos), len(neg))
balanced_df = pd.concat([pos.sample(min_count, random_state=42),
                         neg.sample(min_count, random_state=42)])
combined_c4 = pd.concat([orig_df[['sentence', 'label']],
                          balanced_df[['sentence', 'label']]]).reset_index(drop=True)
train_and_evaluate(combined_c4, 4, 'basap_filtered_balanced', classifier='lr')

# ── Condition 5: Original + BASAP balanced (SVM) ──
# IMPROVEMENT: SVM with RBF kernel — typically outperforms LR on sentence embeddings
print('\nTraining SVM classifier (Condition 5)...')
train_and_evaluate(combined_c4, 5, 'basap_balanced_svm', classifier='svm')