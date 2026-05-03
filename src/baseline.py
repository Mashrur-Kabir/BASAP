# src/baseline.py
# IMPROVEMENTS:
# 1. Upgraded to all-mpnet-base-v2
# 2. Caches embeddings to disk — re-runs are instant
# 3. SVM option for Condition 5

import pandas as pd
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score

os.makedirs('data/embeddings', exist_ok=True)

print('Loading model (all-mpnet-base-v2)...')
model = SentenceTransformer('all-mpnet-base-v2')

# ── Load/cache validation embeddings ──
val_df = pd.read_csv('data/raw/sst2_val.csv')
val_cache = 'data/embeddings/val_emb.npy'
if os.path.exists(val_cache):
    print('Loading cached val embeddings...')
    val_embeddings = np.load(val_cache)
else:
    print('Encoding val set...')
    val_embeddings = model.encode(val_df['sentence'].tolist(), show_progress_bar=True)
    np.save(val_cache, val_embeddings)
y_val = val_df['label'].values

def train_and_evaluate(train_csv_path, condition_num, condition_name, classifier='lr'):
    train_df = pd.read_csv(train_csv_path)

    cache_path = f'data/embeddings/train_c{condition_num}.npy'
    if os.path.exists(cache_path):
        print(f'Loading cached embeddings for condition {condition_num}...')
        X_train = np.load(cache_path)
    else:
        X_train = model.encode(train_df['sentence'].tolist(), show_progress_bar=True)
        np.save(cache_path, X_train)

    y_train = train_df['label'].values

    if classifier == 'svm':
        clf = SVC(kernel='rbf', C=1.0, random_state=42)
    else:
        clf = LogisticRegression(max_iter=1000, random_state=42)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(val_embeddings)
    f1 = f1_score(y_val, y_pred, average='macro')
    print(f'\n=== CONDITION {condition_num}: {condition_name} ({classifier.upper()}) ===')
    print(classification_report(y_val, y_pred))
    print(f'Macro F1: {f1:.4f}')

    with open(f'results/condition{condition_num}.json', 'w') as f:
        json.dump({'condition': condition_num, 'name': condition_name,
                   'f1_macro': f1, 'classifier': classifier}, f, indent=2)

    return clf, y_val, y_pred

# ── Condition 1: Original only (LR) ──
train_and_evaluate('data/raw/sst2_train.csv', 1, 'original_only', classifier='lr')

# ── Condition 2: Original + raw synthetic (LR) ──
if os.path.exists('data/synthetic/sst2_raw_augmented.csv'):
    original_df = pd.read_csv('data/raw/sst2_train.csv')
    synthetic_df = pd.read_csv('data/synthetic/sst2_raw_augmented.csv')
    combined_df = pd.concat([original_df[['sentence','label']],
                              synthetic_df[['sentence','label']]])
    combined_df.to_csv('data/synthetic/combined_c2.csv', index=False)
    train_and_evaluate('data/synthetic/combined_c2.csv', 2, 'raw_llm_augmentation', classifier='lr')