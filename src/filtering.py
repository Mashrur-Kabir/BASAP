# src/filtering.py
import json, re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, f1_score
import os
from config import TRAIN_PATH, VAL_PATH, CONTROLLED_PATH, FILTERED_PATH, MISMATCHES_PATH, RESULTS_DIR, EMBEDDINGS_DIR

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs('data/filtered', exist_ok=True)

model = SentenceTransformer('all-mpnet-base-v2')

val_df = pd.read_csv(VAL_PATH)
val_cache = f'{EMBEDDINGS_DIR}/val_emb.npy'
val_emb = np.load(val_cache) if os.path.exists(val_cache) else model.encode(val_df['sentence'].tolist(), show_progress_bar=True)
if not os.path.exists(val_cache): np.save(val_cache, val_emb)
y_val = val_df['label'].values

def train_and_evaluate(train_df, condition_num, condition_name, classifier='lr'):
    cache_path = f'{EMBEDDINGS_DIR}/train_c{condition_num}.npy'
    if os.path.exists(cache_path):
        X_train = np.load(cache_path)
    else:
        X_train = model.encode(train_df['sentence'].tolist(), show_progress_bar=True)
        np.save(cache_path, X_train)

    y_train = train_df['label'].values
    if classifier == 'svm': clf = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    elif classifier == 'rf': clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif classifier == 'nb': clf = GaussianNB()
    else: clf = LogisticRegression(max_iter=1000, random_state=42)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(val_emb)
    f1 = f1_score(y_val, y_pred, average='macro')
    print(f'\n=== CONDITION {condition_num}: {condition_name} ({classifier.upper()}) ===')
    print(classification_report(y_val, y_pred))
    print(f'Macro F1: {f1:.4f}')
    with open(f'{RESULTS_DIR}/condition{condition_num}.json', 'w') as f:
        json.dump({'condition': condition_num, 'name': condition_name, 'f1_macro': f1, 'classifier': classifier}, f, indent=2)
    return clf

def filter_synthetic(synth_df, label_mismatch_indices=None, keep_per_label=200):
    synth_emb = model.encode(synth_df['sentence'].tolist(), show_progress_bar=True)
    keep_mask = np.ones(len(synth_df), dtype=bool)
    if label_mismatch_indices:
        keep_mask[label_mismatch_indices] = False
    clean_df = synth_df[keep_mask].copy().reset_index(drop=True)
    clean_emb = synth_emb[keep_mask]

    selected = []
    for label in [0, 1]:
        idx = np.where(clean_df['label'].values == label)[0]
        emb = clean_emb[idx]
        if len(idx) <= keep_per_label:
            selected.extend(idx.tolist()); continue
        mean_emb = emb.mean(axis=0, keepdims=True)
        first = idx[np.argmax(cosine_similarity(emb, mean_emb).flatten())]
        chosen, chosen_emb = [first], [clean_emb[first]]
        for _ in range(keep_per_label - 1):
            sims = cosine_similarity(emb, np.vstack(chosen_emb)).max(axis=1)
            for c in chosen:
                li = np.where(idx == c)[0]
                if len(li): sims[li[0]] = 1.0
            nxt = idx[np.argmin(sims)]
            chosen.append(nxt); chosen_emb.append(clean_emb[nxt])
        selected.extend(chosen)

    filtered = clean_df.iloc[selected].reset_index(drop=True)
    print(f'Kept {len(filtered)} / {len(synth_df)} | Removed {len(synth_df)-len(filtered)}')
    return filtered

with open(MISMATCHES_PATH) as f:
    mismatches = json.load(f)

orig_df = pd.read_csv(TRAIN_PATH)
ctrl_df = pd.read_csv(CONTROLLED_PATH)

print('\nFiltering...')
filtered_df = filter_synthetic(ctrl_df, label_mismatch_indices=mismatches, keep_per_label=200)
filtered_df.to_csv(FILTERED_PATH, index=False)

combined_c3 = pd.concat([orig_df[['sentence','label']], filtered_df[['sentence','label']]]).reset_index(drop=True)
train_and_evaluate(combined_c3, 3, 'basap_filtered', classifier='lr')

pos = filtered_df[filtered_df.label==1]; neg = filtered_df[filtered_df.label==0]
n = min(len(pos), len(neg))
balanced_df = pd.concat([pos.sample(n, random_state=42), neg.sample(n, random_state=42)])
combined_c4 = pd.concat([orig_df[['sentence','label']], balanced_df[['sentence','label']]]).reset_index(drop=True)
train_and_evaluate(combined_c4, 4, 'basap_filtered_balanced', classifier='lr')
train_and_evaluate(combined_c4, 5, 'basap_balanced_svm', classifier='svm')
train_and_evaluate(combined_c4, 6, 'basap_balanced_rf', classifier='rf')
train_and_evaluate(combined_c4, 7, 'basap_balanced_nb', classifier='nb')