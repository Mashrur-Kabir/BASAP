# src/baseline.py
import pandas as pd
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score
from config import TRAIN_PATH, VAL_PATH, RAW_SYNTHETIC_PATH, COMBINED_C2_PATH, RESULTS_DIR, EMBEDDINGS_DIR

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print('Loading model (all-mpnet-base-v2)...')
model = SentenceTransformer('all-mpnet-base-v2')

val_df = pd.read_csv(VAL_PATH)
val_cache = f'{EMBEDDINGS_DIR}/val_emb.npy'
if os.path.exists(val_cache):
    print('Loading cached val embeddings...')
    val_embeddings = np.load(val_cache)
else:
    val_embeddings = model.encode(val_df['sentence'].tolist(), show_progress_bar=True)
    np.save(val_cache, val_embeddings)
y_val = val_df['label'].values

def train_and_evaluate(train_csv_path, condition_num, condition_name, classifier='lr'):
    train_df = pd.read_csv(train_csv_path)
    cache_path = f'{EMBEDDINGS_DIR}/train_c{condition_num}.npy'
    if os.path.exists(cache_path):
        print(f'Loading cached embeddings for condition {condition_num}...')
        X_train = np.load(cache_path)
    else:
        X_train = model.encode(train_df['sentence'].tolist(), show_progress_bar=True)
        np.save(cache_path, X_train)

    y_train = train_df['label'].values
    clf = SVC(kernel='rbf', C=1.0, random_state=42) if classifier == 'svm' else LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(val_embeddings)
    f1 = f1_score(y_val, y_pred, average='macro')
    print(f'\n=== CONDITION {condition_num}: {condition_name} ({classifier.upper()}) ===')
    print(classification_report(y_val, y_pred))
    print(f'Macro F1: {f1:.4f}')

    with open(f'{RESULTS_DIR}/condition{condition_num}.json', 'w') as f:
        json.dump({'condition': condition_num, 'name': condition_name, 'f1_macro': f1, 'classifier': classifier}, f, indent=2)
    return clf, y_val, y_pred

# Condition 1
train_and_evaluate(TRAIN_PATH, 1, 'original_only', classifier='lr')

# Condition 2
if os.path.exists(RAW_SYNTHETIC_PATH):
    orig_df = pd.read_csv(TRAIN_PATH)
    synth_df = pd.read_csv(RAW_SYNTHETIC_PATH)
    combined_df = pd.concat([orig_df[['sentence','label']], synth_df[['sentence','label']]])
    combined_df.to_csv(COMBINED_C2_PATH, index=False)
    train_and_evaluate(COMBINED_C2_PATH, 2, 'raw_llm_augmentation', classifier='lr')