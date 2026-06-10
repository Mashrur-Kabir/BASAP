# src/evaluation.py
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
from sklearn.metrics import f1_score, roc_auc_score, cohen_kappa_score
import os
from config import TRAIN_PATH, VAL_PATH, DOMAIN, RAW_SYNTHETIC_PATH, FILTERED_PATH, RESULTS_DIR, EMBEDDINGS_DIR

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print('Loading model (all-mpnet-base-v2)...')
model = SentenceTransformer('all-mpnet-base-v2')

# ── Load/cache val embeddings ──
val_df = pd.read_csv(VAL_PATH)
val_cache = f'{EMBEDDINGS_DIR}/val_emb.npy'
if os.path.exists(val_cache):
    print('Loading cached val embeddings...')
    val_emb = np.load(val_cache)
else:
    val_emb = model.encode(val_df['sentence'].tolist(), show_progress_bar=True)
    np.save(val_cache, val_emb)
y_val = val_df['label'].values

def perturbation_flip_rate(clf, embedding_model, texts, swap_type='gender'):
    if swap_type == 'gender':
        source_words = ['he', 'him', 'his', 'man', 'men', 'boy', 'male', 'father', 'husband']
        target_words = ['she', 'her', 'hers', 'woman', 'women', 'girl', 'female', 'mother', 'wife']
    else:
        source_words = ['young', 'child', 'teenager', 'adolescent', 'youth']
        target_words = ['elderly', 'senior', 'aged', 'older', 'geriatric']
    flip_count = 0
    total = 0
    for text in texts:
        if not isinstance(text, str): continue
        clean = re.sub(r'[^\w\s]', ' ', text.lower())
        words = clean.split()
        swapped = []
        changed = False
        for w in words:
            if w in source_words:
                swapped.append(target_words[source_words.index(w)])
                changed = True
            elif w in target_words:
                swapped.append(source_words[target_words.index(w)])
                changed = True
            else:
                swapped.append(w)
        if not changed:
            continue
        swapped_text = ' '.join(swapped)
        orig_pred = clf.predict(embedding_model.encode([text]))[0]
        pert_pred = clf.predict(embedding_model.encode([swapped_text]))[0]
        if orig_pred != pert_pred:
            flip_count += 1
        total += 1
    return round(flip_count / total, 4) if total > 0 else 0.0, total

# ── Setup training data ──
orig_df = pd.read_csv(TRAIN_PATH)
filtered_df = pd.read_csv(FILTERED_PATH)

pos = filtered_df[filtered_df.label == 1]
neg = filtered_df[filtered_df.label == 0]
min_count = min(len(pos), len(neg))
balanced_df = pd.concat([pos.sample(min_count, random_state=42),
                         neg.sample(min_count, random_state=42)])

train_dfs = {
    1: orig_df[['sentence', 'label']],
    2: pd.concat([orig_df[['sentence', 'label']],
                  pd.read_csv(RAW_SYNTHETIC_PATH)[['sentence', 'label']]]),
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

print(f'\n=== PHASE 1: PERFORMANCE EVALUATION ({DOMAIN.upper()}) ===')
performance_results = []
trained_clfs = {}

for cond_num, train_df in train_dfs.items():
    label = condition_names[cond_num].replace('\n', ' ')
    print(f'\nTraining Condition {cond_num}: {label}...')

    cache_path = f'{EMBEDDINGS_DIR}/train_c{cond_num}.npy'
    if os.path.exists(cache_path) and cond_num < 5:
        X_train = np.load(cache_path)
    else:
        X_train = model.encode(train_df['sentence'].tolist(), show_progress_bar=True)
        if cond_num < 5:
            np.save(cache_path, X_train)

    y_train = train_df['label'].values

    if classifiers[cond_num] == 'svm':
        clf = SVC(kernel='rbf', C=1.0, random_state=42, probability=True)
    elif classifiers[cond_num] == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif classifiers[cond_num] == 'nb':
        clf = GaussianNB()
    else:
        clf = LogisticRegression(max_iter=1000, random_state=42)

    clf.fit(X_train, y_train)
    trained_clfs[cond_num] = clf

    y_pred = clf.predict(val_emb)
    y_prob = clf.predict_proba(val_emb)[:, 1]
    f1 = f1_score(y_val, y_pred, average='macro')
    try:
        auc = roc_auc_score(y_val, y_prob)
    except Exception:
        auc = 0.0
    kappa = cohen_kappa_score(y_val, y_pred)

    print(f'Macro F1: {f1:.4f} | AUC-ROC: {auc:.4f} | Kappa: {kappa:.4f}')
    performance_results.append({
        'condition': cond_num,
        'name': label,
        'f1_macro': round(f1, 4),
        'auc_roc': round(auc, 4),
        'cohen_kappa': round(kappa, 4),
        'classifier': classifiers[cond_num]
    })

results_df = pd.DataFrame(performance_results)
results_df.to_csv(f'{RESULTS_DIR}/ablation_table.csv', index=False)
print('\n=== ABLATION TABLE ===')
print(results_df.to_string(index=False))

colors = ['#1f4e79', '#e74c3c', '#27ae60', '#2e75b6', '#8e44ad', '#d35400', '#16a085']

def dynamic_ylim(values, padding=0.05):
    vmin, vmax = min(values), max(values)
    span = max(vmax - vmin, 0.02)
    return max(0.0, vmin - span * 0.5), min(1.02, vmax + padding)

# F1 Plot
y_bot, y_top = dynamic_ylim(results_df['f1_macro'].tolist())
plt.figure(figsize=(13, 7))
bars = plt.bar(results_df['name'], results_df['f1_macro'], color=colors)
plt.ylim(y_bot, y_top)
plt.ylabel('Macro F1 Score', fontsize=12)
plt.title(f'BASAP Ablation: F1 Score by Condition\n({DOMAIN.title()})', fontsize=13)
plt.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, results_df['f1_macro']):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + (y_top - y_bot) * 0.01,
             f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/ablation_f1.png', dpi=150)
plt.show()

# AUC Plot
y_bot2, y_top2 = dynamic_ylim(results_df['auc_roc'].tolist())
plt.figure(figsize=(13, 7))
bars_auc = plt.bar(results_df['name'], results_df['auc_roc'], color=colors)
plt.ylim(y_bot2, y_top2)
plt.ylabel('AUC-ROC Score', fontsize=12)
plt.title(f'BASAP Ablation: AUC-ROC by Condition\n({DOMAIN.title()})', fontsize=13)
plt.grid(axis='y', alpha=0.3)
for bar, val in zip(bars_auc, results_df['auc_roc']):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + (y_top2 - y_bot2) * 0.01,
             f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/ablation_auc.png', dpi=150)
plt.show()

# ── PHASE 2: Fairness ──
print('\n=== PHASE 2: FAIRNESS EVALUATION (Civil Comments) ===')
jigsaw_df = pd.read_csv('data/raw/jigsaw_train.csv')
jigsaw_df = jigsaw_df.rename(columns={'comment_text': 'sentence', 'toxic': 'label'})

fairness_results = []
for cond_num, clf in trained_clfs.items():
    label = condition_names[cond_num].replace('\n', ' ')
    gender_rate, gender_n = perturbation_flip_rate(clf, model, jigsaw_df['sentence'].tolist(), 'gender')
    age_rate, age_n = perturbation_flip_rate(clf, model, jigsaw_df['sentence'].tolist(), 'age')
    print(f'Condition {cond_num} ({label}) | Gender Flip: {gender_rate:.4f} (n={gender_n}) | Age Flip: {age_rate:.4f} (n={age_n})')
    fairness_results.append({
        'condition': cond_num,
        'name': label,
        'gender_flip_rate': gender_rate,
        'age_flip_rate': age_rate,
    })

fairness_df = pd.DataFrame(fairness_results)
fairness_df.to_csv(f'{RESULTS_DIR}/fairness_jigsaw.csv', index=False)

# Gender Flip Plot
flip_vals = fairness_df['gender_flip_rate'].tolist()
yb, yt = dynamic_ylim(flip_vals, padding=0.03)
plt.figure(figsize=(13, 7))
bars2 = plt.bar(fairness_df['name'], fairness_df['gender_flip_rate'], color=colors)
plt.ylim(max(0, yb), yt)
plt.ylabel('Gender Flip Rate (lower = fairer)', fontsize=12)
plt.title(f'BASAP Fairness: Gender Perturbation Flip Rate\n(Civil Comments — {DOMAIN.title()})', fontsize=12)
plt.grid(axis='y', alpha=0.3)
for bar, val in zip(bars2, fairness_df['gender_flip_rate']):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + yt * 0.01,
             f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/fairness_jigsaw.png', dpi=150)
plt.show()

# Age Flip Plot
age_vals = fairness_df['age_flip_rate'].tolist()
yb3, yt3 = dynamic_ylim(age_vals, padding=0.03)
plt.figure(figsize=(13, 7))
bars3 = plt.bar(fairness_df['name'], fairness_df['age_flip_rate'], color=colors)
plt.ylim(max(0, yb3), yt3)
plt.ylabel('Age Flip Rate (lower = fairer)', fontsize=12)
plt.title(f'BASAP Fairness: Age Perturbation Flip Rate\n(Civil Comments — {DOMAIN.title()})', fontsize=12)
plt.grid(axis='y', alpha=0.3)
for bar, val in zip(bars3, fairness_df['age_flip_rate']):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + yt3 * 0.01,
             f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/fairness_age.png', dpi=150)
plt.show()

print(f'\nDone! Results saved to {RESULTS_DIR}/')