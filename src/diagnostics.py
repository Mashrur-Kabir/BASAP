# src/diagnostics.py
import json
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from transformers import pipeline
import matplotlib.pyplot as plt
import os
from config import TRAIN_PATH, LABEL_1_NAME, LABEL_0_NAME, DOMAIN, RAW_SYNTHETIC_PATH, RESULTS_DIR, EMBEDDINGS_DIR, MISMATCHES_PATH, DIAGNOSTICS_PATH

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

synth_df = pd.read_csv(RAW_SYNTHETIC_PATH)
orig_df = pd.read_csv(TRAIN_PATH)

print('Loading embedding model (all-mpnet-base-v2)...')
model = SentenceTransformer('all-mpnet-base-v2')

diagnostic_report = {}  # ENHANCEMENT: Save all findings to JSON

# ── STEP 11: Lexical Bias ──
def lexical_bias_check(df, text_col='sentence', label_col='label'):
    gendered_words = {
        'male': ['he', 'him', 'his', 'man', 'men', 'boy', 'father', 'husband', 'sir', 'male'],
        'female': ['she', 'her', 'hers', 'woman', 'women', 'girl', 'mother', 'wife', "ma'am", 'female']
    }
    results = {}
    for label in df[label_col].unique():
        subset = df[df[label_col] == label][text_col].str.lower().str.split()
        all_words = [word for sent in subset for word in sent]
        all_words = [re.sub(r'[^\w]', '', w) for w in all_words]
        total = len(all_words)
        results[f'label_{label}'] = {}
        for group, words in gendered_words.items():
            count = sum(all_words.count(w) for w in words)
            results[f'label_{label}'][group] = round(count / total, 6) if total > 0 else 0
    return pd.DataFrame(results)

orig_bias = lexical_bias_check(orig_df)
synth_bias = lexical_bias_check(synth_df)
print('=== LEXICAL BIAS: Original Data ===')
print(orig_bias.to_string())
print('\n=== LEXICAL BIAS: Raw Synthetic Data ===')
print(synth_bias.to_string())

diagnostic_report['lexical_bias_original'] = orig_bias.to_dict()
diagnostic_report['lexical_bias_synthetic'] = synth_bias.to_dict()

# ── ENHANCEMENT: Vocabulary Diversity (Type-Token Ratio) ──
def type_token_ratio(df, text_col='sentence'):
    all_words = []
    for text in df[text_col].dropna():
        words = re.sub(r'[^\w\s]', ' ', str(text).lower()).split()
        all_words.extend(words)
    if not all_words:
        return 0
    return round(len(set(all_words)) / len(all_words), 4)

orig_ttr = type_token_ratio(orig_df)
synth_ttr = type_token_ratio(synth_df)
print(f'\n=== VOCABULARY DIVERSITY (Type-Token Ratio) ===')
print(f'Original data TTR:  {orig_ttr:.4f}')
print(f'Synthetic data TTR: {synth_ttr:.4f}')
print(f'-> Higher TTR = more diverse vocabulary. Lower synthetic TTR = repetitiveness artifact.')
diagnostic_report['ttr_original'] = orig_ttr
diagnostic_report['ttr_synthetic'] = synth_ttr

# ── STEP 12: Embedding Distribution ──
print('\nEncoding original data...')
orig_emb = model.encode(orig_df['sentence'].tolist(), show_progress_bar=True, batch_size=32)
np.save('data/embeddings/orig_emb.npy', orig_emb)

print('Encoding synthetic data...')
synth_emb = model.encode(synth_df['sentence'].tolist(), show_progress_bar=True, batch_size=32)
np.save('data/embeddings/synth_emb.npy', synth_emb)

intra_synth_sim = float(cosine_similarity(synth_emb).mean())
intra_orig_sim = float(cosine_similarity(orig_emb).mean())
ratio = round(intra_synth_sim / intra_orig_sim, 2) if intra_orig_sim > 0 else 0
print(f'\nIntra-similarity (synthetic): {intra_synth_sim:.4f}')
print(f'Intra-similarity (original):  {intra_orig_sim:.4f}')
print(f'Ratio: {ratio}x more repetitive' if ratio > 1 else f'Ratio: {ratio}x')
print('-> High synthetic similarity = low diversity (ARTIFACT)')

diagnostic_report['intra_similarity_synthetic'] = intra_synth_sim
diagnostic_report['intra_similarity_original'] = intra_orig_sim
diagnostic_report['repetitiveness_ratio'] = ratio

# t-SNE plot — use min of 200 or actual count
n_plot = min(200, len(orig_emb), len(synth_emb))
all_emb = np.vstack([orig_emb[:n_plot], synth_emb[:n_plot]])
tsne_labels = ['original']*n_plot + ['synthetic']*n_plot
coords = TSNE(n_components=2, random_state=42).fit_transform(all_emb)

plt.figure(figsize=(8, 6))
for label, color in [('original', 'blue'), ('synthetic', 'red')]:
    mask = [l == label for l in tsne_labels]
    plt.scatter(coords[mask, 0], coords[mask, 1], c=color, label=label, alpha=0.5, s=20)
plt.legend()
plt.title(f'Embedding Space: Original vs Raw Synthetic ({DOMAIN.title()})')
plt.savefig(f'{RESULTS_DIR}/tsne_raw.png', dpi=150, bbox_inches='tight')
plt.show()

# ── STEP 13: Label Consistency ──
# ENHANCEMENT: Uses actual class names from config for accurate zero-shot verification
verifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli', device=-1)
candidate_labels = [LABEL_1_NAME, LABEL_0_NAME]
print(f'\nLabel consistency check using:')
print(f'  Label 1: "{LABEL_1_NAME[:60]}..."')
print(f'  Label 0: "{LABEL_0_NAME[:60]}..."')

def check_label_consistency(df, text_col='sentence', label_col='label'):
    mismatches = []
    for i, row in df.iterrows():
        result = verifier(row[text_col], candidate_labels)
        predicted = 1 if result['labels'][0] == LABEL_1_NAME else 0
        if predicted != row[label_col]:
            mismatches.append(i)
    return mismatches

bad_indices = check_label_consistency(synth_df)
mismatch_rate = round(len(bad_indices)/len(synth_df)*100, 1)
print(f'\nLabel mismatches: {len(bad_indices)} / {len(synth_df)}')
print(f'Mismatch rate: {mismatch_rate}%')

diagnostic_report['label_mismatches'] = len(bad_indices)
diagnostic_report['mismatch_rate_pct'] = mismatch_rate
diagnostic_report['total_synthetic'] = len(synth_df)

with open(MISMATCHES_PATH, 'w') as f:
    json.dump(bad_indices, f)

# ── STEP 14: Perturbation Test (Gender + Age) ──
def perturbation_test(texts, clf, embedding_model, swap_type='gender'):
    if swap_type == 'gender':
        source_words = ['he', 'him', 'his', 'man', 'men', 'boy', 'male', 'father', 'husband']
        target_words = ['she', 'her', 'hers', 'woman', 'women', 'girl', 'female', 'mother', 'wife']
    else:  # age
        source_words = ['young', 'child', 'teenager', 'adolescent', 'youth']
        target_words = ['elderly', 'senior', 'aged', 'older', 'geriatric']

    flip_count = 0
    total = 0
    for text in texts:
        if not isinstance(text, str): continue
        words = re.sub(r'[^\w\s]', ' ', text.lower()).split()
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
        orig_pred = clf.predict(embedding_model.encode([text], batch_size=1))[0]
        pert_pred = clf.predict(embedding_model.encode([swapped_text], batch_size=1))[0]
        if orig_pred != pert_pred:
            flip_count += 1
        total += 1
    return round(flip_count / total, 4) if total > 0 else 0, total

X_train = model.encode(orig_df['sentence'].tolist(), batch_size=32)
y_train = orig_df['label'].values
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

gender_flip, gender_total = perturbation_test(synth_df['sentence'].tolist(), clf, model, 'gender')
age_flip, age_total = perturbation_test(synth_df['sentence'].tolist(), clf, model, 'age')

print(f'\n=== PERTURBATION TESTS (on synthetic data) ===')
print(f'Gender flip rate: {gender_flip:.4f} (tested {gender_total} sentences with gendered words)')
print(f'Age flip rate:    {age_flip:.4f} (tested {age_total} sentences with age words)')
print('Interpretation: > 0.10 indicates significant demographic bias')

diagnostic_report['gender_flip_rate_synthetic'] = gender_flip
diagnostic_report['gender_sentences_tested'] = gender_total
diagnostic_report['age_flip_rate_synthetic'] = age_flip
diagnostic_report['age_sentences_tested'] = age_total

# ── Save full diagnostic report ──
with open(DIAGNOSTICS_PATH, 'w') as f:
    json.dump(diagnostic_report, f, indent=2)
print(f'\nFull diagnostic report saved to {DIAGNOSTICS_PATH}')