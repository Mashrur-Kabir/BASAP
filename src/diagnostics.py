# src/diagnostics.py
# IMPROVEMENT: Fixed perturbation test to strip punctuation — catches "he," "she." etc.
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

os.makedirs('data/embeddings', exist_ok=True)

synth_df = pd.read_csv('data/synthetic/sst2_raw_augmented.csv')
orig_df = pd.read_csv('data/raw/sst2_train.csv')

# IMPROVEMENT: Upgraded to all-mpnet-base-v2 (stronger, still CPU-feasible)
print('Loading embedding model (all-mpnet-base-v2)...')
model = SentenceTransformer('all-mpnet-base-v2')

# ── STEP 11: Lexical Bias ──
def lexical_bias_check(df, text_col='sentence', label_col='label'):
    gendered_words = {
        'male': ['he', 'him', 'his', 'man', 'men', 'boy', 'father', 'husband', 'sir'],
        'female': ['she', 'her', 'hers', 'woman', 'women', 'girl', 'mother', 'wife', "ma'am"]
    }
    results = {}
    for label in df[label_col].unique():
        subset = df[df[label_col] == label][text_col].str.lower().str.split()
        all_words = [word for sent in subset for word in sent]
        # IMPROVEMENT: Strip punctuation from words before counting
        all_words = [re.sub(r'[^\w]', '', w) for w in all_words]
        total = len(all_words)
        results[f'label_{label}'] = {}
        for group, words in gendered_words.items():
            count = sum(all_words.count(w) for w in words)
            results[f'label_{label}'][group] = count / total if total > 0 else 0
    return pd.DataFrame(results)

print('=== LEXICAL BIAS: Original Data ===')
print(lexical_bias_check(orig_df).to_string())
print('\n=== LEXICAL BIAS: Raw Synthetic Data ===')
print(lexical_bias_check(synth_df).to_string())

# ── STEP 12: Embedding Distribution ──
print('\nEncoding original data...')
orig_emb = model.encode(orig_df['sentence'].tolist(), show_progress_bar=True)
np.save('data/embeddings/orig_emb.npy', orig_emb)

print('Encoding synthetic data...')
synth_emb = model.encode(synth_df['sentence'].tolist(), show_progress_bar=True)
np.save('data/embeddings/synth_emb.npy', synth_emb)

intra_synth_sim = cosine_similarity(synth_emb).mean()
intra_orig_sim = cosine_similarity(orig_emb).mean()
print(f'\nIntra-similarity (synthetic): {intra_synth_sim:.4f}')
print(f'Intra-similarity (original):  {intra_orig_sim:.4f}')
print('-> High synthetic similarity = low diversity (ARTIFACT)')

all_emb = np.vstack([orig_emb[:200], synth_emb[:200]])
tsne_labels = ['original']*200 + ['synthetic']*200
coords = TSNE(n_components=2, random_state=42).fit_transform(all_emb)

plt.figure(figsize=(8, 6))
for label, color in [('original', 'blue'), ('synthetic', 'red')]:
    mask = [l == label for l in tsne_labels]
    plt.scatter(coords[mask, 0], coords[mask, 1], c=color, label=label, alpha=0.5, s=20)
plt.legend()
plt.title('Embedding Space: Original vs Raw Synthetic')
plt.savefig('results/tsne_raw.png', dpi=150, bbox_inches='tight')
plt.show()

# ── STEP 13: Label Consistency ──
verifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli', device=-1)
candidate_labels = ['positive review', 'negative review']

def check_label_consistency(df, text_col='sentence', label_col='label'):
    mismatches = []
    for i, row in df.iterrows():
        result = verifier(row[text_col], candidate_labels)
        predicted = 1 if result['labels'][0] == 'positive review' else 0
        if predicted != row[label_col]:
            mismatches.append(i)
    return mismatches

bad_indices = check_label_consistency(synth_df)
print(f'\nLabel mismatches: {len(bad_indices)} / {len(synth_df)}')
print(f'Mismatch rate: {len(bad_indices)/len(synth_df)*100:.1f}%')

with open('results/label_mismatches.json', 'w') as f:
    json.dump(bad_indices, f)

# ── STEP 14: Perturbation Test (FIXED: strips punctuation) ──
def perturbation_test(texts, clf, embedding_model):
    male_words = ['he', 'him', 'his', 'man', 'boy']
    female_words = ['she', 'her', 'hers', 'woman', 'girl']
    flip_count = 0
    total = 0
    for text in texts:
        if not isinstance(text, str): continue
        # IMPROVEMENT: Strip punctuation so "he," and "she." are caught
        words = re.sub(r'[^\w\s]', ' ', text.lower()).split()
        female_version = ' '.join(
            female_words[male_words.index(w)] if w in male_words else w for w in words
        )
        if female_version == ' '.join(words):
            continue
        orig_pred = clf.predict(embedding_model.encode([text]))[0]
        pert_pred = clf.predict(embedding_model.encode([female_version]))[0]
        if orig_pred != pert_pred:
            flip_count += 1
        total += 1
    return flip_count / total if total > 0 else 0

X_train = model.encode(orig_df['sentence'].tolist())
y_train = orig_df['label'].values
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

flip_rate = perturbation_test(synth_df['sentence'].tolist(), clf, model)
print(f'\nLabel flip rate under gender perturbation: {flip_rate:.4f}')
print('Interpretation: > 0.10 indicates significant gender bias in classifier')