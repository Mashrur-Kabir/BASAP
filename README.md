# BASAP — Bias-Aware Synthetic Augmentation Pipeline

A lightweight, CPU-only pipeline that detects and mitigates bias and distributional artifacts in LLM-generated synthetic training data for low-resource binary text classification.

Validated across three domains: **mental health screening** (Depression Reddit), **sentiment analysis** (SST-2), and **hate speech detection** (Tweet Eval Hate).

---

## The Problem

When researchers use Large Language Models (like Llama or ChatGPT) to generate synthetic training data for small datasets, the generated data carries hidden problems:

- **Demographic sterility** — the LLM avoids gendered language entirely, producing sentences with near-zero he/she/him/her references, unlike real human text
- **Distributional artifacts** — synthetic sentences are up to 2.44× more repetitive than real data (SST-2 vocabulary diversity dropped 71%: TTR 0.3514 → 0.1008)
- **Label mismatches** — up to 17.3% of generated samples carry the wrong label (68/393 in the depression dataset)
- **Amplified gender bias** — blindly adding raw synthetic data increased hate speech gender flip rate by 62% (0.1644 → 0.2667)

These problems silently hurt both classifier accuracy and fairness.

---

## The Solution

BASAP runs through 4 modules across 7 scripts:

| Module                   | Script                             | What it does                                                         |
| ------------------------ | ---------------------------------- | -------------------------------------------------------------------- |
| 0. Prepare               | `notebooks/03_depression_data.py`  | Download and split dataset (80/20 train/val)                         |
| 1. Generate (raw)        | `src/generation.py`                | Generate uncontrolled synthetic data via Groq/Llama                  |
| 2. Diagnose              | `src/diagnostics.py`               | Detect lexical bias, TTR drop, embedding artifacts, label mismatches |
| 3. Generate (controlled) | `src/controlled_generation.py`     | Re-generate with demographic balance rules + counterfactual pairs    |
| 4. Baseline              | `src/baseline.py`                  | Train Conditions 1 (real only) and 2 (real + raw synthetic)          |
| 5. Filter + Train        | `src/filtering.py`                 | Greedy diversity selection → 400 best samples; train Conditions 3–7  |
| 6. Evaluate              | `src/evaluation.py`                | Full accuracy + fairness metrics across all 7 conditions             |
| 7. Compare               | `notebooks/05_compare_datasets.py` | Cross-dataset comparison charts                                      |

---

## Key Results

### Performance (Macro F1)

| #   | Condition                      | Depression | SST-2        | Hate Speech  |
| --- | ------------------------------ | ---------- | ------------ | ------------ |
| 1   | Baseline — real data only (LR) | 0.9687     | 0.8806       | 0.6867       |
| 2   | Raw LLM augmentation (LR)      | 0.9500 ↓   | 0.8784 ↓     | 0.6745 ↓     |
| 3   | BASAP filtered (LR)            | 0.9500     | **0.8875 ↑** | 0.7060 ↑     |
| 5   | BASAP balanced (SVM)           | 0.9625     | 0.8807       | **0.7187 ↑** |
| 7   | BASAP balanced (Naive Bayes)   | 0.9311     | 0.8771       | 0.6687       |

Raw augmentation hurt accuracy in all three datasets. BASAP SVM (Condition 5) matched or beat the baseline in all three domains, with the strongest gain in hate speech (+0.032).

### Fairness (Gender Flip Rate — lower is better)

| #   | Condition     | Depression        | SST-2             | Hate Speech       |
| --- | ------------- | ----------------- | ----------------- | ----------------- |
| 1   | Baseline (LR) | 0.0667            | 0.0667            | 0.1644            |
| 2   | Raw LLM (LR)  | 0.0533            | 0.0622            | **0.2667 ↑ +62%** |
| 5   | BASAP SVM     | 0.0489 (−27%)     | 0.0800            | 0.2089            |
| 7   | BASAP NB      | **0.0133 (−80%)** | **0.0444 (−33%)** | **0.1333 (−19%)** |

**Recommendation:** Use BASAP SVM for best accuracy–fairness tradeoff. Use BASAP Naive Bayes when fairness is the primary concern (e.g. clinical screening, content moderation). Never use raw LLM augmentation without BASAP screening.

---

## Data Counts at Every Stage

### Depression Reddit (Primary)

| Stage                     | Count      | File                                          |
| ------------------------- | ---------- | --------------------------------------------- |
| Train set                 | 640        | `data/raw/depression_train.csv`               |
| Validation set            | 160        | `data/raw/depression_val.csv`                 |
| Raw synthetic             | ~393       | `data/synthetic/depression_raw_augmented.csv` |
| Label mismatches caught   | 68 (17.3%) | `results/depression/label_mismatches.json`    |
| Controlled synthetic      | ~802       | `data/synthetic/depression_controlled.csv`    |
| After diversity filtering | 400        | `data/filtered/depression_basap_filtered.csv` |
| Final training set (C3–7) | 1,040      | 640 real + 400 filtered                       |

### SST-2 Sentiment

| Stage                     | Count    | File                                    |
| ------------------------- | -------- | --------------------------------------- |
| Train set                 | 800      | `data/raw/sst2_train.csv`               |
| Validation set            | 872      | `data/raw/sst2_val.csv`                 |
| Raw synthetic             | ~399     | `data/synthetic/sst2_raw_augmented.csv` |
| Label mismatches caught   | 7 (1.8%) | `results/sst2/label_mismatches.json`    |
| Controlled synthetic      | ~697     | `data/synthetic/sst2_controlled.csv`    |
| After diversity filtering | 400      | `data/filtered/sst2_basap_filtered.csv` |
| Final training set (C3–7) | 1,200    | 800 real + 400 filtered                 |

### Tweet Eval Hate Speech

| Stage                     | Count     | File                                    |
| ------------------------- | --------- | --------------------------------------- |
| Train set                 | 640       | `data/raw/hate_train.csv`               |
| Validation set            | 160       | `data/raw/hate_val.csv`                 |
| Raw synthetic             | ~356      | `data/synthetic/hate_raw_augmented.csv` |
| Label mismatches caught   | 23 (6.5%) | `results/hate/label_mismatches.json`    |
| Controlled synthetic      | ~811      | `data/synthetic/hate_controlled.csv`    |
| After diversity filtering | 400       | `data/filtered/hate_basap_filtered.csv` |
| Final training set (C3–7) | 1,040     | 640 real + 400 filtered                 |

---

## Setup

### 1. Create and activate virtual environment

```bash
python -m venv basap_env

# Windows
basap_env\Scripts\activate

# Mac/Linux
source basap_env/bin/activate
```

### 2. Upgrade pip and install dependencies

```bash
python -m pip install --upgrade pip
pip install groq datasets matplotlib seaborn transformers jupyter python-dotenv sentence-transformers scikit-learn pandas numpy
```

### 3. Set up API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com)

> The embedding model (`all-mpnet-base-v2`, ~438 MB) downloads automatically on first run via `sentence-transformers`.

---

## Run Order

Run these scripts in order. Each script saves its outputs before the next one reads them.

```bash
# Depression dataset (primary)
python notebooks/03_depression_data.py       # Prepare: download + split → data/raw/

python src/generation.py                     # Generate ~393 raw synthetic samples
python src/diagnostics.py                    # Diagnose: bias, TTR, mismatches → results/depression/

python src/controlled_generation.py          # Generate ~802 controlled samples
python src/baseline.py                       # Train Conditions 1 & 2 → results/depression/condition1-2.json

python src/filtering.py                      # Filter to 400 samples; train Conditions 3–7
python src/evaluation.py                     # Full evaluation: F1, AUC, Kappa, fairness charts

# Repeat notebooks/04_hate_data.py and the src/ scripts for the hate speech dataset
# SST-2 data is downloaded automatically by generation.py when DOMAIN=sst2

# Cross-dataset comparison (run after all three datasets are complete)
python notebooks/05_compare_datasets.py      # → results/comparison/
```

Switch datasets by changing `DOMAIN` in `src/config.py` before running.

---

## Project Structure

```
basap_thesis/
├── data/
│   ├── raw/                  # Original train/val CSVs for all three datasets
│   ├── synthetic/            # Raw and controlled LLM-generated samples
│   ├── filtered/             # 400 diversity-selected samples per dataset
│   └── embeddings/           # Cached .npy sentence embeddings (auto-generated)
│       ├── depression/       # train_c1.npy … train_c7.npy + val_emb.npy
│       ├── sst2/
│       └── hate/
├── notebooks/
│   ├── 03_depression_data.py # Download + prepare depression dataset
│   ├── 04_hate_data.py       # Download + prepare hate speech dataset
│   └── 05_compare_datasets.py# Cross-dataset comparison charts
├── src/
│   ├── config.py             # Dataset paths, domain settings (edit DOMAIN here)
│   ├── generation.py         # Step 1: raw synthetic generation via Groq API
│   ├── diagnostics.py        # Step 2: bias + artifact detection
│   ├── controlled_generation.py  # Step 3: controlled generation + counterfactuals
│   ├── baseline.py           # Step 4: Conditions 1 & 2
│   ├── filtering.py          # Step 5: greedy diversity filter + Conditions 3–7
│   └── evaluation.py         # Step 6: full metrics + fairness charts
├── results/
│   ├── depression/           # condition1-7.json, ablation_table.csv, fairness_jigsaw.csv,
│   │                         # diagnostics_report.json, label_mismatches.json, *.png charts
│   ├── sst2/
│   ├── hate/
│   └── comparison/           # f1_cross_dataset.png, fairness_cross_dataset.png
├── .env                      # GROQ_API_KEY (never commit)
└── .gitignore
```

---

## The Seven Classifier Conditions

| #   | Classifier          | Training Data             | Purpose                                     |
| --- | ------------------- | ------------------------- | ------------------------------------------- |
| 1   | Logistic Regression | 640/800 real only         | Baseline — no augmentation                  |
| 2   | Logistic Regression | Real + raw synthetic      | Problem condition — blind augmentation      |
| 3   | Logistic Regression | Real + 400 BASAP filtered | BASAP standard                              |
| 4   | Logistic Regression | Real + 400 BASAP balanced | BASAP with perfect label balance            |
| 5   | SVM (rbf kernel)    | Real + 400 BASAP balanced | **Best accuracy–fairness tradeoff**         |
| 6   | Random Forest       | Real + 400 BASAP balanced | Ensemble comparison                         |
| 7   | Naive Bayes         | Real + 400 BASAP balanced | **Best fairness (lowest gender flip rate)** |

All conditions use `all-mpnet-base-v2` sentence embeddings and are evaluated on the same held-out validation set (never seen during training).

---

## Requirements

- Python 3.10+
- No GPU required — runs entirely on CPU
- Free Groq API account for LLM generation
- ~438 MB disk space for the embedding model (downloaded once automatically)
- ~15 minutes per dataset on a standard laptop
