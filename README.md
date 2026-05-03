# BASAP — Bias-Aware Synthetic Augmentation Pipeline

A lightweight, CPU-only pipeline that detects and reduces bias and artifacts in LLM-generated synthetic training data for low-resource text classification.

---

## The Problem

When researchers use Large Language Models (like ChatGPT or Llama) to generate fake training data for small datasets, the generated data carries hidden problems:
- **Demographic sterility** — the AI avoids writing about people in a gendered way entirely, unlike real human text
- **Low diversity** — AI sentences are 2.37× more repetitive than real human sentences
These problems silently hurt classifier accuracy and fairness.

## The Solution

BASAP runs in 4 steps:
1. **Diagnose** — detect lexical bias, diversity gaps, and label mismatches in raw synthetic data
2. **Controlled generation** — prompt the LLM with explicit balance rules and generate counterfactual pairs
3. **Filter** — select the most diverse samples using greedy diversity selection
4. **Evaluate** — compare performance (Macro F1) and fairness (gender flip rate) across 5 conditions

## Key Results

| Condition | Macro F1 |
|---|---|
| Baseline — real data only | 0.8806 |
| Raw LLM augmentation (no controls) | 0.8772 ↓ hurts |
| BASAP filtered (LR) | 0.8829 ↑ beats baseline |
| BASAP balanced (LR) | 0.8829 ↑ beats baseline |
| BASAP balanced (SVM) | 0.8829 ↑ beats baseline |

Blind augmentation hurt accuracy. BASAP-controlled augmentation improved it. No GPU used at any stage.

---

## Setup

### 1. Create virtual environment
```bash
python -m venv basap_env
```

### 2. Activate
```bash
# Windows
basap_env\Scripts\activate

# Mac/Linux
source basap_env/bin/activate
```

### 3. Deactivate (when done)
```bash
deactivate
```

### 4. Upgrade pip
```bash
python -m pip install --upgrade pip
```

### 5. Install dependencies
```bash
pip install fairlearn groq datasets matplotlib seaborn transformers jupyter python-dotenv sentence-transformers
```

### 6. Set up API key
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at [console.groq.com](https://console.groq.com)

---

## Run Order

```bash
python src/generation.py          # Generate raw synthetic data (400 samples)
python src/diagnostics.py         # Detect bias and artifacts in raw data
python src/controlled_generation.py  # Generate bias-controlled data (794 samples)
python src/baseline.py            # Train conditions 1 and 2
python src/filtering.py           # Filter + train conditions 3, 4, 5
python src/evaluation.py          # Build ablation table and charts
```

Results are saved to the `results/` folder.

---

## Project Structure

```
basap_thesis/
   data/
      raw/          # Original SST-2 and Civil Comments datasets
      synthetic/    # LLM-generated samples (raw and controlled)
      filtered/     # After BASAP diversity filtering
      embeddings/   # Cached sentence embeddings (auto-generated)
   notebooks/       # Data download notebook
   src/             # Pipeline modules
   results/         # Metrics, plots, ablation tables
   .env             # API key (never commit this)
```

---

## Requirements
- Python 3.10+
- No GPU required — runs entirely on CPU
- Free Groq API account for LLM generation
