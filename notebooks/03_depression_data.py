# notebooks/03_depression_data.py
# Depression Reddit dataset — binary classification
# Dataset: mrjunos/depression-reddit-cleaned (~7,000 samples)
# Labels: 1 = depressed, 0 = not depressed
# Source: https://huggingface.co/datasets/mrjunos/depression-reddit-cleaned

from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import os

os.makedirs('data/raw', exist_ok=True)

print("Loading depression-reddit-cleaned dataset...")
dataset = load_dataset("mrjunos/depression-reddit-cleaned", split="train")
df = dataset.to_pandas()

print(f"Total samples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Label distribution:\n{df['label'].value_counts()}")
print(f"\nSample text:\n{df['text'].iloc[0][:200]}")

# Rename to BASAP format
df = df.rename(columns={'text': 'sentence'})
df['label'] = df['label'].astype(int)
df = df[['sentence', 'label']].dropna()

# Remove very short texts
df = df[df['sentence'].str.len() > 30].reset_index(drop=True)
print(f"\nAfter cleaning: {len(df)} samples")

# Balance classes — take equal numbers from each
df_pos = df[df['label'] == 1]
df_neg = df[df['label'] == 0]
n_each = min(len(df_pos), len(df_neg), 400)  # 400 per class = 800 total

df_balanced = pd.concat([
    df_pos.sample(n=n_each, random_state=42),
    df_neg.sample(n=n_each, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Balanced dataset: {len(df_balanced)} samples")

# 80/20 stratified split
train_df, val_df = train_test_split(
    df_balanced,
    test_size=0.2,
    random_state=42,
    stratify=df_balanced['label']
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

train_df.to_csv('data/raw/depression_train.csv', index=False)
val_df.to_csv('data/raw/depression_val.csv', index=False)

print(f"\nTrain: {len(train_df)} samples")
print(f"Val:   {len(val_df)} samples")
print(f"Train label distribution: {train_df['label'].value_counts().to_dict()}")
print(f"Val label distribution:   {val_df['label'].value_counts().to_dict()}")
print("\nSaved to data/raw/depression_train.csv and data/raw/depression_val.csv")
print("\nNext: open src/config.py and set DATASET = 'depression'")
print("Then delete all .npy files in data/embeddings/ and rerun the pipeline.")