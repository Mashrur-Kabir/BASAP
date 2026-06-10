# notebooks/04_hate_data.py
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import os

os.makedirs('data/raw', exist_ok=True)

print("Loading hate speech dataset (cardiffnlp/tweet_eval)...")
dataset = load_dataset("cardiffnlp/tweet_eval", "hate", split="train")
df = dataset.to_pandas()

print(f"Total: {len(df)} | Columns: {df.columns.tolist()}")
print(f"Label distribution:\n{df['label'].value_counts()}")

# Already binary: 0 = not hate, 1 = hate
df = df.rename(columns={'text': 'sentence'})
df = df[['sentence', 'label']].dropna()
df = df[df['sentence'].str.len() > 20].reset_index(drop=True)

# Balance: take equal from each class
df_pos = df[df['label'] == 1]
df_neg = df[df['label'] == 0]
n_each = min(len(df_pos), len(df_neg), 400)

df_balanced = pd.concat([
    df_pos.sample(n=n_each, random_state=42),
    df_neg.sample(n=n_each, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

train_df, val_df = train_test_split(
    df_balanced, test_size=0.2, random_state=42, stratify=df_balanced['label']
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

train_df.to_csv('data/raw/hate_train.csv', index=False)
val_df.to_csv('data/raw/hate_val.csv', index=False)

print(f"\nTrain: {len(train_df)} | Val: {len(val_df)}")
print(f"Train labels: {train_df['label'].value_counts().to_dict()}")