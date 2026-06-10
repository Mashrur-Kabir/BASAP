# notebooks/02_medical_data.py
# Prepares the medical dataset for BASAP pipeline
# FIX: Uses proper 80/20 train/val split — previous version left val empty

from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import os

os.makedirs('data/raw', exist_ok=True)

print("Loading dataset...")
# Load both train and test splits and combine to get more samples
train_split = load_dataset("gretelai/symptom_to_diagnosis", split="train")
test_split = load_dataset("gretelai/symptom_to_diagnosis", split="test")

df_train = train_split.to_pandas()
df_test = test_split.to_pandas()
df = pd.concat([df_train, df_test], ignore_index=True)

print(f"Total samples loaded: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Rename columns to match BASAP format
df = df.rename(columns={
    'input_text': 'sentence',
    'output_text': 'label'
})

# Show all available conditions
print(f"\nAll conditions and counts:")
print(df['label'].value_counts().to_string())

# Pick the two most common conditions for binary classification
top2 = df['label'].value_counts().head(10).index.tolist()
print(f"\nSelected top 2 conditions: {top2}")

df_binary = df[df['label'].isin(top2)].copy()
print(f"Total samples for these 2 conditions: {len(df_binary)}")

# Map to 0 and 1
label_map = {cond: 1 for cond in top2[:5]}
label_map.update({cond: 0 for cond in top2[5:]})
df_binary['label'] = df_binary['label'].map(label_map)
df_binary = df_binary[['sentence', 'label']].reset_index(drop=True)

# FIX: Use proper 80/20 stratified split instead of trying to sample fixed numbers
train_df, val_df = train_test_split(
    df_binary,
    test_size=0.2,
    random_state=42,
    stratify=df_binary['label']
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# Save
train_df.to_csv('data/raw/medical_train.csv', index=False)
val_df.to_csv('data/raw/medical_val.csv', index=False)

print(f"\nTrain: {len(train_df)} samples")
print(f"Val:   {len(val_df)} samples")
print(f"Train label distribution: {train_df['label'].value_counts().to_dict()}")
print(f"Val label distribution:   {val_df['label'].value_counts().to_dict()}")
print("\nFiles saved to data/raw/medical_train.csv and data/raw/medical_val.csv")