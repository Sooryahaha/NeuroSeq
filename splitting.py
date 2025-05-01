import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("final_protein_dataset.csv")

# Shuffle and split: 70% train, 30% temp
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['Label'])

# Split the 30% temp into 15% val and 15% test (i.e., 50% each of temp)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['Label'])

# Save the splits
train_df.to_csv("train_protein_data.csv", index=False)
val_df.to_csv("val_protein_data.csv", index=False)
test_df.to_csv("test_protein_data.csv", index=False)

print("✅ Dataset successfully split and saved:")
print(f"→ Training set: {len(train_df)} samples")
print(f"→ Validation set: {len(val_df)} samples")
print(f"→ Test set: {len(test_df)} samples")