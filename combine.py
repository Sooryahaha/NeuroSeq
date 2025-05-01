# import pandas as pd

# # Load the Alzheimer's dataset
# alzheimers_df = pd.read_csv('alzheimers_misfolded_proteins.csv')

# # Add a new column 'Label' to represent Alzheimer's Disease
# # You can assign Alzheimer's disease as label 0
# alzheimers_df['Label'] = 0  # Label for Alzheimer's Disease

# # Load the Parkinson's dataset
# parkinsons_df = pd.read_csv('parkinsons_misfolded_proteins.csv')

# # Add a new column 'Label' to represent Parkinson's Disease
# # You can assign Parkinson's disease as label 1
# parkinsons_df['Label'] = 1  # Label for Parkinson's Disease

# # Combine the two datasets
# combined_df = pd.concat([alzheimers_df, parkinsons_df], ignore_index=True)

# # Shuffle the combined dataset to ensure randomness
# combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# # Save the combined dataset to a new CSV file
# combined_df.to_csv('combined_protein_disease_data.csv', index=False)

# print("Datasets combined and saved as 'combined_protein_disease_data.csv'")



# -------------final----------------

import pandas as pd
from sklearn.utils import shuffle

# Load the original combined Alzheimer’s + Parkinson’s dataset
df_combined = pd.read_csv("combined_protein_disease_data.csv")

# Load the new datasets
df_other = pd.read_csv("other_neurodegenerative_diseases.csv")
df_healthy = pd.read_csv("healthy_individuals.csv")

# Ensure all datasets have the same columns
required_columns = [
    "Sequence_ID", "Protein_Type", "Sequence", "Mutations", 
    "PTMs", "Misfolding_Risk_Score", "Reference", "Label"
]

# Downsample if necessary to ensure class balance (e.g., max 10,000 per class)
def balance_classes(df, label_value, max_samples=10000):
    class_df = df[df["Label"] == label_value]
    if len(class_df) > max_samples:
        return class_df.sample(n=max_samples, random_state=42)
    return class_df

# Add missing labels if needed for Alzheimer’s and Parkinson’s
if "Label" not in df_combined.columns:
    df_combined["Label"] = df_combined["Protein_Type"].apply(
        lambda x: 0 if x == "Amyloid-beta" else 1
    )

# Balance all datasets
df_alzheimers = balance_classes(df_combined, 0)
df_parkinsons = balance_classes(df_combined, 1)
df_other = balance_classes(df_other, 2)
df_healthy = balance_classes(df_healthy, 3)

# Combine all datasets
final_df = pd.concat([df_alzheimers, df_parkinsons, df_other, df_healthy], ignore_index=True)

# Shuffle to ensure mixed data order
final_df = shuffle(final_df, random_state=42)

# Save to CSV
final_df.to_csv("final_protein_dataset.csv", index=False)

print("✅ Final dataset created and saved as 'final_protein_dataset.csv'")