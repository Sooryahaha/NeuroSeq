# import pandas as pd
# import random

# def random_sequence(length, base_seq):
#     base_seq = base_seq[:length]
#     sequence = list(base_seq)
#     for _ in range(random.randint(0, 3)):
#         if len(sequence) > 0:
#             idx = random.randint(0, len(sequence) - 1)
#             sequence[idx] = random.choice("ACDEFGHIKLMNPQRSTVWY")
#     return ''.join(sequence)

# def random_mutation():
#     mutations = [
#         "A53T", "E46K", "D62H", "R262Q", "G2019S", "L166P", "PINK1 (R50Q)", "None"
#     ]
#     return random.choice(mutations)

# def random_ptm():
#     ptms = [
#         "None", "Phosphorylation (Ser129)", "Oxidation (Cys123)",
#         "Acetylation (Lys195)", "Ubiquitination (Lys48)", "Phosphorylation (Ser87)"
#     ]
#     return random.choice(ptms)

# def random_protein_type():
#     return random.choice(["Alpha-synuclein", "Parkin"])

# def generate_sequence_id(i):
#     return f"PD_{i:05d}"

# def random_misfolding_score():
#     return round(random.uniform(0.6, 0.99), 2)

# def random_reference():
#     return f"PMID:{random.randint(10000000, 99999999)}"

# # Dataset size
# n = 10000
# data = []

# for i in range(n):
#     protein_type = random_protein_type()
#     base_seq = (
#         "AGGGGQGGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGA" if protein_type == "Alpha-synuclein"
#         else "MELVQITLQKTHLGSMKESTGDYGRLIPLDGIKPSPPEKSPFKGV"  # Example sequence for Parkin
#     )
#     seq_length = len(base_seq)

#     data.append({
#         "Sequence_ID": generate_sequence_id(i),
#         "Protein_Type": protein_type,
#         "Sequence": random_sequence(seq_length, base_seq),
#         "Mutations": random_mutation(),
#         "PTMs": random_ptm(),
#         "Misfolding_Risk_Score": random_misfolding_score(),
#         "Reference": random_reference()
#     })

# # Save to CSV
# df = pd.DataFrame(data)
# df.to_csv("parkinsons_misfolded_proteins.csv", index=False)
# print("Dataset saved to parkinsons_misfolded_proteins.csv")





# -------------Other--------------

import pandas as pd
import random

# Amino acid alphabet
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Generate a random sequence with small mutations
def random_sequence(length):
    sequence = [random.choice(AMINO_ACIDS) for _ in range(length)]
    # Introduce a few mutations
    for _ in range(random.randint(0, 3)):
        idx = random.randint(0, length - 1)
        sequence[idx] = random.choice(AMINO_ACIDS)
    return ''.join(sequence)

# Mutation types
def random_mutation(label):
    if label == "Other Neurodegenerative Diseases":
        return random.choice([
            "A152T", "G51D", "E46K", "Q331K", "None", "D178N", "P105L"
        ])
    else:  # Healthy individuals have fewer or no known mutations
        return "None"

# PTMs
def random_ptm():
    return random.choice([
        "None", "Phosphorylation (Ser129)", "Ubiquitination (Lys63)", "Acetylation (Lys28)"
    ])

# Protein types (diverse for synthetic variety)
def random_protein_type(label):
    if label == "Other Neurodegenerative Diseases":
        return random.choice(["Alpha-synuclein", "TDP-43", "Prion Protein", "Huntingtin"])
    else:
        return random.choice(["Alpha-synuclein", "Tau", "Amyloid Precursor Protein"])

# Generate synthetic data
def generate_dataset(label, num_samples):
    data = []
    for i in range(num_samples):
        sequence_length = random.randint(40, 60)
        sequence = random_sequence(sequence_length)
        data.append({
            "Sequence_ID": f"{label[:3].upper()}_{i:05d}",
            "Protein_Type": random_protein_type(label),
            "Sequence": sequence,
            "Mutations": random_mutation(label),
            "PTMs": random_ptm(),
            "Misfolding_Risk_Score": round(random.uniform(0.5, 0.99), 2) if label != "Healthy Individuals" else round(random.uniform(0.0, 0.3), 2),
            "Reference": f"PMID:{random.randint(10000000, 99999999)}",
            "Label": 2 if label == "Other Neurodegenerative Diseases" else 3
        })
    return pd.DataFrame(data)

# Generate and save the datasets
df_other = generate_dataset("Other Neurodegenerative Diseases", 10000)
df_healthy = generate_dataset("Healthy Individuals", 10000)

df_other.to_csv("other_neurodegenerative_diseases.csv", index=False)
df_healthy.to_csv("healthy_individuals.csv", index=False)

print("✅ Datasets saved as 'other_neurodegenerative_diseases.csv' and 'healthy_individuals.csv'")