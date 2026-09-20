import pandas as pd

# Path to anomaly dataset
file_path = "data/aws_anomaly_dataset.csv"

# Load CSV file
df = pd.read_csv(file_path)

print("ANOMALY DATASET LOADED")

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nShape:")
print(df.shape)

print("\nAnomaly count:")
print(df["is_anomaly"].value_counts())