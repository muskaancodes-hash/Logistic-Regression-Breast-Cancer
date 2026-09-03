import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data 2.csv")

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Dataset information
print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())

# Target variable distribution
print("\nDiagnosis distribution:")
print(df["diagnosis"].value_counts())

# Remove unnecessary columns
df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

# Convert diagnosis into binary values
# M = 1 (Malignant)
# B = 0 (Benign)
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

print("\nDataset after preprocessing:")
print(df.head())

print("\nFinal shape:")
print(df.shape)
