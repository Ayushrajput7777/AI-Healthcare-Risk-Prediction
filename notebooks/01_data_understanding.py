import pandas as pd

print("Healthcare Risk Prediction Project")

df = pd.read_csv("data/heart.csv")

print(df.head())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove rows with missing values
df = df.dropna()

print("\nAfter Cleaning:")
print(df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Target column
print("\nTarget Values:")
print(df["num"].value_counts())

print("\nTarget Values:")
print(df["num"].value_counts())

# Separate features and target

X = df.drop("num", axis=1)
y = df["num"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())