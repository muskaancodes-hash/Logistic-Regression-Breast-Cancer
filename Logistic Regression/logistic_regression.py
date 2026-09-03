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
# Separate features and target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Train-test split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

# Standardize features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeatures standardized successfully!")
# Train Logistic Regression model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully!")

# Make predictions
y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred[:10])
# Part 4: Model Evaluation

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Display Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# Precision
precision = precision_score(y_test, y_pred)

# Recall
recall = recall_score(y_test, y_pred)

# ROC-AUC
y_probability = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_probability)

print("\nPrecision:", precision)
print("Recall:", recall)
print("ROC-AUC:", roc_auc)
