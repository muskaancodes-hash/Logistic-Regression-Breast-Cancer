# Logistic-Regression-Breast-Cancer
Binary classification project using Logistic Regression to predict breast cancer diagnosis using the Breast Cancer Wisconsin dataset.

## 📌 Project Overview

This project implements a binary classification model using Logistic Regression to predict whether a breast tumor is *Malignant (M)* or *Benign (B)*.

The project uses the Breast Cancer Wisconsin (Diagnostic) dataset from Kaggle.

## 🎯 Objective

The main objective is to build and evaluate a Logistic Regression classifier for binary classification.

## 🛠️ Tools & Libraries

- Python
- Pandas
- Scikit-learn
- Matplotlib

## 📊 Dataset

Dataset: Breast Cancer Wisconsin (Diagnostic)

- 569 records
- 30 numerical features
- Target variable: diagnosis
- M = Malignant
- B = Benign

## 🔧 Steps Performed

1. Loaded and explored the dataset.
2. Removed unnecessary columns such as id and Unnamed: 32.
3. Converted the diagnosis into binary values:
   - Malignant = 1
   - Benign = 0
4. Split the dataset into training and testing sets.
5. Standardized the features using StandardScaler.
6. Trained a Logistic Regression model.
7. Evaluated the model using:
   - Confusion Matrix
   - Precision
   - Recall
   - ROC-AUC
8. Tuned the classification threshold.
9. Explained the sigmoid function used in Logistic Regression.

## 📈 Model Evaluation

The model is evaluated using multiple classification metrics to understand its performance and ability to distinguish between malignant and benign cases.

## 📉 Sigmoid Function

Logistic Regression uses the sigmoid function to convert model output into a probability between 0 and 1.

A threshold is then used to convert the probability into a binary prediction.

## 📁 Project Structure

```text
Logistic Regression/
│
├── data2.csv
├── logistic_regression.py
└── README.md
