# EDA Script for Credit Risk Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('data/credit_risk_dataset.csv')
print(f"Dataset Shape: {df.shape}")
print(df.head())

print("\n--- Column Types ---")
print(df.dtypes)

print("\n--- Missing Values ---")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\n--- Summary Statistics ---")
print(df.describe())

print("\n--- Age Distribution Check ---")
print(f"Min age: {df['person_age'].min()}")
print(f"Max age: {df['person_age'].max()}")
print(f"Ages > 100: {len(df[df['person_age'] > 100])}")

print("\n--- Target Variable ---")
print(df['loan_status'].value_counts())
print(f"Default Rate: {df['loan_status'].mean()*100:.1f}%")

default_ratio = df['loan_status'].value_counts()[0] / df['loan_status'].value_counts()[1]
print(f"Imbalance ratio: {default_ratio:.1f}:1")

print("\n--- Categorical Columns ---")
for col in ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']:
    print(f"\n{col}:")
    print(df[col].value_counts())

os.makedirs('outputs', exist_ok=True)

# plots
plt.figure(figsize=(8, 5))
df['loan_status'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Loan Status Distribution')
plt.savefig('outputs/target_distribution.png')
plt.close()

df_encoded = df.copy()
df_encoded['person_home_ownership'] = df_encoded['person_home_ownership'].astype('category').cat.codes
df_encoded['loan_intent'] = df_encoded['loan_intent'].astype('category').cat.codes
df_encoded['loan_grade'] = df_encoded['loan_grade'].astype('category').cat.codes
df_encoded['cb_person_default_on_file'] = df_encoded['cb_person_default_on_file'].map({'N': 0, 'Y': 1})

plt.figure(figsize=(10, 8))
corr_matrix = df_encoded.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', center=0)
plt.title('Correlation Heatmap')
plt.savefig('outputs/correlation_heatmap.png')
plt.close()

print("\n--- Correlations with Default ---")
print(corr_matrix['loan_status'].drop('loan_status').sort_values(ascending=False))

print("\nEDA complete - plots saved to outputs/")
