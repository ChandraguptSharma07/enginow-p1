# Data Preprocessing Script

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

df = pd.read_csv('data/credit_risk_dataset.csv')
print(f"Original shape: {df.shape}")

# fill missing
df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)
for grade in df['loan_grade'].unique():
    mask = (df['loan_grade'] == grade) & (df['loan_int_rate'].isnull())
    median_rate = df[df['loan_grade'] == grade]['loan_int_rate'].median()
    df.loc[mask, 'loan_int_rate'] = median_rate

# fix outliers
df.loc[df['person_age'] > 100, 'person_age'] = df['person_age'].median()
df['person_emp_length'] = df.apply(lambda x: min(x['person_emp_length'], x['person_age'] - 18), axis=1)

# feature engineering
df['age_group'] = pd.cut(df['person_age'], bins=[0, 25, 35, 45, 55, 100], labels=['18-25', '26-35', '36-45', '46-55', '55+'])
grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
df['grade_score'] = df['loan_grade'].map(grade_map)
df['prev_default'] = df['cb_person_default_on_file'].map({'N': 0, 'Y': 1})
df['risk_score'] = df['grade_score'] + df['prev_default'] * 2
df['income_stability'] = np.where((df['person_home_ownership'].isin(['OWN', 'MORTGAGE'])) & (df['person_emp_length'] > 3), 1, 0)

# encode
home_dummies = pd.get_dummies(df['person_home_ownership'], prefix='home')
intent_dummies = pd.get_dummies(df['loan_intent'], prefix='intent')
age_dummies = pd.get_dummies(df['age_group'], prefix='age')
df = pd.concat([df, home_dummies, intent_dummies, age_dummies], axis=1)

drop_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file', 'age_group']
df_clean = df.drop(columns=drop_cols)

X = df_clean.drop('loan_status', axis=1)
y = df_clean['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scale_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'grade_score', 'risk_score']
scaler = StandardScaler()
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols] = scaler.transform(X_test[scale_cols])

os.makedirs('data/processed', exist_ok=True)
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

with open('data/processed/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print("Preprocessing complete")
