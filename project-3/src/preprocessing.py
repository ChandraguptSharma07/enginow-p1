import pandas as pd
import numpy as np

def clean_data(df):
    """
    Basic data cleaning:
    - Fix TotalCharges dtype
    - Drop missing values
    """
    # Force numeric, coerce errors
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Drop the small number of missing values
    df.dropna(subset=['TotalCharges'], inplace=True)
    return df

def engineer_features(df):
    """
    Creates new features:
    - TenureGroup
    - TotalServices
    - ContractRisk
    - AvgMonthlySpend
    """
    # 1. Tenure Cohorts
    labels = ['0-12', '12-24', '24-48', '48-60', '60+']
    bins = [0, 12, 24, 48, 60, 1000]
    df['TenureGroup'] = pd.cut(df['Tenure'], bins=bins, labels=labels)

    # 2. Total Services
    services = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['TotalServices'] = (df[services] == 'Yes').sum(axis=1)

    # 3. Contract Risk
    risk_map = {'Month-to-month': 'High', 'One year': 'Medium', 'Two year': 'Low'}
    df['ContractRisk'] = df['Contract'].map(risk_map)

    # 4. Avg Monthly Spend
    df['AvgMonthlySpend'] = df['TotalCharges'] / df['Tenure'].replace(0, 1)

    return df

def get_processed_data(filepath):
    """Full pipeline."""
    df = pd.read_csv(filepath)
    df = clean_data(df)
    df = engineer_features(df)
    return df
