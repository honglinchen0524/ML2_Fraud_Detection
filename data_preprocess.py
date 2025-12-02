import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

np.random.seed(2025)

# Load
df = pd.read_csv('creditcard.csv')
print(f"Loaded: {df.shape[0]} samples, Fraud: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")

X = df.drop('Class', axis=1)
y = df['Class']

# Standardize Amount and Time
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])
X['Time'] = scaler.fit_transform(X[['Time']])

# Using Awoyemi paper's data preprocessing strategy
# 70:30 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2025, stratify=y)
print(f"Train: {len(y_train)}, Test: {len(y_test)}")

# Hybrid sampling for 34:66
n_fraud = y_train.sum()
n_fraud_target = n_fraud * 3
n_legit_target = min(int(n_fraud_target * 0.66 / 0.34), len(y_train) - n_fraud)
smote = SMOTE(sampling_strategy={1: n_fraud_target}, random_state=2025)
X_temp, y_temp = smote.fit_resample(X_train, y_train)
under = RandomUnderSampler(sampling_strategy={0: n_legit_target}, random_state=2025)
X_34, y_34 = under.fit_resample(X_temp, y_temp)

print(f"34:66 -> {len(y_34)} samples, fraud: {y_34.mean()*100:.1f}%")

# Hybrid sampling for 10:90
n_legit_target = min(int(n_fraud_target * 0.90 / 0.10), len(y_train) - n_fraud)
smote = SMOTE(sampling_strategy={1: n_fraud_target}, random_state=2025)
X_temp, y_temp = smote.fit_resample(X_train, y_train)
under = RandomUnderSampler(sampling_strategy={0: n_legit_target}, random_state=2025)
X_10, y_10 = under.fit_resample(X_temp, y_temp)
print(f"10:90 -> {len(y_10)} samples, fraud: {y_10.mean()*100:.1f}%")

# Save
X_test.to_csv('data/X_test.csv', index=False)
pd.DataFrame({'y': y_test}).to_csv('data/y_test.csv', index=False)

X_34.to_csv('data/X_train_34.csv', index=False)
pd.DataFrame({'y': y_34}).to_csv('data/y_train_34.csv', index=False)

X_10.to_csv('data/X_train_10.csv', index=False)
pd.DataFrame({'y': y_10}).to_csv('data/y_train_10.csv', index=False)