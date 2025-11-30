import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler


# Make sure you download Kaggle csv from readme before running!!
df = pd.read_csv('creditcard.csv')

print(df.shape)
print(df['Class'].value_counts(normalize=True))  # class imbalance


X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Sampling methods (all are resampled to 1:1 class ratio)


# Undersampling
rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
X_train_under, y_train_under = rus.fit_resample(X_train, y_train)
print(f"Post-Undersampling training class balance: {y_train_under.value_counts(normalize=True)}")

# Oversampling
ros = RandomOverSampler(sampling_strategy=1.0, random_state=42)
X_train_over, y_train_over = ros.fit_resample(X_train, y_train)
print(f"Post-Oversampling training class balance: {y_train_over.value_counts(normalize=True)}")

# SMOTE
smote = SMOTE(sampling_strategy=1.0, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"Post-SMOTE training class balance: {y_train_smote.value_counts(normalize=True)}")
