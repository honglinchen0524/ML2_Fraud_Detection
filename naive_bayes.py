import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score, precision_score, matthews_corrcoef, roc_auc_score
import sys

# Modify suffix
suffix = sys.argv[1] if len(sys.argv) > 1 else '_34'

# Load data
X_train = pd.read_csv(f'data/X_train{suffix}.csv')
y_train = pd.read_csv(f'data/y_train{suffix}.csv')['y']
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')['y']

print(f"Train: {len(y_train)}, Test: {len(y_test)}")

# Train and fit
nb = GaussianNB()
nb.fit(X_train, y_train)
probs = nb.predict_proba(X_test)[:, 1]
preds = (probs >= 0.5).astype(int)

# Metrics
print(f"Sensitivity: {recall_score(y_test, preds): f}")
print(f"Precision: {precision_score(y_test, preds, zero_division=0): f}")
print(f"MCC: {matthews_corrcoef(y_test, preds): f}")
print(f"AUC-ROC: {roc_auc_score(y_test, probs): f}")

# Save
pd.DataFrame({'prob': probs}).to_csv(f'results/nb{suffix}.csv', index=False)
