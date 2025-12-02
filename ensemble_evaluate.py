import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    recall_score, precision_score, matthews_corrcoef,
    roc_auc_score, roc_curve, confusion_matrix
)

# Load
y_test = pd.read_csv('data/y_test.csv')['y'].values

# 34:66
mlp_34 = pd.read_csv('results/mlp_34.csv')['prob'].values
nb_34 = pd.read_csv('results/nb_34.csv')['prob'].values
# Soft voting
ens_34 = (mlp_34 + nb_34) / 2

# 10:90
mlp_10 = pd.read_csv('results/mlp_10.csv')['prob'].values
nb_10 = pd.read_csv('results/nb_10.csv')['prob'].values
# Soft voting
ens_10 = (mlp_10 + nb_10) / 2

# Evaluate models
results = []
for suffix, probs_list in [('34:66', [mlp_34, nb_34, ens_34]), ('10:90', [mlp_10, nb_10, ens_10])]:
    for name, probs in zip(['MLP', 'Naive Bayes', 'Ensemble'], probs_list):
        preds = (probs >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        results.append({
            'Distribution': suffix,
            'Model': name,
            'Sensitivity': recall_score(y_test, preds),
            'Specificity': tn / (tn + fp),
            'Precision': precision_score(y_test, preds, zero_division=0),
            'MCC': matthews_corrcoef(y_test, preds),
            'AUC-ROC': roc_auc_score(y_test, probs)
        })

df = pd.DataFrame(results)
print(df[['Distribution', 'Model', 'Sensitivity', 'Precision', 'MCC', 'AUC-ROC']].round(4).to_string(index=False))

df.to_csv('results/model_result.csv', index=False)

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

# ROC - 34:66
ax = axes[0, 0]
for probs, name, c in [(mlp_34, 'MLP', 'blue'), (nb_34, 'NB', 'green'), (ens_34, 'Ensemble', 'red')]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    ax.plot(fpr, tpr, c, label=f'{name} ({roc_auc_score(y_test, probs):.3f})')
ax.plot([0,1], [0,1], 'k--', alpha=0.3)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('ROC 34:66')
ax.legend(loc='lower right')

# ROC - 10:90
ax = axes[0, 1]
for probs, name, c in [(mlp_10, 'MLP', 'blue'), (nb_10, 'NB', 'green'), (ens_10, 'Ensemble', 'red')]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    ax.plot(fpr, tpr, c, label=f'{name} ({roc_auc_score(y_test, probs):.3f})')
ax.plot([0,1], [0,1], 'k--', alpha=0.3)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('ROC 10:90')
ax.legend(loc='lower right')

# Metrics bar chart
ax = axes[0, 2]
df_34 = df[df['Distribution'] == '34:66']
metrics = ['Sensitivity', 'Precision', 'MCC', 'AUC-ROC']
x = np.arange(len(metrics))
w = 0.25
for i, (_, row) in enumerate(df_34.iterrows()):
    ax.bar(x + i*w, [row[m] for m in metrics], w, label=row['Model'])
ax.set_xticks(x + w)
ax.set_xticklabels(metrics, rotation=15)
ax.set_title('Metrics 34:66')
ax.legend()
ax.set_ylim(0, 1.1)

# Confusion matrices
for idx, (probs, name, cmap) in enumerate([(mlp_34, 'MLP', 'Blues'), (nb_34, 'NB', 'Greens'), (ens_34, 'Ensemble', 'Reds')]):
    ax = axes[1, idx]
    cm = confusion_matrix(y_test, (probs >= 0.5).astype(int))
    ax.imshow(cm, cmap=cmap)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'{name} 34:66')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i,j], ha='center', va='center', fontsize=12)

plt.tight_layout()
plt.savefig('results/evaluation.png', dpi=150)
plt.show()
