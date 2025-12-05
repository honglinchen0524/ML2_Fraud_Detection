import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    recall_score, precision_score, matthews_corrcoef,
    roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve, average_precision_score, f1_score
)

# Load
y_test = pd.read_csv('data/y_test.csv')['y'].values
n_fraud = y_test.sum()
n_legit = len(y_test) - n_fraud

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
plt.savefig('results/evaluation.png')
plt.show()


###### additional evaluations
def compute_metrics(y_true, probs, threshold=0.5):
    """Compute all metrics for given predictions."""
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    return {
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'F1-Score': f1,
        'MCC': matthews_corrcoef(y_true, preds),
        'AUC-ROC': roc_auc_score(y_true, probs),
        'AUC-PR': average_precision_score(y_true, probs),
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
    }
    
    
models = {
    '34:66': {'MLP': mlp_34, 'Naive Bayes': nb_34, 'Ensemble': ens_34},
    '10:90': {'MLP': mlp_10, 'Naive Bayes': nb_10, 'Ensemble': ens_10}
}

## Additional metrics
results = []
for dist, model_dict in models.items():
    for name, probs in model_dict.items():
        metrics = compute_metrics(y_test, probs)
        metrics['Distribution'] = dist
        metrics['Model'] = name
        results.append(metrics)

df_results = pd.DataFrame(results)

cols_display = ['Distribution', 'Model', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score', 'MCC', 'AUC-ROC', 'AUC-PR']
print(df_results[cols_display].round(4).to_string(index=False))
df_results.to_csv('results/all_results.csv', index=False)

## Different Threshold
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
print("\nEnsemble - 34:66:")
print(f"{'Threshold':>10} {'Sensitivity':>12} {'Precision':>12} {'MCC':>12} {'F1':>12} {'AUC-ROC':>12}") 
print("-" * 60)
for thresh in thresholds:
    m = compute_metrics(y_test, ens_34, threshold=thresh)
    print(f"{thresh:>10.1f} {m['Sensitivity']:>12.4f} {m['Precision']:>12.4f} {m['MCC']:>12.4f} {m['F1-Score']:>12.4f} {m['AUC-ROC']:>12.4f}")

print("\nEnsemble - 10:90:")
print(f"{'Threshold':>10} {'Sensitivity':>12} {'Precision':>12} {'MCC':>12} {'F1':>12} {'AUC-ROC':>12}")
print("-" * 60)
for thresh in thresholds:
    m = compute_metrics(y_test, ens_10, threshold=thresh)
    print(f"{thresh:>10.1f} {m['Sensitivity']:>12.4f} {m['Precision']:>12.4f} {m['MCC']:>12.4f} {m['F1-Score']:>12.4f} {m['AUC-ROC']:>12.4f}")


## Confusion Matrix
for _, row in df_results.iterrows():
    print(f"\n{row['Model']} ({row['Distribution']}):")
    print(f"TP (Frauds Caught):    {int(row['TP']):,}")
    print(f"TN (Legit no Faud):    {int(row['TN']):,}")
    print(f"FP (False Alarms):    {int(row['FP']):,}")
    print(f"FN (Frauds Missed):   {int(row['FN']):,}")
    
## Best Model for each metrix
metrics_to_compare = ['Sensitivity', 'Specificity', 'Precision', 'MCC', 'AUC-ROC']
print("\nBest model per metric:")
for metric in metrics_to_compare:
    best_idx = df_results[metric].idxmax()
    best_row = df_results.loc[best_idx]
    print(f"  {metric:12}: {best_row['Model']} ({best_row['Distribution']}) = {best_row[metric]:f}")
    
    
## Check if ensemble improves
for dist in ['34:66', '10:90']:
    df_dist = df_results[df_results['Distribution'] == dist]
    ens_row = df_dist[df_dist['Model'] == 'Ensemble'].iloc[0]
    mlp_row = df_dist[df_dist['Model'] == 'MLP'].iloc[0]
    nb_row = df_dist[df_dist['Model'] == 'Naive Bayes'].iloc[0]
    
    print(f"\n{dist} Distribution:")
    for metric in ['Sensitivity', 'Precision', 'MCC', 'AUC-ROC']:
        ens_val = ens_row[metric]
        mlp_val = mlp_row[metric]
        nb_val = nb_row[metric]
        best_individual = max(mlp_val, nb_val)
        improvement = ens_val - best_individual
        
        status = "Improve" if improvement > 0 else ("= No Change" if improvement == 0 else "Worse")
        print(f"{metric:12}: Ensemble={ens_val:f}, Best Individual={best_individual:f} | {status} ({improvement:+f})")


## Plot precision-recal 
for dist, probs_dict in [('34_66', {'MLP': mlp_34, 'Naive Bayes': nb_34, 'Ensemble': ens_34}),
                          ('10_90', {'MLP': mlp_10, 'Naive Bayes': nb_10, 'Ensemble': ens_10})]:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, probs in probs_dict.items():
        precision, recall, _ = precision_recall_curve(y_test, probs)
        ap = average_precision_score(y_test, probs)
        ax.plot(recall, precision, label=f'{name} (AP={ap:.4f})')
    
    # Baseline
    baseline = n_fraud / len(y_test)
    ax.axhline(y=baseline, color='k', linestyle='--', label=f'Random Baseline ({baseline:.4f})')
    
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision Recall Curve ({dist.replace("_", ":")})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'results/fig3_precision_recall_{dist}.png')
    plt.close()

## Plot Probability Distribution
for dist, probs_dict in [('34_66', {'MLP': mlp_34, 'Naive Bayes': nb_34, 'Ensemble': ens_34}),
                          ('10_90', {'MLP': mlp_10, 'Naive Bayes': nb_10, 'Ensemble': ens_10})]:
    for name, probs in probs_dict.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Separate by class
        probs_fraud = probs[y_test == 1]
        probs_legit = probs[y_test == 0]
        
        ax.hist(probs_legit, bins=50, alpha=0.6, color='blue', label=f'Legit', density=True)
        ax.hist(probs_fraud, bins=50, alpha=0.6, color='red', label=f'Fraud', density=True)
        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, label='Threshold (0.5)')
        
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Density')
        ax.set_title(f'{name} ({dist.replace("_", ":")}) - Probability Distribution')
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f'results/fig4_prob_dist_{name.replace(" ", "_")}_{dist}.png')
        plt.close()


## plot Confusion Matrix 
cmaps = {'MLP': 'Blues', 'Naive Bayes': 'Purples', 'Ensemble': 'Oranges'}
for dist, probs_dict in [('34_66', {'MLP': mlp_34, 'Naive Bayes': nb_34, 'Ensemble': ens_34}),
                          ('10_90', {'MLP': mlp_10, 'Naive Bayes': nb_10, 'Ensemble': ens_10})]:
    for name, probs in probs_dict.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        preds = (probs >= 0.5).astype(int)
        cm = confusion_matrix(y_test, preds)
        
        im = ax.imshow(cm, cmap=cmaps[name])
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
        ax.set_title(f'{name} ({dist.replace("_", ":")}) - Confusion Matrix', fontsize=11, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Legit', 'Fraud'])
        ax.set_yticklabels(['Legit', 'Fraud'])
        
        for i in range(2):
            for j in range(2):
                color = 'white' if cm[i, j] > cm.max()/2 else 'black'
                ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center', 
                       fontsize=12, fontweight='bold', color=color)
        plt.tight_layout()
        plt.savefig(f'results/fig5_confusion_{name.replace(" ", "_")}_{dist}.png')
        plt.close()