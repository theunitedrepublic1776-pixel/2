# Screenshots – Employee Turnover Analytics
**Portobello Tech | HR Department Machine Learning Project**

This document reproduces the key code snippets, visualisation descriptions, and output results from `Employee_Turnover_Analytics.ipynb`. Because the notebook is run in an interactive Jupyter environment, the items below serve as the complete reference for all significant outputs generated during the analysis.

---

## Table of Contents
1. [Data Loading & Initial Inspection](#1-data-loading--initial-inspection)
2. [Data Quality Checks](#2-data-quality-checks)
3. [Correlation Heatmap](#3-correlation-heatmap)
4. [Distribution Plots](#4-distribution-plots)
5. [Project Count – Left vs. Stayed](#5-project-count--left-vs-stayed)
6. [K-Means Clustering Scatter Plot](#6-k-means-clustering-scatter-plot)
7. [Class Distribution Before & After SMOTE](#7-class-distribution-before--after-smote)
8. [5-Fold CV Classification Reports](#8-5-fold-cv-classification-reports)
9. [ROC / AUC Curves](#9-roc--auc-curves)
10. [Confusion Matrices](#10-confusion-matrices)
11. [Test-Set Classification Reports](#11-test-set-classification-reports)
12. [Risk Zone Distribution Bar Chart](#12-risk-zone-distribution-bar-chart)

---

## 1. Data Loading & Initial Inspection

### Code Snippet
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE

%matplotlib inline
sns.set_theme(style='whitegrid')
print('Libraries loaded successfully.')

df = pd.read_csv('HR_comma_sep.csv')
print('Shape:', df.shape)
df.head()
```

### Output
```
Libraries loaded successfully.
Shape: (14999, 10)
```

| | satisfaction_level | last_evaluation | number_project | average_montly_hours | time_spend_company | Work_accident | left | promotion_last_5years | sales | salary |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.38 | 0.53 | 2 | 157 | 3 | 0 | 1 | 0 | sales | low |
| 1 | 0.80 | 0.86 | 5 | 262 | 6 | 0 | 1 | 0 | sales | medium |
| 2 | 0.11 | 0.88 | 7 | 272 | 4 | 0 | 1 | 0 | sales | medium |
| 3 | 0.72 | 0.87 | 5 | 223 | 5 | 0 | 1 | 0 | sales | low |
| 4 | 0.37 | 0.52 | 2 | 159 | 3 | 0 | 1 | 0 | sales | low |

---

## 2. Data Quality Checks

### Code Snippet
```python
print('=== Data Types ===')
print(df.dtypes)
print('\n=== Missing Values ===')
missing = df.isnull().sum()
print(missing)
print('\nTotal missing values:', missing.sum())
print('\n=== Descriptive Statistics ===')
df.describe()
```

### Output – Data Types
```
satisfaction_level       float64
last_evaluation          float64
number_project             int64
average_montly_hours       int64
time_spend_company         int64
Work_accident              int64
left                       int64
promotion_last_5years      int64
sales                        str
salary                       str
dtype: object
```

### Output – Missing Values
```
satisfaction_level       0
last_evaluation          0
number_project           0
average_montly_hours     0
time_spend_company       0
Work_accident            0
left                     0
promotion_last_5years    0
sales                    0
salary                   0
Total missing values: 0
```

✅ **No missing values detected across all 10 columns.**

### Output – Descriptive Statistics

| | satisfaction_level | last_evaluation | number_project | average_montly_hours | time_spend_company |
|---|---|---|---|---|---|
| count | 14999 | 14999 | 14999 | 14999 | 14999 |
| mean | 0.6128 | 0.7161 | 3.8031 | 201.05 | 3.498 |
| std | 0.2486 | 0.1712 | 1.2326 | 49.94 | 1.460 |
| min | 0.09 | 0.36 | 2 | 96 | 2 |
| 25 % | 0.44 | 0.56 | 3 | 156 | 3 |
| 50 % | 0.64 | 0.72 | 4 | 200 | 3 |
| 75 % | 0.82 | 0.87 | 5 | 245 | 4 |
| max | 1.00 | 1.00 | 7 | 310 | 10 |

---

## 3. Correlation Heatmap

### Code Snippet
```python
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, square=True)
plt.title('Correlation Matrix – Numerical Features', fontsize=14)
plt.tight_layout()
plt.show()
```

### Visualisation Description
A 8×8 colour-coded correlation matrix (coolwarm palette) showing Pearson correlation coefficients between all numerical features. Warm (red) cells indicate positive correlation; cool (blue) cells indicate negative correlation.

**Key values highlighted:**

| Feature Pair | Correlation | Interpretation |
|---|---|---|
| `satisfaction_level` ↔ `left` | −0.39 | Strong negative – less satisfied employees leave more |
| `number_project` ↔ `average_montly_hours` | +0.42 | More projects → more hours worked |
| `last_evaluation` ↔ `number_project` | +0.35 | High performers handle more projects |
| `time_spend_company` ↔ `left` | +0.14 | Slight positive – longer tenure not immune to leaving |

---

## 4. Distribution Plots

### Code Snippet
```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(df['satisfaction_level'], kde=True, ax=axes[0], color='steelblue')
axes[0].set_title('Distribution of Employee Satisfaction', fontsize=13)
axes[0].set_xlabel('Satisfaction Level')

sns.histplot(df['last_evaluation'], kde=True, ax=axes[1], color='seagreen')
axes[1].set_title('Distribution of Employee Evaluation', fontsize=13)
axes[1].set_xlabel('Last Evaluation Score')

sns.histplot(df['average_montly_hours'], kde=True, ax=axes[2], color='coral')
axes[2].set_title('Distribution of Avg Monthly Hours', fontsize=13)
axes[2].set_xlabel('Average Monthly Hours')

plt.tight_layout()
plt.show()
```

### Visualisation Description
A 1×3 grid of histogram + KDE overlay plots (18×5 inches):

| Plot | Colour | Key Observation |
|---|---|---|
| **Satisfaction Level** | Steel Blue | Bimodal – spike near 0.1 (very dissatisfied) and peak at 0.6–0.8 |
| **Last Evaluation** | Sea Green | Roughly uniform; slight concentration at 0.7–0.9 (high performers retained) |
| **Avg Monthly Hours** | Coral | Bimodal – peaks around 150 hrs (normal load) and 250 hrs (overworked) |

---

## 5. Project Count – Left vs. Stayed

### Code Snippet
```python
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='number_project',
              hue='left', palette={0: 'steelblue', 1: 'tomato'})
plt.title('Employee Project Count: Left vs. Stayed', fontsize=14)
plt.xlabel('Number of Projects')
plt.ylabel('Count')
plt.legend(title='Left', labels=['Stayed (0)', 'Left (1)'])
plt.tight_layout()
plt.show()
```

### Visualisation Description
Grouped bar chart showing employee counts by number of projects (2–7), coloured by whether the employee stayed (steel blue) or left (tomato red).

| Number of Projects | Turnover Observation |
|---|---|
| 2 | Highest departure rate – disengaged / underutilised |
| 3–5 | Lowest turnover – optimal workload "sweet spot" |
| 6–7 | High turnover – burnout from overwork |

---

## 6. K-Means Clustering Scatter Plot

### Code Snippet
```python
df_left = df[df['left'] == 1][['satisfaction_level', 'last_evaluation', 'left']].copy()
print('Employees who left:', len(df_left))

X_cluster = df_left[['satisfaction_level', 'last_evaluation']]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_left['cluster'] = kmeans.fit_predict(X_cluster)

print('\nCluster Sizes:')
print(df_left['cluster'].value_counts())
print('\nCluster Centers:')
centers = pd.DataFrame(kmeans.cluster_centers_,
                        columns=['satisfaction_level', 'last_evaluation'])
centers.index.name = 'cluster'
print(centers)

palette = {0: 'tomato', 1: 'steelblue', 2: 'seagreen'}
plt.figure(figsize=(9, 6))
sns.scatterplot(data=df_left, x='satisfaction_level', y='last_evaluation',
                hue='cluster', palette=palette, alpha=0.6, s=60)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c='black', marker='X', zorder=5, label='Centroids')
plt.title('K-Means Clustering (k=3) – Employees Who Left', fontsize=14)
plt.xlabel('Satisfaction Level')
plt.ylabel('Last Evaluation')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()
```

### Output – Cluster Summary
```
Employees who left: 3571

Cluster Sizes:
cluster
0    1650
1     977
2     944

Cluster Centers (satisfaction_level, last_evaluation):
         satisfaction_level  last_evaluation
cluster
0                  0.410145         0.516982
1                  0.808516         0.911709
2                  0.111155         0.869301
```

### Visualisation Description
Scatter plot (9×6 inches) of the 3,571 leavers in satisfaction–evaluation space, coloured by cluster. Black X markers denote centroids.

| Cluster | Colour | Centre (Sat, Eval) | Profile |
|---|---|---|---|
| 0 | Tomato Red | (0.41, 0.52) | Dissatisfied High-Performers |
| 1 | Steel Blue | (0.81, 0.91) | High-Performers Who Left |
| 2 | Sea Green | (0.11, 0.87) | Dissatisfied Top-Performers (Burnout) |

---

## 7. Class Distribution Before & After SMOTE

### Code Snippet
```python
print('Class distribution (left):')
print(df_processed['left'].value_counts())
print('\nClass proportion:')
print(df_processed['left'].value_counts(normalize=True).round(3))

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print('After SMOTE – Train size:', X_train_sm.shape)
print('Class distribution after SMOTE:')
print(pd.Series(y_train_sm).value_counts())
```

### Output
```
Class distribution (left):
left
0    11428
1     3571

Class proportion:
left
0    0.762
1    0.238

Train size (before SMOTE): (11999, 20)  |  Test size: (3000, 20)

After SMOTE – Train size: (18284, 20)
Class distribution after SMOTE:
left
0    9142
1    9142
```

SMOTE oversampled the minority class (Left = 1) from 2,857 to 9,142 training samples, producing a perfectly balanced training set.

---

## 8. 5-Fold CV Classification Reports

### Code Snippet
```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def train_evaluate_cv(model, model_name):
    y_pred_cv = cross_val_predict(model, X_train_sm, y_train_sm, cv=skf)
    print(f'=== {model_name} – 5-Fold CV Classification Report (Train) ===')
    print(classification_report(y_train_sm, y_pred_cv,
                                 target_names=['Stayed', 'Left']))
    return model.fit(X_train_sm, y_train_sm)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr = train_evaluate_cv(lr, 'Logistic Regression')

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf = train_evaluate_cv(rf, 'Random Forest Classifier')

gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb = train_evaluate_cv(gb, 'Gradient Boosting Classifier')
```

### Output – Logistic Regression (5-Fold CV)
```
=== Logistic Regression – 5-Fold CV Classification Report (Train) ===
              precision    recall  f1-score   support

      Stayed       0.81      0.80      0.80      9142
        Left       0.80      0.82      0.81      9142

    accuracy                           0.81     18284
   macro avg       0.81      0.81      0.81     18284
weighted avg       0.81      0.81      0.81     18284
```

### Output – Random Forest (5-Fold CV)
```
=== Random Forest Classifier – 5-Fold CV Classification Report (Train) ===
              precision    recall  f1-score   support

      Stayed       0.97      0.99      0.98      9142
        Left       0.99      0.97      0.98      9142

    accuracy                           0.98     18284
   macro avg       0.98      0.98      0.98     18284
weighted avg       0.98      0.98      0.98     18284
```

### Output – Gradient Boosting (5-Fold CV)
```
=== Gradient Boosting Classifier – 5-Fold CV Classification Report (Train) ===
              precision    recall  f1-score   support

      Stayed       0.95      0.98      0.96      9142
        Left       0.98      0.95      0.96      9142

    accuracy                           0.96     18284
   macro avg       0.96      0.96      0.96     18284
weighted avg       0.96      0.96      0.96     18284
```

---

## 9. ROC / AUC Curves

### Code Snippet
```python
plt.figure(figsize=(9, 7))
roc_results = {}

for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    roc_results[name] = roc_auc
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves – Employee Turnover Prediction', fontsize=14)
plt.legend(loc='lower right', fontsize=11)
plt.tight_layout()
plt.show()

print('\nAUC Scores:')
for name, score in sorted(roc_results.items(), key=lambda x: -x[1]):
    print(f'  {name}: {score:.4f}')
```

### Output – AUC Scores
```
AUC Scores:
  Random Forest: 0.9951
  Gradient Boosting: 0.9852
  Logistic Regression: 0.8209
```

### Visualisation Description
A single 9×7 inch line plot showing the ROC curves for all three models against the diagonal random-classifier baseline. Random Forest (AUC = 0.9951) and Gradient Boosting (AUC = 0.9852) hug the top-left corner closely, while Logistic Regression (AUC = 0.8209) curves more gradually. The legend is positioned in the lower-right.

---

## 10. Confusion Matrices

### Code Snippet
```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=['Stayed', 'Left'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'{name}\nConfusion Matrix', fontsize=12)

plt.tight_layout()
plt.show()
```

### Visualisation Description
Three side-by-side (18×5 inches) blue-gradient confusion matrices, one per model. Each matrix shows True Positives, True Negatives, False Positives, and False Negatives for `Stayed` and `Left` classes on the 3,000-row test set.

| Model | TN (Stayed→Stayed) | FP (Stayed→Left) | FN (Left→Stayed) | TP (Left→Left) |
|---|---|---|---|---|
| Logistic Regression | ~1829 | ~457 | ~200 | ~514 |
| Random Forest | ~2264 | ~22 | ~14 | ~700 |
| Gradient Boosting | ~2240 | ~46 | ~40 | ~674 |

---

## 11. Test-Set Classification Reports

### Code Snippet
```python
for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f'=== {name} – Test Set Classification Report ===')
    print(classification_report(y_test, y_pred, target_names=['Stayed', 'Left']))
```

### Output – Logistic Regression (Test Set)
```
=== Logistic Regression – Test Set Classification Report ===
              precision    recall  f1-score   support

      Stayed       0.90      0.80      0.85      2286
        Left       0.53      0.72      0.61       714

    accuracy                           0.78      3000
   macro avg       0.71      0.76      0.73      3000
weighted avg       0.81      0.78      0.79      3000
```

### Output – Random Forest (Test Set)
```
=== Random Forest – Test Set Classification Report ===
              precision    recall  f1-score   support

      Stayed       0.99      0.99      0.99      2286
        Left       0.97      0.98      0.97       714

    accuracy                           0.99      3000
   macro avg       0.98      0.98      0.98      3000
weighted avg       0.99      0.99      0.99      3000
```

### Output – Gradient Boosting (Test Set)
```
=== Gradient Boosting – Test Set Classification Report ===
              precision    recall  f1-score   support

      Stayed       0.98      0.98      0.98      2286
        Left       0.94      0.94      0.94       714

    accuracy                           0.97      3000
   macro avg       0.96      0.96      0.96      3000
weighted avg       0.97      0.97      0.97      3000
```

### Summary Metrics Table

| Model | Accuracy | AUC | Precision (Left) | Recall (Left) | F1 (Left) |
|---|---|---|---|---|---|
| Logistic Regression | 78 % | 0.8209 | 0.53 | 0.72 | 0.61 |
| Gradient Boosting | 97 % | 0.9852 | 0.94 | 0.94 | 0.94 |
| **Random Forest** | **99 %** | **0.9951** | **0.97** | **0.98** | **0.97** |

---

## 12. Risk Zone Distribution Bar Chart

### Code Snippet
```python
def assign_zone(prob):
    if prob < 0.20:
        return 'Safe Zone (Green)'
    elif prob < 0.60:
        return 'Low-Risk Zone (Yellow)'
    elif prob < 0.90:
        return 'Medium-Risk Zone (Orange)'
    else:
        return 'High-Risk Zone (Red)'

results_df['risk_zone'] = results_df['prob_leaving'].apply(assign_zone)

zone_order = [
    'Safe Zone (Green)',
    'Low-Risk Zone (Yellow)',
    'Medium-Risk Zone (Orange)',
    'High-Risk Zone (Red)'
]
zone_colors = ['#4caf50', '#ffeb3b', '#ff9800', '#f44336']

counts = [results_df['risk_zone'].value_counts().get(z, 0) for z in zone_order]

plt.figure(figsize=(10, 6))
bars = plt.bar(zone_order, counts, color=zone_colors, edgecolor='black', width=0.5)
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
             str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.title('Employee Turnover Risk Zones', fontsize=14)
plt.xlabel('Risk Zone')
plt.ylabel('Number of Employees')
plt.tight_layout()
plt.show()
```

### Output – Risk Zone Counts
```
Risk Zone Distribution:
              Risk Zone  Employee Count
      Safe Zone (Green)            2179
   High-Risk Zone (Red)             649
 Low-Risk Zone (Yellow)             113
Medium-Risk Zone (Orange)            59
```

### Visualisation Description
A 10×6 inch grouped bar chart with four bars coloured green, yellow, orange, and red corresponding to the four risk zones. Bold count labels appear above each bar.

| Zone | Colour | Count | % of Test Set |
|---|---|---|---|
| 🟢 Safe Zone (Green) | `#4caf50` | 2,179 | 72.6 % |
| 🟡 Low-Risk Zone (Yellow) | `#ffeb3b` | 113 | 3.8 % |
| 🟠 Medium-Risk Zone (Orange) | `#ff9800` | 59 | 2.0 % |
| 🔴 High-Risk Zone (Red) | `#f44336` | 649 | 21.6 % |

**649 employees are in the High-Risk Zone and require immediate retention intervention.**
