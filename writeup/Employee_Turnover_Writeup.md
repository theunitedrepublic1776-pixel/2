# Employee Turnover Analytics – Project Writeup
**Portobello Tech | HR Department Machine Learning Project**

---

## 1. Problem Statement and Objectives

Employee turnover is one of the most costly challenges faced by modern organisations. At Portobello Tech, the HR department identified a need to move from reactive to proactive talent management by leveraging data science techniques on existing HR records.

**Objectives:**

1. Understand the key drivers that lead employees to leave the organisation.
2. Segment employees who have already left into meaningful behavioural clusters to identify distinct risk profiles.
3. Build, evaluate, and compare predictive machine learning models that can classify whether a current employee is likely to leave.
4. Assign every employee in the test set a risk zone (Safe / Low / Medium / High) based on their predicted probability of leaving.
5. Provide evidence-based, actionable retention recommendations tailored to each risk zone and employee cluster.

---

## 2. Data Description and Source

| Attribute | Details |
|---|---|
| **Dataset** | `HR_comma_sep.csv` |
| **Source** | Internal HR records (Portobello Tech) |
| **Rows** | 14,999 employees |
| **Columns** | 10 features |
| **Target variable** | `left` (1 = employee left, 0 = stayed) |

### Feature Descriptions

| Feature | Type | Description |
|---|---|---|
| `satisfaction_level` | Float (0–1) | Employee self-reported satisfaction score |
| `last_evaluation` | Float (0–1) | Most recent performance evaluation score |
| `number_project` | Integer | Number of projects currently assigned |
| `average_montly_hours` | Integer | Average monthly hours worked |
| `time_spend_company` | Integer | Years spent at the company |
| `Work_accident` | Binary | Whether employee had a workplace accident (1 = yes) |
| `left` | Binary | **Target** – whether employee left (1 = yes) |
| `promotion_last_5years` | Binary | Whether promoted in the last 5 years (1 = yes) |
| `sales` | Categorical | Department name |
| `salary` | Categorical | Salary level (low / medium / high) |

**Class distribution:**
- Stayed (0): 11,428 employees (76.2 %)
- Left (1): 3,571 employees (23.8 %)

The dataset is complete with **no missing values**, requiring no imputation.

---

## 3. Methodology / Approach

The analysis followed a structured pipeline with seven stages:

### 3.1 Data Quality Checks
- Verified data types and confirmed no null values across all 10 columns.
- Generated descriptive statistics (mean, std, min, max, quartiles) to understand feature distributions.

### 3.2 Exploratory Data Analysis (EDA)
- **Correlation heatmap** – identified linear relationships among numerical features.
- **Distribution plots** – examined the shape of key variables (`satisfaction_level`, `last_evaluation`, `average_montly_hours`).
- **Project count bar chart** – compared turnover rates across employees with different project loads.

### 3.3 K-Means Clustering (k = 3) on Leavers
- Filtered the 3,571 employees who left.
- Applied K-Means on `satisfaction_level` and `last_evaluation` to uncover distinct departure archetypes.
- Visualised clusters and described each profile.

### 3.4 Data Pre-processing for Modelling
- One-hot encoded categorical columns (`sales`, `salary`) using `pd.get_dummies`.
- Resulting processed dataset: 14,999 rows × 21 columns.
- Applied a **stratified 80:20 train/test split** (train: 11,999 | test: 3,000).

### 3.5 Class Imbalance Handling with SMOTE
- The training set was imbalanced (~76 % stayed, ~24 % left).
- Applied **Synthetic Minority Oversampling Technique (SMOTE)** to the training set, producing a balanced set of 18,284 samples (9,142 per class).

### 3.6 5-Fold Cross-Validation Model Training
Trained three classifiers using Stratified 5-Fold CV on the SMOTE-upsampled training set:
- **Logistic Regression** (`max_iter=1000`)
- **Random Forest Classifier** (`n_estimators=100`)
- **Gradient Boosting Classifier** (`n_estimators=100`)

### 3.7 Model Evaluation & Selection
- Plotted **ROC / AUC curves** on the held-out test set.
- Generated **confusion matrices** and **classification reports** for all three models.
- Selected the best model based on AUC score and recall for the "Left" class.

### 3.8 Risk Zone Assignment & Retention Strategies
- Used the best model to predict turnover probability for every employee in the test set.
- Categorised employees into four zones: Safe (< 20 %), Low-Risk (20–60 %), Medium-Risk (60–90 %), and High-Risk (> 90 %).
- Formulated retention strategies for each zone and cluster.

---

## 4. Key Findings and Insights

### 4.1 Correlation Analysis
- `satisfaction_level` has a strong **negative correlation with `left`** (~−0.39): lower satisfaction → higher turnover.
- `number_project` and `average_montly_hours` are moderately correlated (~0.42), pointing to a workload relationship.
- `last_evaluation` and `number_project` are positively correlated (~0.35): high-performers tend to carry more projects.
- `time_spend_company` shows a weak positive correlation with leaving, suggesting long-tenured employees are not immune to attrition.

### 4.2 Distribution Insights
- **Satisfaction Level** is bimodal: a small cluster is very dissatisfied (~0.1) and a larger group is moderately to highly satisfied (0.6–0.8).
- **Last Evaluation** is fairly uniform, slightly concentrated at 0.7–0.9, showing the organisation retains high performers on average.
- **Average Monthly Hours** is bimodal: peaks around 150 and 250 hours/month, revealing two distinct work-style groups—normal-load and overworked employees.

### 4.3 Project Count Findings
| Number of Projects | Observation |
|---|---|
| 2 | Highest turnover – employees feel underutilised and disengaged |
| 3–5 | Lowest turnover – optimal workload range |
| 6–7 | High turnover – burnout from excessive workload |

### 4.4 K-Means Cluster Profiles (Employees Who Left)

| Cluster | Satisfaction | Evaluation | Profile | Size |
|---|---|---|---|---|
| **0** | Low (~0.10–0.45) | Medium–High (~0.45–0.90) | **Dissatisfied High-Performers** – Strong evaluations but very low satisfaction; likely overworked or under-rewarded | 1,650 |
| **1** | Medium (~0.40–0.70) | Low (~0.45–0.57) | **Disengaged Low-Performers** – Average satisfaction with low evaluations; stagnation or voluntary exit | 977 |
| **2** | High (~0.70–1.00) | High (~0.77–1.00) | **High-Performers Who Left** – Highly satisfied and well-evaluated; likely left for better external opportunities | 944 |

**3,571 total leavers** are distributed across all three archetypes, indicating that attrition cannot be attributed to a single cause.

### 4.5 Model Performance Comparison

| Model | CV Accuracy (Train) | AUC (Test) | Test Accuracy |
|---|---|---|---|
| Logistic Regression | 81 % | 0.8209 | 78 % |
| Gradient Boosting | 96 % | 0.9852 | ~96 % |
| **Random Forest** | **98 %** | **0.9951** | **~99 %** |

- **Random Forest** achieved the best AUC (0.9951) and near-perfect precision/recall on the test set (Stayed: 99 %, Left: 97–98 %).
- Logistic Regression, while interpretable, falls short—particularly in recall for the "Left" class (72 %).

### 4.6 Risk Zone Distribution (Test Set, 3,000 employees)

| Risk Zone | Probability Range | Count |
|---|---|---|
| 🟢 Safe Zone (Green) | < 20 % | 2,179 |
| 🟡 Low-Risk Zone (Yellow) | 20 %–60 % | 113 |
| 🟠 Medium-Risk Zone (Orange) | 60 %–90 % | 59 |
| 🔴 High-Risk Zone (Red) | > 90 % | 649 |

649 employees (~21.6 % of the test set) are in the High-Risk Zone and require immediate intervention.

---

## 5. Conclusions and Recommendations

### 5.1 Best Predictive Model
**Random Forest Classifier** is the recommended model for deployment, achieving AUC = 0.9951 and recall of 98 % for the "Left" class. Recall is the priority metric here: missing a flight-risk employee (false negative) is far costlier than triggering an unnecessary retention conversation (false positive).

### 5.2 Retention Strategies by Risk Zone

| Zone | Strategy |
|---|---|
| 🟢 Safe | Maintain status quo – regular check-ins, competitive compensation, recognition |
| 🟡 Low-Risk | Personalised development plans, mentorship, flexible arrangements, stay interviews |
| 🟠 Medium-Risk | Immediate 1-on-1 with HR/manager, workload review, compensation review, clear career path |
| 🔴 High-Risk | Emergency escalation to leadership, fast-track promotions, salary adjustments, retention bonuses for top performers, succession planning |

### 5.3 Cross-Cutting Recommendations

1. **Workload Management** – Keep project assignments in the 3–5 range. Redistribute projects from overloaded employees (6–7 projects). Under-utilised employees (2 projects) need more engagement.
2. **Recognition Programs** – Dissatisfied high-performers (Cluster 0) leave despite strong evaluations. Introduce reward and recognition programs tied to performance scores.
3. **Career Path Clarity** – High-performers who left (Cluster 2) likely sought better external opportunities. Transparent promotion criteria and growth roadmaps are critical.
4. **Exit Interview Analysis** – Analyse why disengaged low-performers (Cluster 1) left to surface management or process issues.
5. **Salary Review** – Employees in the `low` salary band are disproportionately represented among leavers. A structured compensation benchmarking exercise is recommended.
6. **Periodic Re-scoring** – The model should be retrained quarterly as new HR data becomes available to keep predictions current and accurate.
