# 📉 Customer Churn Prediction

> **Internship Project | Codec Technologies**  
> Predict which customers are likely to stop using a telecom service using Machine Learning.

---

## 📌 Project Overview

Customer churn — when customers stop using a service — directly impacts revenue. This project builds a full ML pipeline to:

- Analyze churn patterns through **Exploratory Data Analysis (EDA)**
- Engineer meaningful features from raw telecom data
- Train & compare three classification models
- Evaluate with **Accuracy, Recall, F1-Score, and ROC-AUC**

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── telco_churn.csv              # Dataset (Telecom, 7,043 customers, 21 features)
├── generate_dataset.py          # Script to (re)generate the dataset
├── churn_prediction.py          # Main ML pipeline (EDA + Models + Evaluation)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
└── plots/                       # All generated visualizations
    ├── 01_churn_distribution.png
    ├── 02_churn_by_category.png
    ├── 03_numerical_distributions.png
    ├── 04_churn_by_tenure.png
    ├── 05_correlation_heatmap.png
    ├── 06_roc_curves.png
    ├── 07_confusion_matrices.png
    ├── 08_metrics_comparison.png
    └── 09_feature_importance.png
```

---

## 📊 Dataset

**Telecom Customer Dataset** — 7,043 rows, 21 features

| Feature | Description |
|---|---|
| `tenure` | Months the customer has been with the company |
| `Contract` | Month-to-month / One year / Two year |
| `InternetService` | DSL / Fiber optic / No |
| `MonthlyCharges` | Current monthly charge amount |
| `TotalCharges` | Total amount charged |
| `Churn` | **Target** — Yes / No |

**Overall Churn Rate: ~12.1%**

---

## 🔍 EDA Findings

| Insight | Detail |
|---|---|
| Month-to-month vs Two-year churn | **17.7%** vs **4.9%** |
| Fiber optic internet churn rate | **16.5%** |
| Senior citizen churn rate | **18.8%** |
| Low tenure (0–12 months) | Highest churn risk |

---

## 🤖 Models Used

| Model | Role |
|---|---|
| **Logistic Regression** | Linear baseline, interpretable coefficients |
| **Random Forest** | Ensemble, handles non-linearity, feature importance |
| **Gradient Boosting** | Boosting equivalent of XGBoost |

---

## 📈 Results

| Model | Accuracy | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.6643 | **0.7294** | 0.2250 | 0.3440 | **0.7595** |
| Random Forest | 0.7700 | 0.5118 | 0.2652 | 0.3494 | 0.7451 |
| Gradient Boosting | 0.8815 | 0.0412 | 0.6364 | 0.0773 | 0.7556 |

> ✅ **Best Model: Logistic Regression** (highest ROC-AUC = 0.7595 and best Recall)  
> In churn prediction, **Recall** is critical — missing a churner is more costly than a false alarm.

---

## 🛠️ Installation & Usage

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Regenerate dataset
python generate_dataset.py

# 4. Run the full pipeline
python churn_prediction.py
```

---

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

---

## 📊 Sample Visualizations

All 9 plots are auto-saved to the `plots/` folder when you run the pipeline:

- Churn distribution (pie + count)
- Churn rate by Contract, Internet, Payment method, etc.
- Numerical feature distributions (histograms + boxplots)
- Churn rate by tenure group
- Correlation heatmap
- ROC curves for all 3 models
- Confusion matrices
- Metrics comparison bar chart
- Random Forest feature importances

---

## 💡 Business Recommendations

1. **Target month-to-month customers** with loyalty incentives to switch to annual plans.
2. **Investigate Fiber optic service quality** — churn rate is disproportionately high.
3. **Focus retention efforts on new customers (< 12 months)** — highest churn risk window.
4. **Senior citizens** are a vulnerable segment; consider dedicated support programs.

---


