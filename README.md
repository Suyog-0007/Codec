# 🏥 Healthcare Predictive Analytics — Diabetes Risk Detection

> **Internship Project | Codec Technologies**  
> Predict the risk of Type-2 Diabetes using machine learning on anonymised patient records.

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Models & Results](#models--results)
- [Feature Importance](#feature-importance)
- [Ethical Data Handling](#ethical-data-handling)
- [Installation & Usage](#installation--usage)
- [Sample Visualisations](#sample-visualisations)

---

## Project Overview

This project builds a complete end-to-end ML pipeline to predict whether a patient is at **high or low risk of diabetes** based on routine clinical measurements (glucose, BMI, blood pressure, etc.).

Key highlights:
- **6 classification models** benchmarked and compared
- **SMOTE** to handle class imbalance fairly
- **Permutation + built-in feature importance** analysis
- Best model saved for live inference via `predict.py`
- Full compliance with ethical data-handling principles (HIPAA-aligned)

---

## Dataset

| Property       | Detail                                          |
|----------------|-------------------------------------------------|
| Source         | Pima Indians Diabetes — UCI / Kaggle            |
| Records        | 768 patients                                    |
| Features       | 8 clinical + 4 engineered = 12 total            |
| Target         | Binary — Diabetic (1) / Non-diabetic (0)        |
| Class balance  | ≈36% positive (diabetes) before SMOTE          |

**Features used:**

| Feature                     | Description                         |
|-----------------------------|-------------------------------------|
| `Pregnancies`               | Number of pregnancies               |
| `Glucose`                   | Plasma glucose concentration (mg/dL)|
| `BloodPressure`             | Diastolic blood pressure (mm Hg)    |
| `SkinThickness`             | Triceps skinfold thickness (mm)     |
| `Insulin`                   | 2-hour serum insulin (µU/ml)        |
| `BMI`                       | Body Mass Index (kg/m²)             |
| `DiabetesPedigreeFunction`  | Family history scoring function     |
| `Age`                       | Age in years                        |
| `GlucoseBMI` *(engineered)* | Glucose × BMI interaction           |
| `BMI_Category` *(engineered)*| Underweight / Normal / Overweight / Obese |
| `AgeGroup` *(engineered)*   | 4 age bands                         |
| `HighGlucose` *(engineered)*| Binary flag: Glucose > 140          |

---

## Project Structure

```
healthcare_analytics/
│
├── data/
│   └── diabetes.csv               # Anonymised dataset
│
├── models/
│   ├── best_model.pkl             # Best serialised model
│   └── scaler.pkl                 # Fitted RobustScaler
│
├── outputs/
│   ├── eda_distributions.png      # Feature distribution plots
│   ├── correlation_heatmap.png    # Correlation matrix
│   ├── roc_pr_curves.png          # ROC & Precision-Recall curves
│   ├── confusion_matrices.png     # Confusion matrices (all models)
│   ├── cv_comparison.png          # Cross-validation AUC comparison
│   ├── feature_importance.png     # Feature importance (3 methods)
│   └── model_summary.csv          # All model metrics
│
├── diabetes_prediction.py         # ★ Main ML pipeline
├── predict.py                     # Inference on new patient data
├── requirements.txt
└── README.md
```

---

## Pipeline

```
Raw CSV
  │
  ▼
Zero → NaN replacement (physiologically impossible zeros)
  │
  ▼
Median imputation  →  Feature Engineering (4 new features)
  │
  ▼
Train / Test split  (80 / 20, stratified)
  │
  ▼
RobustScaler  →  SMOTE oversampling (training set only)
  │
  ├── Logistic Regression
  ├── Random Forest (200 trees)
  ├── Gradient Boosting (200 estimators)
  ├── SVM (RBF kernel)
  ├── KNN (k=7)
  └── Ensemble Voting (LR + RF + GB, soft voting)
        │
        ▼
   5-fold Stratified CV → Test evaluation → Feature Importance
        │
        ▼
   Best model saved (joblib)  →  predict.py
```

---

## Models & Results

| Model               | Accuracy | F1 Score | ROC-AUC | CV AUC (5-fold)   |
|---------------------|----------|----------|---------|-------------------|
| Logistic Regression | 0.7208   | 0.6261   | **0.7578** | 0.8287 ± 0.034 |
| Random Forest       | 0.7078   | 0.6087   | 0.7540  | 0.8631 ± 0.028    |
| Ensemble Voting     | 0.7208   | 0.6325   | 0.7516  | 0.8648 ± 0.028    |
| SVM (RBF)           | 0.7078   | 0.5946   | 0.7369  | 0.8490 ± 0.032    |
| Gradient Boosting   | 0.7078   | 0.6087   | 0.7367  | 0.8659 ± 0.025    |
| KNN                 | 0.6429   | 0.5378   | 0.6723  | 0.8364 ± 0.038    |

> ★ **Best Model (test ROC-AUC): Logistic Regression — 0.7578**

---

## Feature Importance

Top predictors identified by permutation importance:

| Rank | Feature                    | Δ AUC  |
|------|----------------------------|--------|
| 1    | BMI_Category               | 0.0293 |
| 2    | BMI                        | 0.0285 |
| 3    | Glucose                    | 0.0230 |
| 4    | HighGlucose (flag)         | 0.0159 |
| 5    | DiabetesPedigreeFunction   | 0.0144 |

BMI and Glucose dominate — consistent with clinical literature on Type-2 diabetes risk.

---

## Ethical Data Handling

This project takes patient privacy seriously:

| Principle              | Implementation                                     |
|------------------------|----------------------------------------------------|
| **Anonymisation**      | No PII (names, IDs, addresses) in dataset          |
| **De-identification**  | HIPAA safe-harbour compliant                       |
| **Fairness**           | SMOTE ensures minority class is not under-served   |
| **Transparency**       | Feature importance explains model decisions        |
| **Clinical caveat**    | Output is probabilistic, NOT a clinical diagnosis  |
| **Data minimisation**  | Only clinically relevant features are used         |
| **No leakage**         | Scaler/SMOTE fitted on training data only          |

> ⚠️ **Disclaimer:** Predictions from this model are for **research and educational purposes only**.  
> They are **not a substitute for professional medical advice, diagnosis, or treatment.**

---

## Installation & Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full ML pipeline
```bash
python diabetes_prediction.py
```
This will:
- Load and preprocess the data
- Run EDA (saves plots to `outputs/`)
- Train all 6 models with cross-validation
- Evaluate on held-out test set
- Generate feature importance plots
- Save the best model to `models/`

### 3. Predict for a new (anonymised) patient
```bash
python predict.py --glucose 148 --bmi 33.6 --age 50 --pregnancies 2 --dp 0.627
```

### 4. Predict on a batch CSV file
```bash
python predict.py --csv data/new_patients.csv
```
Outputs a new CSV with `DiabetesRisk_Proba` and `RiskLevel` columns.

---

## Sample Visualisations

| Plot | Description |
|------|-------------|
| `eda_distributions.png` | Per-feature histograms by diabetes class |
| `correlation_heatmap.png` | Pearson correlation matrix |
| `roc_pr_curves.png` | ROC & Precision-Recall curves for all models |
| `confusion_matrices.png` | Confusion matrices for all models |
| `cv_comparison.png` | Cross-validation AUC bar chart |
| `feature_importance.png` | RF, GB, and permutation importance |

---

## Technologies Used

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![pandas](https://img.shields.io/badge/pandas-2.0-green)
![seaborn](https://img.shields.io/badge/seaborn-0.12-lightblue)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.11-red)

---

## Author

**Codec Technologies Internship Project**  
📧 Contact: vaishali@codectechnologies.in

---

*This project was completed as part of the Codec Technologies Data Science Internship.*
