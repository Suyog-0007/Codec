# =============================================================================
# CUSTOMER CHURN PREDICTION
# Telecom Dataset | EDA + Classification Models + Evaluation
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score, recall_score, precision_score, f1_score
)
from sklearn.pipeline import Pipeline

import os
os.makedirs('plots', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 | LOAD & INSPECT DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print(" CUSTOMER CHURN PREDICTION PIPELINE")
print("=" * 65)

df = pd.read_csv('telco_churn.csv')

print(f"\n📂 Dataset Shape : {df.shape}")
print(f"📋 Columns       : {list(df.columns)}\n")
print("── First 5 rows ──")
print(df.head())
print("\n── Data Types ──")
print(df.dtypes)
print("\n── Missing Values ──")
print(df.isnull().sum()[df.isnull().sum() > 0])

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 | DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n[1/6] DATA CLEANING")
print("-" * 40)

# Drop customerID (not a feature)
df.drop(columns=['customerID'], inplace=True)

# Fix TotalCharges: fill NaN with median (new customers with tenure=0)
missing_count = df['TotalCharges'].isnull().sum()
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
print(f"  ✔ Filled {missing_count} missing TotalCharges values with median")

# Encode target
df['Churn_binary'] = (df['Churn'] == 'Yes').astype(int)
churn_rate = df['Churn_binary'].mean()
print(f"  ✔ Overall churn rate: {churn_rate:.1%}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 | EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n[2/6] EXPLORATORY DATA ANALYSIS")
print("-" * 40)

# ── Plot 1: Churn Distribution ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Churn Distribution Overview', fontsize=15, fontweight='bold', y=1.02)

churn_counts = df['Churn'].value_counts()
colors = ['#2ecc71', '#e74c3c']
axes[0].pie(churn_counts, labels=['Not Churned', 'Churned'], autopct='%1.1f%%',
            colors=colors, startangle=90, shadow=True,
            textprops={'fontsize': 12})
axes[0].set_title('Overall Churn Split')

sns.countplot(x='Churn', data=df, palette={'No': '#2ecc71', 'Yes': '#e74c3c'}, ax=axes[1])
axes[1].set_title('Churn Count')
axes[1].set_xlabel('Churn')
axes[1].set_ylabel('Count')
for p in axes[1].patches:
    axes[1].annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/01_churn_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✔ Saved: plots/01_churn_distribution.png")

# ── Plot 2: Churn by Categorical Features ────────────────────────────────
cat_cols = ['Contract', 'InternetService', 'PaymentMethod', 'gender',
            'SeniorCitizen', 'Partner', 'PaperlessBilling']

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle('Churn Rate by Categorical Features', fontsize=15, fontweight='bold')
axes_flat = axes.flatten()

for i, col in enumerate(cat_cols):
    churn_by_col = df.groupby(col)['Churn_binary'].mean().reset_index()
    churn_by_col.columns = [col, 'Churn Rate']
    churn_by_col = churn_by_col.sort_values('Churn Rate', ascending=False)
    bars = axes_flat[i].bar(churn_by_col[col].astype(str), churn_by_col['Churn Rate'],
                            color=['#e74c3c' if v > churn_rate else '#3498db'
                                   for v in churn_by_col['Churn Rate']])
    axes_flat[i].axhline(churn_rate, color='black', linestyle='--', linewidth=1,
                         label=f'Overall ({churn_rate:.1%})')
    axes_flat[i].set_title(col, fontsize=11, fontweight='bold')
    axes_flat[i].set_ylabel('Churn Rate')
    axes_flat[i].tick_params(axis='x', rotation=20)
    for bar, val in zip(bars, churn_by_col['Churn Rate']):
        axes_flat[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                          f'{val:.1%}', ha='center', fontsize=8)
    axes_flat[i].legend(fontsize=8)

for j in range(len(cat_cols), len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.tight_layout()
plt.savefig('plots/02_churn_by_category.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✔ Saved: plots/02_churn_by_category.png")

# ── Plot 3: Numerical Features Distribution ────────────────────────────
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Numerical Features by Churn Status', fontsize=15, fontweight='bold')

for i, col in enumerate(num_cols):
    # KDE plot
    for label, color in zip(['No', 'Yes'], ['#2ecc71', '#e74c3c']):
        subset = df[df['Churn'] == label][col]
        axes[0][i].hist(subset, bins=30, alpha=0.6, color=color, label=label, density=True)
    axes[0][i].set_title(f'{col} Distribution', fontweight='bold')
    axes[0][i].set_xlabel(col)
    axes[0][i].legend(title='Churn')

    # Box plot
    df.boxplot(column=col, by='Churn', ax=axes[1][i],
               boxprops=dict(color='navy'),
               medianprops=dict(color='red', linewidth=2))
    axes[1][i].set_title(f'{col} Boxplot')
    axes[1][i].set_xlabel('Churn')

plt.tight_layout()
plt.savefig('plots/03_numerical_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✔ Saved: plots/03_numerical_distributions.png")

# ── Plot 4: Tenure Segments Churn Rate ────────────────────────────────
df['tenure_group'] = pd.cut(df['tenure'],
                            bins=[0, 12, 24, 36, 48, 60, 72],
                            labels=['0-12m', '13-24m', '25-36m', '37-48m', '49-60m', '60+m'])

fig, ax = plt.subplots(figsize=(10, 5))
churn_by_tenure = df.groupby('tenure_group')['Churn_binary'].mean()
bars = ax.bar(churn_by_tenure.index.astype(str), churn_by_tenure.values,
              color=['#e74c3c' if v > churn_rate else '#3498db' for v in churn_by_tenure])
ax.axhline(churn_rate, color='black', linestyle='--', linewidth=1.5,
           label=f'Avg churn ({churn_rate:.1%})')
ax.set_title('Churn Rate by Tenure Group', fontsize=13, fontweight='bold')
ax.set_xlabel('Tenure Group')
ax.set_ylabel('Churn Rate')
ax.legend()
for bar, val in zip(bars, churn_by_tenure):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f'{val:.1%}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/04_churn_by_tenure.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✔ Saved: plots/04_churn_by_tenure.png")

# ── Plot 5: Correlation Heatmap ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
num_df = df[['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn_binary']]
corr = num_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', mask=mask,
            linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 11, 'weight': 'bold'})
ax.set_title('Correlation Heatmap', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/05_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✔ Saved: plots/05_correlation_heatmap.png")

# Key EDA Findings
print("\n  📊 Key EDA Findings:")
print(f"     • Month-to-month contract churn: "
      f"{df[df['Contract']=='Month-to-month']['Churn_binary'].mean():.1%} vs "
      f"{df[df['Contract']=='Two year']['Churn_binary'].mean():.1%} (Two-year)")
print(f"     • Fiber optic internet churn: "
      f"{df[df['InternetService']=='Fiber optic']['Churn_binary'].mean():.1%}")
print(f"     • Senior citizen churn rate: "
      f"{df[df['SeniorCitizen']==1]['Churn_binary'].mean():.1%}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 | FEATURE ENGINEERING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n[3/6] FEATURE ENGINEERING")
print("-" * 40)

# Encode categorical features
df_model = df.drop(columns=['Churn', 'tenure_group'])

le = LabelEncoder()
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
               'PaperlessBilling', 'Churn_binary']

ordinal_map = {
    'MultipleLines':    {'No phone service': 0, 'No': 1, 'Yes': 2},
    'OnlineSecurity':   {'No internet service': 0, 'No': 1, 'Yes': 2},
    'OnlineBackup':     {'No internet service': 0, 'No': 1, 'Yes': 2},
    'DeviceProtection': {'No internet service': 0, 'No': 1, 'Yes': 2},
    'TechSupport':      {'No internet service': 0, 'No': 1, 'Yes': 2},
    'StreamingTV':      {'No internet service': 0, 'No': 1, 'Yes': 2},
    'StreamingMovies':  {'No internet service': 0, 'No': 1, 'Yes': 2},
}

# Binary encode
for col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
    df_model[col] = (df_model[col] == 'Yes').astype(int)
df_model['gender'] = (df_model['gender'] == 'Female').astype(int)

# Ordinal encode
for col, mapping in ordinal_map.items():
    df_model[col] = df_model[col].map(mapping)

# One-hot encode remaining categoricals
ohe_cols = ['InternetService', 'Contract', 'PaymentMethod']
df_model = pd.get_dummies(df_model, columns=ohe_cols, drop_first=False)

# Feature engineering
df_model['charges_per_month'] = df_model['TotalCharges'] / (df_model['tenure'] + 1)
df_model['is_new_customer'] = (df_model['tenure'] <= 6).astype(int)
df_model['streaming_both'] = ((df_model['StreamingTV'] == 2) &
                               (df_model['StreamingMovies'] == 2)).astype(int)

print(f"  ✔ Total features after encoding: {df_model.shape[1] - 1}")

# Impute any remaining NaNs
df_model.fillna(df_model.median(numeric_only=True), inplace=True)

X = df_model.drop(columns=['Churn_binary'])
y = df_model['Churn_binary']

feature_names = X.columns.tolist()
print(f"  ✔ Features: {feature_names[:8]} ... (+{len(feature_names)-8} more)")

# Train / Test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print(f"  ✔ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"  ✔ Train churn rate: {y_train.mean():.1%} | Test: {y_test.mean():.1%}")

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 | MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n[4/6] MODEL TRAINING")
print("-" * 40)

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, class_weight='balanced', C=1.0, random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1),
    'Gradient Boosting\n(XGBoost-style)': GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, random_state=42),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    clean_name = name.replace('\n', ' ')
    use_scaled = 'Logistic' in name
    Xtr = X_train_scaled if use_scaled else X_train.values
    Xte = X_test_scaled  if use_scaled else X_test.values

    model.fit(Xtr, y_train)
    y_pred  = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    recall  = recall_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cv_auc  = cross_val_score(model, Xtr, y_train, cv=cv,
                               scoring='roc_auc').mean()
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    results[clean_name] = {
        'model': model, 'y_pred': y_pred, 'y_proba': y_proba,
        'accuracy': acc, 'recall': recall, 'precision': prec,
        'f1': f1, 'roc_auc': roc_auc, 'cv_auc': cv_auc,
        'fpr': fpr, 'tpr': tpr,
        'use_scaled': use_scaled
    }
    print(f"\n  ── {clean_name} ──")
    print(f"     Accuracy  : {acc:.4f}")
    print(f"     Recall    : {recall:.4f}")
    print(f"     Precision : {prec:.4f}")
    print(f"     F1-Score  : {f1:.4f}")
    print(f"     ROC-AUC   : {roc_auc:.4f}  (CV: {cv_auc:.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 | EVALUATION & VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n[5/6] EVALUATION & PLOTS")
print("-" * 40)

# ── Plot 6: ROC Curves ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
colors_roc = ['#3498db', '#e67e22', '#9b59b6']
for (name, res), color in zip(results.items(), colors_roc):
    ax.plot(res['fpr'], res['tpr'],
            label=f"{name} (AUC = {res['roc_auc']:.3f})", color=color, lw=2)
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves – All Models', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/06_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✔ Saved: plots/06_roc_curves.png")

# ── Plot 7: Confusion Matrices ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'],
                annot_kws={'size': 13, 'weight': 'bold'})
    ax.set_title(name, fontsize=10, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('plots/07_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✔ Saved: plots/07_confusion_matrices.png")

# ── Plot 8: Metrics Comparison Bar Chart ──────────────────────────────
metrics = ['accuracy', 'recall', 'precision', 'f1', 'roc_auc']
model_names = list(results.keys())
x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(13, 6))
for i, (name, res) in enumerate(results.items()):
    vals = [res[m] for m in metrics]
    bars = ax.bar(x + i * width, vals, width, label=name, alpha=0.85,
                  color=colors_roc[i])
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

ax.set_xticks(x + width)
ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/08_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✔ Saved: plots/08_metrics_comparison.png")

# ── Plot 9: Feature Importances (Random Forest) ────────────────────────
rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=feature_names)
top20 = importances.sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 8))
colors_fi = ['#e74c3c' if v > top20.mean() else '#3498db' for v in top20.values]
top20.sort_values().plot(kind='barh', ax=ax, color=colors_fi[::-1])
ax.set_title('Top 20 Feature Importances (Random Forest)', fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score')
ax.axvline(top20.mean(), color='black', linestyle='--', linewidth=1.5,
           label=f'Mean ({top20.mean():.4f})')
ax.legend()
plt.tight_layout()
plt.savefig('plots/09_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✔ Saved: plots/09_feature_importance.png")

# ── Summary Table ──────────────────────────────────────────────────────
print("\n\n[6/6] FINAL SUMMARY")
print("=" * 65)

summary = pd.DataFrame({
    name: {
        'Accuracy':  f"{res['accuracy']:.4f}",
        'Recall':    f"{res['recall']:.4f}",
        'Precision': f"{res['precision']:.4f}",
        'F1-Score':  f"{res['f1']:.4f}",
        'ROC-AUC':   f"{res['roc_auc']:.4f}",
        'CV AUC':    f"{res['cv_auc']:.4f}",
    }
    for name, res in results.items()
}).T

print(summary.to_string())

best_model_name = max(results, key=lambda k: results[k]['roc_auc'])
print(f"\n🏆 Best Model : {best_model_name}")
print(f"   ROC-AUC    : {results[best_model_name]['roc_auc']:.4f}")
print(f"   Recall     : {results[best_model_name]['recall']:.4f}")
print(f"\n📁 All plots saved in ./plots/")
print("=" * 65)

# ── Classification Report for best model ──────────────────────────────
print(f"\nDetailed Classification Report — {best_model_name}")
print(classification_report(y_test, results[best_model_name]['y_pred'],
                             target_names=['No Churn', 'Churn']))

print("\n✅ Churn Prediction Pipeline Complete!\n")
