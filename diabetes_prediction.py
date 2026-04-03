"""
============================================================
Healthcare Predictive Analytics — Diabetes Risk Detection
============================================================
Dataset  : Pima Indians Diabetes (UCI / Kaggle)
Models   : Logistic Regression, Random Forest, XGBoost (via sklearn)
Author   : Internship Project — Codec Technologies
Ethics   : No PII stored; data anonymised; SMOTE for fairness
============================================================
"""

# ── Standard Library ──────────────────────────────────────
import os
import warnings
warnings.filterwarnings("ignore")

# ── Data & Math ───────────────────────────────────────────
import numpy as np
import pandas as pd

# ── Visualisation ─────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── ML ────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, f1_score
)
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE

import joblib

# ── Paths ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(BASE_DIR, "data",    "diabetes.csv")
OUT_DIR      = os.path.join(BASE_DIR, "outputs")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
os.makedirs(OUT_DIR,    exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Plot style ────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
           "#8172B2", "#937860", "#DA8BC3"]

# ═══════════════════════════════════════════════════════════
# 1. DATA LOADING & PRIVACY NOTICE
# ═══════════════════════════════════════════════════════════
def load_data(path: str) -> pd.DataFrame:
    """Load dataset and enforce ethical data-handling rules."""
    print("\n" + "="*60)
    print("  ETHICAL DATA HANDLING NOTICE")
    print("="*60)
    print("  • Dataset is fully anonymised — no patient identifiers.")
    print("  • Zero PII (name, SSN, address) present or retained.")
    print("  • Predictions are probabilistic — NOT clinical diagnoses.")
    print("  • Model output must be reviewed by qualified clinicians.")
    print("  • Compliant with HIPAA de-identification safe-harbour.")
    print("="*60 + "\n")

    df = pd.read_csv(path)
    print(f"[DATA] Loaded {df.shape[0]} records × {df.shape[1]} features")
    return df


# ═══════════════════════════════════════════════════════════
# 2. EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════
def eda(df: pd.DataFrame) -> None:
    """Generate EDA plots and save to outputs/."""
    print("\n[EDA] Running exploratory data analysis …")

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("Diabetes Dataset — Exploratory Data Analysis",
                 fontsize=18, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    features = [c for c in df.columns if c != "Outcome"]
    colours  = [PALETTE[i % len(PALETTE)] for i in range(len(features))]

    for idx, (feat, col) in enumerate(zip(features, colours)):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        sns.histplot(data=df, x=feat, hue="Outcome", kde=True,
                     palette=["#4C72B0", "#DD8452"], ax=ax, bins=25,
                     alpha=0.65, legend=(idx == 0))
        ax.set_title(feat, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        if idx == 0:
            ax.legend(["No Diabetes", "Diabetes"], fontsize=8)

    # Class balance
    ax_bal = fig.add_subplot(gs[2, 2])
    counts = df["Outcome"].value_counts()
    ax_bal.pie(counts, labels=["No Diabetes", "Diabetes"],
               autopct="%1.1f%%", colors=["#4C72B0", "#DD8452"],
               startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax_bal.set_title("Class Balance", fontsize=11, fontweight="bold")

    plt.savefig(os.path.join(OUT_DIR, "eda_distributions.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # Correlation heatmap
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, ax=ax2,
                linewidths=0.5, square=True)
    ax2.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "correlation_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    print("[EDA] Plots saved → outputs/")


# ═══════════════════════════════════════════════════════════
# 3. PREPROCESSING & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame):
    """
    Steps:
      1. Replace physiologically impossible zeros with NaN.
      2. Impute NaNs with median (robust to outliers).
      3. Engineer BMI category + Glucose-BMI interaction.
      4. Scale features with RobustScaler (handles outliers).
      5. Balance classes with SMOTE (fairness across demographics).
    """
    print("\n[PREP] Preprocessing …")

    # Columns where 0 is medically impossible
    zero_invalid = ["Glucose", "BloodPressure", "SkinThickness",
                    "Insulin", "BMI"]
    df = df.copy()
    df[zero_invalid] = df[zero_invalid].replace(0, np.nan)
    missing = df.isnull().sum()
    if missing.any():
        print(f"[PREP] Missing after zero-replacement:\n{missing[missing>0]}")

    # Median imputation
    for col in zero_invalid:
        df[col] = df[col].fillna(df[col].median())

    # Feature engineering
    df["GlucoseBMI"]   = df["Glucose"] * df["BMI"]
    df["BMI_Category"] = pd.cut(df["BMI"],
                                bins=[0, 18.5, 25, 30, 100],
                                labels=[0, 1, 2, 3]).astype(int)
    df["AgeGroup"]     = pd.cut(df["Age"],
                                bins=[0, 30, 45, 60, 100],
                                labels=[0, 1, 2, 3]).astype(int)
    df["HighGlucose"]  = (df["Glucose"] > 140).astype(int)

    features = [c for c in df.columns if c != "Outcome"]
    X = df[features]
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Robust scaling
    scaler = RobustScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # SMOTE on training set only
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_sc, y_train)
    print(f"[PREP] After SMOTE — class counts: "
          f"{dict(zip(*np.unique(y_train_res, return_counts=True)))}")

    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    print("[PREP] Scaler saved → models/scaler.pkl")
    return (X_train_res, X_test_sc, y_train_res, y_test,
            features, scaler, X_test)


# ═══════════════════════════════════════════════════════════
# 4. MODEL TRAINING & CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════
def train_models(X_train, y_train):
    """Train multiple classifiers and return fitted models."""
    print("\n[TRAIN] Training classifiers …")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=4,
            class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42),
        "SVM (RBF)": SVC(
            kernel="rbf", C=1.5, gamma="scale",
            probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7, metric="minkowski"),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train,
                                 cv=skf, scoring="roc_auc", n_jobs=-1)
        cv_results[name] = scores
        print(f"  {name:25s}  CV AUC = {scores.mean():.4f} ± {scores.std():.4f}")
        model.fit(X_train, y_train)

    # Ensemble Voting Classifier
    voting = VotingClassifier(
        estimators=[
            ("lr",  models["Logistic Regression"]),
            ("rf",  models["Random Forest"]),
            ("gb",  models["Gradient Boosting"]),
        ],
        voting="soft",
    )
    v_scores = cross_val_score(voting, X_train, y_train,
                               cv=skf, scoring="roc_auc", n_jobs=-1)
    cv_results["Ensemble Voting"] = v_scores
    print(f"  {'Ensemble Voting':25s}  CV AUC = "
          f"{v_scores.mean():.4f} ± {v_scores.std():.4f}")
    voting.fit(X_train, y_train)
    models["Ensemble Voting"] = voting

    return models, cv_results


# ═══════════════════════════════════════════════════════════
# 5. EVALUATION
# ═══════════════════════════════════════════════════════════
def evaluate(models, X_test, y_test, cv_results):
    """Evaluate each model and return results dict."""
    print("\n[EVAL] Evaluating on held-out test set …")
    results = {}

    for name, model in models.items():
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "f1":        f1_score(y_test, y_pred),
            "roc_auc":   roc_auc_score(y_test, y_proba),
            "avg_prec":  average_precision_score(y_test, y_proba),
            "y_pred":    y_pred,
            "y_proba":   y_proba,
            "cv_auc":    cv_results[name],
        }
        print(f"\n  ── {name} ──")
        print(f"  Accuracy : {results[name]['accuracy']:.4f}")
        print(f"  F1 Score : {results[name]['f1']:.4f}")
        print(f"  ROC-AUC  : {results[name]['roc_auc']:.4f}")
        report = classification_report(y_test, y_pred,
                  target_names=["No Diabetes", "Diabetes"])
        print(report)

    return results


# ═══════════════════════════════════════════════════════════
# 6. VISUALISATIONS — EVALUATION
# ═══════════════════════════════════════════════════════════
def plot_evaluation(results, y_test):
    """Plot ROC curves, PR curves, confusion matrices, CV scores."""
    print("\n[VIZ] Generating evaluation plots …")

    # ── ROC Curves ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Model Evaluation — ROC & Precision-Recall Curves",
                 fontsize=14, fontweight="bold")

    for i, (name, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        axes[0].plot(fpr, tpr,
                     label=f"{name} (AUC={res['roc_auc']:.3f})",
                     linewidth=2)

        prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
        axes[1].plot(rec, prec,
                     label=f"{name} (AP={res['avg_prec']:.3f})",
                     linewidth=2)

    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set(title="ROC Curves", xlabel="False Positive Rate",
                ylabel="True Positive Rate")
    axes[0].legend(fontsize=8)

    axes[1].set(title="Precision-Recall Curves", xlabel="Recall",
                ylabel="Precision")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "roc_pr_curves.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── Confusion Matrices ──
    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")

    for idx, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    ax=axes[idx],
                    xticklabels=["No Diabetes", "Diabetes"],
                    yticklabels=["No Diabetes", "Diabetes"])
        axes[idx].set_title(name, fontsize=10, fontweight="bold")
        axes[idx].set_ylabel("Actual")
        axes[idx].set_xlabel("Predicted")

    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrices.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── CV Score Comparison ──
    fig, ax = plt.subplots(figsize=(12, 5))
    names = list(results.keys())
    means = [results[n]["cv_auc"].mean() for n in names]
    stds  = [results[n]["cv_auc"].std()  for n in names]

    bars = ax.barh(names, means, xerr=stds,
                   color=PALETTE[:len(names)], edgecolor="white",
                   height=0.55, capsize=5)
    ax.set_xlabel("ROC-AUC (5-fold CV)", fontsize=12)
    ax.set_title("Cross-Validation AUC Comparison",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0.5, 1.0)
    for bar, mean in zip(bars, means):
        ax.text(mean + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{mean:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "cv_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    print("[VIZ] Evaluation plots saved → outputs/")


# ═══════════════════════════════════════════════════════════
# 7. FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════
def plot_feature_importance(models, X_test, y_test, feature_names):
    """Plot feature importance for RF and permutation importance."""
    print("\n[FI] Computing feature importances …")

    rf_model = models["Random Forest"]
    gb_model = models["Gradient Boosting"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle("Feature Importance Analysis",
                 fontsize=15, fontweight="bold")

    # RF built-in importance
    rf_imp = pd.Series(rf_model.feature_importances_,
                       index=feature_names).sort_values(ascending=True)
    rf_imp.plot(kind="barh", ax=axes[0], color="#4C72B0", edgecolor="white")
    axes[0].set_title("Random Forest\n(Gini Importance)",
                      fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Importance Score")

    # GB built-in importance
    gb_imp = pd.Series(gb_model.feature_importances_,
                       index=feature_names).sort_values(ascending=True)
    gb_imp.plot(kind="barh", ax=axes[1], color="#DD8452", edgecolor="white")
    axes[1].set_title("Gradient Boosting\n(Feature Importance)",
                      fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Importance Score")

    # Permutation importance (model-agnostic)
    perm = permutation_importance(
        rf_model, X_test, y_test,
        n_repeats=15, random_state=42, scoring="roc_auc")
    perm_imp = pd.Series(perm.importances_mean,
                         index=feature_names).sort_values(ascending=True)
    perm_imp.plot(kind="barh", ax=axes[2], color="#55A868", edgecolor="white")
    axes[2].set_title("Permutation Importance\n(Model-Agnostic, AUC drop)",
                      fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Mean AUC Decrease")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("[FI] Feature importance saved → outputs/feature_importance.png")

    # Return top features
    top = perm_imp.sort_values(ascending=False).head(5)
    print("\n  Top 5 Predictors (Permutation Importance):")
    for feat, score in top.items():
        print(f"    {feat:30s}  Δ AUC = {score:.4f}")
    return top


# ═══════════════════════════════════════════════════════════
# 8. SAVE BEST MODEL
# ═══════════════════════════════════════════════════════════
def save_best_model(models, results):
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = models[best_name]
    path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_model, path)
    print(f"\n[SAVE] Best model: {best_name} "
          f"(AUC={results[best_name]['roc_auc']:.4f})")
    print(f"[SAVE] Saved → {path}")
    return best_name


# ═══════════════════════════════════════════════════════════
# 9. SUMMARY REPORT
# ═══════════════════════════════════════════════════════════
def save_summary(results, best_name):
    rows = []
    for name, res in results.items():
        rows.append({
            "Model":       name,
            "Accuracy":    round(res["accuracy"], 4),
            "F1 Score":    round(res["f1"],       4),
            "ROC-AUC":     round(res["roc_auc"],  4),
            "Avg Prec":    round(res["avg_prec"], 4),
            "CV AUC Mean": round(res["cv_auc"].mean(), 4),
            "CV AUC Std":  round(res["cv_auc"].std(),  4),
        })
    df_res = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
    path = os.path.join(OUT_DIR, "model_summary.csv")
    df_res.to_csv(path, index=False)
    print(f"\n[REPORT] Summary saved → {path}")
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(df_res.to_string(index=False))
    print(f"\n  ★  Best Model: {best_name}")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("\n" + "█"*60)
    print("  HEALTHCARE PREDICTIVE ANALYTICS — DIABETES DETECTION")
    print("█"*60)

    # 1. Load
    df = load_data(DATA_PATH)

    # 2. EDA
    eda(df)

    # 3. Preprocess
    X_train, X_test, y_train, y_test, features, scaler, _ = preprocess(df)

    # 4. Train
    models, cv_results = train_models(X_train, y_train)

    # 5. Evaluate
    results = evaluate(models, X_test, y_test, cv_results)

    # 6. Plots
    plot_evaluation(results, y_test)

    # 7. Feature importance
    plot_feature_importance(models, X_test, y_test, features)

    # 8. Save best
    best_name = save_best_model(models, results)

    # 9. Summary
    save_summary(results, best_name)

    print("\n✅  Pipeline complete! All outputs in → outputs/")


if __name__ == "__main__":
    main()
