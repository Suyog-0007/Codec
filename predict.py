"""
predict.py — Run inference on new (anonymised) patient data.

Usage:
    python predict.py --glucose 148 --bmi 33.6 --age 50 --pregnancies 2
    python predict.py --csv new_patients.csv
"""
import argparse
import joblib
import numpy as np
import pandas as pd
import os

BASE    = os.path.dirname(os.path.abspath(__file__))
MODEL   = joblib.load(os.path.join(BASE, "models", "best_model.pkl"))
SCALER  = joblib.load(os.path.join(BASE, "models", "scaler.pkl"))

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
    "GlucoseBMI", "BMI_Category", "AgeGroup", "HighGlucose",
]


def engineer(row: dict) -> pd.DataFrame:
    row["GlucoseBMI"]   = row["Glucose"] * row["BMI"]
    row["BMI_Category"] = (
        0 if row["BMI"] < 18.5 else
        1 if row["BMI"] < 25 else
        2 if row["BMI"] < 30 else 3
    )
    row["AgeGroup"] = (
        0 if row["Age"] < 30 else
        1 if row["Age"] < 45 else
        2 if row["Age"] < 60 else 3
    )
    row["HighGlucose"] = int(row["Glucose"] > 140)
    return pd.DataFrame([row])[FEATURES]


def predict_single(args):
    row = {
        "Pregnancies":            args.pregnancies,
        "Glucose":                args.glucose,
        "BloodPressure":          args.bp,
        "SkinThickness":          args.skin,
        "Insulin":                args.insulin,
        "BMI":                    args.bmi,
        "DiabetesPedigreeFunction": args.dpf,
        "Age":                    args.age,
    }
    df_row = engineer(row)
    X_sc   = SCALER.transform(df_row)
    proba  = MODEL.predict_proba(X_sc)[0][1]
    risk   = "HIGH" if proba > 0.5 else "LOW"

    print("\n" + "="*50)
    print("  DIABETES RISK ASSESSMENT  (Anonymised)")
    print("="*50)
    for k, v in row.items():
        print(f"  {k:30s}: {v}")
    print("-"*50)
    print(f"  Diabetes Probability : {proba:.2%}")
    print(f"  Risk Level           : {risk}")
    print("="*50)
    print("  ⚠  This is NOT a clinical diagnosis.")
    print("  ⚠  Please consult a qualified physician.")
    print("="*50 + "\n")


def predict_csv(path: str):
    df   = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        eng  = engineer(r.to_dict())
        X_sc = SCALER.transform(eng)
        rows.append(MODEL.predict_proba(X_sc)[0][1])
    df["DiabetesRisk_Proba"] = rows
    df["RiskLevel"] = df["DiabetesRisk_Proba"].apply(
        lambda p: "HIGH" if p > 0.5 else "LOW")
    out = path.replace(".csv", "_predictions.csv")
    df.to_csv(out, index=False)
    print(f"Predictions saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",          default=None)
    parser.add_argument("--pregnancies",  type=int,   default=1)
    parser.add_argument("--glucose",      type=float, default=120)
    parser.add_argument("--bp",           type=float, default=70)
    parser.add_argument("--skin",         type=float, default=20)
    parser.add_argument("--insulin",      type=float, default=80)
    parser.add_argument("--bmi",          type=float, default=28.0)
    parser.add_argument("--dpf",          type=float, default=0.4)
    parser.add_argument("--age",          type=int,   default=35)
    args = parser.parse_args()

    if args.csv:
        predict_csv(args.csv)
    else:
        predict_single(args)
