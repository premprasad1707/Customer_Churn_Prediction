import pandas as pd
import numpy as np
import json
import joblib
import os

MODEL_DIR = "models"
PROCESSED_DIR = "data/processed"
EXPORT_DIR = "exports"
os.makedirs(EXPORT_DIR, exist_ok=True)


def load_model():
    model = joblib.load(f"{MODEL_DIR}/best_model.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    with open(f"{MODEL_DIR}/metrics.json") as f:
        meta = json.load(f)
    return model, scaler, meta["features"]


def get_risk_level(prob):
    if prob >= 0.6:
        return "High"
    elif prob >= 0.3:
        return "Medium"
    return "Low"


def get_retention_action(risk):
    actions = {
        "High": "Call customer immediately + offer discount",
        "Medium": "Send re-engagement email + loyalty reward",
        "Low": "Send newsletter + satisfaction survey"
    }
    return actions[risk]


def predict_dataset():
    model, scaler, features = load_model()
    df = pd.read_csv(f"{PROCESSED_DIR}/cleaned.csv")

    with open(f"{PROCESSED_DIR}/columns.json") as f:
        meta = json.load(f)
    target = meta["target"]

    X = df[features]
    X_scaled = scaler.transform(X)

    proba = model.predict_proba(X_scaled)[:, 1]
    preds = model.predict(X_scaled)

    results = df.copy()
    results["churn_prediction"] = preds
    results["churn_probability_%"] = (proba * 100).round(1)
    results["risk_level"] = [get_risk_level(p) for p in proba]
    results["retention_action"] = [get_retention_action(r) for r in results["risk_level"]]

    results.to_csv(f"{EXPORT_DIR}/churn_predictions.csv", index=False)

    print(f"[DONE] Predictions saved -> {EXPORT_DIR}/churn_predictions.csv")
    print(f"\n[SUMMARY]")
    print(f"  Total customers  : {len(results)}")
    print(f"  Predicted churn  : {preds.sum()}")
    print(f"  Churn rate       : {preds.mean()*100:.1f}%")
    print(f"\n  Risk Distribution:")
    print(results["risk_level"].value_counts().to_string())

    return results


def predict_single(input_dict: dict):
    """Predict churn for a single customer. Input is a dict of feature values."""
    model, scaler, features = load_model()

    row = pd.DataFrame([input_dict])
    for col in features:
        if col not in row.columns:
            row[col] = 0

    row = row[features]
    row_scaled = scaler.transform(row)

    prob = model.predict_proba(row_scaled)[0][1]
    pred = model.predict(row_scaled)[0]
    risk = get_risk_level(prob)

    return {
        "prediction": int(pred),
        "probability": round(prob * 100, 1),
        "risk_level": risk,
        "retention_action": get_retention_action(risk)
    }


if __name__ == "__main__":
    predict_dataset()
