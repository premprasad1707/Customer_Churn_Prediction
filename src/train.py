import pandas as pd
import numpy as np
import json
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, accuracy_score, f1_score)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
EXPORT_DIR = "exports"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(f"{PROCESSED_DIR}/cleaned.csv")
    with open(f"{PROCESSED_DIR}/columns.json") as f:
        meta = json.load(f)
    target = meta["target"]
    features = meta["features"]
    X = df[features]
    y = df[target]
    print(f"[INFO] Features: {len(features)} | Samples: {len(df)}")
    return X, y, features


def apply_smote(X_train, y_train):
    ratio = y_train.value_counts().max() / y_train.value_counts().min()
    if ratio > 2:
        print(f"[INFO] Class imbalance detected (ratio={ratio:.1f}). Applying SMOTE...")
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"[INFO] After SMOTE: {pd.Series(y_train).value_counts().to_dict()}")
    return X_train, y_train


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    print(f"\n{'='*40}")
    print(f"Model: {name}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(classification_report(y_test, preds))
    return {"model": name, "accuracy": round(acc, 4), "f1": round(f1, 4), "roc_auc": round(auc, 4)}


def train():
    X, y, features = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    X_train_s, y_train = apply_smote(X_train_s, y_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42,
                                  eval_metric="logloss", verbosity=0),
        "LightGBM": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    }

    results = []
    trained = {}

    for name, model in models.items():
        print(f"\n[TRAIN] Training {name}...")
        model.fit(X_train_s, y_train)
        res = evaluate(name, model, X_test_s, y_test)
        results.append(res)
        trained[name] = model

    # Pick best model
    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    print(f"\n{'='*40}")
    print("MODEL COMPARISON:")
    print(results_df.to_string(index=False))
    results_df.to_csv(f"{EXPORT_DIR}/model_comparison.csv", index=False)

    best_name = results_df.iloc[0]["model"]
    best_model = trained[best_name]
    print(f"\n[BEST] Best model: {best_name} (ROC-AUC={results_df.iloc[0]['roc_auc']})")

    # Tune best model
    print(f"\n[TUNE] Hyperparameter tuning {best_name}...")
    if best_name == "Random Forest":
        param_grid = {"n_estimators": [100, 200], "max_depth": [None, 10, 20]}
    elif best_name in ["XGBoost", "LightGBM", "Gradient Boosting"]:
        param_grid = {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]}
    else:
        param_grid = {"C": [0.01, 0.1, 1, 10]}

    try:
        gs = GridSearchCV(best_model, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_train_s, y_train)
        best_model = gs.best_estimator_
        print(f"[TUNE] Best params: {gs.best_params_}")
        final_res = evaluate(f"{best_name} (Tuned)", best_model, X_test_s, y_test)
    except Exception as e:
        print(f"[WARN] Tuning failed: {e}. Using default best model.")

    # Save
    joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

    meta = {
        "best_model_name": best_name,
        "features": features,
        "roc_auc": float(results_df.iloc[0]["roc_auc"]),
        "accuracy": float(results_df.iloc[0]["accuracy"]),
        "f1": float(results_df.iloc[0]["f1"]),
        "all_results": results
    }
    with open(f"{MODEL_DIR}/metrics.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Feature importance
    if hasattr(best_model, "feature_importances_"):
        fi = pd.DataFrame({
            "feature": features,
            "importance": best_model.feature_importances_
        }).sort_values("importance", ascending=False)
        fi.to_csv(f"{EXPORT_DIR}/feature_importance.csv", index=False)
        print(f"\n[INFO] Top 10 Features:\n{fi.head(10).to_string(index=False)}")

    # Confusion matrix
    cm = confusion_matrix(y_test, best_model.predict(X_test_s))
    pd.DataFrame(cm).to_csv(f"{EXPORT_DIR}/confusion_matrix.csv", index=False)

    print(f"\n[DONE] Model saved -> {MODEL_DIR}/best_model.pkl")
    print(f"[DONE] Scaler saved -> {MODEL_DIR}/scaler.pkl")
    return best_model, scaler, features


if __name__ == "__main__":
    train()
