import pandas as pd
import numpy as np
import json
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = "data/processed"
EXPORT_DIR = "exports"
MODEL_DIR = "models"
os.makedirs(EXPORT_DIR, exist_ok=True)


def segment_customers(n_clusters=4):
    df = pd.read_csv(f"{PROCESSED_DIR}/cleaned.csv")

    with open(f"{PROCESSED_DIR}/columns.json") as f:
        meta = json.load(f)
    target = meta["target"]
    features = meta["features"]

    # Pick behavioral features for clustering
    seg_keywords = ["tenure", "charge", "usage", "monthly", "total", "contract"]
    seg_features = [f for f in features if any(k in f.lower() for k in seg_keywords)]

    if len(seg_features) < 2:
        seg_features = features[:5]

    print(f"[INFO] Segmentation features: {seg_features}")

    X_seg = df[seg_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_seg)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # Auto-label clusters by churn rate + value
    cluster_stats = df.groupby("cluster").agg(
        churn_rate=(target, "mean"),
        size=("cluster", "count")
    )

    sorted_by_churn = cluster_stats["churn_rate"].sort_values()
    label_map = {}
    labels = ["Loyal Customers", "Passive Users", "At-Risk Customers", "New Users"]

    # Assign labels based on churn rate order
    for i, (cluster_id, _) in enumerate(sorted_by_churn.items()):
        label_map[cluster_id] = labels[i % len(labels)]

    df["segment"] = df["cluster"].map(label_map)

    print(f"\n[SEGMENTS]")
    for seg in df["segment"].unique():
        seg_df = df[df["segment"] == seg]
        print(f"  {seg}: {len(seg_df)} customers | Churn Rate: {seg_df[target].mean()*100:.1f}%")

    # Check if predictions exist and merge
    pred_path = f"{EXPORT_DIR}/churn_predictions.csv"
    if os.path.exists(pred_path):
        preds_df = pd.read_csv(pred_path)
        if "segment" not in preds_df.columns:
            preds_df["segment"] = df["segment"].values
            preds_df.to_csv(pred_path, index=False)

    df[seg_features + ["cluster", "segment", target]].to_csv(
        f"{EXPORT_DIR}/segments.csv", index=False
    )
    print(f"\n[DONE] Segments saved -> {EXPORT_DIR}/segments.csv")

    return df


if __name__ == "__main__":
    segment_customers()
