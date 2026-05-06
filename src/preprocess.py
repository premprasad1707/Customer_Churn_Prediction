import pandas as pd
import numpy as np
import json
import os
import glob

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

TARGET_KEYWORDS = ["churn", "target", "label", "outcome", "result"]
ID_KEYWORDS = ["id", "customerid", "customer_id", "userid", "user_id"]


def find_csv():
    files = glob.glob(f"{RAW_DIR}/*.csv")
    if not files:
        raise FileNotFoundError(f"No CSV found in {RAW_DIR}/. Please place your dataset there.")
    print(f"[INFO] Found: {files[0]}")
    return files[0]


def detect_target(df):
    cols_lower = {c.lower(): c for c in df.columns}
    for kw in TARGET_KEYWORDS:
        for col_lower, col_orig in cols_lower.items():
            if kw in col_lower:
                print(f"[INFO] Target column detected: '{col_orig}'")
                return col_orig
    raise ValueError("Could not auto-detect target column. Rename it to 'Churn' or 'target'.")


def detect_id_cols(df):
    drop_cols = []
    for col in df.columns:
        if col.lower() in ID_KEYWORDS:
            drop_cols.append(col)
    if drop_cols:
        print(f"[INFO] Dropping ID columns: {drop_cols}")
    return drop_cols


def preprocess():
    csv_path = find_csv()
    df = pd.read_csv(csv_path)

    print(f"\n[DATA] Shape: {df.shape}")
    print(f"[DATA] Columns: {list(df.columns)}")
    print(f"[DATA] Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    # Drop ID columns
    id_cols = detect_id_cols(df)
    df.drop(columns=id_cols, inplace=True)

    # Detect target
    target_col = detect_target(df)

    # Fix mixed-type numeric columns (e.g. TotalCharges with spaces)
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > len(df) * 0.8:
                df[col] = converted
                print(f"[INFO] Converted '{col}' to numeric")

    # Encode target
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].map({"Yes": 1, "No": 0, "yes": 1, "no": 0, "1": 1, "0": 0})
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    # Class balance
    balance = df[target_col].value_counts()
    print(f"\n[DATA] Class balance:\n{balance}")

    # Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

    # Encode categoricals
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    print(f"[INFO] Encoded categorical columns: {cat_cols}")

    # Save
    df.to_csv(f"{PROCESSED_DIR}/cleaned.csv", index=False)
    meta = {
        "target": target_col,
        "features": [c for c in df.columns if c != target_col],
        "shape": list(df.shape),
        "cat_cols": cat_cols
    }
    with open(f"{PROCESSED_DIR}/columns.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[DONE] Saved cleaned data -> {PROCESSED_DIR}/cleaned.csv")
    print(f"[DONE] Saved metadata   -> {PROCESSED_DIR}/columns.json")
    return df, target_col


if __name__ == "__main__":
    preprocess()
