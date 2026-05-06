# 🛒 Customer Churn Prediction System

An end-to-end ML system to predict customer churn with a Streamlit dashboard.

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost, LightGBM
- Streamlit, Plotly
- Power BI (optional export)

## Setup
venv\Scripts\activate
```bash
pip install -r requirements.txt
```

## Dataset
Download from Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Place the CSV file inside `data/raw/`

## Run Full Pipeline

```bash
python run_pipeline.py
```

This runs all steps:
1. Preprocess → cleans and encodes data
2. Train → trains 5 models, picks best
3. Predict → scores all customers
4. Segment → groups customers into 4 segments

## Launch Dashboard

```bash
streamlit run dashboard/app.py
```

## Run Steps Individually

```bash
python src/preprocess.py
python src/train.py
python src/predict.py
python src/segment.py
```

## Power BI
Connect `exports/churn_predictions.csv` and `exports/segments.csv` to Power BI Desktop.
Build: KPI cards, risk breakdown bar, churn pie chart, segment scatter.

## Model Accuracy
Typical results on Telco dataset:
- Logistic Regression: ~80%
- Random Forest: ~82%
- XGBoost / LightGBM: ~83-85%
- ROC-AUC: ~0.85+
