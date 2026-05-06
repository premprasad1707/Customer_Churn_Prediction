import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

MODEL_DIR = "models"
EXPORT_DIR = "exports"
PROCESSED_DIR = "data/processed"


@st.cache_data
def load_predictions():
    path = f"{EXPORT_DIR}/churn_predictions.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_metrics():
    path = f"{MODEL_DIR}/metrics.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_model_comparison():
    path = f"{EXPORT_DIR}/model_comparison.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_resource
def load_model():
    try:
        model = joblib.load(f"{MODEL_DIR}/best_model.pkl")
        scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
        with open(f"{MODEL_DIR}/metrics.json") as f:
            meta = json.load(f)
        return model, scaler, meta["features"]
    except Exception as e:
        return None, None, []


# ─── Sidebar Navigation ───────────────────────────────────────────
st.sidebar.title("📊 Churn Prediction")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📋 Predictions Table",
    "👥 Customer Segments",
    "🔍 Predict New Customer",
    "📈 Model Performance"
])
st.sidebar.markdown("---")
metrics = load_metrics()
if metrics:
    st.sidebar.metric("Best Model", metrics.get("best_model_name", "N/A"))
    st.sidebar.metric("ROC-AUC", metrics.get("roc_auc", "N/A"))
    st.sidebar.metric("Accuracy", metrics.get("accuracy", "N/A"))


# ─── PAGE 1: OVERVIEW ─────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("🛒 Customer Churn Prediction Dashboard")
    st.markdown("Real-time churn analysis and retention strategy system.")

    df = load_predictions()

    if df is None:
        st.warning("No predictions found. Run `python src/predict.py` first.")
        st.stop()

    total = len(df)
    churned = int(df["churn_prediction"].sum()) if "churn_prediction" in df.columns else 0
    churn_rate = round(churned / total * 100, 1)
    high_risk = int((df["risk_level"] == "High").sum()) if "risk_level" in df.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total:,}")
    col2.metric("Predicted Churn", f"{churned:,}")
    col3.metric("Churn Rate", f"{churn_rate}%")
    col4.metric("High Risk", f"{high_risk:,}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Churn Distribution")
        churn_counts = df["churn_prediction"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        churn_counts["Churn"] = churn_counts["Churn"].map({0: "Not Churned", 1: "Churned"})
        fig = px.pie(churn_counts, values="Count", names="Churn",
                     color_discrete_map={"Churned": "#EF553B", "Not Churned": "#00CC96"},
                     hole=0.4)
        fig.update_layout(height=320, margin=dict(t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Risk Level Breakdown")
        if "risk_level" in df.columns:
            risk_counts = df["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]
            color_map = {"High": "#EF553B", "Medium": "#FFA15A", "Low": "#00CC96"}
            fig2 = px.bar(risk_counts, x="Risk Level", y="Count",
                          color="Risk Level", color_discrete_map=color_map,
                          text="Count")
            fig2.update_layout(height=320, margin=dict(t=30, b=0), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    # Churn probability distribution
    if "churn_probability_%" in df.columns:
        st.subheader("Churn Probability Distribution")
        fig3 = px.histogram(df, x="churn_probability_%", nbins=30,
                            color_discrete_sequence=["#636EFA"])
        fig3.update_layout(height=280, margin=dict(t=30, b=0))
        st.plotly_chart(fig3, use_container_width=True)


# ─── PAGE 2: PREDICTIONS TABLE ────────────────────────────────────
elif page == "📋 Predictions Table":
    st.title("📋 Customer Predictions")

    df = load_predictions()
    if df is None:
        st.warning("No predictions found. Run `python src/predict.py` first.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        risk_filter = st.selectbox("Filter by Risk Level", ["All", "High", "Medium", "Low"])
    with col2:
        sort_by = st.selectbox("Sort by", ["churn_probability_%", "risk_level"])

    filtered = df.copy()
    if risk_filter != "All" and "risk_level" in filtered.columns:
        filtered = filtered[filtered["risk_level"] == risk_filter]

    if sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by, ascending=False)

    st.markdown(f"Showing **{len(filtered):,}** customers")

    display_cols = ["churn_prediction", "churn_probability_%", "risk_level", "retention_action"]
    display_cols = [c for c in display_cols if c in filtered.columns]
    other_cols = [c for c in filtered.columns if c not in display_cols]
    st.dataframe(filtered[display_cols + other_cols[:6]].reset_index(drop=True),
                 use_container_width=True, height=450)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download CSV", csv, "filtered_churn.csv", "text/csv")


# ─── PAGE 3: SEGMENTS ─────────────────────────────────────────────
elif page == "👥 Customer Segments":
    st.title("👥 Customer Segmentation")

    seg_path = f"{EXPORT_DIR}/segments.csv"
    pred_path = f"{EXPORT_DIR}/churn_predictions.csv"

    if not os.path.exists(seg_path):
        st.warning("No segment data found. Run `python src/segment.py` first.")
        st.stop()

    seg_df = pd.read_csv(seg_path)
    pred_df = load_predictions()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Segment Distribution")
        seg_counts = seg_df["segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig = px.pie(seg_counts, values="Count", names="Segment", hole=0.3)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Segment Details")
        if pred_df is not None and "segment" in pred_df.columns:
            seg_summary = pred_df.groupby("segment").agg(
                Customers=("segment", "count"),
                Avg_Churn_Prob=("churn_probability_%", "mean"),
                High_Risk=("risk_level", lambda x: (x == "High").sum())
            ).reset_index()
            seg_summary["Avg_Churn_Prob"] = seg_summary["Avg_Churn_Prob"].round(1)
            st.dataframe(seg_summary, use_container_width=True)

    # Scatter
    num_cols = seg_df.select_dtypes(include=np.number).columns.tolist()
    num_cols = [c for c in num_cols if c not in ["cluster"]]

    if len(num_cols) >= 2:
        st.subheader("Segment Scatter Plot")
        c1, c2 = st.columns(2)
        with c1:
            x_col = st.selectbox("X axis", num_cols, index=0)
        with c2:
            y_col = st.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1))

        fig2 = px.scatter(seg_df, x=x_col, y=y_col, color="segment",
                          opacity=0.7, size_max=8)
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)


# ─── PAGE 4: PREDICT NEW CUSTOMER ────────────────────────────────
elif page == "🔍 Predict New Customer":
    st.title("🔍 Predict Churn for a New Customer")

    model, scaler, features = load_model()

    if model is None:
        st.warning("Model not found. Run `python src/train.py` first.")
        st.stop()

    df = load_predictions()

    st.markdown("Enter customer details below:")
    st.markdown("---")

    input_dict = {}
    cols = st.columns(3)

    with open(f"{PROCESSED_DIR}/columns.json") as f:
        col_meta = json.load(f)
        cat_cols = col_meta.get("cat_cols", [])

    import glob
    raw_files = glob.glob("data/raw/*.csv")
    raw_df = pd.read_csv(raw_files[0]) if raw_files else None

    for i, feat in enumerate(features):
        col = cols[i % 3]
        with col:
            # Categorical handling
            if feat in cat_cols and raw_df is not None and feat in raw_df.columns:
                options = sorted(raw_df[feat].astype(str).unique().tolist())
                selected_val = col.selectbox(feat, options)
                # Map back to integer index exactly like LabelEncoder does
                val = options.index(selected_val)
            # Binary fallback or numerical handling
            elif df is not None and feat in df.columns:
                min_v = float(df[feat].min())
                max_v = float(df[feat].max())
                mean_v = float(df[feat].mean())
                unique_vals = df[feat].nunique()

                if unique_vals <= 2 and feat not in cat_cols:
                    val = col.selectbox(feat, [0, 1],
                                        format_func=lambda x: "No" if x == 0 else "Yes")
                else:
                    # e.g., MonthlyCharges, TotalCharges, tenure
                    val = col.number_input(feat, min_value=min_v,
                                           max_value=max_v, value=round(mean_v, 2))
            else:
                val = col.number_input(feat, value=0.0)
            input_dict[feat] = val

    st.markdown("---")
    if st.button("🔍 Predict Churn Risk", use_container_width=True):
        row = pd.DataFrame([input_dict])
        row_scaled = scaler.transform(row[features])
        prob = model.predict_proba(row_scaled)[0][1]
        pred = model.predict(row_scaled)[0]

        prob_pct = round(prob * 100, 1)

        if prob >= 0.6:
            risk = "High"
            action = "Call customer immediately + offer discount"
            st.error(f"⚠️ HIGH RISK — This customer is likely to churn!")
        elif prob >= 0.3:
            risk = "Medium"
            action = "Send re-engagement email + loyalty reward"
            st.warning(f"⚡ MEDIUM RISK — Monitor this customer closely.")
        else:
            risk = "Low"
            action = "Send newsletter + satisfaction survey"
            st.success(f"✅ LOW RISK — Customer is likely to stay.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Probability", f"{prob_pct}%")
        c2.metric("Risk Level", risk)
        c3.metric("Prediction", "Will Churn" if pred == 1 else "Will Stay")

        st.progress(int(prob_pct))

        st.subheader("💡 Recommended Action")
        st.info(f"➡️ {action}")

        if prob >= 0.6:
            st.markdown("""
            **Urgent retention steps:**
            - Call within 24 hours and offer a personalized discount
            - Assign a dedicated customer success manager
            - Identify the root cause (billing, service quality, competitor)
            """)
        elif prob >= 0.3:
            st.markdown("""
            **Preventive steps:**
            - Send a personalized re-engagement email
            - Offer a loyalty reward or bonus credits
            - Schedule a check-in call within 1 week
            """)
        else:
            st.markdown("""
            **Maintenance tips:**
            - Include in monthly newsletter
            - Send satisfaction survey
            - Offer upgrade options
            """)


# ─── PAGE 5: MODEL PERFORMANCE ────────────────────────────────────
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    comp_df = load_model_comparison()
    metrics = load_metrics()

    if comp_df is None or metrics is None:
        st.warning("No model metrics found. Run `python src/train.py` first.")
        st.stop()

    st.subheader("All Models Comparison")
    fig = px.bar(comp_df.sort_values("roc_auc"),
                 x="roc_auc", y="model", orientation="h",
                 color="roc_auc", color_continuous_scale="Blues",
                 text="roc_auc", labels={"roc_auc": "ROC-AUC"})
    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Best Model", metrics["best_model_name"])
    col2.metric("ROC-AUC", metrics["roc_auc"])
    col3.metric("Accuracy", metrics["accuracy"])

    # Confusion matrix
    cm_path = f"{EXPORT_DIR}/confusion_matrix.csv"
    if os.path.exists(cm_path):
        st.subheader("Confusion Matrix")
        cm = pd.read_csv(cm_path).values
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                           labels=dict(x="Predicted", y="Actual"),
                           x=["No Churn", "Churn"], y=["No Churn", "Churn"])
        fig_cm.update_layout(height=350)
        st.plotly_chart(fig_cm, use_container_width=True)

    # Feature importance
    fi_path = f"{EXPORT_DIR}/feature_importance.csv"
    if os.path.exists(fi_path):
        st.subheader("Top 15 Feature Importances")
        fi = pd.read_csv(fi_path).head(15)
        fig_fi = px.bar(fi.sort_values("importance"),
                        x="importance", y="feature", orientation="h",
                        color="importance", color_continuous_scale="Teal")
        fig_fi.update_layout(height=450)
        st.plotly_chart(fig_fi, use_container_width=True)
