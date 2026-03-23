"""
dashboard.py
Interactive Streamlit dashboard for the Telecom Churn Prediction project.
Run: streamlit run dashboard.py  (after running model.py first)
"""

import streamlit as st
import pandas as pd
import sqlite3
import os
import plotly.express as px
import plotly.graph_objects as go

DB_PATH = "data/churn.db"
EXPORT_DIR = "data/exports"

st.set_page_config(page_title="Churn Prediction Dashboard", page_icon="📉", layout="wide")

st.markdown("<style>.block-container{padding-top:1.5rem;}</style>", unsafe_allow_html=True)


@st.cache_data
def load_data():
    if not os.path.exists(DB_PATH):
        return None, None, None
    conn = sqlite3.connect(DB_PATH)
    try:
        preds = pd.read_sql("SELECT * FROM predictions", conn)
        fi = pd.read_sql("SELECT * FROM feature_importance", conn)
        comp = pd.read_sql("SELECT * FROM model_comparison", conn)
        conn.close()
        return preds, fi, comp
    except Exception:
        conn.close()
        return None, None, None


def load_sample_data():
    fi = pd.DataFrame({
        "feature": ["Contract_Month-to-month", "tenure", "MonthlyCharges",
                     "TotalCharges", "InternetService_Fiber optic",
                     "OnlineSecurity_No", "TechSupport_No", "PaymentMethod_Electronic check",
                     "PaperlessBilling", "Dependents"],
        "importance": [0.18, 0.16, 0.13, 0.11, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]
    })
    comp = pd.DataFrame({
        "model": ["logistic_regression", "random_forest"],
        "accuracy": [0.8012, 0.8134],
        "roc_auc": [0.8421, 0.8672],
        "cv_accuracy": [0.7988, 0.8097]
    })
    import numpy as np
    np.random.seed(42)
    n = 1409
    preds = pd.DataFrame({
        "actual_churn": np.random.choice([0, 1], n, p=[0.735, 0.265]),
        "rf_predicted": np.random.choice([0, 1], n, p=[0.74, 0.26]),
        "rf_probability": np.random.beta(2, 5, n),
        "tenure": np.random.randint(1, 72, n),
        "MonthlyCharges": np.random.uniform(18, 119, n),
    })
    return preds, fi, comp


preds, fi, comp = load_data()
if preds is None:
    st.info("Running in demo mode — showing sample data. Run `python model.py` to load real results.")
    preds, fi, comp = load_sample_data()

st.title("📉 Telecom Customer Churn Prediction")
st.caption("Machine Learning model comparing Logistic Regression vs Random Forest · IBM Telco Dataset")

st.markdown("---")
st.subheader("Model Performance Comparison")
cols = st.columns(len(comp))
for i, row in comp.iterrows():
    with cols[i]:
        label = "Logistic Regression" if "logistic" in row["model"] else "Random Forest"
        st.markdown(f"""
        <div style="background:{'#E1F5EE' if i==1 else '#F1EFE8'};border-radius:10px;padding:1rem;text-align:center;">
            <div style="font-size:0.85rem;color:#555;">{label}</div>
            <div style="font-size:1.8rem;font-weight:600;color:{'#0F6E56' if i==1 else '#444'};">
                {row['accuracy']*100:.1f}%
            </div>
            <div style="font-size:0.8rem;color:#777;">Accuracy</div>
            <div style="margin-top:0.5rem;font-size:0.85rem;">ROC-AUC: <b>{row['roc_auc']:.3f}</b></div>
            <div style="font-size:0.85rem;">CV Acc: <b>{row['cv_accuracy']*100:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)
st.markdown("---")

row1a, row1b = st.columns(2)

with row1a:
    st.subheader("Top Predictive Features")
    fi_top = fi.head(10).copy()
    fig = px.bar(
        fi_top, x="importance", y="feature", orientation="h",
        color="importance", color_continuous_scale="Teal",
        labels={"importance": "Importance Score", "feature": "Feature"}
    )
    fig.update_layout(coloraxis_showscale=False,
                      yaxis={"categoryorder": "total ascending"},
                      margin=dict(l=0, r=0, t=10, b=0), height=370)
    st.plotly_chart(fig, use_container_width=True)

with row1b:
    st.subheader("Predicted Churn Probability Distribution")
    churned = preds[preds["actual_churn"] == 1]["rf_probability"]
    stayed = preds[preds["actual_churn"] == 0]["rf_probability"]
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=stayed, name="No Churn", opacity=0.7,
                                marker_color="#1D9E75", nbinsx=30))
    fig2.add_trace(go.Histogram(x=churned, name="Churned", opacity=0.7,
                                marker_color="#E24B4A", nbinsx=30))
    fig2.update_layout(barmode="overlay", xaxis_title="Predicted Churn Probability",
                       yaxis_title="Count", legend=dict(x=0.7, y=0.95),
                       margin=dict(l=0, r=0, t=10, b=0), height=370)
    st.plotly_chart(fig2, use_container_width=True)

row2a, row2b = st.columns(2)

with row2a:
    st.subheader("Actual vs Predicted (Random Forest)")
    if "rf_predicted" in preds.columns and "actual_churn" in preds.columns:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(preds["actual_churn"], preds["rf_predicted"])
        fig3 = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["No Churn", "Churn"], y=["No Churn", "Churn"],
            color_continuous_scale="Greens",
            text_auto=True
        )
        fig3.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=330)
        st.plotly_chart(fig3, use_container_width=True)

with row2b:
    st.subheader("Tenure vs Churn Probability")
    if "tenure" in preds.columns:
        sample = preds.sample(min(500, len(preds)))
        sample = sample.copy()
        sample["Churn Status"] = sample["actual_churn"].astype(str).map({"0": "No Churn", "1": "Churned"})
        fig4 = px.scatter(
            sample,
            x="tenure",
            y="rf_probability",
            color="Churn Status",
            color_discrete_map={"No Churn": "#1D9E75", "Churned": "#E24B4A"},
            opacity=0.5,
            labels={"tenure": "Tenure (months)", "rf_probability": "Churn Probability"}
        )
        fig4.update_layout(legend_title="Actual",
                           margin=dict(l=0, r=0, t=10, b=0), height=330)
        st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")
st.subheader("Key Business Insights")
c1, c2, c3 = st.columns(3)
c1.info("**Month-to-month contracts** are the #1 churn predictor. Customers on these plans churn at ~43% vs ~3% for 2-year contracts.")
c2.warning("**Short tenure** customers (< 12 months) are at highest risk. Early engagement programs could reduce churn significantly.")
c3.success("**Random Forest outperforms** Logistic Regression on ROC-AUC, making it the preferred model for probability-ranked intervention lists.")

st.caption("Built by Simrah Ayan · IBM Telco Churn Dataset · github.com/simrahayan")
