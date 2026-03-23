"""
model.py
Preprocesses the Telco Churn dataset, trains Logistic Regression and Random Forest,
evaluates both, and exports predictions + feature importances.

Run: python model.py
Dataset: Download from https://www.kaggle.com/datasets/blastchar/telco-customer-churn
         Save as: data/WA_Fn-UseC_-Telco-Customer-Churn.csv
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score
)

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
DB_PATH = "data/churn.db"
EXPORT_DIR = "data/exports"


def load_and_preprocess():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df)} rows, {df.shape[1]} columns")

    # Fix TotalCharges (has spaces for new customers)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["MonthlyCharges"], inplace=True)

    # Drop customerID (not a feature)
    df.drop("customerID", axis=1, inplace=True)

    # Encode target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Binary yes/no columns
    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling",
                   "MultipleLines", "OnlineSecurity", "OnlineBackup",
                   "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0, "No phone service": 0,
                                   "No internet service": 0}).fillna(0).astype(int)

    # Encode gender
    df["gender"] = (df["gender"] == "Male").astype(int)

    # One-hot encode contract, internet service, payment method
    df = pd.get_dummies(df, columns=["Contract", "InternetService", "PaymentMethod"],
                        drop_first=False)

    print(f"  After preprocessing: {df.shape[1]} features")
    return df


def train_and_evaluate(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # --- Logistic Regression ---
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_auc = roc_auc_score(y_test, lr_prob)
    cv_lr = cross_val_score(lr, X_train_scaled, y_train, cv=5, scoring="accuracy").mean()

    print(f"  Accuracy: {lr_acc:.4f}")
    print(f"  ROC-AUC:  {lr_auc:.4f}")
    print(f"  CV Accuracy (5-fold): {cv_lr:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, lr_pred, target_names=["No Churn", "Churn"]))

    results["logistic_regression"] = {
        "accuracy": round(lr_acc, 4),
        "roc_auc": round(lr_auc, 4),
        "cv_accuracy": round(cv_lr, 4)
    }

    # --- Random Forest ---
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_prob)
    cv_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring="accuracy").mean()

    print(f"  Accuracy: {rf_acc:.4f}")
    print(f"  ROC-AUC:  {rf_auc:.4f}")
    print(f"  CV Accuracy (5-fold): {cv_rf:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, rf_pred, target_names=["No Churn", "Churn"]))

    results["random_forest"] = {
        "accuracy": round(rf_acc, 4),
        "roc_auc": round(rf_auc, 4),
        "cv_accuracy": round(cv_rf, 4)
    }

    # --- Feature Importance ---
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False).head(15)

    print(f"\nTop 10 Features (Random Forest):")
    print(fi.head(10).to_string(index=False))

    # --- Save predictions to DB and CSV ---
    os.makedirs(EXPORT_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    test_df = X_test.copy()
    test_df["actual_churn"] = y_test.values
    test_df["lr_predicted"] = lr_pred
    test_df["lr_probability"] = lr_prob.round(4)
    test_df["rf_predicted"] = rf_pred
    test_df["rf_probability"] = rf_prob.round(4)
    test_df.to_csv(f"{EXPORT_DIR}/predictions.csv", index=False)

    fi.to_csv(f"{EXPORT_DIR}/feature_importance.csv", index=False)

    # Model comparison
    comparison = pd.DataFrame(results).T.reset_index()
    comparison.columns = ["model", "accuracy", "roc_auc", "cv_accuracy"]
    comparison.to_csv(f"{EXPORT_DIR}/model_comparison.csv", index=False)

    # Save to SQLite
    conn = sqlite3.connect(DB_PATH)
    test_df.to_sql("predictions", conn, if_exists="replace", index=False)
    fi.to_sql("feature_importance", conn, if_exists="replace", index=False)
    comparison.to_sql("model_comparison", conn, if_exists="replace", index=False)
    conn.close()

    print(f"\nAll outputs saved to {EXPORT_DIR}/ and {DB_PATH}")
    print("\n=== FINAL MODEL COMPARISON ===")
    print(comparison.to_string(index=False))

    return results, fi


if __name__ == "__main__":
    print("=" * 55)
    print("Telecom Customer Churn Prediction")
    print("=" * 55)
    df = load_and_preprocess()
    results, fi = train_and_evaluate(df)
    print("\nDone. Run `streamlit run dashboard.py` to view the dashboard.")
