# 📉 Telecom Customer Churn Prediction

An end-to-end machine learning project that predicts customer churn using the IBM Telco dataset. Compares Logistic Regression vs Random Forest, exports predictions to SQLite, and visualizes insights in an interactive Streamlit dashboard.

---

## Features

- **EDA Notebook** — exploratory analysis with charts: churn distribution, churn by contract type, tenure vs charges, correlation heatmap
- **ML Pipeline** — preprocessing, feature engineering, model training, cross-validation, and evaluation for two classifiers
- **Model Comparison** — Logistic Regression vs Random Forest with accuracy, ROC-AUC, and 5-fold CV scores
- **SQLite Export** — predictions, feature importances, and model metrics saved to a local database
- **Streamlit Dashboard** — interactive visualization of model results, feature importance, confusion matrix, and business insights

---

## Results

| Model | Accuracy | ROC-AUC | CV Accuracy |
|---|---|---|---|
| Logistic Regression | 80.1% | 0.842 | 79.9% |
| **Random Forest** | **81.3%** | **0.867** | **81.0%** |

**Random Forest is the better model** for this use case — higher ROC-AUC means better ability to rank customers by churn risk for targeted intervention.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Analysis | Python, Pandas, NumPy |
| Visualization (EDA) | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Storage | SQLite |
| Dashboard | Streamlit, Plotly |
| Notebook | Jupyter |

---

## Project Structure

```
telecom-churn-prediction/
├── model.py                        # Preprocessing, training, evaluation, export
├── dashboard.py                    # Streamlit interactive dashboard
├── requirements.txt
├── notebooks/
│   └── 01_eda.ipynb               # Exploratory Data Analysis
├── data/
│   └── exports/                   # Auto-generated after running model.py
│       ├── predictions.csv
│       ├── feature_importance.csv
│       └── model_comparison.csv
└── plots/                         # Auto-generated EDA charts
```

---

## Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/simrahayan/telecom-churn-prediction.git
cd telecom-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
- Go to: [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Save it to the `data/` folder

### 4. Run the ML pipeline
```bash
python model.py
```
This cleans the data, trains both models, prints evaluation results, and saves outputs to `data/exports/` and `data/churn.db`.

### 5. Run the EDA notebook (optional)
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 6. Launch the dashboard
```bash
streamlit run dashboard.py
```
Open [http://localhost:8501](http://localhost:8501)

> **No dataset yet?** The dashboard shows sample data automatically so you can explore the UI right away.

---

## Key Business Insights

- **Month-to-month contracts** are the #1 churn driver — churn rate of ~43% vs ~3% for 2-year contracts
- **Customers in their first 12 months** are at the highest risk — early loyalty programs are critical
- **Higher monthly charges** combined with short tenure strongly predict churn
- **Electronic check payment** users churn at higher rates than auto-pay customers

---

## Author

**Simrah Ayan**
Durham College — Post-Graduate Diplomas in AI & Data Analytics (Honors)
Microsoft Azure Certified (AI-900 + AZ-900)

[LinkedIn](https://www.linkedin.com/in/simrah-ayan) · [GitHub](https://github.com/simrahayan)
