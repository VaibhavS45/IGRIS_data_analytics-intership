# Week 4 — Customer Churn Prediction

## Project Goal

Build a machine learning pipeline to predict customer churn for a telecommunications company. Identify the key drivers of churn, compare multiple classification models, and provide actionable retention recommendations.

## Dataset

A realistic synthetic Telco customer churn dataset was generated with **3,000 records** and the following features:

| Column | Description |
|--------|-------------|
| `customer_id` | Unique customer identifier |
| `tenure_months` | Months the customer has been with the company |
| `monthly_charges` | Monthly recurring charges ($) |
| `total_charges` | Total charges to date ($) |
| `contract_type` | Month-to-month / One year / Two year |
| `payment_method` | Electronic check / Mailed check / Bank transfer / Credit card |
| `internet_service` | DSL / Fiber optic / No |
| `tech_support` | Yes / No / No internet service |
| `online_security` | Yes / No / No internet service |
| `num_products` | Number of subscribed products (1–4) |
| `senior_citizen` | 0 = No, 1 = Yes |
| `has_partner` | Yes / No |
| `has_dependents` | Yes / No |
| `churn` | Yes = churned, No = retained |

## Methodology

1. **Data Generation** — Synthetic dataset with realistic churn patterns based on contract type, tenure, services, and demographics
2. **Exploratory Data Analysis** — Churn rate analysis by contract type, tenure group, monthly charges, and payment method
3. **Preprocessing** — Label encoding for categorical features, standard scaling for numeric features
4. **Model Training** — 3 classifiers trained on 75/25 stratified split:
   - Logistic Regression
   - Random Forest (200 trees, max depth 15)
   - XGBoost (200 estimators, max depth 8)
5. **Evaluation** — Accuracy, Precision, Recall, F1, AUC-ROC
6. **Feature Importance** — Top predictors identified from the best model
7. **Deployment** — Best model saved as `churn_model.pkl` with encoders

## Model Performance

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|----------|-----------|--------|-----|---------|
| XGBoost | ~0.85 | ~0.78 | ~0.72 | ~0.75 | ~0.90 |
| Random Forest | ~0.84 | ~0.77 | ~0.70 | ~0.73 | ~0.88 |
| Logistic Regression | ~0.80 | ~0.70 | ~0.65 | ~0.67 | ~0.84 |

*(Exact values vary slightly with random seed)*

## Key Findings

1. **Contract type** is the strongest churn predictor — Month-to-month customers churn at 3-4x the rate of annual contract customers
2. **Tenure** is the second most important factor — churn risk is highest in the first 6 months
3. **Tech support** and **online security** significantly reduce churn probability
4. **Electronic check** payers have the highest churn rate among payment methods
5. **Fiber optic** customers churn more than DSL customers, likely due to higher expectations and more competitive alternatives
6. **Senior citizens** show slightly higher churn rates

## Top Retention Strategies

| Strategy | Target Segment | Expected Impact |
|----------|---------------|-----------------|
| Discounted annual plans | Month-to-month customers | High |
| Onboarding program (first 90 days) | New customers (< 6 months) | High |
| Free tech support trial | Customers without support | Medium-High |
| Auto-pay incentive ($2/month credit) | Electronic check users | Medium |
| Security bundle at no extra cost | Fiber optic + no security | Medium |

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the analysis script

```bash
cd week4-customer-churn
python churn_analysis.py
```

This will:
- Generate `telco_churn_data.csv` (3,000 records)
- Print EDA metrics and churn rates
- Train and evaluate 3 models
- Save `churn_model.pkl` and `churn_encoders.pkl`

### 3. Launch the interactive dashboard

```bash
streamlit run app.py
```

The dashboard includes:
- **Sidebar filters** — contract type, payment method, internet service
- **KPI cards** — churn rate, avg tenure of churned customers, revenue at risk
- **6 interactive charts** — contract churn bar, tenure histogram overlay, charges boxplot, feature importance, ROC curves, payment method pie
- **Model comparison table** — all 3 models side by side
- **Predict Churn tab** — input customer details → get churn probability with risk badge (Low/Medium/High) and personalised recommendations
- **Retention Recommendations tab** — high-risk segments and actionable strategies

### Streamlit Community Cloud Deployment

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. **New app** → Repository: `VaibhavS45/igrislab-data-analytics-internship`
4. **Branch:** `main` → **Main file path:** `week4-customer-churn/app.py`
5. Click **Deploy**

---

*Week 4 completed — Customer Churn Prediction.*