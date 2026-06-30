"""
Customer Churn Analysis
========================
Data Analytics Internship @ IGRIS LAB — Week 4

Generates a realistic Telco customer churn dataset,
performs EDA, trains 3 classification models,
evaluates them, and saves the best model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report
)

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent


def generate_churn_data(n=3000, seed=42) -> pd.DataFrame:
    """Generate a realistic synthetic Telco customer churn dataset."""
    np.random.seed(seed)
    customer_id = [f'CUST{str(i).zfill(5)}' for i in range(1, n + 1)]
    senior_citizen = np.random.choice([0, 1], size=n, p=[0.75, 0.25])
    has_partner = np.random.choice(['Yes', 'No'], size=n, p=[0.55, 0.45])
    has_dependents = np.random.choice(['Yes', 'No'], size=n, p=[0.30, 0.70])
    tenure = np.random.choice(range(1, 73), size=n)

    contract_type = []
    for t in tenure:
        if t <= 12: p = [0.70, 0.20, 0.10]
        elif t <= 36: p = [0.30, 0.45, 0.25]
        else: p = [0.10, 0.30, 0.60]
        contract_type.append(np.random.choice(
            ['Month-to-month', 'One year', 'Two year'], p=p))
    payment_method = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
        size=n, p=[0.35, 0.25, 0.20, 0.20])
    internet_service = np.random.choice(
        ['DSL', 'Fiber optic', 'No'], size=n, p=[0.30, 0.45, 0.25])
    tech_support, online_security = [], []
    for iserv in internet_service:
        if iserv == 'No':
            tech_support.append('No internet service')
            online_security.append('No internet service')
        else:
            tech_support.append(np.random.choice(['Yes', 'No'], p=[0.35, 0.65]))
            online_security.append(np.random.choice(['Yes', 'No'], p=[0.40, 0.60]))
    num_products = np.random.choice([1, 2, 3, 4], size=n, p=[0.40, 0.35, 0.20, 0.05])
    base_charge = 20 + np.random.exponential(30, n).clip(10, 100)
    internet_bonus = np.where(np.array(internet_service) == 'Fiber optic', 30,
                              np.where(np.array(internet_service) == 'DSL', 10, 0))
    product_bonus = (np.array(num_products) - 1) * 15
    tech_bonus = np.where(np.array(tech_support) == 'Yes', 10, 0)
    security_bonus = np.where(np.array(online_security) == 'Yes', 8, 0)
    noise = np.random.normal(0, 8, n)
    monthly_charges = (base_charge + internet_bonus + product_bonus
                       + tech_bonus + security_bonus + noise).clip(18, 150).round(2)
    total_charges = (monthly_charges * tenure * (0.95 + 0.1 * np.random.random(n))).round(2)

    churn_prob = np.zeros(n)
    churn_prob += np.where(np.array(contract_type) == 'Month-to-month', 0.30, 0)
    churn_prob += np.where(np.array(contract_type) == 'One year', 0.10, 0)
    churn_prob += np.maximum(0, (12 - np.array(tenure)) / 12 * 0.25)
    churn_prob += np.where(np.array(tech_support) == 'No', 0.10, 0)
    churn_prob += np.where(np.array(tech_support) == 'No internet service', 0.05, 0)
    churn_prob += np.where(np.array(online_security) == 'No', 0.08, 0)
    churn_prob += np.where(np.array(online_security) == 'No internet service', 0.04, 0)
    churn_prob += np.where(np.array(internet_service) == 'Fiber optic', 0.10, 0)
    churn_prob += np.where(np.array(payment_method) == 'Electronic check', 0.08, 0)
    churn_prob += np.array(senior_citizen) * 0.05
    churn_prob -= np.where(np.array(has_partner) == 'Yes', 0.05, 0)
    churn_prob -= np.where(np.array(has_dependents) == 'Yes', 0.04, 0)
    churn_prob = churn_prob.clip(0.02, 0.85)
    churn = np.where(np.random.random(n) < churn_prob, 'Yes', 'No')

    df = pd.DataFrame({
        'customer_id': customer_id, 'tenure_months': tenure,
        'monthly_charges': monthly_charges, 'total_charges': total_charges,
        'contract_type': contract_type, 'payment_method': payment_method,
        'internet_service': internet_service, 'tech_support': tech_support,
        'online_security': online_security, 'num_products': num_products,
        'senior_citizen': senior_citizen, 'has_partner': has_partner,
        'has_dependents': has_dependents, 'churn': churn
    })
    return df


def perform_eda(df_orig: pd.DataFrame):
    """Print key EDA metrics. Does NOT modify the original DataFrame."""
    print("\n" + "=" * 70)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    df = df_orig.copy()
    total = len(df)
    churn_rate = (df['churn'] == 'Yes').mean() * 100
    print(f"\nTotal customers: {total}")
    print(f"Overall churn rate: {churn_rate:.2f}%")
    print(f"Retained: {(df['churn'] == 'No').sum()}  |  Churned: {(df['churn'] == 'Yes').sum()}")

    print(f"\n--- Churn Rate by Contract Type ---")
    contract_churn = df.groupby('contract_type')['churn'].apply(
        lambda x: (x == 'Yes').mean() * 100).sort_values(ascending=False)
    for ct, rate in contract_churn.items():
        print(f"  {ct:20s}: {rate:.2f}%")

    print(f"\n--- Churn Rate by Tenure Group ---")
    df['tenure_group'] = pd.cut(df['tenure_months'],
                                 bins=[0, 6, 12, 24, 48, 72],
                                 labels=['0-6', '7-12', '13-24', '25-48', '49-72'])
    tenure_churn = df.groupby('tenure_group', observed=True)['churn'].apply(
        lambda x: (x == 'Yes').mean() * 100)
    for tg, rate in tenure_churn.items():
        print(f"  {str(tg):>6s} months: {rate:.2f}%")

    print(f"\n--- Churn Rate by Monthly Charges Range ---")
    df['charge_range'] = pd.cut(df['monthly_charges'],
                                 bins=[0, 40, 70, 100, 150],
                                 labels=['$0-40', '$40-70', '$70-100', '$100+'])
    charges_churn = df.groupby('charge_range', observed=True)['churn'].apply(
        lambda x: (x == 'Yes').mean() * 100)
    for cr, rate in charges_churn.items():
        print(f"  {cr:>8s}: {rate:.2f}%")

    print(f"\n--- Revenue at Risk ---")
    revenue_at_risk = df[df['churn'] == 'Yes']['monthly_charges'].sum()
    print(f"  Monthly revenue from churned customers: ${revenue_at_risk:,.2f}")
    avg_tenure_churned = df[df['churn'] == 'Yes']['tenure_months'].mean()
    print(f"  Avg tenure of churned customers: {avg_tenure_churned:.1f} months")

    return {
        'churn_rate': churn_rate, 'contract_churn': contract_churn,
        'tenure_churn': tenure_churn, 'charges_churn': charges_churn,
        'revenue_at_risk': revenue_at_risk, 'avg_tenure_churned': avg_tenure_churned
    }


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train 3 models, evaluate, return results and best model."""
    print("\n" + "=" * 70)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 70)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15,
                                                 random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=200, max_depth=8,
                                  learning_rate=0.1, random_state=42,
                                  eval_metric='logloss')
    }

    results = []
    trained_models = {}
    roc_data = {}

    for name, model in models.items():
        print(f"\n--- {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)

        results.append({
            'Model': name, 'Accuracy': acc, 'Precision': prec,
            'Recall': rec, 'F1': f1, 'AUC-ROC': auc
        })
        trained_models[name] = model
        roc_data[name] = (fpr, tpr, auc)

        print(f"  Accuracy:  {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}  AUC-ROC: {auc:.4f}")

    results_df = pd.DataFrame(results).sort_values('AUC-ROC', ascending=False)
    best_model_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]

    print(f"\n{'=' * 70}")
    print(f"BEST MODEL: {best_model_name}  (AUC-ROC: {results_df.iloc[0]['AUC-ROC']:.4f})")

    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    else:
        importances = np.abs(best_model.coef_[0])

    feature_names = X_train.columns.tolist()
    imp_df = pd.DataFrame({
        'feature': feature_names, 'importance': importances
    }).sort_values('importance', ascending=False)

    print(f"\nTop 5 Churn Predictors ({best_model_name}):")
    for i in range(5):
        row = imp_df.iloc[i]
        print(f"  {i+1}. {row['feature']:25s}: {row['importance']:.4f}")

    return results_df, best_model, trained_models, roc_data, imp_df


if __name__ == "__main__":
    print("=" * 70)
    print("CUSTOMER CHURN ANALYSIS")
    print("Data Analytics Internship @ IGRIS LAB — Week 4")
    print("=" * 70)

    print("\nGenerating synthetic Telco churn dataset (n=3000)...")
    df = generate_churn_data(n=3000)
    print(f"  Generated {df.shape[0]} rows x {df.shape[1]} columns")
    df.to_csv(BASE_DIR / "telco_churn_data.csv", index=False)

    eda_metrics = perform_eda(df)

    # Preprocess
    print("\n" + "=" * 70)
    print("DATA PREPROCESSING")
    print("=" * 70)

    df_model = df.drop(columns=['customer_id']).copy()
    y = (df_model['churn'] == 'Yes').astype(int)
    X = df_model.drop(columns=['churn'])

    # Identify column types BEFORE encoding
    cat_cols = X.select_dtypes(include=['object', 'str']).columns.tolist()
    num_cols_original = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    ALL_FEATURE_NAMES = X.columns.tolist()  # Preserve order: ['tenure_months', ..., 'has_dependents']

    print(f"  Original numeric columns: {num_cols_original}")
    print(f"  Categorical columns: {cat_cols}")
    print(f"  All feature names order: {ALL_FEATURE_NAMES}")

    # Encode categoricals
    le_dict = {}
    X_encoded = X.copy()
    for col in cat_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        le_dict[col] = le

    # After encoding, ALL columns are numeric. Scale ALL of them
    scaler = StandardScaler()
    X_encoded_scaled = pd.DataFrame(
        scaler.fit_transform(X_encoded),
        columns=ALL_FEATURE_NAMES,
        index=X_encoded.index
    )

    print(f"  Encoded features shape: {X_encoded_scaled.shape}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded_scaled, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"\n  Train size: {X_train.shape[0]}  Test size: {X_test.shape[0]}")

    # Train and evaluate
    results_df, best_model, trained_models, roc_data, feature_importance = \
        train_and_evaluate(X_train, X_test, y_train, y_test)

    # Save artifacts
    print("\n" + "=" * 70)
    print("SAVING ARTIFACTS")
    print("=" * 70)

    joblib.dump(best_model, BASE_DIR / 'churn_model.pkl')

    roc_curves_save = {name: (fpr, tpr, auc) for name, (fpr, tpr, auc) in roc_data.items()}

    encoders_data = {
        'le_dict': le_dict,
        'scaler': scaler,
        'cat_cols': cat_cols,
        'num_cols_original': num_cols_original,
        'ALL_FEATURE_NAMES': ALL_FEATURE_NAMES,
        'imp_df': feature_importance,
        'metrics_df': results_df,
        'roc_curves': roc_curves_save
    }
    joblib.dump(encoders_data, BASE_DIR / 'churn_encoders.pkl')

    print("  Saved churn_model.pkl")
    print("  Saved churn_encoders.pkl")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"  Best model: {results_df.iloc[0]['Model']}")
    print(f"  AUC-ROC:    {results_df.iloc[0]['AUC-ROC']:.4f}")
    print(f"  Churn rate: {eda_metrics['churn_rate']:.2f}%")