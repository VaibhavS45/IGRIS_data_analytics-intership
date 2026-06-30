"""
Customer Churn Analytics — Streamlit Dashboard
================================================
Data Analytics Internship @ IGRIS LAB — Week 4

Interactive dashboard with KPIs, 6 charts, churn prediction,
and retention recommendations.

OPTIMISED FOR FAST DEPLOY: Loads pre-trained models from .pkl files
instead of retraining at startup.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📉",
    layout="wide"
)

# ---------------------------
# 1. Generate / Load Data
# ---------------------------

@st.cache_data
def generate_churn_data(n=3000, seed=42):
    """Generate synthetic telco churn dataset."""
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


# ---------------------------
# 2. Load Pre-trained Models (FAST)
# ---------------------------

@st.cache_resource
def load_artifacts():
    """
    Load pre-trained model and encoders from .pkl files.
    This takes ~1 second instead of retraining 3 models from scratch.
    If .pkl files don't exist, trains models and saves them.
    """
    base_dir = Path(__file__).parent
    model_path = base_dir / 'churn_model.pkl'
    encoders_path = base_dir / 'churn_encoders.pkl'

    def train_fresh():
        """Train all models from scratch and return artifacts."""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

        df_temp = generate_churn_data()
        df_temp = df_temp.drop(columns=['customer_id']).copy()
        y = (df_temp['churn'] == 'Yes').astype(int)
        X = df_temp.drop(columns=['churn'])

        cat_cols = X.select_dtypes(include=['object', 'str']).columns.tolist()
        ALL_FEATURE_NAMES = X.columns.tolist()

        le_dict = {}
        X_encoded = X.copy()
        for col in cat_cols:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            le_dict[col] = le

        # Scale ALL columns after encoding
        scaler = StandardScaler()
        X_encoded_scaled = pd.DataFrame(
            scaler.fit_transform(X_encoded),
            columns=ALL_FEATURE_NAMES,
            index=X_encoded.index
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded_scaled, y, test_size=0.25, random_state=42, stratify=y
        )

        best_model = LogisticRegression(max_iter=1000, random_state=42)
        best_model.fit(X_train, y_train)

        imp = np.abs(best_model.coef_[0])
        imp_df = pd.DataFrame({'feature': ALL_FEATURE_NAMES, 'importance': imp})\
            .sort_values('importance', ascending=False)

        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15,
                                                     random_state=42, n_jobs=-1),
            'XGBoost': XGBClassifier(n_estimators=200, max_depth=8,
                                      learning_rate=0.1, random_state=42,
                                      eval_metric='logloss')
        }
        metrics = []
        roc_curves = {}
        for name, m in models.items():
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            y_proba = m.predict_proba(X_test)[:, 1]
            metrics.append({
                'Model': name, 'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1': f1_score(y_test, y_pred),
                'AUC-ROC': roc_auc_score(y_test, y_proba),
            })
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_curves[name] = (fpr, tpr, roc_auc_score(y_test, y_proba))

        metrics_df = pd.DataFrame(metrics).sort_values('AUC-ROC', ascending=False)

        return {
            'model': best_model,
            'le_dict': le_dict, 'scaler': scaler,
            'ALL_FEATURE_NAMES': ALL_FEATURE_NAMES,
            'cat_cols': cat_cols,
            'imp_df': imp_df, 'metrics_df': metrics_df,
            'roc_curves': roc_curves
        }

    artifacts = None

    if model_path.exists() and encoders_path.exists():
        try:
            model = joblib.load(model_path)
            encoders = joblib.load(encoders_path)
            required_keys = ['le_dict', 'scaler', 'ALL_FEATURE_NAMES', 'cat_cols',
                             'imp_df', 'metrics_df', 'roc_curves']
            if all(k in encoders for k in required_keys):
                artifacts = {
                    'model': model, 'le_dict': encoders['le_dict'],
                    'scaler': encoders['scaler'],
                    'ALL_FEATURE_NAMES': encoders['ALL_FEATURE_NAMES'],
                    'cat_cols': encoders['cat_cols'],
                    'imp_df': encoders['imp_df'],
                    'metrics_df': encoders['metrics_df'],
                    'roc_curves': encoders['roc_curves']
                }
        except Exception:
            pass

    if artifacts is None:
        artifacts = train_fresh()
        try:
            joblib.dump(artifacts['model'], model_path)
            joblib.dump(
                {k: v for k, v in artifacts.items() if k != 'model'},
                encoders_path
            )
        except Exception:
            pass

    return artifacts


# ---------------------------
# Load Everything
# ---------------------------

df = generate_churn_data()
artifacts = load_artifacts()

model = artifacts['model']
le_dict = artifacts['le_dict']
scaler = artifacts['scaler']
ALL_FEATURE_NAMES = artifacts['ALL_FEATURE_NAMES']
cat_cols = artifacts['cat_cols']
imp_df = artifacts['imp_df']
metrics_df = artifacts['metrics_df']
roc_curves = artifacts['roc_curves']

# ---------------------------
# Sidebar Filters
# ---------------------------

st.sidebar.title("Data Analytics Internship @ IGRIS LAB")
st.sidebar.markdown("**Week 4 — Customer Churn**")
st.sidebar.markdown("---")

contract_filter = st.sidebar.multiselect(
    "Contract Type", options=df['contract_type'].unique(),
    default=list(df['contract_type'].unique())
)
payment_filter = st.sidebar.multiselect(
    "Payment Method", options=df['payment_method'].unique(),
    default=list(df['payment_method'].unique())
)
internet_filter = st.sidebar.multiselect(
    "Internet Service", options=df['internet_service'].unique(),
    default=list(df['internet_service'].unique())
)

mask = pd.Series(True, index=df.index)
if contract_filter:
    mask &= df['contract_type'].isin(contract_filter)
if payment_filter:
    mask &= df['payment_method'].isin(payment_filter)
if internet_filter:
    mask &= df['internet_service'].isin(internet_filter)

filtered_df = df[mask].copy()

st.sidebar.markdown("---")
st.sidebar.metric("Filtered Customers", filtered_df.shape[0])

# ---------------------------
# Title & KPIs
# ---------------------------

st.title("Customer Churn Analytics")
st.markdown(
    "**Data Analytics Internship @ IGRIS LAB - Week 4**"
)
st.markdown("---")

overall_churn_rate = (df['churn'] == 'Yes').mean() * 100
overall_avg_tenure = df['tenure_months'].mean()

if len(filtered_df) > 0:
    churned = filtered_df[filtered_df['churn'] == 'Yes']
    retained = filtered_df[filtered_df['churn'] == 'No']
    kpi_churn_rate = (len(churned) / len(filtered_df)) * 100
    kpi_avg_tenure_churned = churned['tenure_months'].mean() if len(churned) > 0 else 0
    kpi_revenue_at_risk = churned['monthly_charges'].sum() if len(churned) > 0 else 0
else:
    churned = pd.DataFrame()
    retained = pd.DataFrame()
    kpi_churn_rate = 0
    kpi_avg_tenure_churned = 0
    kpi_revenue_at_risk = 0

col1, col2, col3 = st.columns(3)
with col1:
    delta_churn = kpi_churn_rate - overall_churn_rate if len(filtered_df) > 0 else 0
    st.metric("Churn Rate", f"{kpi_churn_rate:.1f}%",
              delta=f"{delta_churn:.1f}% vs all", delta_color="inverse")
with col2:
    delta_tenure = kpi_avg_tenure_churned - overall_avg_tenure if len(filtered_df) > 0 else 0
    st.metric("Avg Tenure (Churned)", f"{kpi_avg_tenure_churned:.1f} months",
              delta=f"{delta_tenure:.1f} vs overall avg", delta_color="inverse")
with col3:
    st.metric("Monthly Revenue at Risk", f"${kpi_revenue_at_risk:,.0f}")

st.markdown("---")

# ---------------------------
# Tabs
# ---------------------------

tab1, tab2, tab3 = st.tabs(["Churn Patterns", "Model Performance & Retention", "Predict Churn"])

with tab1:
    if len(filtered_df) == 0:
        st.warning("No data matches the current filter selection. Please adjust filters.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            contract_churn = filtered_df.groupby('contract_type')['churn'].apply(
                lambda x: (x == 'Yes').mean() * 100).reset_index()
            contract_churn.columns = ['contract_type', 'churn_rate']
            fig_a = px.bar(contract_churn, x='contract_type', y='churn_rate',
                color='contract_type', title='Churn Rate by Contract Type',
                labels={'churn_rate': 'Churn Rate (%)', 'contract_type': ''},
                text=contract_churn['churn_rate'].round(1),
                color_discrete_sequence=px.colors.qualitative.Set2)
            fig_a.update_traces(texttemplate='%{text}%', textposition='outside')
            fig_a.update_layout(showlegend=False, height=380)
            st.plotly_chart(fig_a, width='stretch')

        with col_b:
            fig_b = go.Figure()
            rt = retained['tenure_months'] if len(retained) > 0 else []
            ct = churned['tenure_months'] if len(churned) > 0 else []
            fig_b.add_trace(go.Histogram(x=rt, name='Retained', marker_color='#2ecc71', opacity=0.7, nbinsx=25))
            fig_b.add_trace(go.Histogram(x=ct, name='Churned', marker_color='#e74c3c', opacity=0.7, nbinsx=25))
            fig_b.update_layout(barmode='overlay', title='Tenure Distribution: Churned vs Retained',
                xaxis_title='Tenure (months)', yaxis_title='Count', height=380, legend=dict(orientation='h', y=1.12))
            st.plotly_chart(fig_b, width='stretch')

        col_c, col_d = st.columns(2)
        with col_c:
            if len(churned) > 0 and len(retained) > 0:
                fig_c = px.box(filtered_df, x='churn', y='monthly_charges',
                    color='churn', color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'},
                    title='Monthly Charges vs Churn', labels={'churn': '', 'monthly_charges': 'Monthly Charges ($)'},
                    points='outliers')
                fig_c.update_layout(showlegend=False, height=380)
            else:
                fig_c = go.Figure()
                fig_c.add_annotation(text="Insufficient data", showarrow=False)
                fig_c.update_layout(height=380)
            st.plotly_chart(fig_c, width='stretch')

        with col_d:
            payment_churn = filtered_df.groupby('payment_method')['churn'].apply(
                lambda x: (x == 'Yes').mean() * 100).reset_index()
            payment_churn.columns = ['payment_method', 'churn_rate']
            fig_f = px.pie(payment_churn, values='churn_rate', names='payment_method',
                title='Churn Rate by Payment Method', hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3)
            fig_f.update_traces(textposition='inside', textinfo='label+percent')
            fig_f.update_layout(height=380)
            st.plotly_chart(fig_f, width='stretch')

        col_e, col_f = st.columns(2)
        with col_e:
            top_features = imp_df.head(10)
            fig_d = px.bar(top_features, x='importance', y='feature',
                orientation='h', title='Top 10 Churn Predictors',
                labels={'importance': 'Importance', 'feature': ''},
                color='importance', color_continuous_scale='Reds')
            fig_d.update_layout(height=420)
            st.plotly_chart(fig_d, width='stretch')

        with col_f:
            fig_e = go.Figure()
            colors = {'Logistic Regression': 'blue', 'Random Forest': 'green', 'XGBoost': 'red'}
            if roc_curves:
                for name, (fpr, tpr, auc) in roc_curves.items():
                    fig_e.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                        name=f'{name} (AUC={auc:.3f})',
                        line=dict(color=colors.get(name, 'gray'), width=2)))
                fig_e.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                    name='Random (AUC=0.5)', line=dict(color='gray', dash='dash', width=1)))
            fig_e.update_layout(title='ROC Curves - All Models',
                xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                height=420, legend=dict(orientation='h', y=1.12, font=dict(size=10)))
            st.plotly_chart(fig_e, width='stretch')

with tab2:
    st.subheader("Model Performance Summary")
    if metrics_df is not None and len(metrics_df) > 0:
        st.dataframe(metrics_df.style.highlight_max(
            axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']
        ).format({'Accuracy': '{:.3f}', 'Precision': '{:.3f}',
                   'Recall': '{:.3f}', 'F1': '{:.3f}', 'AUC-ROC': '{:.3f}'}),
            width='stretch', hide_index=True)
    else:
        st.info("Model metrics not available. Run churn_analysis.py first to train models.")
    st.markdown("---")
    st.subheader("Retention Recommendations")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("### High-Risk Segments")
        st.markdown("""
        | Segment | Churn Risk | Action |
        |---|---|---|
        | Month-to-month contracts | Very High | Offer discounted annual plans |
        | First 6 months of tenure | High | Onboarding calls, check-ins |
        | No tech support | High | Free 30-day tech support trial |
        | Electronic check payers | High | Incentivise auto-pay with 2% discount |
        | Fiber optic + no security | Medium-High | Bundle security at no extra cost |
        """)
    with col_r2:
        st.markdown("### Retention Strategies")
        st.markdown("""
        **1. Contract Conversion** - Offer 10% discount for switching to annual contracts
        **2. Early Intervention** - Trigger outreach if tenure < 6 months
        **3. Service Bundling** - Bundle online security + tech support at reduced rate
        **4. Payment Method Migration** - Target Electronic check users for auto-pay
        **5. Senior Citizen Care** - Dedicated support line for senior customers
        """)
    st.markdown("---")
    st.info("**Expected Impact:** Addressing the top 3 risk factors could reduce churn by 30-40%")

with tab3:
    st.subheader("Predict Customer Churn")
    st.markdown("Enter customer details below to predict churn probability.")

    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        tenure_input = st.slider("Tenure (months)", 1, 72, 12)
        monthly_charges_input = st.number_input("Monthly Charges ($)", 18.0, 150.0, 70.0)
        total_charges_input = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0)
    with col_in2:
        contract_input = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
        payment_input = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])
        internet_input = st.selectbox("Internet Service", ['Fiber optic', 'DSL', 'No'])
    with col_in3:
        tech_input = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
        security_input = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
        num_products_input = st.slider("Number of Products", 1, 4, 2)
    col_in4, col_in5, col_in6 = st.columns(3)
    with col_in4:
        senior_input = st.radio("Senior Citizen", [0, 1], horizontal=True)
    with col_in5:
        partner_input = st.radio("Has Partner", ['Yes', 'No'], horizontal=True)
    with col_in6:
        dependents_input = st.radio("Has Dependents", ['Yes', 'No'], horizontal=True)

    input_data = {
        'tenure_months': tenure_input,
        'monthly_charges': monthly_charges_input,
        'total_charges': total_charges_input,
        'contract_type': contract_input,
        'payment_method': payment_input,
        'internet_service': internet_input,
        'tech_support': tech_input,
        'online_security': security_input,
        'num_products': num_products_input,
        'senior_citizen': senior_input,
        'has_partner': partner_input,
        'has_dependents': dependents_input
    }

    if st.button("Predict Churn", type="primary", use_container_width=True):
        try:
            # Build DataFrame from user input
            input_df = pd.DataFrame([input_data])

            # STEP 1: Label encode categorical columns using saved encoders
            for col in cat_cols:
                le = le_dict[col]
                input_df[col] = le.transform(input_df[col].astype(str))

            # STEP 2: Ensure column ORDER matches training order exactly
            input_df = input_df[ALL_FEATURE_NAMES]

            # STEP 3: Scale ALL columns (scaler was fit on ALL columns)
            input_scaled = pd.DataFrame(
                scaler.transform(input_df),
                columns=ALL_FEATURE_NAMES,
                index=input_df.index
            )

            # STEP 4: Predict
            proba = model.predict_proba(input_scaled)[0, 1]

            st.markdown("---")
            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.metric("Churn Probability", f"{proba:.1%}")

            with col_res2:
                risk_label = "Low" if proba < 0.3 else ("Medium" if proba < 0.6 else "High")
                risk_color = "#2ecc71" if proba < 0.3 else ("#f39c12" if proba < 0.6 else "#e74c3c")
                st.markdown(
                    f"<div style='text-align: center;'>"
                    f"<p style='color: gray; font-size: 0.9rem;'>Risk Level</p>"
                    f"<span style='background: {risk_color}; color: white; padding: 0.5rem 1.5rem; "
                    f"border-radius: 20px; font-weight: bold; font-size: 1.2rem;'>{risk_label}</span>"
                    f"</div>", unsafe_allow_html=True)

            with col_res3:
                action = "Immediate retention intervention needed" if proba >= 0.6 else \
                         "Monitor and offer incentives" if proba >= 0.3 else \
                         "Customer is likely to stay"
                st.markdown(
                    f"<div style='text-align: center;'>"
                    f"<p style='color: gray; font-size: 0.9rem;'>Recommendation</p>"
                    f"<p style='font-weight: bold; font-size: 0.95rem; padding: 0.5rem;'>{action}</p>"
                    f"</div>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### Key Risk Factors for This Customer")
            risk_factors = []
            if contract_input == 'Month-to-month':
                risk_factors.append("Month-to-month contract - highest churn risk segment")
            if tenure_input <= 12:
                risk_factors.append(f"Short tenure ({tenure_input} months) - high early churn risk")
            if tech_input == 'No':
                risk_factors.append("No tech support - customers with support are more loyal")
            if security_input == 'No':
                risk_factors.append("No online security - bundle security to increase retention")
            if payment_input == 'Electronic check':
                risk_factors.append("Electronic check payment - migrate to auto-pay")
            if senior_input == 1:
                risk_factors.append("Senior citizen - consider dedicated support line")
            if monthly_charges_input > 100:
                risk_factors.append(f"High monthly charges (${monthly_charges_input:.0f}) - above average")

            if risk_factors:
                for rf in risk_factors:
                    st.markdown(f"- {rf}")
            else:
                st.markdown("No major risk factors detected for this customer profile.")

        except Exception as e:
            st.error(f"Prediction error: {e}")

st.markdown("---")
st.caption("Data Analytics Internship @ IGRIS LAB - Week 4 | Dataset: Synthetic Telco Customer Churn (3,000 records)")