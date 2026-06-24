"""
Student Performance Analytics — Streamlit Dashboard
=====================================================
Data Analytics Internship @ IGRIS LAB — Week 2

Interactive dashboard with filters, KPI cards, 6 charts, and
top factor analysis using Plotly.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Must be the first Streamlit command
st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="📊",
    layout="wide"
)

# ---------------------------
# 1. Generate / Load Data
# ---------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    """Generate synthetic student data (cached)."""
    np.random.seed(42)
    n = 1000

    age_probs = [0.02, 0.10, 0.25, 0.30, 0.20, 0.08, 0.03, 0.02]
    ages = np.random.choice(range(15, 23), size=n, p=age_probs)
    genders = np.random.choice(['Male', 'Female'], size=n, p=[0.48, 0.52])
    parent_edu = np.random.choice(
        ['High School', 'Associate', 'Bachelor', 'Master', 'PhD'],
        size=n, p=[0.30, 0.25, 0.25, 0.15, 0.05]
    )
    internet = np.random.choice(['Yes', 'No'], size=n, p=[0.85, 0.15])
    extra = np.random.choice(['Yes', 'No'], size=n, p=[0.55, 0.45])

    study_hours = np.random.normal(15, 6, n).clip(1, 40)
    attendance = np.random.normal(80, 12, n).clip(30, 100)
    prev_grades = np.random.normal(65, 15, n).clip(20, 100)

    noise = np.random.normal(0, 8, n)
    score = (
        study_hours * 1.2 + attendance * 0.25 + prev_grades * 0.20
        + np.where(genders == 'Female', 3, 0)
        + np.where(internet == 'Yes', 4, 0)
        + np.array([{'High School': 0, 'Associate': 3, 'Bachelor': 6,
                     'Master': 9, 'PhD': 12}[e] for e in parent_edu])
        + noise
    ).clip(0, 100).round(1)

    pass_fail = np.where(score >= 40, 'Pass', 'Fail')

    df = pd.DataFrame({
        'student_id': [f'STU{str(i).zfill(4)}' for i in range(1, n + 1)],
        'age': ages,
        'gender': genders,
        'study_hours_per_week': study_hours.round(1),
        'attendance_rate': attendance.round(1),
        'previous_grades': prev_grades.round(1),
        'parental_education': parent_edu,
        'internet_access': internet,
        'extracurricular_activities': extra,
        'final_score': score,
        'pass_fail': pass_fail
    })

    # Fill small % of nulls with median for realism
    for col in ['study_hours_per_week', 'attendance_rate', 'previous_grades']:
        nan_idx = np.random.choice(df.index, size=int(n * 0.005), replace=False)
        df.loc[nan_idx, col] = np.nan
        df[col] = df[col].fillna(df[col].median())

    return df


df = load_data()

# ---------------------------
# 2. Sidebar Filters
# ---------------------------

st.sidebar.title("🔍 Filters")

gender_filter = st.sidebar.multiselect(
    "Gender",
    options=['Male', 'Female'],
    default=['Male', 'Female']
)

parent_edu_filter = st.sidebar.multiselect(
    "Parental Education",
    options=['High School', 'Associate', 'Bachelor', 'Master', 'PhD'],
    default=['High School', 'Associate', 'Bachelor', 'Master', 'PhD']
)

pass_fail_filter = st.sidebar.multiselect(
    "Pass / Fail",
    options=['Pass', 'Fail'],
    default=['Pass', 'Fail']
)

# Apply filters
filtered_df = df[
    (df['gender'].isin(gender_filter)) &
    (df['parental_education'].isin(parent_edu_filter)) &
    (df['pass_fail'].isin(pass_fail_filter))
]

# Sidebar summary
st.sidebar.markdown("---")
st.sidebar.metric("Filtered Students", filtered_df.shape[0])

# ---------------------------
# 3. Title & KPIs
# ---------------------------

st.title("📊 Student Performance Analytics")
st.markdown(
    "**Data Analytics Internship @ IGRIS LAB — Week 2**  |  "
    "Interactive dashboard exploring factors that influence student academic performance."
)
st.markdown("---")

# KPI calculations
avg_score = filtered_df['final_score'].mean()
pass_rate = (filtered_df['pass_fail'] == 'Pass').mean() * 100
avg_study = filtered_df['study_hours_per_week'].mean()
avg_attendance = filtered_df['attendance_rate'].mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📈 Average Score",
        value=f"{avg_score:.1f}",
        delta=f"{avg_score - df['final_score'].mean():.1f} vs all"
    )
with col2:
    st.metric(
        label="✅ Pass Rate",
        value=f"{pass_rate:.1f}%",
        delta=f"{pass_rate - (df['pass_fail'] == 'Pass').mean() * 100:.1f}% vs all"
    )
with col3:
    st.metric(
        label="⏱ Avg Study Hours/Week",
        value=f"{avg_study:.1f}h",
        delta=f"{avg_study - df['study_hours_per_week'].mean():.1f}h vs all"
    )
with col4:
    st.metric(
        label="📋 Avg Attendance",
        value=f"{avg_attendance:.1f}%",
        delta=f"{avg_attendance - df['attendance_rate'].mean():.1f}% vs all"
    )

st.markdown("---")

# ---------------------------
# 4. Charts (2x3 grid)
# ---------------------------

# Colour palette
colour_map = {'Pass': '#2ecc71', 'Fail': '#e74c3c'}
edu_order = ['High School', 'Associate', 'Bachelor', 'Master', 'PhD']

st.subheader("📈 Visualisations")

with st.container():
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        # (a) Score distribution histogram
        fig_a = px.histogram(
            filtered_df, x='final_score', nbins=30,
            color='pass_fail', color_discrete_map=colour_map,
            marginal='box',
            title='Distribution of Final Scores',
            labels={'final_score': 'Final Score', 'count': 'Frequency'}
        )
        fig_a.add_vline(
            x=filtered_df['final_score'].mean(),
            line_dash='dash', line_color='red',
            annotation_text=f"Mean: {filtered_df['final_score'].mean():.1f}"
        )
        fig_a.update_layout(height=350, legend=dict(orientation='h', y=1.12))
        st.plotly_chart(fig_a, use_container_width=True)

    with row1_col2:
        # (b) Study hours vs final score scatter
        fig_b = px.scatter(
            filtered_df, x='study_hours_per_week', y='final_score',
            color='pass_fail', color_discrete_map=colour_map,
            opacity=0.6, hover_data=['gender', 'age'],
            title='Study Hours vs Final Score',
            labels={'study_hours_per_week': 'Study Hours/Week', 'final_score': 'Final Score'}
        )
        corr_b = filtered_df['study_hours_per_week'].corr(filtered_df['final_score'])
        fig_b.add_annotation(
            xref='paper', yref='paper', x=0.02, y=0.98,
            text=f"r = {corr_b:.3f}", showarrow=False,
            font=dict(size=14, color='white'),
            bgcolor='rgba(0,0,0,0.5)', bordercolor='black', borderwidth=1
        )
        fig_b.update_layout(height=350)
        st.plotly_chart(fig_b, use_container_width=True)

with st.container():
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        # (c) Attendance rate vs final score scatter
        fig_c = px.scatter(
            filtered_df, x='attendance_rate', y='final_score',
            color='pass_fail', color_discrete_map=colour_map,
            opacity=0.6, hover_data=['gender', 'study_hours_per_week'],
            title='Attendance Rate vs Final Score',
            labels={'attendance_rate': 'Attendance Rate (%)', 'final_score': 'Final Score'}
        )
        corr_c = filtered_df['attendance_rate'].corr(filtered_df['final_score'])
        fig_c.add_annotation(
            xref='paper', yref='paper', x=0.02, y=0.98,
            text=f"r = {corr_c:.3f}", showarrow=False,
            font=dict(size=14, color='white'),
            bgcolor='rgba(0,0,0,0.5)', bordercolor='black', borderwidth=1
        )
        fig_c.update_layout(height=350)
        st.plotly_chart(fig_c, use_container_width=True)

    with row2_col2:
        # (d) Pass/Fail count by gender (grouped bar)
        crosstab = pd.crosstab(filtered_df['gender'], filtered_df['pass_fail'])
        fig_d = go.Figure()
        for result in ['Fail', 'Pass']:
            fig_d.add_trace(go.Bar(
                name=result,
                x=crosstab.index,
                y=crosstab[result],
                marker_color=colour_map[result],
                text=crosstab[result],
                textposition='auto'
            ))
        fig_d.update_layout(
            barmode='group',
            title='Pass/Fail Count by Gender',
            xaxis_title='Gender',
            yaxis_title='Count',
            height=350,
            legend=dict(orientation='h', y=1.12)
        )
        st.plotly_chart(fig_d, use_container_width=True)

with st.container():
    row3_col1, row3_col2 = st.columns(2)

    with row3_col1:
        # (e) Correlation heatmap
        corr_cols = ['study_hours_per_week', 'attendance_rate', 'previous_grades',
                     'final_score', 'age']
        corr_matrix = filtered_df[corr_cols].corr()

        fig_e = px.imshow(
            corr_matrix,
            text_auto='.3f',
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title='Correlation Heatmap of Numeric Features',
            aspect='auto'
        )
        fig_e.update_layout(height=400)
        st.plotly_chart(fig_e, use_container_width=True)

    with row3_col2:
        # (f) Parental education impact (boxplot)
        fig_f = px.box(
            filtered_df, x='parental_education', y='final_score',
            color='parental_education',
            category_orders={'parental_education': edu_order},
            title='Impact of Parental Education on Final Scores',
            labels={'parental_education': 'Parental Education', 'final_score': 'Final Score'},
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig_f.update_layout(
            height=400,
            showlegend=False,
            xaxis={'categoryorder':'array', 'categoryarray': edu_order}
        )
        st.plotly_chart(fig_f, use_container_width=True)

st.markdown("---")

# ---------------------------
# 5. Top Factors Affecting Performance
# ---------------------------

st.subheader("🔬 Top Factors Affecting Performance")

# Recalculate correlations on the filtered data
corr_cols = ['study_hours_per_week', 'attendance_rate', 'previous_grades',
             'final_score', 'age']
corr_matrix = filtered_df[corr_cols].corr()
corr_target = corr_matrix['final_score'].drop('final_score').abs().sort_values(ascending=False)

# Build factor importance DataFrame
factor_df = pd.DataFrame({
    'Factor': [col.replace('_', ' ').title() for col in corr_target.index],
    'Correlation (r)': corr_target.values,
    'Direction': ['Positive' if corr_matrix.loc[col, 'final_score'] > 0
                  else 'Negative' for col in corr_target.index]
})

st.markdown(
    "The table below ranks features by their absolute correlation with `final_score`. "
    "A higher absolute value indicates a stronger linear relationship."
)

col_table, col_note = st.columns([2, 1])

with col_table:
    st.dataframe(
        factor_df,
        column_config={
            'Factor': st.column_config.TextColumn('Factor'),
            'Correlation (r)': st.column_config.NumberColumn(
                'Correlation (r)', format='%.4f'
            ),
            'Direction': st.column_config.TextColumn('Direction')
        },
        hide_index=True,
        use_container_width=True
    )

with col_note:
    top_factor = factor_df.iloc[0]
    st.info(
        f"**Top Factor:** {top_factor['Factor']}\n\n"
        f"Correlation: **{top_factor['Correlation (r)']:.4f}**\n\n"
        f"This means students with higher {top_factor['Factor'].lower()} "
        f"tend to achieve {'higher' if top_factor['Direction'] == 'Positive' else 'lower'} "
        f"final scores."
    )

st.markdown("---")
st.caption(
    "Data Analytics Internship @ IGRIS LAB — Week 2  |  "
    "Dataset: Synthetic Student Performance (1000 records)  |  "
    "Built with Streamlit & Plotly"
)