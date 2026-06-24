"""
Student Performance Analysis
============================
Data Analytics Internship @ IGRIS LAB — Week 2

Generates a realistic synthetic student performance dataset,
performs EDA, creates visualisations, and prints key insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# ---------------------------
# 1. Data Generation
# ---------------------------

def generate_student_data(n=1000, seed=42) -> pd.DataFrame:
    """Generate a realistic synthetic student performance dataset."""
    np.random.seed(seed)

    # Age distribution (15–22, mostly 16–18)
    age_probs = [0.02, 0.10, 0.25, 0.30, 0.20, 0.08, 0.03, 0.02]
    ages = np.random.choice(range(15, 23), size=n, p=age_probs)

    # Gender
    genders = np.random.choice(['Male', 'Female'], size=n, p=[0.48, 0.52])

    # Parental education
    parent_edu = np.random.choice(
        ['High School', 'Associate', 'Bachelor', 'Master', 'PhD'],
        size=n, p=[0.30, 0.25, 0.25, 0.15, 0.05]
    )

    # Internet access
    internet = np.random.choice(['Yes', 'No'], size=n, p=[0.85, 0.15])

    # Extracurricular activities
    extra = np.random.choice(['Yes', 'No'], size=n, p=[0.55, 0.45])

    # Study hours per week (correlated with final score)
    study_hours_base = np.random.normal(15, 6, n).clip(1, 40)

    # Attendance rate (correlated with final score)
    attendance_base = np.random.normal(80, 12, n).clip(30, 100)

    # Previous grades (scale 0–100)
    prev_grades_base = np.random.normal(65, 15, n).clip(20, 100)

    # Build final_score with realistic relationships
    noise = np.random.normal(0, 8, n)
    score_from_study = study_hours_base * 1.2
    score_from_attendance = attendance_base * 0.25
    score_from_prev = prev_grades_base * 0.20
    score_from_gender = np.where(genders == 'Female', 3, 0)

    # Internet bonus
    internet_bonus = np.where(internet == 'Yes', 4, 0)

    # Parental education bonus
    edu_bonus_map = {
        'High School': 0, 'Associate': 3, 'Bachelor': 6,
        'Master': 9, 'PhD': 12
    }
    edu_bonus = np.array([edu_bonus_map[e] for e in parent_edu])

    final_score = (
        score_from_study + score_from_attendance + score_from_prev
        + score_from_gender + internet_bonus + edu_bonus + noise
    )
    final_score = final_score.clip(0, 100).round(1)

    # Pass/Fail (threshold: 40)
    pass_fail = np.where(final_score >= 40, 'Pass', 'Fail')

    df = pd.DataFrame({
        'student_id': [f'STU{str(i).zfill(4)}' for i in range(1, n + 1)],
        'age': ages,
        'gender': genders,
        'study_hours_per_week': study_hours_base.round(1),
        'attendance_rate': attendance_base.round(1),
        'previous_grades': prev_grades_base.round(1),
        'parental_education': parent_edu,
        'internet_access': internet,
        'extracurricular_activities': extra,
        'final_score': final_score,
        'pass_fail': pass_fail
    })

    # Add deliberate missing values (approx 1.5%)
    for col in ['study_hours_per_week', 'attendance_rate', 'previous_grades']:
        nan_idx = np.random.choice(df.index, size=int(n * 0.005), replace=False)
        df.loc[nan_idx, col] = np.nan

    return df


# ---------------------------
# 2. Data Cleaning
# ---------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle nulls and fix dtypes."""
    print("\n" + "=" * 60)
    print("DATA CLEANING")
    print("=" * 60)

    print(f"\nNull counts before cleaning:")
    print(df.isnull().sum().to_string())

    # Fill numeric nulls with median
    for col in ['study_hours_per_week', 'attendance_rate', 'previous_grades']:
        df[col] = df[col].fillna(df[col].median())

    # Ensure correct dtypes
    df['student_id'] = df['student_id'].astype(str)
    df['pass_fail'] = df['pass_fail'].astype(str)
    df['gender'] = df['gender'].astype(str)
    df['internet_access'] = df['internet_access'].astype(str)
    df['extracurricular_activities'] = df['extracurricular_activities'].astype(str)
    df['parental_education'] = df['parental_education'].astype(str)
    df['age'] = df['age'].astype(int)

    print(f"\nNull counts after cleaning:")
    print(df.isnull().sum().to_string())
    print(f"\nDataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    return df


# ---------------------------
# 3. Exploratory Data Analysis
# ---------------------------

def perform_eda(df: pd.DataFrame):
    """Print summary statistics and categorical value counts."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print("\n--- Descriptive Statistics (Numeric) ---")
    print(df.describe().to_string())

    print("\n--- Value Counts: Gender ---")
    print(df['gender'].value_counts().to_string())
    print(f"  Proportions:\n{df['gender'].value_counts(normalize=True).to_string()}")

    print("\n--- Value Counts: Pass/Fail ---")
    print(df['pass_fail'].value_counts().to_string())
    print(f"  Proportions:\n{df['pass_fail'].value_counts(normalize=True).to_string()}")

    print("\n--- Value Counts: Parental Education ---")
    print(df['parental_education'].value_counts().to_string())

    print("\n--- Value Counts: Internet Access ---")
    print(df['internet_access'].value_counts().to_string())

    print("\n--- Value Counts: Extracurricular Activities ---")
    print(df['extracurricular_activities'].value_counts().to_string())

    print(f"\n--- Correlation Matrix ---")
    corr_cols = ['study_hours_per_week', 'attendance_rate', 'previous_grades',
                  'final_score', 'age']
    corr_matrix = df[corr_cols].corr()
    print(corr_matrix.to_string())
    print()

    return corr_matrix


# ---------------------------
# 4. Visualisations (Static — Matplotlib / Seaborn)
# ---------------------------

def create_visualisations(df: pd.DataFrame, output_dir: str = "assets"):
    """
    Generate 6 static plots and save them to output_dir.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Style
    sns.set_style("whitegrid")
    plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 150})

    print("\n" + "=" * 60)
    print("GENERATING VISUALISATIONS")
    print("=" * 60)

    # (a) Score distribution histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['final_score'], bins=30, kde=True, color='steelblue', ax=ax)
    ax.axvline(df['final_score'].mean(), color='red', linestyle='--',
               label=f"Mean: {df['final_score'].mean():.1f}")
    ax.set_title('Distribution of Final Scores', fontsize=14, fontweight='bold')
    ax.set_xlabel('Final Score')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{output_dir}/a_score_distribution.png")
    plt.close(fig)
    print("  [OK] a_score_distribution.png")

    # (b) Study hours vs final score scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x='study_hours_per_week', y='final_score',
                    hue='pass_fail', alpha=0.6, ax=ax)
    corr = df['study_hours_per_week'].corr(df['final_score'])
    ax.set_title(f'Study Hours vs Final Score  (r = {corr:.3f})',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Study Hours per Week')
    ax.set_ylabel('Final Score')
    plt.tight_layout()
    fig.savefig(f"{output_dir}/b_study_vs_score.png")
    plt.close(fig)
    print("  [OK] b_study_vs_score.png")

    # (c) Attendance rate vs final score scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x='attendance_rate', y='final_score',
                    hue='pass_fail', alpha=0.6, ax=ax)
    corr = df['attendance_rate'].corr(df['final_score'])
    ax.set_title(f'Attendance Rate vs Final Score  (r = {corr:.3f})',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Attendance Rate (%)')
    ax.set_ylabel('Final Score')
    plt.tight_layout()
    fig.savefig(f"{output_dir}/c_attendance_vs_score.png")
    plt.close(fig)
    print("  [OK] c_attendance_vs_score.png")

    # (d) Pass/Fail count by gender (grouped bar)
    fig, ax = plt.subplots(figsize=(8, 5))
    crosstab = pd.crosstab(df['gender'], df['pass_fail'])
    crosstab.plot(kind='bar', ax=ax, color=['#e74c3c', '#2ecc71'], edgecolor='black')
    ax.set_title('Pass/Fail Count by Gender', fontsize=14, fontweight='bold')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Count')
    ax.legend(title='Result')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    fig.savefig(f"{output_dir}/d_pass_fail_by_gender.png")
    plt.close(fig)
    print("  [OK] d_pass_fail_by_gender.png")

    # (e) Correlation heatmap of numeric features
    corr_cols = ['study_hours_per_week', 'attendance_rate', 'previous_grades',
                  'final_score', 'age']
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[corr_cols].corr(), annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, square=True, linewidths=1, ax=ax)
    ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f"{output_dir}/e_correlation_heatmap.png")
    plt.close(fig)
    print("  [OK] e_correlation_heatmap.png")

    # (f) Parental education impact on scores (boxplot)
    fig, ax = plt.subplots(figsize=(10, 5))
    order = ['High School', 'Associate', 'Bachelor', 'Master', 'PhD']
    sns.boxplot(data=df, x='parental_education', y='final_score',
                hue='parental_education', order=order, palette='Blues',
                legend=False, ax=ax)
    ax.set_title('Impact of Parental Education on Final Scores',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Parental Education Level')
    ax.set_ylabel('Final Score')
    plt.tight_layout()
    fig.savefig(f"{output_dir}/f_parental_education_boxplot.png")
    plt.close(fig)
    print("  [OK] f_parental_education_boxplot.png")

    print(f"\nAll plots saved to '{output_dir}/'")
    return output_dir


# ---------------------------
# 5. Key Insights
# ---------------------------

def print_insights(df: pd.DataFrame, corr_matrix: pd.DataFrame):
    """Print actionable insights based on the analysis."""
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)

    overall_pass_rate = (df['pass_fail'] == 'Pass').mean() * 100

    # Top 3 correlations with final_score
    corr_target = corr_matrix['final_score'].drop('final_score').abs().sort_values(ascending=False)
    top_factors = corr_target.head(3)

    avg_score_pass = df[df['pass_fail'] == 'Pass']['final_score'].mean()
    avg_score_fail = df[df['pass_fail'] == 'Fail']['final_score'].mean()

    male_pass_rate = (df[df['gender'] == 'Male']['pass_fail'] == 'Pass').mean() * 100
    female_pass_rate = (df[df['gender'] == 'Female']['pass_fail'] == 'Pass').mean() * 100

    internet_yes = df[df['internet_access'] == 'Yes']['final_score'].mean()
    internet_no = df[df['internet_access'] == 'No']['final_score'].mean()

    extracurricular_yes = df[df['extracurricular_activities'] == 'Yes']['final_score'].mean()
    extracurricular_no = df[df['extracurricular_activities'] == 'No']['final_score'].mean()

    insights = [
        f"1. Overall Pass Rate: {overall_pass_rate:.1f}% ({df.shape[0]} students)",
        f"2. Top 3 factors correlated with final score:",
        f"   - {top_factors.index[0]}: r = {corr_target.iloc[0]:.3f}",
        f"   - {top_factors.index[1]}: r = {corr_target.iloc[1]:.3f}",
        f"   - {top_factors.index[2]}: r = {corr_target.iloc[2]:.3f}",
        f"3. Average score — Pass: {avg_score_pass:.1f}  |  Fail: {avg_score_fail:.1f}",
        f"4. Pass rate by gender — Male: {male_pass_rate:.1f}%  |  Female: {female_pass_rate:.1f}%",
        f"5. Students with internet access score {internet_yes:.1f} vs {internet_no:.1f} without (avg)",
        f"6. Extracurricular participants avg: {extracurricular_yes:.1f} vs {extracurricular_no:.1f} non-participants",
    ]

    for ins in insights:
        print(f"  {ins}")

    print("=" * 60)
    print()


# ---------------------------
# 6. Main
# ---------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("STUDENT PERFORMANCE ANALYSIS")
    print("Data Analytics Internship @ IGRIS LAB — Week 2")
    print("=" * 60)

    # Generate synthetic data
    print("\nGenerating synthetic student dataset (n=1000)...")
    df = generate_student_data(n=1000)
    print(f"  Generated {df.shape[0]} rows x {df.shape[1]} columns")

    # Save raw CSV
    df.to_csv("student_data_raw.csv", index=False)
    print(f"  Saved raw data to student_data_raw.csv")

    # Clean
    df_clean = clean_data(df)

    # Save clean CSV
    df_clean.to_csv("student_data_clean.csv", index=False)
    print(f"  Saved clean data to student_data_clean.csv")

    # EDA
    corr_matrix = perform_eda(df_clean)

    # Visualisations
    create_visualisations(df_clean, output_dir="assets")

    # Insights
    print_insights(df_clean, corr_matrix)

    print("Analysis complete.")