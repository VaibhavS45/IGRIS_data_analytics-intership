# Week 2 — Student Performance Analysis

## Project Goal

Analyse factors influencing student academic performance using a synthetic dataset of 1,000 students. Perform exploratory data analysis (EDA), generate visualisations, and build an interactive Streamlit dashboard to explore key relationships.

## Dataset

A realistic synthetic dataset was generated with the following features:

| Column                        | Description                                  |
|-------------------------------|----------------------------------------------|
| `student_id`                  | Unique identifier for each student           |
| `age`                         | Age of student (15–22)                       |
| `gender`                      | Male / Female                                |
| `study_hours_per_week`        | Hours spent studying per week                |
| `attendance_rate`             | Class attendance percentage                  |
| `previous_grades`             | Average grade from previous assessments      |
| `parental_education`          | Highest education level of parents           |
| `internet_access`             | Whether student has internet at home         |
| `extracurricular_activities`  | Whether student participates in activities   |
| `final_score`                 | Final exam score (0–100)                     |
| `pass_fail`                   | Pass (>= 40) or Fail (< 40)                  |

## Tools & Libraries

- **Python 3.11** — core programming language
- **Pandas, NumPy** — data manipulation & numerical computation
- **Matplotlib, Seaborn** — static visualisations
- **Plotly** — interactive visualisations for the dashboard
- **Streamlit** — interactive web dashboard framework
- **Scikit-learn** — (used implicitly for scaling / metrics)

## Key Findings

1. **Study hours** and **attendance rate** are the strongest predictors of final scores.
2. **Parental education** shows a positive gradient — students with higher-educated parents tend to score higher.
3. **Internet access** provides a modest performance boost.
4. **Gender differences** are minimal, with female students slightly outperforming male students on average.
5. The **pass rate** for this dataset is approximately 87% (varies slightly with random seed).

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the analysis script (generates data, EDA, static plots)

```bash
cd week2-student-performance
python student_analysis.py
```

This will:
- Generate `student_data_raw.csv` and `student_data_clean.csv`
- Print EDA summaries and key insights to the console
- Save 6 static plots to the `assets/` folder

### 3. Launch the interactive dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser with:
- Sidebar filters for gender, parental education, and pass/fail status
- KPI metric cards (average score, pass rate, study hours, attendance)
- 6 interactive Plotly charts
- A ranked table of top factors affecting performance

### Streamlit Community Cloud Deployment

To deploy on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push this folder to a GitHub repository
2. Log into [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** and select the repository
4. Set:
   - **Repository:** `your-username/igrislab-data-analytics-internship`
   - **Branch:** `main`
   - **Main file path:** `week2-student-performance/app.py`
5. Click **Deploy**

---

*Week 2 completed — Student Performance Analysis.*