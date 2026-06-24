"""
IGRIS LAB — Intern Profile Dashboard
======================================
Data Analytics Internship — Week 1 (Onboarding)

Displays the intern profile loaded from profile.json
as a clean, professional Streamlit page.
"""

import json
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Vaibhav S — IGRIS LAB Intern",
    page_icon="🧑‍💻",
    layout="centered"
)

# ---------------------------
# Load profile
# ---------------------------

@st.cache_data
def load_profile():
    path = Path(__file__).parent / "profile.json"
    with open(path) as f:
        return json.load(f)


profile = load_profile()

# ---------------------------
# Header
# ---------------------------

st.markdown(
    f"""
    <div style='text-align: center; padding: 2rem 1rem 1rem 1rem;'>
        <div style='
            font-size: 4rem;
            width: 100px;
            height: 100px;
            line-height: 100px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin: 0 auto 1rem auto;
            font-weight: bold;
        '>{profile['name'][0]}</div>
        <h1 style='margin: 0;'>{profile['name']}</h1>
        <h3 style='color: #555; margin-top: 0.3rem;'>{profile['role']}</h3>
        <p style='color: #888; font-size: 1.1rem;'>
            <strong>Intern ID:</strong> {profile['internID']}
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------------------
# Skills
# ---------------------------

st.subheader("🛠️ Skills")

cols = st.columns(3)
for i, skill in enumerate(profile['skills']):
    with cols[i % 3]:
        st.markdown(f"- {skill}")

# ---------------------------
# Tools
# ---------------------------

st.subheader("🔧 Tools")

cols2 = st.columns(3)
for i, tool in enumerate(profile['tools']):
    with cols2[i % 3]:
        st.markdown(f"- {tool}")

st.markdown("---")

# ---------------------------
# Weekly navigation
# ---------------------------

st.subheader("📂 Weekly Projects")

st.markdown("""
| Week | Project | Status |
|------|---------|--------|
| 1 | Onboarding & Setup | ✅ Complete |
| 2 | Student Performance Analysis | ✅ Complete |
| 3 | E-Commerce Sales Dashboard | ⏳ Upcoming |
| 4 | Customer Churn Prediction | ⏳ Upcoming |
""")

st.markdown("---")

# ---------------------------
# Footer
# ---------------------------

st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 0.9rem;'>
        <strong>📅 Internship:</strong> Data Analytics Intern @ IGRIS LAB &nbsp;|&nbsp; Jun 2026 – Aug 2026
        <br>
        <a href='https://github.com/VaibhavS45'>GitHub</a> &nbsp;·&nbsp;
        <a href='https://linkedin.com/in/vaibhavs45'>LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True
)