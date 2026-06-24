# Week 1 — Onboarding & Setup

## Overview

This week focused on setting up the development environment, understanding the internship workflow, and preparing the portfolio repository structure.

## Objectives

- [x] Set up Python virtual environment and install core data science libraries
- [x] Configure Git and connect to GitHub
- [x] Create the repository folder structure for all 4 weekly projects
- [x] Write a professional `README.md` for the main portfolio
- [x] Create a personal `profile.json` with intern details
- [x] Add `.gitignore` and `requirements.txt` for reproducibility

## Tools Configured

| Tool        | Version / Purpose                     |
|-------------|---------------------------------------|
| Python      | 3.11 — core programming language      |
| Git         | Version control                       |
| VS Code     | Primary code editor                   |
| GitHub      | Remote repository hosting             |

## Key Learnings

- Best practices for structuring a data analytics project repository
- Writing clean, markdown-formatted documentation
- Using `.gitignore` to avoid committing unnecessary files (e.g., `__pycache__/`, `.env`, datasets)
- Managing Python dependencies with `requirements.txt`

## Files in this folder

| File             | Description                                |
|------------------|--------------------------------------------|
| `README.md`      | This file — onboarding notes               |
| `profile.json`   | JSON profile with intern details           |
| `app.py`         | Streamlit profile dashboard (deployable)   |

## 🚀 Live Deployment (Streamlit)

This week includes a Streamlit profile page (`app.py`) that displays your intern profile from `profile.json`.

### Run locally

```bash
cd week1-onboarding
streamlit run app.py
```

### Deploy to Streamlit Community Cloud

1. Push the repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"**
4. Set:
   - **Repository:** `VaibhavS45/igrislab-data-analytics-internship`
   - **Branch:** `main`
   - **Main file path:** `week1-onboarding/app.py`
5. Click **Deploy**

Your profile dashboard will be live at:  
`https://igrislab-data-analytics-internship.streamlit.app/`

---

*Week 1 completed — repository ready for upcoming data analysis projects.*