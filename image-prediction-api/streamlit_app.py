"""
Streamlit frontend for the Fruit Freshness Classification API.
Connects to FastAPI backend at http://127.0.0.1:8000
"""

import streamlit as st
import requests
import pandas as pd
from PIL import Image
import io

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fruit Freshness Detector",
    page_icon="🍎",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🍎 Fruit Freshness Detector")
st.subheader("Upload a fruit image to check if it's fresh or rotten")
st.divider()

# ---------------------------------------------------------------------------
# API Base URL — configurable for cloud deployment
# ---------------------------------------------------------------------------
# Local dev: http://127.0.0.1:8000
# Cloud: Your deployed FastAPI URL (set via Streamlit secrets or env var)
API_BASE_URL = "http://127.0.0.1:8000"

# For Streamlit Cloud deployment, use secrets.toml:
# [api]
# base_url = "https://your-api.onrender.com"
try:
    if "api" in st.secrets and "base_url" in st.secrets["api"]:
        API_BASE_URL = st.secrets["api"]["base_url"]
except Exception:
    pass  # No secrets file found — using default local URL

try:
    resp = requests.get(f"{API_BASE_URL}/", timeout=3)
    if resp.status_code == 200:
        st.success(f"✅ API is online")
    else:
        st.error("❌ API returned an unexpected status")
        st.stop()
except requests.exceptions.ConnectionError:
    st.error("❌ API is offline — run: uvicorn app:app --reload")
    st.stop()
except requests.exceptions.Timeout:
    st.error("❌ API timed out — check if the server is running")
    st.stop()

# ---------------------------------------------------------------------------
# Image Upload
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose a fruit image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("👆 Upload an image to get started")
    st.stop()

# ---------------------------------------------------------------------------
# Display uploaded image
# ---------------------------------------------------------------------------
st.image(
    Image.open(io.BytesIO(uploaded_file.getvalue())),
    use_container_width=True,
    caption="Uploaded image",
)

# ---------------------------------------------------------------------------
# Predict button
# ---------------------------------------------------------------------------
if st.button("🔍 Predict"):
    with st.spinner("Analyzing image..."):
        try:
            # Send to API
            files = {
                "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }
            response = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=30)

        except requests.exceptions.ConnectionError:
            st.error("Cannot reach API. Is it running?")
            st.stop()
        except requests.exceptions.Timeout:
            st.error("Request timed out. The model may be too slow.")
            st.stop()
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.stop()

        # Handle non-200 responses
        if response.status_code == 503:
            st.warning(
                "Model not trained yet. Run `python train.py` first."
            )
            st.stop()
        elif response.status_code != 200:
            st.error(f"API Error ({response.status_code}): {response.text}")
            st.stop()

        # Parse response
        data = response.json()
        prediction = data["prediction"]
        confidence = data["confidence"]
        all_scores = data["all_scores"]

        # -------------------------------------------------------------------
        # Display result
        # -------------------------------------------------------------------
        st.divider()
        st.subheader("🧾 Prediction Result")

        result_container = st.container(border=True)

        if prediction == "fresh":
            result_container.success(
                f"✅ **FRESH** —  {confidence:.1%} confidence"
            )
        elif prediction == "rotten":
            result_container.error(
                f"⚠️ **ROTTEN** —  {confidence:.1%} confidence"
            )
        else:
            result_container.info(f"**{prediction}** — {confidence:.1%} confidence")

        # -------------------------------------------------------------------
        # Confidence bar chart
        # -------------------------------------------------------------------
        st.subheader("📊 Confidence Scores")

        # Convert all_scores dict to a pandas Series for the bar chart
        scores_series = pd.Series(all_scores)
        df_scores = scores_series.reset_index()
        df_scores.columns = ["Class", "Score"]

        st.bar_chart(df_scores, x="Class", y="Score", color="Score")

        # -------------------------------------------------------------------
        # Full API response expander
        # -------------------------------------------------------------------
        with st.expander("📊 Full API Response"):
            st.json(data)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("ℹ️ About")
    st.write(
        "This app uses a **MobileNetV2** model trained with **PyTorch** "
        "to classify fruit images as fresh or rotten."
    )

    st.subheader("How to use")
    st.markdown("""
    1. **Upload** an image of a fruit
    2. Click **Predict**  
    3. See the **result** (fresh or rotten)
    """)

    st.subheader("Model info")
    st.markdown("""
    - **Model:** MobileNetV2 (transfer learning)
    - **Classes:** Fresh / Rotten
    - **Input size:** 224×224 px
    - **Framework:** PyTorch
    """)

    st.subheader("API endpoint")
    st.code("POST http://127.0.0.1:8000/predict")

    st.divider()
    st.caption("Built with FastAPI + PyTorch + Streamlit — Intern Project")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption("Built with FastAPI + PyTorch + Streamlit — Intern Project")