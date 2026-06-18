@echo off
echo ============================================
echo  Image Prediction API - Starting Services
echo ============================================
echo.

echo [1/2] Starting FastAPI backend on port 8000...
start cmd /k "uvicorn app:app --reload"

echo Waiting 3 seconds for API to initialize...
timeout /t 3 /nobreak >nul

echo [2/2] Starting Streamlit frontend on port 8501...
start cmd /k "streamlit run streamlit_app.py"

echo.
echo ============================================
echo  Both services launched!
echo  API:      http://127.0.0.1:8000
echo  Swagger:  http://127.0.0.1:8000/docs
echo  Frontend: http://localhost:8501
echo ============================================
echo.
pause