#!/bin/bash
echo "============================================"
echo " Image Prediction API - Starting Services"
echo "============================================"
echo ""

echo "[1/2] Starting FastAPI backend on port 8000..."
uvicorn app:app --reload &
API_PID=$!

echo "Waiting 3 seconds for API to initialize..."
sleep 3

echo "[2/2] Starting Streamlit frontend on port 8501..."
streamlit run streamlit_app.py &
STREAMLIT_PID=$!

echo ""
echo "============================================"
echo " Both services launched!"
echo " API:      http://127.0.0.1:8000"
echo " Swagger:  http://127.0.0.1:8000/docs"
echo " Frontend: http://localhost:8501"
echo "============================================"
echo ""
echo "Press Ctrl+C to stop both services."
trap "kill $API_PID $STREAMLIT_PID 2>/dev/null; exit" SIGINT SIGTERM
wait