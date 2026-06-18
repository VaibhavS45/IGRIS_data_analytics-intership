# Image Prediction API — Fresh vs Rotten Fruit Classifier

A **REST API** that accepts an uploaded fruit image, runs it through a trained **PyTorch MobileNetV2** model, and returns a prediction of **"fresh"** or **"rotten"** with a confidence score — all in JSON.

Built for 2nd-year engineering students as a complete AI/ML intern project covering the full pipeline: dataset preparation, transfer learning, backend API, testing, and deployment.

---

## Problem Statement

Fruit spoilage causes significant food waste globally. Build an automated system that can classify whether a fruit is fresh or rotten from an image, making it useful for quality control in supply chains, grocery stores, and consumer apps.

## Dataset

You need to provide training images. Create the following folder structure and place your own JPG/PNG images:

```
data/
├── fresh/        ← Place ~20 fresh fruit images here
└── rotten/       ← Place ~20 rotten fruit images here
```

**Requirements per class:**
- At least 5–10 images per class (more = better accuracy)
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
- Any fruit type works (apples, bananas, oranges, etc.)

**Free dataset ideas:**
- Take photos of fresh and rotten fruits from your kitchen
- Kaggle: "Fruit Classification" or "Fresh and Rotten Fruits" datasets
- Google Images (ensure you have rights to use them)

## Tech Stack

| Component         | Technology                               |
|-------------------|------------------------------------------|
| Backend Framework | FastAPI                                  |
| Deep Learning     | PyTorch + torchvision                    |
| Model             | MobileNetV2 (pretrained on ImageNet)     |
| Image Processing  | Pillow (PIL)                             |
| Server            | Uvicorn                                  |
| Language          | Python 3.10+                             |

## Installation

### Step 1: Install PyTorch (CPU version — no GPU needed)

PyTorch must be installed first using the official CPU-only channel:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

This avoids Windows long-path issues that occur with TensorFlow.

### Step 2: Install other dependencies

```bash
pip install -r requirements.txt
```

## How to Train

Add your images to `data/fresh/` and `data/rotten/`, then run:

```bash
python train.py
```

The script will:
1. Load images from `data/` subfolders
2. Apply data augmentation (random flip, rotation, color jitter)
3. Load MobileNetV2 pretrained on ImageNet
4. Freeze all layers except the classifier head
5. Train for 5 epochs with Adam optimizer
6. Print training/validation accuracy after each epoch
7. Save the trained model to `model/fruit_model.pth`
8. Save class labels to `model/class_names.json`

**Expected output (example):**
```
Using device: cpu
============================================================
Found classes: ['fresh', 'rotten']
  fresh/: 20 images
  rotten/: 20 images
============================================================
Loading dataset...
Classes: ['fresh', 'rotten']
Total images: 40
Training samples: 32
Validation samples: 8
============================================================
Loading pretrained MobileNetV2...
Model architecture loaded. Classifier head: Linear(1280, 2)
============================================================
Starting training for 5 epochs...
------------------------------------------------------------

Epoch 1/5 — Train Loss: 0.5214 | Train Acc: 78.12% | Val Acc: 87.50%
------------------------------------------------------------
...
============================================================
TRAINING COMPLETE!
============================================================
Model saved to: model/fruit_model.pth
Class names saved to: model/class_names.json
```

### Verify the trained model

```bash
python test_model.py
```

This loads the saved model, runs a dummy inference, and confirms everything works.

## How to Run the API

Start the server with Uvicorn:

```bash
uvicorn app:app --reload
```

The server starts at **http://127.0.0.1:8000**

### Test with Swagger UI (Recommended)

Open **http://127.0.0.1:8000/docs** in your browser. This provides an interactive API playground where you can:
1. Click on the `/predict` endpoint
2. Click **"Try it out"**
3. Upload a fruit image
4. Click **"Execute"** to see the prediction

### Test with curl

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@test_images/apple.jpg"
```

## API Endpoints

### `GET /`

Returns API status and available routes.

**Response:**
```json
{
  "message": "Image Prediction API is running",
  "status": "ok",
  "model_loaded": true,
  "classes": ["fresh", "rotten"],
  "endpoints": {
    "GET /": "This information",
    "GET /health": "Health check",
    "POST /predict": "Upload an image for classification"
  },
  "docs": "http://127.0.0.1:8000/docs"
}
```

### `GET /health`

Simple health check.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `POST /predict`

Upload an image and get a classification.

**Request:**
- Method: `POST`
- URL: `http://127.0.0.1:8000/predict`
- Body: `form-data`
- Key: `file` (type: File)
- Value: Select an image file

**Response:**
```json
{
  "prediction": "fresh",
  "confidence": 0.9345,
  "all_scores": {
    "fresh": 0.9345,
    "rotten": 0.0655
  }
}
```

**Error response (model not trained):**
```json
{
  "detail": "Model not trained yet. Run 'python train.py' first, then restart the server."
}
```

## Running the Streamlit Frontend

The project includes a **Streamlit web interface** that connects to the FastAPI backend.

### Prerequisites

Install the additional frontend dependencies:
```bash
pip install streamlit requests
```

### Start both services

You need **two terminals** running simultaneously:

**Terminal 1 — Start the API:**
```bash
uvicorn app:app --reload
```

**Terminal 2 — Start the Streamlit app:**
```bash
streamlit run streamlit_app.py
```

Then open **http://localhost:8501** in your browser.

### One-click launcher

**Windows:** Double-click `start.bat` in the project folder.

**Mac/Linux:**
```bash
chmod +x start.sh
./start.sh
```

### Streamlit features
- ✅ **API health check** on page load
- 🖼️ **Image upload** with preview
- 🔍 **Predict button** sends image to the API
- 📊 **Confidence bar chart** showing fresh vs rotten scores
- 📋 **Full API response** in an expandable section
- ⚠️ **Graceful error handling** — warns if API is offline or model not trained

## Project Structure

```
image-prediction-api/
│
├── app.py              # FastAPI backend with /predict endpoint
├── train.py            # PyTorch training script (transfer learning)
├── test_model.py       # Quick model verification script
├── streamlit_app.py    # Streamlit frontend UI
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── start.bat           # Windows one-click launcher
├── start.sh            # Mac/Linux one-click launcher
│
├── model/              # Saved model files (created by train.py)
│   ├── fruit_model.pth     # Trained PyTorch state dict
│   └── class_names.json    # Class labels
│
├── data/               # Training data (you add images here)
│   ├── fresh/              # ← Place fresh fruit images here
│   └── rotten/             # ← Place rotten fruit images here
│
└── test_images/        # Sample images for testing the API
```

## Hosting on Streamlit Cloud (Free)

Deploy the frontend on **Streamlit Community Cloud** for free so anyone can try it online.

### Prerequisites
- Push your project to a **public GitHub repository**
- Have the **FastAPI backend** already deployed on Render/Railway (or keep it local)

### Step-by-step deployment

#### 1. Prepare your GitHub repo
```bash
# Initialize git & push
cd image-prediction-api
git init
git add .
git commit -m "Initial commit: Image Prediction API"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/image-prediction-api.git
git push -u origin main
```

#### 2. Create a Render account & deploy the FastAPI backend
1. Go to https://render.com → Sign up with GitHub
2. Click **New +** → **Web Service**
3. Connect your GitHub repo
4. Configure:
   - **Name**: `fruit-classifier-api`
   - **Root Directory**: (leave blank)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free
5. Click **Create Web Service**
6. Wait for deployment (~2-3 min). Your API URL will be:  
   `https://fruit-classifier-api.onrender.com`

#### 3. Deploy Streamlit app on Streamlit Cloud
1. Go to https://streamlit.io/cloud → **Sign in with GitHub**
2. Click **New app** → Connect your repo
3. Configure:
   - **Repository**: Your GitHub repo
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
4. Click **Deploy**

#### 4. Link Streamlit to your hosted API
1. On Streamlit Cloud dashboard, go to your app → **⚙️ Settings** → **Secrets**
2. Add the following secret:
   ```toml
   [api]
   base_url = "https://fruit-classifier-api.onrender.com"
   ```
3. Click **Save** → The app will automatically restart

#### 5. Done! 🎉
- **Streamlit URL**: `https://your-app-name.streamlit.app`
- **FastAPI URL**: `https://fruit-classifier-api.onrender.com`
- **Swagger UI**: `https://fruit-classifier-api.onrender.com/docs`

### Alternative: Deploy ONLY the Streamlit app (Frontend Only)
If you can't deploy the FastAPI backend, the Streamlit app works locally with:
```bash
streamlit run streamlit_app.py
```
The API must be running locally: `uvicorn app:app --reload`

## Resume Line

> **Built an image classification REST API using FastAPI and PyTorch** — Trained a MobileNetV2 model via transfer learning to classify uploaded fruit images as fresh or rotten, with confidence scores returned via a `/predict` endpoint tested through Swagger UI.

---

## Quick Start (3 Steps)

```
Step 1: Add ~20 JPG images each to data/fresh/ and data/rotten/
Step 2: python train.py
Step 3: uvicorn app:app --reload  →  visit http://127.0.0.1:8000/docs
