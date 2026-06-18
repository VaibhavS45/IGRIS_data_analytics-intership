"""
FastAPI backend for Fruit Freshness Classification.
Loads a trained PyTorch model and serves predictions via REST API.

After starting the server, visit http://127.0.0.1:8000/docs to test the API
visually using Swagger UI — upload any fruit image and see the prediction.
"""

import json
import os
from io import BytesIO

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms, models

# ---------------------------------------------------------------------------
# Change to script directory so relative paths work
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = "model/fruit_model.pth"
CLASS_NAMES_PATH = "model/class_names.json"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Image preprocessing (must match training transforms)
# ---------------------------------------------------------------------------
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Image Prediction API — Fresh vs Rotten Fruit",
    description="Upload an image of a fruit. The model predicts whether it's fresh or rotten with a confidence score.",
    version="1.0.0",
)

# Allow all origins for easy frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model loading (lazy — at startup)
# ---------------------------------------------------------------------------
model = None
class_names = []


def load_model():
    """Load the PyTorch model and class names from disk."""
    global model, class_names

    if not os.path.exists(MODEL_PATH):
        return False

    if not os.path.exists(CLASS_NAMES_PATH):
        return False

    # Load class names
    with open(CLASS_NAMES_PATH, "r") as f:
        data = json.load(f)
        class_names = data["classes"]

    num_classes = len(class_names)

    # Build model architecture
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # Load trained weights
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return True


# Try loading model on startup
model_loaded = load_model()


# ---------------------------------------------------------------------------
# Helper: validate image type
# ---------------------------------------------------------------------------
def validate_image(file: UploadFile):
    """Reject non-image uploads."""
    if file.content_type is None:
        raise HTTPException(status_code=400, detail="File content type is unknown")
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Only images are accepted.",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def home():
    """Root endpoint – shows API status and available routes."""
    return {
        "message": "Image Prediction API is running",
        "status": "ok",
        "model_loaded": model_loaded,
        "classes": class_names if model_loaded else [],
        "endpoints": {
            "GET /": "This information",
            "GET /health": "Health check",
            "POST /predict": "Upload an image for classification",
        },
        "docs": "http://127.0.0.1:8000/docs",
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": model_loaded}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an image upload and return a classification prediction.

    Usage:
        curl -X POST http://127.0.0.1:8000/predict \
            -F "file=@apple.jpg"

    Returns:
        {
            "prediction": "fresh",
            "confidence": 0.93,
            "all_scores": {"fresh": 0.93, "rotten": 0.07}
        }
    """
    # ---- Check model availability ----
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run 'python train.py' first, then restart the server.",
        )

    # ---- Validate input ----
    validate_image(file)

    # ---- Read file ----
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    # ---- Preprocess ----
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)  # add batch dim
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")

    # ---- Inference ----
    try:
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    # ---- Build response ----
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    predicted_class = class_names[predicted_index]

    all_scores = {class_names[i]: round(float(probabilities[i]), 4)
                  for i in range(len(class_names))}

    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 4),
        "all_scores": all_scores,
    }