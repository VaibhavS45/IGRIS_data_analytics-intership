"""
FastAPI backend for Fruit Freshness Classification.
Loads a trained model (ONNX preferred, PyTorch fallback) and serves predictions via REST API.

After starting the server, visit http://127.0.0.1:8000/docs to test the API
visually using Swagger UI — upload any fruit image and see the prediction.
"""

import json
import os
from io import BytesIO

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# ---------------------------------------------------------------------------
# Change to script directory so relative paths work
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = "model"
ONNX_PATH = os.path.join(MODEL_DIR, "fruit_model.onnx")
PTH_PATH = os.path.join(MODEL_DIR, "fruit_model.pth")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
IMG_SIZE = 224

# ---------------------------------------------------------------------------
# Try to load model (ONNX first, then PyTorch fallback)
# ---------------------------------------------------------------------------
model = None
class_names = []
model_backend = None
session = None  # ONNX session


def load_model():
    """Load model - ONNX Runtime preferred, PyTorch as fallback."""
    global model, class_names, model_backend, session

    if not os.path.exists(CLASS_NAMES_PATH):
        return False

    with open(CLASS_NAMES_PATH, "r") as f:
        data = json.load(f)
        class_names = data["classes"]

    # Try ONNX first (lightweight, preferred for deployment)
    if os.path.exists(ONNX_PATH):
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(
                ONNX_PATH,
                providers=["CPUExecutionProvider"]
            )
            model_backend = "onnx"
            return True
        except Exception:
            pass

    # Fall back to PyTorch
    if os.path.exists(PTH_PATH):
        try:
            import torch
            import torch.nn as nn
            from torchvision import models

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            m = models.mobilenet_v2(weights=None)
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(class_names))
            m.load_state_dict(
                torch.load(PTH_PATH, map_location=device, weights_only=True)
            )
            m.to(device)
            m.eval()
            model = m
            model_backend = "pytorch"
            return True
        except Exception:
            pass

    return False


model_loaded = load_model()


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert uploaded image bytes to a normalized 224x224 numpy array."""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image, dtype=np.float32) / 255.0
    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    # Convert to NCHW format (batch=1, channels=3, height=224, width=224)
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array


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
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Image Prediction API \u2014 Fresh vs Rotten Fruit",
    description="Upload an image of a fruit. The model predicts whether it's fresh or rotten with a confidence score.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def predict_with_model(input_array: np.ndarray):
    """Run inference using the loaded model (ONNX or PyTorch)."""
    if model_backend == "onnx":
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: input_array})
        logits = result[0]
    elif model_backend == "pytorch":
        import torch
        with torch.no_grad():
            tensor = torch.from_numpy(input_array)
            logits = model(tensor).cpu().numpy()
    else:
        raise RuntimeError("No model loaded")

    # Apply softmax
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    return probabilities[0]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def home():
    return {
        "message": "Image Prediction API is running",
        "status": "ok",
        "model_loaded": model_loaded,
        "model_backend": model_backend,
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
    return {"status": "ok", "model_loaded": model_loaded}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run 'python train.py' first, then restart the server.",
        )

    validate_image(file)

    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        input_array = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")

    try:
        probabilities = predict_with_model(input_array)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    predicted_class = class_names[predicted_index]

    all_scores = {
        class_names[i]: round(float(probabilities[i]), 4)
        for i in range(len(class_names))
    }

    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 4),
        "all_scores": all_scores,
    }