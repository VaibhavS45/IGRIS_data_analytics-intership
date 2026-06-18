"""
Test script to verify the trained model loads and runs inference correctly.
Creates a dummy image tensor to validate the model pipeline without requiring real images.
"""

import torch
import torch.nn as nn
from torchvision import models
import json
import os
import sys


def main():
    # Change to script directory so relative paths work
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    MODEL_PATH = "model/fruit_model.pth"
    CLASS_NAMES_PATH = "model/class_names.json"

    print("=" * 60)
    print("Model Verification Script")
    print("=" * 60)

    # ---- Check model file ----
    if not os.path.exists(MODEL_PATH):
        print(f"  ERROR: Model file not found at '{MODEL_PATH}'")
        print("  Run 'python train.py' first to train the model.")
        sys.exit(1)

    if not os.path.exists(CLASS_NAMES_PATH):
        print(f"  ERROR: Class names file not found at '{CLASS_NAMES_PATH}'")
        print("  Run 'python train.py' first to train the model.")
        sys.exit(1)

    print(f"  Model file:      {MODEL_PATH}")
    print(f"  Class names file: {CLASS_NAMES_PATH}")

    # ---- Load class names ----
    with open(CLASS_NAMES_PATH, "r") as f:
        data = json.load(f)
        class_names = data["classes"]

    print(f"  Classes:          {class_names}")
    print(f"  Number of classes: {len(class_names)}")

    # ---- Load model ----
    print("\n  Loading model architecture (MobileNetV2)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device:           {device}")

    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(class_names))

    print("  Loading trained weights...")
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # ---- Test inference ----
    print("\n  Creating dummy image tensor (224x224)...")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    print("  Running forward pass (inference)...")
    with torch.no_grad():
        logits = model(dummy_input)
        probabilities = torch.softmax(logits, dim=1)

    predicted_index = int(torch.argmax(probabilities[0]))
    confidence = float(torch.max(probabilities[0]))

    print(f"\n  ✓ Model loaded and inference successful!")
    print(f"  ✓ Output shape: {list(logits.shape)}")
    print(f"  ✓ Predicted class index: {predicted_index}")
    print(f"  ✓ Class: {class_names[predicted_index]}")
    print(f"  ✓ Confidence: {confidence:.4f}")
    print(f"  ✓ All class scores:")
    for i, name in enumerate(class_names):
        print(f"      {name}: {float(probabilities[0][i]):.4f}")

    print("\n" + "=" * 60)
    print("Model verification PASSED — ready for API deployment!")
    print("=" * 60)
    print("\nNext step: Start the API server")
    print("  uvicorn app:app --reload")
    print("  Then visit http://127.0.0.1:8000/docs")


if __name__ == "__main__":
    main()