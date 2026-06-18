"""
Convert the trained PyTorch model (.pth) to ONNX format (.onnx).
The ONNX model is lightweight and can be used in the API without PyTorch.

Usage:
    python convert_to_onnx.py

Requires (local machine only): pip install torch torchvision onnxruntime
The API server only needs: pip install onnxruntime
"""

import os
import json
import sys


def main():
    MODEL_DIR = "model"
    PTH_PATH = os.path.join(MODEL_DIR, "fruit_model.pth")
    ONNX_PATH = os.path.join(MODEL_DIR, "fruit_model.onnx")
    CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("=" * 60)
    print("PyTorch to ONNX Converter")
    print("=" * 60)

    if not os.path.exists(PTH_PATH):
        print(f"  ERROR: Model not found at '{PTH_PATH}'")
        print("  Run 'python train.py' first.")
        sys.exit(1)

    if not os.path.exists(CLASS_NAMES_PATH):
        print(f"  ERROR: Class names not found at '{CLASS_NAMES_PATH}'")
        sys.exit(1)

    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)["classes"]
    num_classes = len(class_names)
    print(f"  Classes: {class_names} ({num_classes})")

    # Import PyTorch here (only needed locally, not on server)
    print("  Loading PyTorch model...")
    import torch
    import torch.nn as nn
    from torchvision import models

    device = torch.device("cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(PTH_PATH, map_location=device, weights_only=True))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    print("  Converting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11,
        do_constant_folding=True,
    )

    size_mb = os.path.getsize(ONNX_PATH) / 1e6
    print(f"  Saved: {ONNX_PATH} ({size_mb:.1f} MB)")

    # Quick verification with onnxruntime
    print("  Verifying...")
    import onnxruntime as ort
    session = ort.InferenceSession(ONNX_PATH)
    result = session.run(
        [session.get_outputs()[0].name],
        {session.get_inputs()[0].name: dummy_input.numpy()}
    )
    print(f"  ONNX inference OK! Output shape: {result[0].shape}")

    print("\n  Done! The API will now use fruit_model.onnx (no PyTorch needed).")


if __name__ == "__main__":
    main()