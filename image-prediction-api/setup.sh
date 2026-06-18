#!/bin/bash
# Render.com build script
# 1. Install PyTorch temporarily to convert model to ONNX
# 2. Convert .pth -> .onnx (lighter format for deployment)
# 3. Remove heavy PyTorch packages
# 4. Install remaining lightweight dependencies

set -e  # Exit on error

echo "============================================"
echo " Step 1/4: Install PyTorch (temp, for conversion)"
echo "============================================"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "============================================"
echo " Step 2/4: Convert model to ONNX"
echo "============================================"
pip install onnx onnxscript onnxruntime
python convert_to_onnx.py

echo "============================================"
echo " Step 3/4: Remove heavy PyTorch packages"
echo "============================================"
pip uninstall -y torch torchvision onnx onnxscript

echo "============================================"
echo " Step 4/4: Install remaining dependencies"
echo "============================================"
pip install -r requirements.txt

echo "============================================"
echo " Build complete! ONNX model ready."
echo "============================================"