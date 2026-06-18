#!/bin/bash
# Render.com build script
# Install PyTorch (CPU) first, then the rest of the dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt