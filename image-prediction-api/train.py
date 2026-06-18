"""
Train an image classifier (Fresh vs Rotten Fruits) using PyTorch transfer learning.

This script:
1. Loads images from data/fresh/ and data/rotten/
2. Uses MobileNetV2 pretrained on ImageNet
3. Fine-tunes the classifier head for 2 classes
4. Saves the trained model to model/fruit_model.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, DataLoader
import os
import json
import sys


def main():
    # -------------------------------
    # 0. Change to script directory
    # -------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {script_dir}")

    # -------------------------------
    # 1. Configuration
    # -------------------------------
    DATA_DIR = "data"
    MODEL_DIR = "model"
    MODEL_PATH = os.path.join(MODEL_DIR, "fruit_model.pth")
    CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
    IMG_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 5
    VALIDATION_SPLIT = 0.2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {DEVICE}")
    print("=" * 60)

    # -------------------------------
    # 2. Check dataset
    # -------------------------------
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Data directory '{DATA_DIR}' not found.")
        print("Please create folders: data/fresh/ and data/rotten/ with your images.")
        sys.exit(1)

    # Check if class folders have images
    class_folders = [d for d in os.listdir(DATA_DIR)
                     if os.path.isdir(os.path.join(DATA_DIR, d))]
    if not class_folders:
        print(f"ERROR: No class folders found inside '{DATA_DIR}'.")
        print("Create subfolders like data/fresh/ and data/rotten/ each containing images.")
        sys.exit(1)

    print(f"Found classes: {class_folders}")

    # Count images per class
    for cls in class_folders:
        img_count = len([f for f in os.listdir(os.path.join(DATA_DIR, cls))
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
        print(f"  {cls}/: {img_count} images")
        if img_count == 0:
            print(f"  WARNING: {cls}/ has no images! Add at least 5-10 per class.")

    print("=" * 60)

    # -------------------------------
    # 3. Data transforms
    # -------------------------------
    # Training: augmentation + normalization
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                             std=[0.229, 0.224, 0.225])   # ImageNet std
    ])

    # Validation: just resize + normalize
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # -------------------------------
    # 4. Load dataset
    # -------------------------------
    print("Loading dataset...")
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)

    num_classes = len(full_dataset.classes)
    class_names = full_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Total images: {len(full_dataset)}")

    if len(full_dataset) == 0:
        print("ERROR: No images found. Add JPG/PNG images to data/fresh/ and data/rotten/")
        sys.exit(1)

    # 80/20 train/val split
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size

    # Apply different transforms for train vs val
    full_dataset_train = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)
    full_dataset_val = datasets.ImageFolder(root=DATA_DIR, transform=val_transform)

    # Use same random seed for reproducible split
    torch.manual_seed(42)
    train_indices, val_indices = random_split(
        range(len(full_dataset)),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset = torch.utils.data.Subset(full_dataset_train, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset_val, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print("=" * 60)

    # -------------------------------
    # 5. Load pretrained model
    # -------------------------------
    print("Loading pretrained MobileNetV2...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze all layers (transfer learning: only train the classifier head)
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier head for our 2 classes
    # MobileNetV2's classifier is: Linear(1280, 1000)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier[1] = nn.Linear(in_features, num_classes)

    model = model.to(DEVICE)

    print(f"Model architecture loaded. Classifier head: Linear(1280, {num_classes})")
    print("=" * 60)

    # -------------------------------
    # 6. Training setup
    # -------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # -------------------------------
    # 7. Training loop
    # -------------------------------
    print(f"Starting training for {EPOCHS} epochs...")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        # --- Training phase ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Print mini-batch progress
            if (batch_idx + 1) % 5 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        train_acc = 100.0 * correct / total
        avg_loss = running_loss / len(train_loader)

        # --- Validation phase ---
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100.0 * val_correct / val_total

        print(f"\nEpoch {epoch}/{EPOCHS} — "
              f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}%")
        print("-" * 60)

    # -------------------------------
    # 8. Save model
    # -------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    # Save class names
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump({"classes": class_names}, f)
    print(f"Class names saved to: {CLASS_NAMES_PATH}")

    # -------------------------------
    # 9. Summary
    # -------------------------------
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Final Training Accuracy: {train_acc:.2f}%")
    print(f"Final Validation Accuracy: {val_acc:.2f}%")
    print(f"Number of classes: {num_classes} -> {class_names}")
    print(f"Model file: {MODEL_PATH}")
    print()
    print("Next step: Start the API server")
    print("  uvicorn app:app --reload")
    print("  Then visit http://127.0.0.1:8000/docs to test")


if __name__ == "__main__":
    main()