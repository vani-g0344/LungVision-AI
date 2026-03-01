import gdown
import os

MODEL_PATH = "model/lungvision_model.pth"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(
        id="1c5V9OKM1ONNAXWyWIwrO36qW6sMLhHbK",  # ← paste your ID here
        output=MODEL_PATH,
        quiet=False
    )
    print("Model downloaded successfully!")
else:
    print("Model already exists, skipping download.")