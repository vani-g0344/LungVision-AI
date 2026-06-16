import os
from huggingface_hub import hf_hub_download
import shutil

MODEL_PATH = "model/lungvision_model.pth"
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    path = hf_hub_download(repo_id="vanig/lungvision-model", filename="lungvision_model.pth")
    os.makedirs("model", exist_ok=True)
    shutil.copy(path, MODEL_PATH)
    print("Model downloaded successfully!")
else:
    print("Model already exists, skipping download.")