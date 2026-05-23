import os
import gdown

MODEL_PATH = "models/admet_net_best.pt"
FILE_ID = "https://drive.google.com/file/d/1X6LjhXk4UWYUt5mif0kv64zXvPvWUKAV/view?usp=sharing"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    os.makedirs("models", exist_ok=True)
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
    print("✅ Model downloaded!")
else:
    print("✅ Model already exists, skipping download.")