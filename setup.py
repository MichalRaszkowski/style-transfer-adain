import os
import sys
import subprocess

MODEL_URL = "https://huggingface.co/Michal-Raszkowski/adain-style-transfer/resolve/main/style-transfer-best.ckpt?download=true"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_FILENAME = "style-transfer-best.ckpt"

def install_requirements():
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError:
        print("Failed to install dependencies.")
        sys.exit(1)

def download_model():
    dest_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)
    
    if os.path.exists(dest_path):
        return

    print("Downloading model...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    try:
        import torch
        torch.hub.download_url_to_file(MODEL_URL, dest_path)
    except Exception:
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, dest_path)

def main():
    install_requirements()
    download_model()
    print("Setup complete.")

if __name__ == "__main__":
    main()