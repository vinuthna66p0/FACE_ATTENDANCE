"""Optional: pre-download the AI models (python download_models.py)."""
from model_downloader import ensure_models

if __name__ == "__main__":
    ensure_models()
    print("Done.")
