"""Downloads the two ONNX deep-learning models on first run (one time, ~37 MB)."""
import shutil
import urllib.request

import config

MODELS = {
    "face_detection_yunet_2023mar.onnx": [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
    ],
    "face_recognition_sface_2021dec.onnx": [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
    ],
}
MIN_BYTES = 100_000  # guards against saving an HTML/LFS-pointer error page


def _download(urls, dest):
    last_err = None
    for url in urls:
        try:
            print(f"Downloading {dest.name} from {url} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            tmp = dest.with_suffix(".part")
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            if tmp.stat().st_size < MIN_BYTES:
                raise RuntimeError("downloaded file is too small")
            tmp.replace(dest)
            return
        except Exception as e:  # try next mirror
            last_err = e
            print(f"  failed: {e}")
    raise RuntimeError(
        f"Could not download {dest.name}: {last_err}\n"
        f"Download it manually from one of:\n  " + "\n  ".join(urls) +
        f"\nand place it in: {config.MODELS_DIR}"
    )


def ensure_models():
    """Returns (detector_path, recognizer_path), downloading if missing."""
    paths = []
    for name, urls in MODELS.items():
        p = config.MODELS_DIR / name
        if not p.exists() or p.stat().st_size < MIN_BYTES:
            _download(urls, p)
        paths.append(p)
    return paths[0], paths[1]


if __name__ == "__main__":
    ensure_models()
    print("Models ready in", config.MODELS_DIR)
