"""Central settings. Tweak these to tune the system."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
FACES_DIR = DATA_DIR / "faces"
EXPORT_DIR = DATA_DIR / "exports"
DB_PATH = DATA_DIR / "attendance.db"
for _d in (DATA_DIR, MODELS_DIR, FACES_DIR, EXPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- Camera ----
CAMERA_INDEX = 0          # try 1 or 2 if you have several cameras
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ---- Face detection / recognition ----
DETECTION_SCORE_THRESHOLD = 0.8   # YuNet confidence
MIN_FACE_SIZE = 80                # px; smaller faces are ignored ("move closer")
MATCH_THRESHOLD = 0.40            # cosine similarity (OpenCV suggests 0.363). Higher = stricter
CONSECUTIVE_FRAMES = 3            # same person must be seen N frames in a row before marking
REGISTER_SAMPLES = 12             # face samples captured per person at registration
REGISTER_INTERVAL = 0.4           # seconds between samples

# ---- Attendance rules ----
LATE_AFTER = "09:30"              # 24h HH:MM; check-ins after this are marked "Late"
