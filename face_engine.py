"""Face detection (YuNet) + face recognition (SFace) using OpenCV's deep-learning modules."""
import cv2
import numpy as np

import config
from model_downloader import ensure_models


class FaceEngine:
    def __init__(self):
        det_path, rec_path = ensure_models()
        self.detector = cv2.FaceDetectorYN.create(
            str(det_path), "", (320, 320), config.DETECTION_SCORE_THRESHOLD, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(rec_path), "")

    def detect(self, frame):
        """Returns a list of face rows: [x, y, w, h, 10 landmark values, score]."""
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        return [] if faces is None else list(faces)

    def embed(self, frame, face):
        """128-d L2-normalised embedding for one detected face."""
        aligned = self.recognizer.alignCrop(frame, face)
        feat = self.recognizer.feature(aligned).flatten().astype(np.float32)
        return feat / (np.linalg.norm(feat) + 1e-9)


class Gallery:
    """In-memory store of all known embeddings for fast matching."""

    def __init__(self):
        self.users = {}
        self.owner = np.zeros(0, dtype=np.int64)
        self.mat = np.zeros((0, 128), dtype=np.float32)

    def load(self, users, rows):
        """users: {id: {...}}, rows: [(user_id, embedding), ...]"""
        self.users = users
        if rows:
            self.owner = np.array([r[0] for r in rows], dtype=np.int64)
            self.mat = np.stack([r[1] for r in rows]).astype(np.float32)
        else:
            self.owner = np.zeros(0, dtype=np.int64)
            self.mat = np.zeros((0, 128), dtype=np.float32)

    def __len__(self):
        return len(self.users)

    def identify(self, emb):
        """Returns (user_id or None, best_similarity)."""
        if len(self.mat) == 0:
            return None, 0.0
        sims = self.mat @ emb
        i = int(np.argmax(sims))
        score = float(sims[i])
        if score >= config.MATCH_THRESHOLD:
            return int(self.owner[i]), score
        return None, score
