# AI Face Attendance System

Desktop app that registers people with their webcam and marks attendance automatically
by recognising faces. Uses deep-learning models running locally (no cloud, no API keys):

- **Detection:** YuNet (ONNX)  - **Recognition:** SFace (128-d embeddings, cosine matching)
- **Storage:** SQLite (`data/attendance.db`)  - **GUI:** Tkinter

## Quick start (VS Code)

1. Unzip, then **File → Open Folder…** and choose `face_attendance`.
2. Needs **Python 3.9 – 3.12** and a webcam. (Linux only: `sudo apt install python3-tk`)
3. Open the VS Code terminal (**Ctrl + `**) and run:

   **Windows**
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   python app.py
   ```
   **macOS / Linux**
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   python app.py
   ```
   (Or just run `setup_windows.bat` / `./setup_mac_linux.sh`.)

4. In VS Code, select the interpreter `.venv` (Ctrl+Shift+P → *Python: Select Interpreter*). You can then press **F5** to run.

On the **first launch** the two AI models (~37 MB) are downloaded automatically into `models/`.
Internet is needed only for that step. If it fails, download the files manually
(links are in `model_downloader.py`) and put them in `models/`.
macOS: allow camera access for VS Code / Terminal when prompted.

## How to use

1. **Register new person** – enter name + ID, look at the camera, slowly turn your head slightly. 12 samples are captured.
2. **Start attendance** – faces are recognised live. After 3 consecutive matching frames the person is marked
   *Present* (or *Late* after `LATE_AFTER`). One record per person per day.
3. **Attendance records** – view by date or all dates, **Export CSV**.
4. **Manage people** – delete a person and their data.

## Project layout

```
app.py              GUI + camera loop
face_engine.py      YuNet detector, SFace embedder, matching
database.py         SQLite (users, embeddings, attendance)
model_downloader.py auto-downloads the ONNX models
config.py           thresholds, late time, camera index
data/               database, face thumbnails, CSV exports
```

## Tuning (config.py)

| Setting | Meaning |
|---|---|
| `CAMERA_INDEX` | 0 default webcam; try 1/2 for external cameras |
| `MATCH_THRESHOLD` | 0.40 default. Raise (e.g. 0.45) if strangers get matched; lower (0.35) if people aren't recognised |
| `LATE_AFTER` | Check-ins after this time are "Late" |
| `REGISTER_SAMPLES` | More samples = better accuracy |

## Troubleshooting

- **"Camera not available"** – close other apps using the webcam, change `CAMERA_INDEX`.
- **`AttributeError: FaceDetectorYN`** – upgrade OpenCV: `pip install -U opencv-python` (needs ≥ 4.8).
- **Not recognised** – register in the same lighting you use for attendance; re-register with the face fully visible (no mask/cap).

## Limitations

This is a learning/small-scale project. There is **no liveness detection**, so a printed photo or phone
screen of a registered person could be accepted. For real deployments add anti-spoofing, and get consent
for storing biometric data (face embeddings are stored locally in the SQLite file).
