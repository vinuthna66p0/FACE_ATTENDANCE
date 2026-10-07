@echo off
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_models.py
echo.
echo Setup complete. Run:  .venv\Scripts\python app.py
pause
