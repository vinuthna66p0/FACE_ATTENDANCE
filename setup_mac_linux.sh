#!/usr/bin/env bash
set -e
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_models.py
echo
echo "Setup complete. Run:  source .venv/bin/activate && python app.py"
