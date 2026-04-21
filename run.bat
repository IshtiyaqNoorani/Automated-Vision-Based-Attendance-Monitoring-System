@echo off
cd /d %~dp0

REM Create venv if not exists
if not exist venv (
    echo Creating virtual environment...
    py -3.10 -m venv venv
)

REM Activate venv
call venv\Scripts\activate

REM Upgrade pip (optional but good)
python -m pip install --upgrade pip

REM Install dependencies
pip install pyqt5 opencv-python numpy==1.26.4 insightface onnxruntime

REM Run app
python app.py

pause