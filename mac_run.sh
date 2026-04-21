#!/bin/bash
cd "$(dirname "$0")"

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3.10 -m venv venv || python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install pyqt5 opencv-python numpy==1.26.4 insightface onnxruntime

# Run app
python app.py