#!/bin/bash

echo "Checking for virtual environment..."
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating .venv..."
    # Attempt to use python3, falling back to python if needed
    if command -v python3 &>/dev/null; then
        PYTHON_BIN="python3"
    else
        PYTHON_BIN="python"
    fi
    $PYTHON_BIN -m venv .venv
    
    echo "Installing dependencies from ./ui_client/requirements.txt..."
    ./.venv/bin/pip install -r ./ui_client/requirements.txt
    echo "Setup complete. Virtual environment created and dependencies installed."
else
    echo "Virtual environment .venv already exists."
    echo "Assuming dependencies are up-to-date. If not, activate venv and run:"
    echo "pip install -r ./ui_client/requirements.txt"
fi

echo ""
echo "Starting the application using run.py (manages Docker and UI client)..."
source ./.venv/bin/activate
echo "Activated virtual environment."
python run.py
echo ""
echo "Application (run.py) has been started. It will manage Docker services and the UI client."
echo "If it closed immediately, please check for errors in the console."