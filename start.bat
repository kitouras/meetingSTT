@echo off

echo Checking for virtual environment...
IF NOT EXIST .\.venv (
    echo Virtual environment not found. Creating .\.venv...
    python -m venv .\.venv
    echo Installing dependencies from .\ui_client\requirements.txt...
    .\.venv\Scripts\pip.exe install -r .\ui_client\requirements.txt
    echo Setup complete. Virtual environment created and dependencies installed.
) ELSE (
    echo Virtual environment .\.venv already exists.
    echo Assuming dependencies are up-to-date. If not, activate venv and run:
    echo pip install -r .\ui_client\requirements.txt
)

echo.
echo Starting the application using run.py (manages Docker and UI client)...
call .\.venv\Scripts\activate.bat
echo Activated virtual environment.
python run.py
echo.
echo Application (run.py) has been started. It will manage Docker services and the UI client.
echo If it closed immediately, please check for errors in the console.