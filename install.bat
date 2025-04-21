python -m venv .\.venv
.\.venv\Scripts\pip.exe install -r requirements.txt
git clone https://github.com/salute-developers/GigaAM.git
.\.venv\Scripts\pip.exe install -e .\GigaAM
.\.venv\Scripts\pip.exe uninstall torch torchaudio
.\.venv\Scripts\pip.exe install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118