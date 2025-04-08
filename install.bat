curl --output vosk_model.zip "https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip"
tar -xf .\vosk_model.zip
rm .\vosk_model.zip
python -m venv .\.venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118