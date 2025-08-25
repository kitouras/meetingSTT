"""Script to pre-cache the Faster Whisper model."""
import os
import json
import sys

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Error: faster_whisper is not installed. Please install it using 'pip install faster_whisper'")
    sys.exit(1)

print("Attempting to pre-cache Whisper models...")

settings_path = "settings.json"
whisper_model_size = None
whisper_device = "cpu"
whisper_compute_type = "default"

if not os.path.exists(settings_path):
    print(f"Error: Settings file '{settings_path}' not found in current directory ({os.getcwd()}). Cannot pre-cache Whisper model.")
    sys.exit(1)

try:
    with open(settings_path, 'r', encoding='utf-8') as f:
        settings = json.load(f)
    whisper_model_size = settings.get("whisper_model_size")
    whisper_device = settings.get("whisper_device", "cpu")
    whisper_compute_type = settings.get("whisper_compute_type", "default")
except Exception as e:
    print(f"Error reading or parsing '{settings_path}': {e}")
    sys.exit(1)

if not whisper_model_size:
    print("Error: 'whisper_model_size' not found in settings.json. Cannot pre-cache.")
    sys.exit(1)

print(f"Pre-caching Whisper model: '{whisper_model_size}'...")
print(f"Device: '{whisper_device}', Compute Type: '{whisper_compute_type}'")

try:
    model = WhisperModel(whisper_model_size, device=whisper_device, compute_type=whisper_compute_type)
    
    if model:
        print(f"Whisper model '{whisper_model_size}' appears to be pre-cached/loaded successfully.")
    else:
        print(f"Failed to load/pre-cache Whisper model '{whisper_model_size}'.")
        sys.exit(1)

except Exception as e:
    print(f"An unexpected error occurred during Whisper model ('{whisper_model_size}') pre-caching: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Whisper model pre-caching script completed successfully.")
sys.exit(0)