"""Script to pre-cache the Pyannote.audio pipeline model.

This script reads the model name from settings.json and attempts to
download and cache it to avoid download delays when the main service starts.
"""
import os
import json
import sys

try:
    from diarization_service.models import PyannotePipelineWrapper
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_candidate = os.path.abspath(os.path.join(current_dir, '..'))
    if os.path.basename(project_root_candidate) == 'app' or 'diarization_service' in os.listdir(project_root_candidate):
         sys.path.insert(0, project_root_candidate)
    else:
         sys.path.insert(0, os.path.dirname(current_dir))

    from diarization_service.models import PyannotePipelineWrapper


print("Attempting to pre-cache Pyannote models...")

settings_path = "settings.json"
pyannote_model_name = None
hugging_face_token = None

if not os.path.exists(settings_path):
    print(f"Error: Settings file '{settings_path}' not found in current directory ({os.getcwd()}). Cannot pre-cache Pyannote models.")
    sys.exit(1)

try:
    with open(settings_path, 'r', encoding='utf-8') as f:
        settings = json.load(f)
    pyannote_model_name = settings.get("pyannote_model_name")
    hugging_face_token = settings.get("hugging_face_token")
except Exception as e:
    print(f"Error reading or parsing '{settings_path}': {e}")
    sys.exit(1)

if not pyannote_model_name:
    print("Error: 'pyannote_model_name' not found in settings.json. Cannot pre-cache.")
    sys.exit(1)

print(f"Pre-caching Pyannote model: {pyannote_model_name}...")
if hugging_face_token:
    print("Using Hugging Face token for pre-caching.")
else:
    print("No Hugging Face token provided for pre-caching (will use public models or cached Hugging Face CLI login if any).")

try:
    wrapper = PyannotePipelineWrapper(model_name=pyannote_model_name, auth_token=hugging_face_token)

    if wrapper.pipeline is not None:
        print(f"Pyannote model '{pyannote_model_name}' appears to be pre-cached/loaded successfully.")
    else:
        print(f"Failed to load/pre-cache Pyannote model '{pyannote_model_name}'. Review logs from PyannotePipelineWrapper for details.")
except Exception as e:
    print(f"An unexpected error occurred during the Pyannote model pre-caching script: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Pyannote model pre-caching script completed successfully.")
sys.exit(0)