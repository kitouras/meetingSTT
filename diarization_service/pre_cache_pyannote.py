"""Script to pre-cache the Pyannote.audio pipeline model for Docker builds."""
import os
import json
import sys

try:
    from pyannote.audio import Pipeline
except ImportError:
    print("Could not import Pipeline from pyannote.audio. Make sure it's installed.", file=sys.stderr)
    sys.exit(1)


print("Attempting to pre-cache Pyannote models...")

settings_path = "settings.json"
if not os.path.exists(settings_path):
    print(f"Error: Settings file '{settings_path}' not found. Cannot pre-cache.", file=sys.stderr)
    sys.exit(1)

try:
    with open(settings_path, 'r', encoding='utf-8') as f:
        settings = json.load(f)
    pyannote_model_name = settings.get("pyannote_model_name")
    hugging_face_token = settings.get("hugging_face_token")
except Exception as e:
    print(f"Error reading or parsing settings file: {e}", file=sys.stderr)
    sys.exit(1)

if not pyannote_model_name:
    print("Error: 'pyannote_model_name' not found in settings.json.", file=sys.stderr)
    sys.exit(1)

print(f"Pre-caching Pyannote model: {pyannote_model_name}...")
try:
    # This call downloads the model files to the cache.
    # We don't need to keep the pipeline object.
    _ = Pipeline.from_pretrained(
        pyannote_model_name,
        use_auth_token=hugging_face_token
    )
    print("Pyannote model pre-caching script completed successfully.")
    sys.exit(0)
except Exception as e:
    print(f"An unexpected error occurred during Pyannote model pre-caching: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)