"""Script to pre-cache the GigaAM ASR model.

This script attempts to download and cache the specified GigaAM model
to avoid download delays when the main service starts.
"""
import sys
import gigaam

print("Attempting to pre-cache GigaAM models...")

model_name_to_cache = "rnnt"

try:
    print(f"Pre-caching GigaAM model: '{model_name_to_cache}'...")
    model = gigaam.load_model(model_name_to_cache)

    if model is not None:
        print(f"GigaAM model '{model_name_to_cache}' appears to be pre-cached/loaded successfully.")
    else:
        print(f"Failed to load/pre-cache GigaAM model '{model_name_to_cache}'. The load_model function returned None.")
        sys.exit(1)

except Exception as e:
    print(f"An unexpected error occurred during GigaAM model ('{model_name_to_cache}') pre-caching: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("GigaAM model pre-caching script completed successfully.")
sys.exit(0)