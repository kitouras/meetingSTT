"""Flask API for the Diarization and Transcription service."""
import os
import json
import tempfile
from typing import Tuple

import psutil
import librosa
import numpy as np
import noisereduce as nr
from flask import Flask, request, jsonify, Response

try:
    import pynvml
except ImportError:
    pynvml = None

from .transcription import process_audio_pipeline

DIARIZATION_HOST = '0.0.0.0'
DIARIZATION_PORT = 5002
UPLOAD_FOLDER = 'diarization_uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a'}
TARGET_SAMPLE_RATE = 16000

settings: dict = None

def load_service_settings() -> None:
    """
    Loads settings from settings.json. This is called once at startup.
    """
    global settings
    settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "settings.json")
    try:
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"FATAL: Could not load settings file at {settings_path}. Error: {e}")
        exit(1)

def preprocess_audio_from_stream(file_stream) -> Tuple[np.ndarray, int]:
    """
    Loads audio from a file stream, preprocesses it in memory, and returns a NumPy array.
    """
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_file:
        file_stream.save(temp_file.name)
        temp_file.seek(0)
        
        audio, sample_rate = librosa.load(temp_file.name, sr=TARGET_SAMPLE_RATE, mono=True)
        
    print("Applying noise reduction in-memory...")
    reduced_noise_audio = nr.reduce_noise(y=audio, sr=sample_rate, prop_decrease=0.6)
    
    return reduced_noise_audio, sample_rate

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 ** 3
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename: str) -> bool:
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check() -> Tuple[Response, int]:
    """Provides a health check of the service, including resource usage."""
    process = psutil.Process(os.getpid())
    cpu_percent = psutil.cpu_percent(interval=0.1)
    service_mem_percent = psutil.virtual_memory().percent
    service_process_mem_mb = process.memory_info().rss / (1024 * 1024)

    gpu_utilization, gpu_mem_percent, gpu_mem_used_mb, gpu_mem_total_mb, gpu_error = None, None, None, None, None

    if pynvml:
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization = utilization.gpu
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_mem_total_mb = round(mem_info.total / (1024**2), 2)
                gpu_mem_used_mb = round(mem_info.used / (1024**2), 2)
                gpu_mem_percent = round((gpu_mem_used_mb / gpu_mem_total_mb) * 100, 2) if gpu_mem_total_mb > 0 else 0
            else:
                gpu_error = "No NVIDIA GPU detected by NVML."
            pynvml.nvmlShutdown()
        except pynvml.NVMLError as e:
            gpu_error = f"NVML Error: {str(e)}. GPU info not available."
    else:
        gpu_error = "pynvml library not installed. GPU info not available."

    return jsonify({
        "status": "healthy",
        "model_status": "Models are loaded on-demand for each request.",
        "system_cpu_percent": cpu_percent,
        "system_mem_percent": service_mem_percent,
        "service_process_mem_mb": round(service_process_mem_mb, 2),
        "gpu_utilization_percent": gpu_utilization,
        "gpu_mem_percent": gpu_mem_percent,
        "gpu_mem_used_mb": gpu_mem_used_mb,
        "gpu_mem_total_mb": gpu_mem_total_mb,
        "gpu_error": gpu_error
    }), 200

@app.route('/process_audio', methods=['POST'])
def process_audio_endpoint() -> Tuple[Response, int]:
    """Processes an uploaded audio file using on-demand parallel model execution."""
    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio_file part in the request"}), 400

    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        try:
            audio_array, sample_rate = preprocess_audio_from_stream(file)
            
            transcribed_segments, error_message = process_audio_pipeline(
                audio_array, sample_rate, settings
            )

            if error_message:
                print(f"Error from pipeline: {error_message}")
                return jsonify({"error": f"Failed during processing: {error_message}"}), 500
            
            return jsonify({
                "message": "Audio processed successfully.",
                "transcribed_segments": transcribed_segments
            }), 200

        except Exception as e:
            import traceback
            print(f"An unexpected error occurred in /process_audio: {e}\n{traceback.format_exc()}")
            return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500
    else:
        return jsonify({"error": f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400

if __name__ == '__main__':
    load_service_settings()
    print(f"Starting Diarization & Transcription Service on http://{DIARIZATION_HOST}:{DIARIZATION_PORT}")
    app.run(host=DIARIZATION_HOST, port=DIARIZATION_PORT, debug=False)