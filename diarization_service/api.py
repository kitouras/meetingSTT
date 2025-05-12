import os
import json
import tempfile
import gigaam
import psutil
try:
    import pynvml
except ImportError:
    pynvml = None
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from .models import PyannotePipelineWrapper
from .transcription import diarize_and_transcribe_audio_segments

DIARIZATION_HOST = '0.0.0.0'
DIARIZATION_PORT = 5002
UPLOAD_FOLDER = 'diarization_uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a'}

pyannote_wrapper = None
gigaam_model = None
settings = None

def load_service_settings_and_models():
    global pyannote_wrapper, gigaam_model, settings
    
    project_root_diarization_service = os.path.dirname(os.path.abspath(__file__))
    settings_path = os.path.join(project_root_diarization_service, "..", "settings.json")

    try:
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
    except FileNotFoundError:
        print(f"Error: Settings file not found at {settings_path}")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from settings file {settings_path}")
        exit()

    pyannote_model_name = settings.get("pyannote_model_name")
    hugging_face_token = settings.get("hugging_face_token")

    if not pyannote_model_name:
        print("Error: 'pyannote_model_name' not found in settings.json. Diarization service cannot start.")
        exit()

    print("Initializing Pyannote pipeline for Diarization Service...")
    pyannote_wrapper = PyannotePipelineWrapper(pyannote_model_name, hugging_face_token)
    if pyannote_wrapper.pipeline is None:
        print("Pyannote pipeline failed to load. Diarization service cannot start.")
        exit()
    print("Pyannote pipeline initialized for Diarization Service.")

    print("Initializing GigaAM model for Diarization Service...")
    try:
        gigaam_model = gigaam.load_model("rnnt")
        print("GigaAM model initialized for Diarization Service.")
    except Exception as e:
        print(f"Error initializing GigaAM model for Diarization Service: {e}")
        gigaam_model = None
        print("Warning: GigaAM model failed to load. Transcription will not be available.")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 ** 3

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    pyannote_ok = pyannote_wrapper and pyannote_wrapper.pipeline is not None
    gigaam_ok = gigaam_model is not None
    
    overall_status = "healthy"
    if not pyannote_ok or not gigaam_ok:
        overall_status = "degraded"
    if not pyannote_ok and not gigaam_ok:
        overall_status = "error"

    process = psutil.Process(os.getpid())
    cpu_percent = psutil.cpu_percent(interval=0.1)
    service_mem_percent = psutil.virtual_memory().percent
    service_process_mem_mb = process.memory_info().rss / (1024 * 1024)

    gpu_utilization = None
    gpu_mem_percent = None
    gpu_mem_used_mb = None
    gpu_mem_total_mb = None
    gpu_error = None

    if pynvml:
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                utilization_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization = utilization_rates.gpu
                
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_mem_total_mb = round(mem_info.total / (1024 * 1024), 2)
                gpu_mem_used_mb = round(mem_info.used / (1024 * 1024), 2)
                if gpu_mem_total_mb > 0:
                    gpu_mem_percent = round((gpu_mem_used_mb / gpu_mem_total_mb) * 100, 2)
                else:
                    gpu_mem_percent = 0
            else:
                gpu_error = "No NVIDIA GPU detected by NVML."
            
            pynvml.nvmlShutdown()
        except pynvml.NVMLError as e:
            gpu_error = f"NVML Error: {str(e)}. GPU info not available."
            print(gpu_error)
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass
        except Exception as e_gpu:
            gpu_error = f"Unexpected error getting GPU info: {str(e_gpu)}"
            print(gpu_error)
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass
    else:
        gpu_error = "pynvml library not installed or import failed. GPU info not available."

    return jsonify({
        "status": overall_status,
        "pyannote_pipeline": "OK" if pyannote_ok else "Error: Not loaded",
        "gigaam_model": "OK" if gigaam_ok else "Error: Not loaded",
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
def process_audio_endpoint():
    if not pyannote_wrapper or pyannote_wrapper.pipeline is None:
        return jsonify({"error": "Pyannote pipeline not initialized on server."}), 503
    if gigaam_model is None:
        return jsonify({"error": "GigaAM model not initialized on server."}), 503

    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio_file part in the request"}), 400

    file = request.files['audio_file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp(dir=app.config['UPLOAD_FOLDER'])
        temp_audio_path = os.path.join(temp_dir, original_filename)
        cleaned_audio_path = None

        try:
            file.save(temp_audio_path)
            print(f"Audio file saved temporarily to: {temp_audio_path}")

            # cleaned_audio_filename = "cleaned_" + original_filename
            # cleaned_audio_path = os.path.join(temp_dir, cleaned_audio_filename)
            
            # print(f"Attempting to preprocess audio to: {cleaned_audio_path}")
            # preprocess_success = preprocess_audio(temp_audio_path, cleaned_audio_path)

            # if not preprocess_success:
            #     print("Audio preprocessing failed.")
            #     print(f"Warning: Audio preprocessing failed. Attempting to use original file: {temp_audio_path}")
            #     target_audio_path_for_diarization = temp_audio_path
            # else:
            #     print("Audio preprocessing successful.")
            #     target_audio_path_for_diarization = cleaned_audio_path


            print(f"Starting diarization and transcription on: {temp_audio_path}")
            transcribed_segments, error_message = diarize_and_transcribe_audio_segments(
                temp_audio_path,
                pyannote_wrapper,
                gigaam_model
            )

            if error_message:
                return jsonify({"error": f"Failed during transcription/diarization: {error_message}"}), 500
            
            if transcribed_segments is None:
                 return jsonify({"error": "An unknown error occurred during transcription/diarization"}), 500

            print("Processing complete. Returning transcribed segments.")
            return jsonify({
                "message": "Audio processed successfully.",
                "transcribed_segments": transcribed_segments
            }), 200

        except Exception as e:
            print(f"An unexpected error occurred in /process_audio: {e}")
            return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500
        finally:
            if os.path.exists(temp_dir):
                if temp_audio_path and os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                    print(f"Removed temporary audio file: {temp_audio_path}")
                if cleaned_audio_path and os.path.exists(cleaned_audio_path):
                    os.remove(cleaned_audio_path)
                    print(f"Removed temporary cleaned audio file: {cleaned_audio_path}")
                try:
                    os.rmdir(temp_dir)
                    print(f"Removed temporary directory: {temp_dir}")
                except OSError as e:
                    print(f"Error removing temp directory {temp_dir}: {e} (might not be empty if other files were created)")
    else:
        return jsonify({"error": f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400

if __name__ == '__main__':
    load_service_settings_and_models()
    if pyannote_wrapper and pyannote_wrapper.pipeline and gigaam_model:
        print(f"Starting Diarization & Transcription Service on http://{DIARIZATION_HOST}:{DIARIZATION_PORT}")
        app.run(host=DIARIZATION_HOST, port=DIARIZATION_PORT, debug=False)
    else:
        print("Diarization & Transcription Service could not start due to model loading issues.")