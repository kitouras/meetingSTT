import os
import json
import tempfile
import gigaam
import psutil
import pynvml
import webbrowser
from threading import Timer
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from models import PyannotePipelineWrapper, LLMClientWrapper
from main import diarize_and_transcribe, summarize_text_with_llm, preprocess_audio

HOST = '127.0.0.1'
PORT = 5001

try:
    with open("./settings.json", 'r', encoding='utf-8') as f:
        settings = json.load(f)
except FileNotFoundError:
    print("Error: Settings file not found at ./settings.json")
    exit()
except json.JSONDecodeError:
    print("Error: Could not decode JSON from settings file ./settings.json")
    exit()

pyannote_model_name = settings.get("pyannote_model_name")
hugging_face_token = settings.get("hugging_face_token")
llm_api_endpoint = settings.get("llm_api_endpoint")
llm_api_key = settings.get("llm_api_key", "")
llm_api_auth = settings.get("llm_api_auth", False)
llm_api_model = settings.get("llm_api_model")
print("Initializing models for API...")
pyannote_wrapper = PyannotePipelineWrapper(pyannote_model_name, hugging_face_token)
llm_wrapper = LLMClientWrapper(llm_api_endpoint, llm_api_key, llm_api_auth, llm_api_model)

print("Initializing GigaAM model for API...")
try:
    gigaam_model = gigaam.load_model("rnnt")
    print("GigaAM model initialized for API.")
except Exception as e:
    print(f"Error initializing GigaAM model for API: {e}")
    gigaam_model = None

print("Core models initialized for API.")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'wav'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/resources', methods=['GET'])
def get_resources():
    """Endpoint to get current server resource usage, including GPU if available."""
    process = psutil.Process(os.getpid())
    cpu_percent = psutil.cpu_percent(interval=0.1)
    mem_percent = psutil.virtual_memory().percent
    process_mem_mb = process.memory_info().rss / (1024 * 1024)

    gpu_utilization = None
    gpu_mem_percent = None
    gpu_mem_used_mb = None
    gpu_mem_total_mb = None

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_utilization = utilization.gpu
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_mem_percent = round((mem_info.used / mem_info.total) * 100, 2)
        gpu_mem_used_mb = round(mem_info.used / (1024 * 1024), 2)
        gpu_mem_total_mb = round(mem_info.total / (1024 * 1024), 2)
        pynvml.nvmlShutdown()
    except pynvml.NVMLError as e:
        print(f"NVML Error: {e}. GPU info not available.")
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass
    except Exception as e:
        print(f"Unexpected error getting GPU info: {e}")
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass

    return jsonify({
        "cpu_percent": cpu_percent,
        "mem_percent": mem_percent,
        "process_mem_mb": round(process_mem_mb, 2),
        "gpu_utilization_percent": gpu_utilization,
        "gpu_mem_percent": gpu_mem_percent,
        "gpu_mem_used_mb": gpu_mem_used_mb,
        "gpu_mem_total_mb": gpu_mem_total_mb
    })

@app.route('/summarize', methods=['POST'])
def summarize_audio():
    if gigaam_model is None:
        return jsonify({"error": "GigaAM model failed to load on server startup. Cannot process request."}), 503
    if pyannote_wrapper.pipeline is None:
        return jsonify({"error": "Pyannote pipeline failed to load on server startup. Cannot process request."}), 503

    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file part in the request"}), 400

    file = request.files['audio_file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp(dir=app.config['UPLOAD_FOLDER'])
        temp_audio_path = os.path.join(temp_dir, filename)
        try:
            file.save(temp_audio_path)
            print(f"Audio file saved temporarily to: {temp_audio_path}")

            cleaned_audio_path = os.path.join(temp_dir, "cleaned_" + filename)
            print(f"Attempting to preprocess audio to: {cleaned_audio_path}")
            preprocess_success = preprocess_audio(temp_audio_path, cleaned_audio_path)

            if not preprocess_success:
                print("Audio preprocessing failed.")
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                return jsonify({"error": "Failed during audio preprocessing"}), 500
            print("Audio preprocessing successful.")


            print("Starting transcription and diarization using wrappers on cleaned audio...")
            transcription = diarize_and_transcribe(
                cleaned_audio_path,
                pyannote_wrapper,
                gigaam_model
            )
            print(transcription)
            
            transcription_file = "last_transcription.txt"
            try:
                with open(transcription_file, 'w', encoding='utf-8') as f:
                    f.write(transcription)
                print(f"Transcription saved to {transcription_file}")
            except IOError as e:
                print(f"Warning: Failed to save transcription to file: {e}")

            if transcription is None:
                return jsonify({"error": "Failed during transcription/diarization"}), 500
            elif not transcription:
                 print("Transcription resulted in empty text.")
                 summary = ""
            else:
                print("Starting summarization using wrapper...")
                summary = summarize_text_with_llm(
                    transcription,
                    llm_wrapper
                )
                print(summary)

                if summary is None:
                    return jsonify({"error": "Failed during summarization"}), 500
                
                summary_file = "last_summary.txt"
                try:
                    with open(summary_file, 'w', encoding='utf-8') as f:
                        f.write(summary)
                    print(f"Summary saved to {summary_file}")
                except IOError as e:
                    print(f"Warning: Failed to save summary to file: {e}")

            print("Processing complete. Returning summary.")
            return jsonify({"summary": summary})

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return jsonify({"error": f"An internal server error occurred: {e}"}), 500
        finally:
            if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                print(f"Removed temporary audio file: {temp_audio_path}")
            if 'cleaned_audio_path' in locals() and os.path.exists(cleaned_audio_path):
                os.remove(cleaned_audio_path)
                print(f"Removed temporary cleaned audio file: {cleaned_audio_path}")
                print(f"Removed temporary audio file: {temp_audio_path}")
            if os.path.exists(temp_dir):
                 os.rmdir(temp_dir)
                 print(f"Removed temporary directory: {temp_dir}")


    else:
        return jsonify({"error": "Invalid file type. Only .wav files are allowed."}), 400

def open_browser():
    webbrowser.open_new(f"http://127.0.0.1:{PORT}")
    
if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(host=HOST, port=PORT)