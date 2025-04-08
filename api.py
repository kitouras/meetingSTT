import os
import json
import tempfile
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from models import VoskModelWrapper, PyannotePipelineWrapper, LLMClientWrapper
from main import diarize_and_transcribe, summarize_text_with_llm

try:
    with open("./settings.json", 'r', encoding='utf-8') as f:
        settings = json.load(f)
except FileNotFoundError:
    print("Error: Settings file not found at ./settings.json")
    exit()
except json.JSONDecodeError:
    print("Error: Could not decode JSON from settings file ./settings.json")
    exit()

vosk_model_path = settings.get("vosk_model_path")
pyannote_model_name = settings.get("pyannote_model_name")
hugging_face_token = settings.get("hugging_face_token")
llm_api_endpoint = settings.get("llm_api_endpoint")
llm_api_key = settings.get("llm_api_key", "")
llm_api_auth = settings.get("llm_api_auth", False)
llm_api_model = settings.get("llm_api_model")
vosk_log_level = settings.get("vosk_log_level", -1)
chunk_duration_seconds = settings.get("vosk_chunk_duration", 30)
print("Initializing models for API...")
vosk_wrapper = VoskModelWrapper(vosk_model_path, vosk_log_level)
pyannote_wrapper = PyannotePipelineWrapper(pyannote_model_name, hugging_face_token)
llm_wrapper = LLMClientWrapper(llm_api_endpoint, llm_api_key, llm_api_auth, llm_api_model)
print("Models initialized for API.")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'wav'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/summarize', methods=['POST'])
def summarize_audio():
    if vosk_wrapper.model is None:
        return jsonify({"error": "Vosk model failed to load on server startup. Cannot process request."}), 503
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

            print("Starting transcription and diarization using wrappers...")
            transcription = diarize_and_transcribe(
                temp_audio_path,
                vosk_wrapper,
                pyannote_wrapper,
                chunk_duration_seconds
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
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                print(f"Removed temporary audio file: {temp_audio_path}")
            if os.path.exists(temp_dir):
                 os.rmdir(temp_dir)
                 print(f"Removed temporary directory: {temp_dir}")


    else:
        return jsonify({"error": "Invalid file type. Only .wav files are allowed."}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)