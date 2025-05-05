import os
import json
import tempfile
import gigaam
import psutil
import pynvml
import webbrowser
from threading import Timer
import io
import uuid
from flask import Flask, request, jsonify, render_template, send_file
import markdown
from fpdf import FPDF
from models import PyannotePipelineWrapper, LLMClientWrapper
from main import diarize_and_transcribe, summarize_text_with_llm #, preprocess_audio

project_root = os.path.dirname(os.path.abspath(__file__))

HOST = '0.0.0.0'
PORT = 5001

try:
    with open(os.path.join(project_root, "settings.json"), 'r', encoding='utf-8') as f:
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
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024 * 10

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
        original_filename = file.filename
        _, ext = os.path.splitext(original_filename)
        unique_filename = f"{uuid.uuid4()}{ext}"
        temp_dir = tempfile.mkdtemp(dir=app.config['UPLOAD_FOLDER'])
        temp_audio_path = os.path.join(temp_dir, unique_filename)
        try:
            file.save(temp_audio_path)
            print(f"Audio file saved temporarily to: {temp_audio_path}")

            # cleaned_audio_path = os.path.join(temp_dir, "cleaned_" + filename)
            # print(f"Attempting to preprocess audio to: {cleaned_audio_path}")
            # preprocess_success = preprocess_audio(temp_audio_path, cleaned_audio_path)

            # if not preprocess_success:
            #     print("Audio preprocessing failed.")
            #     if os.path.exists(temp_audio_path):
            #         os.remove(temp_audio_path)
            #     return jsonify({"error": "Failed during audio preprocessing"}), 500
            # print("Audio preprocessing successful.")


            print("Starting transcription and diarization using wrappers on audio...")
            transcription = diarize_and_transcribe(
                temp_audio_path,
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
            # if 'cleaned_audio_path' in locals() and os.path.exists(cleaned_audio_path):
            #     os.remove(cleaned_audio_path)
            #     print(f"Removed temporary cleaned audio file: {cleaned_audio_path}")
            #     print(f"Removed temporary audio file: {temp_audio_path}")
            if os.path.exists(temp_dir):
                 os.rmdir(temp_dir)
                 print(f"Removed temporary directory: {temp_dir}")


    else:
        return jsonify({"error": "Invalid file type. Only .wav files are allowed."}), 400

def _create_and_send_pdf(source_txt_path, output_pdf_filename):
    """Reads a text file, converts its content to PDF, and sends it."""
    if not os.path.exists(source_txt_path):
        return jsonify({"error": f"{os.path.basename(source_txt_path)} not found. Please generate the content first."}), 404

    try:
        with open(source_txt_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

        if source_txt_path.endswith("last_transcription.txt"):
            html_content = file_content.replace('\n', '<br>')
        elif source_txt_path.endswith("last_summary.txt"):
            html_content = markdown.markdown(file_content, extensions=['nl2br'])

        pdf = FPDF()
        pdf_font_family = 'Helvetica'

        try:
            font_path_regular = os.path.join(project_root, 'DejaVuSans.ttf')
            font_path_bold = os.path.join(project_root, 'DejaVuSans-Bold.ttf')

            font_regular_exists = os.path.exists(font_path_regular)
            font_bold_exists = os.path.exists(font_path_bold)

            if font_regular_exists:
                pdf.add_font('DejaVu', '', font_path_regular)
                pdf.add_font('DejaVu', 'I', font_path_regular)
                pdf.add_font('DejaVu', 'BI', font_path_regular)
                pdf.set_font('DejaVu', '', 12)
                pdf_font_family = 'DejaVu'
                print(f"Loaded {font_path_regular} for regular, italic, bold-italic styles.")
                if font_bold_exists:
                    pdf.add_font('DejaVu', 'B', font_path_bold)
                    print(f"Loaded {font_path_bold} for bold style.")
                else:
                    pdf.add_font('DejaVu', 'B', font_path_regular)
                    print(f"Warning: {font_path_bold} not found. Using regular font for bold style.")
            elif font_bold_exists:
                pdf.add_font('DejaVu', '', font_path_bold)
                pdf.add_font('DejaVu', 'B', font_path_bold)
                pdf.add_font('DejaVu', 'I', font_path_bold)
                pdf.add_font('DejaVu', 'BI', font_path_bold)
                pdf.set_font('DejaVu', '', 12)
                pdf_font_family = 'DejaVu'
                print(f"Warning: {font_path_regular} not found. Using {font_path_bold} for all styles.")
            else:
                print(f"Warning: Neither {font_path_regular} nor {font_path_bold} found. Falling back to Helvetica.")
                pdf.set_font('helvetica', '', 12)

        except Exception as font_error:
            print(f"Error adding font: {font_error}. Falling back to default font.")
            pdf.set_font('helvetica', '', 12)
            pdf_font_family = 'Helvetica'

        pdf.add_page()

        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head><meta charset="UTF-8"></head>
        <body style="font-family: '{pdf_font_family}', sans-serif; line-height: 5.0;">
            {html_content}
        </body>
        </html>
        """

        try:
            pdf.write_html(styled_html)
        except Exception as e:
            print(f"Error generating PDF from HTML content for {source_txt_path} with fpdf2: {e}")
            if "Character" in str(e) and "outside the range" in str(e):
                print(f"Unicode character error in PDF generation: {e}. Ensure a Unicode font is correctly loaded and set.")
            return jsonify({"error": f"Failed to generate PDF from content using fpdf2: {e}"}), 500

        try:
            pdf_output_bytes = pdf.output()
            pdf_buffer = io.BytesIO(pdf_output_bytes)
            pdf_buffer.seek(0)
        except Exception as e:
            print(f"Error getting PDF output bytes: {e}")
            return jsonify({"error": f"Failed to finalize PDF generation: {e}"}), 500

        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=output_pdf_filename,
            mimetype='application/pdf'
        )

    except FileNotFoundError:
         return jsonify({"error": f"{os.path.basename(source_txt_path)} not found."}), 404
    except Exception as e:
        print(f"Error processing request for {output_pdf_filename}: {e}")
        return jsonify({"error": f"Failed to generate or send PDF: {e}"}), 500

@app.route('/download/summary', methods=['GET'])
def download_summary():
    """Endpoint to download the last generated summary as a PDF."""
    return _create_and_send_pdf("last_summary.txt", "summary.pdf")

@app.route('/download/transcription', methods=['GET'])
def download_transcription():
    """Endpoint to download the last generated transcription as a PDF."""
    return _create_and_send_pdf("last_transcription.txt", "transcription.pdf")


def open_browser():
    webbrowser.open_new(f"http://127.0.0.1:{PORT}")
    
if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(host=HOST, port=PORT)