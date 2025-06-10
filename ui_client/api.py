"""Flask API for the UI client, handling user requests and orchestrating services."""
import os
import json
import tempfile
import uuid
import webbrowser
from threading import Timer
import io
import signal
from typing import Tuple

from flask import Flask, request, jsonify, render_template, send_file, Response
import markdown
from fpdf import FPDF

from .llm_client import LLMClientWrapper
from .diarization_client import DiarizationServiceClient

UI_HOST = '0.0.0.0'
UI_PORT = 5001
DIARIZATION_SERVICE_URL = os.environ.get("DIARIZATION_SERVICE_URL", "http://localhost:5002")

llm_wrapper = None
diarization_service_client = None
settings = None
project_root_ui_client = os.path.dirname(os.path.abspath(__file__))

def load_ui_settings_and_clients() -> None:
    """Loads settings and initializes service clients.

    Reads the main settings.json, initializes the LLMClientWrapper and
    the DiarizationServiceClient. Exits the application if essential
    settings are missing.
    """
    global llm_wrapper, diarization_service_client, settings
    
    settings_path = os.path.join(os.path.dirname(project_root_ui_client), "settings.json")

    try:
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
    except FileNotFoundError:
        print(f"Error: Settings file not found at {settings_path}")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from settings file {settings_path}")
        exit()

    llm_service_url = settings.get("llm_service_url")
    llm_api_endpoint = settings.get("llm_api_endpoint")
    llm_api_key = settings.get("llm_api_key", "")
    llm_api_auth = settings.get("llm_api_auth", False)
    llm_api_model = settings.get("llm_api_model")

    required_llm_settings = [llm_service_url, llm_api_endpoint, llm_api_model]
    if not all(required_llm_settings):
        print("Error: LLM settings (llm_service_url, llm_api_endpoint, llm_api_model) are missing. UI Client cannot start.")
        exit()

    print("Initializing LLM Client Wrapper for UI Service...")
    llm_wrapper = LLMClientWrapper(llm_service_url, llm_api_endpoint, llm_api_key, llm_api_auth, llm_api_model)
    print("LLM Client Wrapper initialized.")

    print(f"Initializing Diarization Service Client for UI Service (Target: {DIARIZATION_SERVICE_URL})...")
    diarization_service_client = DiarizationServiceClient(service_base_url=DIARIZATION_SERVICE_URL)
    health = diarization_service_client.check_health()
    print(f"Diarization Service Health: {health}")
    if not health or health.get("status") != "healthy" or health.get("pyannote_pipeline") != "OK" or health.get("gigaam_model") != "OK":
        print("Warning: Diarization service is not healthy or fully operational. UI client might face issues.")
    else:
        print("Diarization Service Client initialized and service is healthy.")


app = Flask(__name__, template_folder=os.path.join(os.path.dirname(project_root_ui_client), "templates"),
            static_folder=os.path.join(os.path.dirname(project_root_ui_client), "static"))

app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(project_root_ui_client), "uploads")
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 ** 3

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a'}

def allowed_file(filename: str) -> bool:
    """Checks if the uploaded file has an allowed extension.

    Args:
        filename: The name of the file to check.

    Returns:
        True if the file extension is in the allowed list, False otherwise.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def loading_page() -> Response:
    """Serves the initial loading page that checks service status.

    Returns:
        The rendered HTML for the loading page.
    """
    return render_template('loading.html')

@app.route('/app')
def application_page() -> Response:
    """Serves the main application page once services are ready.

    Returns:
        The rendered HTML for the main application page.
    """
    return render_template('index.html')

@app.route('/service_status', methods=['GET'])
def service_status_endpoint() -> Tuple[Response, int]:
    """Endpoint for the loading page to poll diarization and LLM service health.

    Returns:
        A tuple containing the Flask JSON response with the service statuses
        and the HTTP status code.
    """
    diarization_ready = False
    diarization_message = "Diarization client not initialized on UI server."
    diarization_details = None

    if diarization_service_client:
        ds_health = diarization_service_client.check_health()
        diarization_details = ds_health
        if (
            ds_health and
            ds_health.get("status") == "healthy" and
            ds_health.get("pyannote_pipeline") == "OK" and
            ds_health.get("gigaam_model") == "OK"
        ):
            diarization_ready = True
            diarization_message = "Diarization service is ready."
        else:
            diarization_message = "Diarization service is not yet ready or an error occurred."
            if ds_health and "error" in ds_health:
                diarization_message = f"Diarization service error: {ds_health.get('details', ds_health['error'])}"
            elif ds_health:
                pyannote_status = ds_health.get('pyannote_pipeline', 'Unknown')
                gigaam_status = ds_health.get('gigaam_model', 'Unknown')
                overall_status = ds_health.get('status', 'Unknown')
                diarization_message = f"Diarization service status: Overall: {overall_status}, Pyannote: {pyannote_status}, GigaAM: {gigaam_status}."
            else:
                diarization_message = "Diarization service health could not be determined."
    
    llm_ready = False
    llm_message = "LLM client not initialized on UI server."
    llm_details = None

    if llm_wrapper:
        llm_health = llm_wrapper.check_health()
        llm_details = llm_health
        if llm_health and llm_health.get("status") == "healthy":
            llm_ready = True
            llm_message = "LLM service is ready."
        else:
            llm_message = "LLM service is not yet ready or an error occurred."
            if llm_health and "error" in llm_health:
                llm_message = f"LLM service error: {llm_health.get('details', llm_health.get('error', 'Unknown LLM error'))}"
            elif llm_health and "status" in llm_health:
                 llm_message = f"LLM service status: {llm_health.get('status')}. Details: {llm_health.get('details', 'No details')}"
            else:
                llm_message = "LLM service health could not be determined."

    is_fully_ready = diarization_ready and llm_ready
    
    combined_message = f"Diarization: {diarization_message} LLM: {llm_message}"
    
    status_code = 200 if is_fully_ready else 503

    return jsonify({
        "ready": is_fully_ready,
        "message": combined_message,
        "diarization_status": {
            "ready": diarization_ready,
            "message": diarization_message,
            "details": diarization_details
        },
        "llm_status": {
            "ready": llm_ready,
            "message": llm_message,
            "details": llm_details
        }
    }), status_code

@app.route('/resources', methods=['GET'])
def get_resources() -> Tuple[Response, int]:
    """Provides resource usage data from the diarization service.

    Fetches CPU, memory, and GPU usage statistics from the health endpoint
    of the diarization service and returns them as a JSON response.

    Returns:
        A tuple containing the Flask JSON response and the HTTP status code.
    """
    if not diarization_service_client:
        return jsonify({
            "status": "error",
            "error_message": "Diarization service client not initialized in UI.",
            "cpu_percent": None,
            "mem_percent": None,
            "process_mem_mb": None,
            "gpu_utilization_percent": None,
            "gpu_mem_percent": None,
            "gpu_mem_used_mb": None,
            "gpu_mem_total_mb": None,
            "gpu_error": "Diarization service client not initialized in UI.",
            "pyannote_pipeline_status": "Unknown",
            "gigaam_model_status": "Unknown"
        }), 503

    ds_health_response = diarization_service_client.check_health()

    if not ds_health_response:
        return jsonify({
            "status": "error",
            "error_message": "Failed to get health/resource data from diarization service (no response or empty response).",
            "cpu_percent": None,
            "mem_percent": None,
            "process_mem_mb": None,
            "gpu_utilization_percent": None,
            "gpu_mem_percent": None,
            "gpu_mem_used_mb": None,
            "gpu_mem_total_mb": None,
            "gpu_error": "No response or empty response from diarization service health check.",
            "pyannote_pipeline_status": "Unknown",
            "gigaam_model_status": "Unknown"
        }), 503
        
    response_data = {
        "status": ds_health_response.get("status", "Unknown"),
        
        "cpu_percent": ds_health_response.get("system_cpu_percent"),
        "mem_percent": ds_health_response.get("system_mem_percent"),
        "process_mem_mb": ds_health_response.get("service_process_mem_mb"),
        
        "gpu_utilization_percent": ds_health_response.get("gpu_utilization_percent"),
        "gpu_mem_percent": ds_health_response.get("gpu_mem_percent"),
        "gpu_mem_used_mb": ds_health_response.get("gpu_mem_used_mb"),
        "gpu_mem_total_mb": ds_health_response.get("gpu_mem_total_mb"),
        "gpu_error": ds_health_response.get("gpu_error"),

        "pyannote_pipeline_status": ds_health_response.get("pyannote_pipeline"),
        "gigaam_model_status": ds_health_response.get("gigaam_model"),
        "error_message": ds_health_response.get("details") if "error" in ds_health_response else ds_health_response.get("error_message")
    }
    
    expected_keys = [
        "cpu_percent", "mem_percent", "process_mem_mb",
        "gpu_utilization_percent", "gpu_mem_percent", "gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_error"
    ]
    for key in expected_keys:
        if key not in response_data:
            response_data[key] = None
            
    if "error_message" in ds_health_response and ds_health_response.get("status") != "healthy" and not response_data.get("error_message"):
         response_data["error_message"] = ds_health_response.get("details", ds_health_response.get("error", "Unknown error from diarization service."))


    return jsonify(response_data)

@app.route('/summarize', methods=['POST'])
def summarize_meeting_endpoint() -> Tuple[Response, int]:
    """Handles audio file upload, transcription, and summarization.

    Receives an audio file, sends it to the diarization service for
    transcription, then sends the transcription to the LLM service for
    summarization.

    Returns:
        A tuple containing the Flask JSON response with the summary and
        the HTTP status code.
    """
    if not diarization_service_client:
        return jsonify({"error": "Diarization service client not initialized."}), 503
    if not llm_wrapper:
        return jsonify({"error": "LLM wrapper not initialized."}), 503

    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio_file part in the request"}), 400

    file = request.files['audio_file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        original_filename = file.filename
        
        temp_dir_ui = tempfile.mkdtemp(dir=app.config['UPLOAD_FOLDER'])
        temp_audio_path_ui = os.path.join(temp_dir_ui, f"{uuid.uuid4()}_{original_filename}")
        
        try:
            file.save(temp_audio_path_ui)
            print(f"UI Client: Audio file saved temporarily to: {temp_audio_path_ui}")

            print("UI Client: Calling Diarization Service...")
            diarization_response = diarization_service_client.process_audio_file(temp_audio_path_ui)

            if not diarization_response or "transcribed_segments" not in diarization_response:
                error_detail = "Unknown error from diarization service."
                if diarization_response and "error" in diarization_response:
                    error_detail = diarization_response["error"]
                    if "details" in diarization_response:
                         error_detail += f" Details: {diarization_response['details']}"
                print(f"UI Client: Error from Diarization Service: {error_detail}")
                return jsonify({"error": f"Failed during transcription/diarization: {error_detail}"}), 500

            transcribed_segments = diarization_response["transcribed_segments"]
            
            full_transcription_text = ""
            current_speaker = None
            speaker_utterance = ""
            for seg in transcribed_segments:
                if seg.get('speaker') == "ERROR" or not seg.get('text'):
                    continue
                if current_speaker is None:
                    current_speaker = seg['speaker']
                    speaker_utterance += seg['text'] + " "
                elif seg['speaker'] == current_speaker:
                    speaker_utterance += seg['text'] + " "
                else:
                    full_transcription_text += f"{current_speaker}: {speaker_utterance.strip()}\n"
                    current_speaker = seg['speaker']
                    speaker_utterance = seg['text'] + " "
            if current_speaker and speaker_utterance.strip():
                full_transcription_text += f"{current_speaker}: {speaker_utterance.strip()}\n"
            
            full_transcription_text = full_transcription_text.strip()
            print(f"UI Client: Received transcription from service. Length: {len(full_transcription_text)}")

            transcription_file_path = os.path.join(os.path.dirname(project_root_ui_client), "last_transcription.txt")
            try:
                with open(transcription_file_path, 'w', encoding='utf-8') as f:
                    f.write(full_transcription_text)
                print(f"UI Client: Transcription saved to {transcription_file_path}")
            except IOError as e:
                print(f"UI Client: Warning: Failed to save transcription to file: {e}")

            summary = ""
            if not full_transcription_text:
                print("UI Client: Transcription is empty, skipping summarization.")
                summary = "No content to summarize (transcription was empty)."
            else:
                print("UI Client: Calling LLM for summarization...")
                summary = llm_wrapper.summarize(full_transcription_text)
                if summary is None:
                    print("UI Client: Summarization failed.")
                    return jsonify({"error": "Failed during summarization by LLM"}), 500
                print("UI Client: Summarization successful.")

            summary_file_path = os.path.join(os.path.dirname(project_root_ui_client), "last_summary.txt")
            try:
                with open(summary_file_path, 'w', encoding='utf-8') as f:
                    f.write(summary)
                print(f"UI Client: Summary saved to {summary_file_path}")
            except IOError as e:
                print(f"UI Client: Warning: Failed to save summary to file: {e}")
            
            print("UI Client: Processing complete. Returning summary.")
            return jsonify({"summary": summary, "transcription_available": bool(full_transcription_text)})

        except Exception as e:
            print(f"UI Client: An unexpected error occurred in /summarize: {str(e)}")
            return jsonify({"error": f"An internal server error occurred in UI client: {str(e)}"}), 500
        finally:
            if os.path.exists(temp_dir_ui):
                if temp_audio_path_ui and os.path.exists(temp_audio_path_ui):
                    os.remove(temp_audio_path_ui)
                    print(f"UI Client: Removed temporary audio file: {temp_audio_path_ui}")
                try:
                    os.rmdir(temp_dir_ui)
                    print(f"UI Client: Removed temporary directory: {temp_dir_ui}")
                except OSError as e:
                     print(f"UI Client: Error removing temp_dir_ui {temp_dir_ui}: {e}")
    else:
        return jsonify({"error": f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400


def _create_and_send_pdf_ui(source_txt_filename: str, output_pdf_filename: str) -> Response:
    """Creates and sends a PDF from a given text file.

    Args:
        source_txt_filename: The name of the source .txt file (e.g., "last_summary.txt").
        output_pdf_filename: The desired filename for the downloaded PDF.

    Returns:
        A Flask response object to send the generated PDF file.
    """
    source_txt_path = os.path.join(os.path.dirname(project_root_ui_client), source_txt_filename)
    
    if not os.path.exists(source_txt_path):
        return jsonify({"error": f"{source_txt_filename} not found. Please generate the content first."}), 404

    try:
        with open(source_txt_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

        html_content = ""
        if source_txt_filename == "last_transcription.txt":
            html_content = file_content.replace('\n', '<br>')
        elif source_txt_filename == "last_summary.txt":
            html_content = markdown.markdown(file_content, extensions=['nl2br'])

        pdf = FPDF()
        pdf_font_family = 'Helvetica'

        font_dir = os.path.join(os.path.dirname(project_root_ui_client), "resources")
        font_path_regular = os.path.join(font_dir, 'DejaVuSans.ttf')
        font_path_bold = os.path.join(font_dir, 'DejaVuSans-Bold.ttf')

        try:
            if os.path.exists(font_path_regular):
                pdf.add_font('DejaVu', '', font_path_regular)
                pdf.add_font('DejaVu', 'I', font_path_regular)
                pdf.add_font('DejaVu', 'BI', font_path_regular)
                if os.path.exists(font_path_bold):
                    pdf.add_font('DejaVu', 'B', font_path_bold)
                else:
                    pdf.add_font('DejaVu', 'B', font_path_regular) 
                pdf.set_font('DejaVu', '', 12)
                pdf_font_family = 'DejaVu'
                print(f"UI Client: PDF using DejaVu font from {font_dir}")
            else:
                print(f"UI Client: Warning: DejaVuSans.ttf not found at {font_path_regular}. Falling back to Helvetica for PDF.")
                pdf.set_font('helvetica', '', 12)
        except Exception as font_error:
            print(f"UI Client: Error adding font to PDF: {font_error}. Falling back to Helvetica.")
            pdf.set_font('helvetica', '', 12)
        
        pdf.add_page()
        
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head><meta charset="UTF-8"></head>
        <body style="font-family: '{pdf_font_family}', sans-serif; line-height: 1.5;">
            {html_content}
        </body>
        </html>
        """
        try:
            pdf.write_html(styled_html)
        except Exception as e_write_html:
            print(f"UI Client: Error generating PDF from HTML content for {source_txt_filename} with fpdf2: {e_write_html}")
            pdf.set_font(pdf_font_family, '', 12)
            pdf.multi_cell(0, 10, file_content)


        pdf_output_bytes = pdf.output()
        pdf_buffer = io.BytesIO(pdf_output_bytes)
        pdf_buffer.seek(0)

        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=output_pdf_filename,
            mimetype='application/pdf'
        )

    except FileNotFoundError:
         return jsonify({"error": f"{os.path.basename(source_txt_path)} not found."}), 404
    except Exception as e:
        print(f"UI Client: Error processing PDF request for {output_pdf_filename}: {e}")
        return jsonify({"error": f"Failed to generate or send PDF from UI client: {str(e)}"}), 500

@app.route('/download/summary', methods=['GET'])
def download_summary_ui() -> Response:
    """Endpoint to download the meeting summary as a PDF.

    Returns:
        A Flask response to download the generated PDF file.
    """
    return _create_and_send_pdf_ui("last_summary.txt", "summary.pdf")

@app.route('/download/transcription', methods=['GET'])
def download_transcription_ui() -> Response:
    """Endpoint to download the full transcription as a PDF.

    Returns:
        A Flask response to download the generated PDF file.
    """
    return _create_and_send_pdf_ui("last_transcription.txt", "transcription.pdf")

@app.route('/shutdown', methods=['POST', 'GET'])
def shutdown_server() -> Tuple[str, int]:
    """Shuts down the Flask server gracefully.

    Returns:
        A tuple containing a confirmation message and the HTTP status code.
    """
    print("UI Client: Shutdown request received from browser. Sending SIGINT to self to terminate UI client process.")
    os.kill(os.getpid(), signal.SIGINT)
    return "UI Server is shutting down...", 200

def open_browser_ui() -> None:
    """Opens the application's URL in a new browser tab."""
    webbrowser.open_new(f"http://127.0.0.1:{UI_PORT}/")
    
if __name__ == '__main__':
    load_ui_settings_and_clients()
    if llm_wrapper and diarization_service_client:
        Timer(1, open_browser_ui).start()
        print(f"Starting UI Client Service on http://{UI_HOST}:{UI_PORT}")
        app.run(host=UI_HOST, port=UI_PORT, debug=False)
    else:
        print("UI Client Service could not start due to missing LLM wrapper or Diarization client.")