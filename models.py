import os
import json
import wave
import math
import requests
import torch
from vosk import Model, KaldiRecognizer, SetLogLevel
from pyannote.audio import Pipeline
from pyannote.core import Segment

class VoskModelWrapper:
    def __init__(self, model_path, log_level=-1):
        self.model_path = model_path
        self.log_level = log_level
        self.model = self._load_model()
        SetLogLevel(self.log_level)

    def _load_model(self):
        print(f"Loading Vosk model from: {self.model_path}")
        if not os.path.exists(self.model_path):
            print(f"FATAL ERROR: Vosk model path does not exist: {self.model_path}")
            return None
        try:
            model = Model(self.model_path)
            print("Vosk model loaded successfully.")
            return model
        except Exception as e:
            print(f"FATAL ERROR: Could not load Vosk model: {e}")
            return None

    def transcribe_audio_file(self, audio_path, chunk_duration_sec=30, diarization=None):
        if not self.model:
            print("Error: Vosk model not loaded.")
            return None
        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found at {audio_path}")
            return None

        try:
            wf = wave.open(audio_path, "rb")
        except Exception as e:
            print(f"Error opening audio file: {e}")
            return None

        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            print("Audio file must be WAV format mono PCM.")
            wf.close()
            return None

        sample_rate = wf.getframerate()
        rec = KaldiRecognizer(self.model, sample_rate)
        rec.SetWords(True)
        rec.SetPartialWords(False)

        chunk_size_frames = chunk_duration_sec * sample_rate
        total_frames = wf.getnframes()
        all_words = []

        if diarization:
            print("Processing audio with Vosk using diarization-informed chunks...")
            current_pos = 0
            while current_pos < total_frames:
                next_boundary = min(current_pos + chunk_size_frames, total_frames)
                boundary_time = next_boundary / sample_rate
                
                overlapping_segments = diarization.crop(Segment(boundary_time - 0.1, boundary_time + 0.1))
                
                if overlapping_segments:
                    next_silence = None
                    for segment in diarization.itersegments():
                        if segment.start > boundary_time and not diarization.crop(segment):
                            next_silence = segment.start
                            break
                    
                    if next_silence:
                        next_boundary = min(int(next_silence * sample_rate), total_frames)
                
                frames_to_read = next_boundary - current_pos
                data = wf.readframes(frames_to_read)
                if len(data) == 0:
                    break
                    
                if rec.AcceptWaveform(data):
                    result_json = rec.Result()
                    result = json.loads(result_json)
                    if 'result' in result:
                        all_words.extend(result['result'])
                
                current_pos = next_boundary
        else:
            num_chunks = math.ceil(total_frames / chunk_size_frames)
            print(f"Processing audio with Vosk in {num_chunks} fixed chunks...")
            for i in range(num_chunks):
                data = wf.readframes(chunk_size_frames)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result_json = rec.Result()
                    result = json.loads(result_json)
                    if 'result' in result:
                        all_words.extend(result['result'])

        final_result_json = rec.FinalResult()
        final_result = json.loads(final_result_json)
        if 'result' in final_result:
             all_words.extend(final_result['result'])

        print("Vosk transcription processing complete.")
        wf.close()

        if not all_words:
            print("Warning: No words were transcribed by Vosk.")
            return []

        return all_words

    def transcribe_segment(self, audio_bytes, sample_rate):
        if not self.model:
            print("Error: Vosk model not loaded.")
            return None
        if not audio_bytes:
            print("Warning: No audio bytes provided for transcription.")
            return []

        rec = KaldiRecognizer(self.model, sample_rate)
        rec.SetWords(True)
        rec.SetPartialWords(False)

        segment_words = []
        if rec.AcceptWaveform(audio_bytes):
            result_json = rec.Result()
            result = json.loads(result_json)
            if 'result' in result:
                segment_words.extend(result['result'])
        else:
            final_result_json = rec.FinalResult()
            final_result = json.loads(final_result_json)
            if 'result' in final_result:
                 segment_words.extend(final_result['result'])

        return segment_words


class PyannotePipelineWrapper:
    def __init__(self, model_name, auth_token=None):
        self.model_name = model_name
        self.auth_token = auth_token
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self):
        print(f"Loading Pyannote pipeline: {self.model_name}")
        try:
            pipeline_args = {"use_auth_token": self.auth_token} if self.auth_token else {}
            pipeline = Pipeline.from_pretrained(self.model_name, **pipeline_args)

            if torch.cuda.is_available():
                pipeline = pipeline.to(torch.device("cuda"))
                print("Pyannote pipeline moved to GPU.")
            else:
                 print("Pyannote pipeline running on CPU.")
            print("Pyannote pipeline loaded successfully.")
            return pipeline
        except Exception as e:
            print(f"Error loading Pyannote pipeline: {e}")
            return None

    def diarize(self, audio_path):
        if not self.pipeline:
            print("Error: Pyannote pipeline not loaded.")
            return None
        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found at {audio_path}")
            return None

        print("Starting speaker diarization...")
        try:
            diarization = self.pipeline(audio_path)
            print("Speaker diarization complete.")
            return diarization
        except Exception as e:
            print(f"Error during diarization: {e}")
            return None


class LLMClientWrapper:
    def __init__(self, api_endpoint, api_key=None, use_auth=False, model_name="gemma-3-4b-it"):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.use_auth = use_auth
        self.model_name = model_name

    def summarize(self, text, system_prompt="You are a helpful assistant that summarizes transcripts.",
                  user_prompt_template="Please summarize the following transcript, give the answer in Russian:\n\n{}", temperature=0.7, max_tokens=4096):
        if not text:
            print("Error: No text provided for summarization.")
            return None

        headers = {"Content-Type": "application/json"}
        if self.use_auth and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.use_auth and not self.api_key:
             print("Warning: LLM API authentication is enabled, but no API key was provided.")


        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_template.format(text)}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        print(f"Sending text to LLM ({self.model_name}) at {self.api_endpoint} for summarization...")
        try:
            response = requests.post(self.api_endpoint, headers=headers, json=payload)
            response.raise_for_status()

            result = response.json()
            summary = result.get('choices', [{}])[0].get('message', {}).get('content', '')

            if summary:
                print("Summarization complete.")
                return summary.strip()
            else:
                print("Error: Could not extract summary from LLM response.")
                print("LLM Response:", result)
                return None

        except requests.exceptions.RequestException as e:
            print(f"Error calling LLM API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                 print(f"LLM Response Status Code: {e.response.status_code}")
                 print(f"LLM Response Text: {e.response.text}")
            return None
        except json.JSONDecodeError:
            print("Error: Could not decode JSON response from LLM API.")
            print("Raw Response:", response.text)
            return None
        except Exception as e:
             print(f"An unexpected error occurred during summarization: {e}")
             return None