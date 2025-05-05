import os
import json
import requests
import torch
from pyannote.audio import Pipeline

project_root = os.path.dirname(os.path.abspath(__file__))

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

    def summarize(self, text, temperature=0.7, max_tokens=4096):
        if not text:
            print("Error: No text provided for summarization.")
            return None

        try:
            template_path = os.path.join(project_root, "summarize_template.txt")
            with open(template_path, "r", encoding="utf-8") as f:
                user_prompt_template = f.read()
        except FileNotFoundError:
            print(f"Error: summarize_template.txt not found at {template_path}")
            return None
        except Exception as e:
            print(f"Error reading summarize_template.txt: {e}")
            return None

        headers = {"Content-Type": "application/json"}
        if self.use_auth and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.use_auth and not self.api_key:
             print("Warning: LLM API authentication is enabled, but no API key was provided.")


        payload = {
            "model": self.model_name,
            "messages": [
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