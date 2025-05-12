import os
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