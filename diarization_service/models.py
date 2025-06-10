"""Wrappers for machine learning models used in the service."""
import os
from typing import Optional

import torch
from pyannote.audio import Pipeline
from pyannote.core import Annotation

project_root = os.path.dirname(os.path.abspath(__file__))


class PyannotePipelineWrapper:
    """A wrapper for the pyannote.audio Pipeline to handle loading and execution."""

    def __init__(self, model_name: str, auth_token: Optional[str] = None) -> None:
        """Initializes the wrapper and loads the pyannote pipeline.

        Args:
            model_name: The name of the pyannote pipeline model to load.
            auth_token: The Hugging Face authentication token, if required.
        """
        self.model_name = model_name
        self.auth_token = auth_token
        self.pipeline: Optional[Pipeline] = self._load_pipeline()

    def _load_pipeline(self) -> Optional[Pipeline]:
        """Loads the pyannote pipeline model.

        Handles moving the model to GPU if available.

        Returns:
            The loaded pyannote.audio.Pipeline object, or None if loading fails.
        """
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

    def diarize(self, audio_path: str) -> Optional[Annotation]:
        """Performs speaker diarization on an audio file.

        Args:
            audio_path: The path to the audio file to be processed.

        Returns:
            A pyannote.core.Annotation object containing the diarization results,
            or None if an error occurs.
        """
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