import os
import torch
import noisereduce as nr
import soundfile as sf
import numpy as np
from pyannote.core import Segment, Annotation
from .models import PyannotePipelineWrapper
from typing import List, Tuple
from gigaam.preprocess import load_audio, SAMPLE_RATE
from gigaam.vad_utils import audiosegment_to_tensor
from pydub import AudioSegment

def preprocess_audio(input_path, output_path):
    """
    Reads an audio file, applies noise reduction, and saves the cleaned audio.
    Returns True on success, False on failure.
    """
    try:
        print(f"Preprocessing audio: {input_path}")
        audio_data, sample_rate = sf.read(input_path)

        if audio_data.ndim > 1 and audio_data.shape[1] > 1:
             print("Audio is stereo, converting to mono by averaging channels.")
             audio_data = np.mean(audio_data, axis=1)

        print("Applying noise reduction...")
        reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=sample_rate, stationary=True, prop_decrease=1.0)

        sf.write(output_path, reduced_noise_audio, sample_rate)
        print(f"Cleaned audio saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error during audio preprocessing: {e}")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        return False

def segment_audio_from_diarization(
    wav_tensor: torch.Tensor,
    sample_rate: int,
    diarization: Annotation,
) -> Tuple[List[torch.Tensor], List[Tuple[float, float]]]:
    """Segments audio tensor based on Pyannote diarization annotation."""
    audio = AudioSegment(
        wav_tensor.numpy().tobytes(),
        frame_rate=sample_rate,
        sample_width=wav_tensor.dtype.itemsize,
        channels=1,
    )
    segments: List[torch.Tensor] = []
    boundaries: List[Tuple[float, float]] = []

    for segment_info in diarization.get_timeline().support():
        start = segment_info.start
        end = segment_info.end

        audio_duration_sec = len(audio) / 1000.0
        start = max(0.0, start)
        end = min(audio_duration_sec, end)

        if start >= end:
            continue

        start_ms = int(start * 1000)
        end_ms = int(end * 1000)

        audio_segment_chunk = audio[start_ms:end_ms]

        if len(audio_segment_chunk) == 0:
            print(f"Warning: Skipping empty audio segment from {start:.2f}s to {end:.2f}s")
            continue

        try:
            segment_tensor = audiosegment_to_tensor(audio_segment_chunk)
            segments.append(segment_tensor)
            boundaries.append((start, end))
        except Exception as e:
            print(f"Warning: Error converting segment {start:.2f}s-{end:.2f}s to tensor: {e}")

    return segments, boundaries

def diarize_and_transcribe_audio_segments(audio_path: str, pyannote_wrapper: PyannotePipelineWrapper, gigaam_model):
    """
    Performs speaker diarization using Pyannote and then transcribes
    each speaker segment using the GigaAM model.
    Returns a list of transcribed segments with speaker, start, end, and text.
    """
    if pyannote_wrapper.pipeline is None:
        print("Error: Pyannote pipeline within wrapper was not loaded successfully.")
        return None, "Pyannote pipeline not loaded"
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return None, f"Audio file not found at {audio_path}"
    if gigaam_model is None:
         print("Error: GigaAM model was not provided or loaded.")
         return None, "GigaAM model not loaded"

    print("Performing diarization...")
    diarization = pyannote_wrapper.diarize(audio_path)
    if diarization is None:
        print("Diarization failed.")
        return None, "Diarization failed"
    print("Diarization complete.")

    print("Loading audio tensor for transcription...")
    try:
        wav_tensor = load_audio(audio_path, return_format="int")
    except Exception as e:
        print(f"Error loading audio tensor from {audio_path}: {e}")
        return None, f"Error loading audio tensor: {e}"

    print("Segmenting audio based on diarization...")
    segments, boundaries = segment_audio_from_diarization(
        wav_tensor, SAMPLE_RATE, diarization
    )

    if not segments:
        print("Warning: No audio segments generated from diarization.")
        return [], None

    print(f"Transcribing {len(segments)} segments using GigaAM...")
    transcribed_results = []
    for i, (segment_tensor, (start, end)) in enumerate(zip(segments, boundaries)):
        print(f"  Transcribing segment {i+1}/{len(segments)} ({start:.2f}s - {end:.2f}s)")
        try:
            wav = segment_tensor.to(gigaam_model._device).unsqueeze(0).to(gigaam_model._dtype)
            length = torch.full([1], wav.shape[-1], device=gigaam_model._device)
            encoded, encoded_len = gigaam_model.forward(wav, length)
            result_text = gigaam_model.decoding.decode(gigaam_model.head, encoded, encoded_len)[0]

            segment_for_speaker_lookup = Segment(start, end)
            try:
                speakers = diarization.crop(segment_for_speaker_lookup).labels()
                if speakers:
                    speaker = speakers[0]
                else:
                    overlapping_speakers = diarization.overlapping(segment_for_speaker_lookup).labels()
                    if overlapping_speakers:
                        speaker = overlapping_speakers[0]
                    else:
                        speaker = "UNKNOWN"
                        print(f"Warning: No speaker found for segment {start:.2f}s - {end:.2f}s")
            except Exception as e_speaker:
                 print(f"Error during speaker lookup for segment {start:.2f}s - {end:.2f}s: {e_speaker}")
                 speaker = "UNKNOWN"

            transcribed_results.append({
                "speaker": speaker,
                "text": result_text.strip(),
                "start_time": round(start, 3),
                "end_time": round(end, 3)
            })
        except Exception as e_transcribe:
            print(f"Error transcribing segment {i+1} ({start:.2f}s - {end:.2f}s): {e_transcribe}")
            transcribed_results.append({
                "speaker": "ERROR",
                "text": "[Transcription Error]",
                "start_time": round(start, 3),
                "end_time": round(end, 3)
            })

    print("Transcription and alignment complete.")
    return transcribed_results, None