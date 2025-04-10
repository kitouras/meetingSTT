import os
import json
import argparse
import torch
import gigaam
import noisereduce as nr
import soundfile as sf
import numpy as np
import tempfile
from pyannote.core import Segment, Annotation
from models import PyannotePipelineWrapper, LLMClientWrapper
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

    for segment in diarization.get_timeline().support():
        start = segment.start
        end = segment.end

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


def diarize_and_transcribe(audio_path, pyannote_wrapper: PyannotePipelineWrapper, gigaam_model):
    """
    Performs speaker diarization using Pyannote and then transcribes
    each speaker segment using the GigaAM model.
    Returns a formatted string with speaker labels and transcriptions.
    """
    if pyannote_wrapper.pipeline is None:
        print("Error: Pyannote pipeline within wrapper was not loaded successfully.")
        return None
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return None
    if gigaam_model is None:
         print("Error: GigaAM model was not provided or loaded.")
         return None

    print("Performing diarization...")
    diarization = pyannote_wrapper.diarize(audio_path)
    if diarization is None:
        print("Diarization failed.")
        return None
    print("Diarization complete.")

    print("Loading audio tensor for transcription...")
    try:
        wav_tensor = load_audio(audio_path, return_format="int")
    except Exception as e:
        print(f"Error loading audio tensor from {audio_path}: {e}")
        return None

    print("Segmenting audio based on diarization...")
    segments, boundaries = segment_audio_from_diarization(
        wav_tensor, SAMPLE_RATE, diarization
    )

    if not segments:
        print("Warning: No audio segments generated from diarization.")
        return ""

    print(f"Transcribing {len(segments)} segments using GigaAM...")
    transcribed_results = []
    for i, (segment_tensor, (start, end)) in enumerate(zip(segments, boundaries)):
        print(f"  Transcribing segment {i+1}/{len(segments)} ({start:.2f}s - {end:.2f}s)")
        try:
            wav = segment_tensor.to(gigaam_model._device).unsqueeze(0).to(gigaam_model._dtype)
            length = torch.full([1], wav.shape[-1], device=gigaam_model._device)
            encoded, encoded_len = gigaam_model.forward(wav, length)
            result = gigaam_model.decoding.decode(gigaam_model.head, encoded, encoded_len)[0]

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
            except Exception as e:
                 print(f"Error during speaker lookup for segment {start:.2f}s - {end:.2f}s: {e}")
                 speaker = "UNKNOWN"

            transcribed_results.append({
                "speaker": speaker,
                "transcription": result,
                "start": start,
                "end": end
            })
        except Exception as e:
            print(f"Error transcribing segment {i+1} ({start:.2f}s - {end:.2f}s): {e}")
            transcribed_results.append({
                "speaker": "ERROR",
                "transcription": "[Transcription Error]",
                "start": start,
                "end": end
            })

    print("Formatting final transcription...")
    final_transcription = ""
    current_speaker = None
    speaker_utterance = ""

    for result in transcribed_results:
        speaker = result["speaker"]
        transcription = result["transcription"].strip()

        if not transcription or speaker == "ERROR":
            continue

        if current_speaker is None:
            current_speaker = speaker
            speaker_utterance += transcription + " "
        elif speaker == current_speaker:
            speaker_utterance += transcription + " "
        else:
            final_transcription += f"{current_speaker}: {speaker_utterance.strip()}\n"
            current_speaker = speaker
            speaker_utterance = transcription + " "

    if current_speaker and speaker_utterance.strip():
        final_transcription += f"{current_speaker}: {speaker_utterance.strip()}\n"

    print("Transcription and alignment complete.")
    return final_transcription.strip()


def summarize_text_with_llm(text: str, llm_wrapper: LLMClientWrapper):
    summary = llm_wrapper.summarize(text)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Diarize, transcribe, and summarize a meeting audio file.")
    parser.add_argument("audio_file", help="Path to the audio file to process.")
    parser.add_argument("-s", "--settings", default="settings.json", help="Path to the settings JSON file (default: settings.json).")
    args = parser.parse_args()

    audio_file_path = args.audio_file

    try:
        with open(args.settings, 'r', encoding='utf-8') as f:
            settings = json.load(f)
    except FileNotFoundError:
        print(f"Error: Settings file not found at {args.settings}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from settings file {args.settings}")
        return

    pyannote_model_name = settings.get("pyannote_model_name")
    hugging_face_token = settings.get("hugging_face_token")
    llm_api_endpoint = settings.get("llm_api_endpoint")
    llm_api_key = settings.get("llm_api_key", "")
    llm_api_auth = settings.get("llm_api_auth", False)
    llm_api_model = settings.get("llm_api_model")

    required_settings = [pyannote_model_name, llm_api_endpoint, llm_api_model]
    if not all(required_settings):
        print("Error: One or more required settings (pyannote_model_name, llm_api_endpoint, llm_api_model) are missing in the settings file.")
        return

    print("Starting meeting summarization pipeline...")
    print(f"Processing audio file: {audio_file_path}")
    print(f"Using settings from: {args.settings}")


    print("Initializing models...")
    pyannote_wrapper = PyannotePipelineWrapper(pyannote_model_name, hugging_face_token)
    print("Initializing GigaAM model...")
    try:
        gigaam_model = gigaam.load_model("rnnt")
        print("GigaAM model initialized.")
    except Exception as e:
        print(f"Error initializing GigaAM model: {e}")
        print("Pipeline cannot start because GigaAM model failed to load.")
        return

    llm_wrapper = LLMClientWrapper(llm_api_endpoint, llm_api_key, llm_api_auth, llm_api_model)

    if pyannote_wrapper.pipeline is None:
        print("Pipeline cannot start because Pyannote pipeline failed to load.")
        return

    print("Models initialized successfully.")
    cleaned_audio_path = None
    preprocess_success = False
    transcription = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            cleaned_audio_path = tmp_file.name
        print(f"Created temporary file for cleaned audio: {cleaned_audio_path}")

        preprocess_success = preprocess_audio(audio_file_path, cleaned_audio_path)

        if not preprocess_success:
            print("Audio preprocessing failed. Exiting.")
            if cleaned_audio_path and os.path.exists(cleaned_audio_path):
                 try:
                    os.remove(cleaned_audio_path)
                 except OSError:
                    pass
            return

        print(f"Starting diarization and transcription on cleaned audio: {cleaned_audio_path}")
        transcription = diarize_and_transcribe(cleaned_audio_path, pyannote_wrapper, gigaam_model)

    finally:
        if cleaned_audio_path and os.path.exists(cleaned_audio_path):
            try:
                print(f"Cleaning up temporary file: {cleaned_audio_path}")
                os.remove(cleaned_audio_path)
            except OSError as e:
                print(f"Warning: Could not remove temporary file {cleaned_audio_path}: {e}")

    if not preprocess_success or transcription is None:
        print("Pipeline cannot continue because preprocessing or transcription failed.")
        return

    if not transcription:
        print("Warning: Transcription result is empty.")

    print("\n--- Diarized Transcription ---")
    print(transcription)
    print("-------------------------\n")

    transcription_file = "last_transcription.txt"
    try:
        with open(transcription_file, 'w', encoding='utf-8') as f:
            f.write(transcription)
        print(f"Transcription saved to {transcription_file}")
    except IOError as e:
        print(f"Warning: Failed to save transcription to file: {e}")

    if transcription:
        print("Starting summarization...")
        summary = summarize_text_with_llm(transcription, llm_wrapper)

        if not summary:
            print("Pipeline failed during summarization.")
        else:
            print("\n--- Meeting Summary ---")
            print(summary)
            print("-----------------------\n")

            summary_file = "last_summary.txt"
            try:
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(summary)
                print(f"Summary saved to {summary_file}")
            except IOError as e:
                print(f"Warning: Failed to save summary to file: {e}")
    else:
        print("Skipping summarization because transcription was empty.")


    print("Pipeline finished.")

if __name__ == "__main__":
    main()
