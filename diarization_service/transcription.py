"""Functions for audio transcription and diarization correlation."""
import collections
import multiprocessing
from typing import List, Dict, Any, Tuple

import torch
import numpy as np

def transcribe_audio_process(
    audio_array: np.ndarray,
    settings: dict,
    result_queue: multiprocessing.Queue
):
    """
    Transcription process: loads the Whisper model, transcribes, and puts result in a queue.
    """
    try:
        from faster_whisper import WhisperModel

        whisper_model_size = settings.get("whisper_model_size")
        whisper_device = settings.get("whisper_device", "cuda" if torch.cuda.is_available() else "cpu")
        whisper_compute_type = settings.get("whisper_compute_type", "int8")

        model = WhisperModel(whisper_model_size, device=whisper_device, compute_type=whisper_compute_type)
        
        segments, _ = model.transcribe(audio_array, beam_size=5, language="ru")
        
        full_transcription = [
            {"start": s.start, "end": s.end, "text": s.text} for s in segments
        ]
        result_queue.put(full_transcription)
    except Exception as e:
        print(f"Error in transcription process: {e}")
        result_queue.put([])


def diarize_audio_process(
    audio_array: np.ndarray,
    settings: dict,
    sample_rate: int,
    result_queue: multiprocessing.Queue
):
    """
    Diarization process: loads the Pyannote model, diarizes, and puts result in a queue.
    """
    try:
        from pyannote.audio import Pipeline as PyannotePipeline

        pyannote_model_name = settings.get("pyannote_model_name")
        hf_token = settings.get("hugging_face_token")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        pipeline = PyannotePipeline.from_pretrained(
            pyannote_model_name, token=hf_token
        ).to(torch.device(device))

        audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
        audio_input = {"waveform": audio_tensor, "sample_rate": sample_rate}
        
        diarization = pipeline(audio_input)

        diarization_result = [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, speaker in diarization.speaker_diarization
        ]
        result_queue.put(diarization_result)
    except Exception as e:
        print(f"Error in diarization process: {e}")
        result_queue.put([])


def correlate_and_merge_segments(
    transcription_segments: List[Dict[str, Any]],
    diarization_segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Correlate transcription with diarization and merge consecutive segments 
    from the same speaker.
    """
    if not transcription_segments:
        return []

    diar_timeline = sorted(diarization_segments, key=lambda x: x['start'])

    for t_seg in transcription_segments:
        t_start, t_end = t_seg["start"], t_seg["end"]
        t_duration = t_end - t_start
        overlaps = collections.defaultdict(float)
        
        for d_seg in diar_timeline:
            d_start, d_end, speaker = d_seg["start"], d_seg["end"], d_seg["speaker"]
            if d_end < t_start: continue
            if d_start > t_end: break
            
            overlap_duration = max(0, min(t_end, d_end) - max(t_start, d_start))
            if overlap_duration > 0:
                overlaps[speaker] += overlap_duration
        
        total_known_overlap = sum(overlaps.values())
        if total_known_overlap < t_duration / 2 or not overlaps:
            assigned_speaker = "SPEAKER_UNKNOWN"
        else:
            assigned_speaker = max(overlaps, key=overlaps.get)
        t_seg["speaker"] = assigned_speaker
    
    merged_segments = []
    if transcription_segments:
        current_segment = transcription_segments[0].copy()
        for next_segment in transcription_segments[1:]:
            if next_segment["speaker"] == current_segment["speaker"]:
                current_segment["text"] += " " + next_segment["text"]
                current_segment["end"] = next_segment["end"]
            else:
                merged_segments.append(current_segment)
                current_segment = next_segment.copy()
        merged_segments.append(current_segment)
    
    return merged_segments

def process_audio_pipeline(
    audio_array: np.ndarray,
    sample_rate: int,
    settings: dict
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Orchestrates transcription and diarization in parallel processes.
    """
    try:
        ctx = multiprocessing.get_context("spawn")
        transcription_queue = ctx.Queue()
        diarization_queue = ctx.Queue()

        p_transcribe = ctx.Process(
            target=transcribe_audio_process,
            args=(audio_array, settings, transcription_queue)
        )
        p_diarize = ctx.Process(
            target=diarize_audio_process,
            args=(audio_array, settings, sample_rate, diarization_queue)
        )

        p_transcribe.start()
        p_diarize.start()

        transcription_segments = transcription_queue.get()
        diarization_segments = diarization_queue.get()

        p_transcribe.join()
        p_diarize.join()

        if transcription_segments is None or diarization_segments is None:
            return [], "A child process (transcription or diarization) failed."

        final_segments = correlate_and_merge_segments(
            transcription_segments, diarization_segments
        )
        return final_segments, ""
    except Exception as e:
        import traceback
        return [], f"Error in main processing pipeline: {e}\n{traceback.format_exc()}"