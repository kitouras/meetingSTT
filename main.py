import os
import json
import argparse
from pyannote.core import Segment
from models import VoskModelWrapper, PyannotePipelineWrapper, LLMClientWrapper


def diarize_and_transcribe(audio_path, vosk_wrapper: VoskModelWrapper, pyannote_wrapper: PyannotePipelineWrapper, chunk_duration_sec: int):
        
    if vosk_wrapper.model is None:
        print("Error: Vosk model within wrapper was not loaded successfully.")
        return None
    if pyannote_wrapper.pipeline is None:
        print("Error: Pyannote pipeline within wrapper was not loaded successfully.")
        return None
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return None

    diarization = pyannote_wrapper.diarize(audio_path)
    if diarization is None:
        print("Diarization failed.")
        return None


    all_words = vosk_wrapper.transcribe_audio_file(audio_path, chunk_duration_sec, diarization)
    if all_words is None:
        print("Transcription failed.")
        return None

    if not all_words:
        print("Warning: No words were transcribed by Vosk (returned empty list).")
        return ""

    print("Aligning transcription with speaker segments...")
    diarized_transcription = ""
    current_speaker = None
    speaker_utterance = ""
    
    print(all_words)
    print(diarization)

    for word_info in all_words:
        if not isinstance(word_info, dict) or not all(k in word_info for k in ['start', 'end', 'word']):
            print(f"Warning: Skipping invalid word_info item: {word_info}")
            continue

        word_start = word_info['start']
        word_end = word_info['end']
        word_text = word_info['word']

        word_mid_time = word_start + (word_end - word_start) / 2

        lookup_segment = Segment(word_mid_time - 1e-4, word_mid_time + 1e-4)
        try:
            speakers_at_time = diarization.crop(lookup_segment).labels()
            if speakers_at_time:
                speaker = speakers_at_time[0]
            else:
                speakers_during_word = diarization.crop(Segment(word_start, word_end)).labels()
                if speakers_during_word:
                    speaker = speakers_during_word[0]
                else:
                    speaker = "UNKNOWN"
                    print(f"Warning: No speaker found for word '{word_text}' between {word_start:.2f}s and {word_end:.2f}s")
        except Exception as e:
            print(f"Error during speaker lookup for word '{word_text}': {e}")
            speaker = "UNKNOWN"
        if current_speaker is None:
            current_speaker = speaker
            speaker_utterance += word_text + " "
        elif speaker == current_speaker:
            speaker_utterance += word_text + " "
        else:
            diarized_transcription += f"{current_speaker}: {speaker_utterance.strip()}\n"
            current_speaker = speaker
            speaker_utterance = word_text + " "

    if current_speaker and speaker_utterance:
        diarized_transcription += f"{current_speaker}: {speaker_utterance.strip()}\n"

    print("Alignment complete.")
    return diarized_transcription.strip()

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

    vosk_model_path = settings.get("vosk_model_path")
    pyannote_model_name = settings.get("pyannote_model_name")
    hugging_face_token = settings.get("hugging_face_token")
    llm_api_endpoint = settings.get("llm_api_endpoint")
    llm_api_key = settings.get("llm_api_key", "")
    llm_api_auth = settings.get("llm_api_auth", False)
    llm_api_model = settings.get("llm_api_model")
    vosk_log_level = settings.get("vosk_log_level", -1)
    chunk_duration_seconds = settings.get("vosk_chunk_duration", 30)

    required_settings = [vosk_model_path, pyannote_model_name, llm_api_endpoint, llm_api_model]
    if not all(required_settings):
        print("Error: One or more required settings are missing in the settings file.")
        return

    print("Starting meeting summarization pipeline...")
    print(f"Processing audio file: {audio_file_path}")
    print(f"Using settings from: {args.settings}")


    print("Initializing models...")
    vosk_wrapper = VoskModelWrapper(vosk_model_path, vosk_log_level)
    pyannote_wrapper = PyannotePipelineWrapper(pyannote_model_name, hugging_face_token)
    llm_wrapper = LLMClientWrapper(llm_api_endpoint, llm_api_key, llm_api_auth, llm_api_model)

    if vosk_wrapper.model is None:
        print("Pipeline cannot start because Vosk model failed to load.")
        return
    if pyannote_wrapper.pipeline is None:
        print("Pipeline cannot start because Pyannote pipeline failed to load.")
        return

    print("Models initialized.")

    transcription = diarize_and_transcribe(audio_file_path, vosk_wrapper, pyannote_wrapper, chunk_duration_seconds)

    if not transcription:
        print("Pipeline failed during transcription.")
        return

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

    summary = summarize_text_with_llm(transcription, llm_wrapper)

    if not summary:
        print("Pipeline failed during summarization.")
        return

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

    print("Pipeline finished successfully.")

if __name__ == "__main__":
    main()
