import os
import sys
import json
import cv2
import numpy as np
from datetime import datetime
import shutil
import wave  # Library for writing valid WAV files
import glob

# --- Setup for Testing ---
# Add the airflow folder to the path so we can import our normalizers
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'airflow'))

from dags.utils.normalizers import normalize_video, normalize_audio, process_language_file


# --- PART 1: SYNTHETIC DATA GENERATION (REVISED) ---

def generate_synthetic_session(output_base_path):
    """
    Generates a single synthetic session folder with valid placeholder data.
    Returns the path to the newly created session folder.
    """
    print("--- 1. Generating a Synthetic Data Session ---")
    session_id = f"session_synthetic_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(output_base_path, session_id)
    os.makedirs(output_dir, exist_ok=True)
    print(f"  [Generator] Created session folder: {session_id}")

    # --- Video Generation (Random Noise) ---
    video_path = os.path.join(output_dir, "synthetic_video.mp4")
    height, width = 480, 640
    image_sequence = [np.random.randint(0, 256, (height, width, 3), dtype=np.uint8) for _ in range(30)]
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))
    for frame in image_sequence:
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video_writer.release()
    print(f"  [Generator] Created placeholder video: {os.path.basename(video_path)}")

    # --- Audio Generation (Silent but Valid WAV file) ---
    audio_path = os.path.join(output_dir, "synthetic_audio.wav")
    with wave.open(audio_path, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(44100)  # Standard sample rate
        wf.writeframes(b'\x00' * 44100 * 2)  # 1 second of silence
    print(f"  [Generator] Created valid silent audio: {os.path.basename(audio_path)}")

    # --- Metadata Generation ---
    manifest = {
        "goal": "Test synthetic data processing.",
        "language_prompt": "Robot, process this valid synthetic data.",
        "user_intent": "End-to-end pipeline validation with a synthetic source."
    }
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(manifest, f, indent=4)
    print(f"  [Generator] Created metadata.json.")

    return output_dir, session_id


# --- PART 2: PHASE 2 PROCESSING (REVISED) ---

def process_session(session_path, session_id, processed_base_path):
    """
    Takes a raw session folder, dynamically finds the media files, normalizes them,
    and creates the final structured JSON record.
    """
    print("\n--- 2. Running Phase 2 Processing on Synthetic Data ---")

    # --- Dynamic File Discovery (Fix for hardcoded names) ---
    # Find any video file and any audio file in the session folder
    raw_video_path = next(iter(glob.glob(os.path.join(session_path, '*.mp4'))), None)
    raw_audio_path = next(iter(glob.glob(os.path.join(session_path, '*.wav'))), None)  # Look for WAV now
    raw_metadata_path = os.path.join(session_path, 'metadata.json')

    if not all([raw_video_path, raw_audio_path, raw_metadata_path]):
        print("  [Processor] ERROR: Missing one or more required files in the session folder.")
        return

    processed_session_folder = os.path.join(processed_base_path, session_id)
    os.makedirs(processed_session_folder, exist_ok=True)

    # Run the normalizers from our utils
    normalized_video_path = normalize_video(raw_video_path, processed_session_folder)
    normalized_audio_path = normalize_audio(raw_audio_path, processed_session_folder)
    task_context = process_language_file(raw_metadata_path)

    # Merge and create the final record
    final_record = {
        "schema_version": "1.0",
        "episode_id": session_id,
        "task_context": task_context,
        "sensor_data": [
            {"modality": "video", "file_path": normalized_video_path},
            {"modality": "audio", "file_path": normalized_audio_path}
        ]
    }

    output_json_path = os.path.join(processed_base_path, f"{session_id}.json")
    with open(output_json_path, 'w') as f:
        json.dump(final_record, f, indent=4)

    print(f"\n--- Phase 2 Processing Complete. Final record saved to: {output_json_path} ---")


def main():
    """Runs the full generate-and-process workflow."""
    print("====== STARTING SYNTHETIC DATA PIPELINE TEST ======\n")

    RAW_SYNTHETIC_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw_synthetic')
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

    # Step 1: Generate a new synthetic data session
    session_path, session_id = generate_synthetic_session(RAW_SYNTHETIC_DIR)

    # Step 2: Process that exact session
    process_session(session_path, session_id, PROCESSED_DIR)

    print("\n====== TEST COMPLETE ======")
    # Clean up the raw synthetic data folder for the next run
    shutil.rmtree(session_path)


if __name__ == "__main__":
    main()
