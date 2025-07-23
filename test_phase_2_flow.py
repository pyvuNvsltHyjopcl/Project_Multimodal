import os
import json
from datetime import datetime, timezone

# Import the functions we built for Phase 2
from airflow.dags.utils.data_validators import perform_pre_flight_checks
from airflow.dags.utils.normalizers import normalize_video, normalize_audio, process_language_file

# --- Configuration for the Test Run ---
# Define the paths relative to this test script
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_LAKE_ROOT = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_FOLDER = os.path.join(DATA_LAKE_ROOT, 'raw')
PROCESSED_DATA_FOLDER = os.path.join(DATA_LAKE_ROOT, 'processed')

# The specific raw data session we are going to process
SESSION_TO_PROCESS = "session_20250720_153000"


def main():
    """Simulates the entire Phase 2 data ingestion and normalization workflow."""
    print("====== STARTING PHASE 2 WORKFLOW SIMULATION ======\n")

    session_path = os.path.join(RAW_DATA_FOLDER, SESSION_TO_PROCESS)

    # --- 1. Ingestion & Pre-flight Checks ---
    raw_file_paths = perform_pre_flight_checks(session_path)

    if not raw_file_paths:
        print("\nHalting workflow due to failed pre-flight checks.")
        return

    # Create a dedicated folder for the processed output of this session
    processed_session_folder = os.path.join(PROCESSED_DATA_FOLDER, SESSION_TO_PROCESS)
    os.makedirs(processed_session_folder, exist_ok=True)
    print(f"\nCreated output directory: {processed_session_folder}")

    # --- 2. Parallel Normalization ---
    print("\n--- Starting Normalization Tasks (simulating in sequence) ---")
    normalized_video_path = normalize_video(raw_file_paths['video'], processed_session_folder)
    normalized_audio_path = normalize_audio(raw_file_paths['audio'], processed_session_folder)
    task_context = process_language_file(raw_file_paths['metadata'])
    print("--- Normalization Tasks Complete ---\n")

    # --- 3. Merge & Structure ---
    print("--- Starting Merge & Structure Task ---")
    if not all([normalized_video_path, normalized_audio_path, task_context]):
        print("Halting workflow because one or more normalization steps failed.")
        return

    final_record = {
        "schema_version": "1.0",
        "episode_id": SESSION_TO_PROCESS,
        "session_info": {
            "start_time_utc": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
            "end_time_utc": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        },
        "task_context": task_context,
        "sensor_data": [
            {"modality": "video", "source_id": "front_cam_SN12345", "file_path": normalized_video_path},
            {"modality": "audio", "source_id": "microphone_SN67890", "file_path": normalized_audio_path}
        ],
        "cross_modality_tags": []
    }

    output_json_path = os.path.join(PROCESSED_DATA_FOLDER, f"{SESSION_TO_PROCESS}.json")
    with open(output_json_path, 'w') as f:
        json.dump(final_record, f, indent=4)

    print(f"--- Successfully created final structured record at: {output_json_path} ---")

    print("\n====== PHASE 2 WORKFLOW SIMULATION COMPLETE ======")


if __name__ == "__main__":
    main()
