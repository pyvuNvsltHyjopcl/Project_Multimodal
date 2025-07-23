from __future__ import annotations
import pendulum
import os
import json
from datetime import datetime, timezone

from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator

# Import our custom tools
from utils.data_validators import perform_pre_flight_checks
from utils.normalizers import normalize_video, normalize_audio, process_language_file

# --- Configuration ---
# Define the base paths for our data lake
# This assumes the DAG is run from a context where the 'data' folder is at the root
# In a real Airflow setup, these would be variables or connections.
DATA_LAKE_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
RAW_DATA_FOLDER = os.path.join(DATA_LAKE_ROOT, 'raw')
PROCESSED_DATA_FOLDER = os.path.join(DATA_LAKE_ROOT, 'processed')
# For this example, we will process a specific session
SESSION_TO_PROCESS = "session_20250720_153000"


# --- Task Functions ---
# These are the Python functions that our Airflow tasks will execute.

def ingest_task(**kwargs):
    """
    Task 1: Performs pre-flight checks on the session folder and pushes file paths.
    """
    session_path = os.path.join(RAW_DATA_FOLDER, SESSION_TO_PROCESS)
    file_paths = perform_pre_flight_checks(session_path)
    if file_paths is None:
        raise ValueError("Pre-flight checks failed. Halting pipeline.")

    # Push the dictionary of file paths so other tasks can access it
    kwargs['ti'].xcom_push(key='raw_file_paths', value=file_paths)


def merge_and_structure_task(**kwargs):
    """
    Final Task: Gathers results from all normalization tasks and builds the final JSON record.
    """
    print("--- Merging and Structuring Final Record ---")
    ti = kwargs['ti']

    # Pull the results (file paths) from all the upstream tasks
    normalized_video_path = ti.xcom_pull(task_ids='normalize_video_task', key='return_value')
    normalized_audio_path = ti.xcom_pull(task_ids='normalize_audio_task', key='return_value')
    task_context = ti.xcom_pull(task_ids='process_language_task', key='return_value')

    # Create the final JSON structure based on our Phase 1 schema
    episode_id = SESSION_TO_PROCESS
    final_record = {
        "schema_version": "1.0",
        "episode_id": episode_id,
        "session_info": {
            "start_time_utc": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
            "end_time_utc": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        },
        "task_context": task_context,
        "sensor_data": [
            {
                "modality": "video",
                "source_id": "front_cam_SN12345",  # Example ID
                "file_path": normalized_video_path,
                "calibration_info": {"notes": "Example calibration data"}
            },
            {
                "modality": "audio",
                "source_id": "microphone_SN67890",  # Example ID
                "file_path": normalized_audio_path,
                "calibration_info": {"notes": "Example calibration data"}
            }
        ],
        "cross_modality_tags": []  # Example, would be populated in a later phase
    }

    # Save the final JSON record to the 'processed' data lake
    output_path = os.path.join(PROCESSED_DATA_FOLDER, f"{episode_id}.json")
    os.makedirs(PROCESSED_DATA_FOLDER, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(final_record, f, indent=4)

    print(f"--- Successfully created final record at: {output_path} ---")


# --- DAG Definition ---
with DAG(
        dag_id="phase_2_ingest_and_normalize",
        start_date=pendulum.datetime(2025, 7, 20, tz="America/Los_Angeles"),
        schedule="@daily",  # Example: run this pipeline once a day
        catchup=False,
        tags=["origami-ai", "phase-2", "ingestion"],
        doc_md="""
    ### Phase 2: Real-World Data Ingestion & Normalization DAG
    This pipeline ingests raw session data, normalizes it in parallel,
    and merges it into a final, structured JSON record.
    """,
) as dag:
    # Task 1: Check the raw data
    ingest_data = PythonOperator(
        task_id="ingest_task",
        python_callable=ingest_task,
    )

    # Task 2a: Normalize Video
    normalize_video_op = PythonOperator(
        task_id="normalize_video_task",
        python_callable=normalize_video,
        op_kwargs={
            "source_path": "{{ ti.xcom_pull(task_ids='ingest_task', key='raw_file_paths')['video'] }}",
            "output_folder": os.path.join(PROCESSED_DATA_FOLDER, SESSION_TO_PROCESS)
        }
    )

    # Task 2b: Normalize Audio
    normalize_audio_op = PythonOperator(
        task_id="normalize_audio_task",
        python_callable=normalize_audio,
        op_kwargs={
            "source_path": "{{ ti.xcom_pull(task_ids='ingest_task', key='raw_file_paths')['audio'] }}",
            "output_folder": os.path.join(PROCESSED_DATA_FOLDER, SESSION_TO_PROCESS)
        }
    )

    # Task 2c: Process Language
    process_language_op = PythonOperator(
        task_id="process_language_task",
        python_callable=process_language_file,
        op_kwargs={
            "source_path": "{{ ti.xcom_pull(task_ids='ingest_task', key='raw_file_paths')['metadata'] }}"
        }
    )

    # Task 3: Merge everything together
    merge_results = PythonOperator(
        task_id="merge_and_structure_task",
        python_callable=merge_and_structure_task,
    )

    # --- Task Dependencies (The Assembly Line Order) ---
    # The ingest task runs first.
    # Then, the three normalization tasks run in parallel.
    # Finally, the merge task runs only after all three normalization tasks are complete.
    ingest_data >> [normalize_video_op, normalize_audio_op, process_language_op] >> merge_results

