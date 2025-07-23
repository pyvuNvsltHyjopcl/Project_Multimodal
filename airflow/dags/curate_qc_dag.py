from __future__ import annotations
import pendulum
import os
import glob
import json
import shutil

from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator

# Import our new inspection tools
from .utils.curator import run_quality_checks

# --- Configuration ---
DATA_LAKE_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
PROCESSED_FOLDER = os.path.join(DATA_LAKE_ROOT, 'processed')
CURATED_FOLDER = os.path.join(DATA_LAKE_ROOT, 'curated')
REJECTED_FOLDER = os.path.join(DATA_LAKE_ROOT, 'rejected')

# --- Quality Thresholds ---
BLUR_THRESHOLD = 100.0  # Videos with a score below this are considered blurry
AUDIO_THRESHOLD = 0.01  # Audio with an RMS below this is considered silent


def find_aligned_records_task():
    """Finds aligned JSON records that need to be curated."""
    print(f"Scanning for aligned records in: {PROCESSED_FOLDER}")
    # Find all files starting with 'aligned_'
    records_to_curate = glob.glob(os.path.join(PROCESSED_FOLDER, "aligned_*.json"))
    print(f"Found {len(records_to_curate)} records to curate.")
    return records_to_curate


def filter_and_route_task(json_path, quality_scores):
    """Sorts a record into 'curated' or 'rejected' based on its quality scores."""
    if not quality_scores:
        print(f"  [ROUTER] No quality scores for {os.path.basename(json_path)}, rejecting.")
        destination_folder = REJECTED_FOLDER
    else:
        # Check if the record passes our quality gates
        passes_blur_check = quality_scores.get("video_blur_score", 0) >= BLUR_THRESHOLD
        passes_audio_check = quality_scores.get("audio_rms", 0) >= AUDIO_THRESHOLD

        if passes_blur_check and passes_audio_check:
            print(f"  [ROUTER] PASSED: {os.path.basename(json_path)}")
            destination_folder = CURATED_FOLDER
        else:
            print(f"  [ROUTER] FAILED: {os.path.basename(json_path)}")
            destination_folder = REJECTED_FOLDER

    # Move the JSON file and its associated media folder
    shutil.move(json_path, os.path.join(destination_folder, os.path.basename(json_path)))
    media_folder_name = os.path.splitext(os.path.basename(json_path))[0].replace('aligned_', '')
    media_folder_path = os.path.join(PROCESSED_FOLDER, media_folder_name)
    if os.path.exists(media_folder_path):
        shutil.move(media_folder_path, os.path.join(destination_folder, media_folder_name))


def export_gold_dataset_task():
    """Selects the best records from the curated set and tags them as 'gold_standard'."""
    print("--- Exporting Gold Standard Dataset ---")
    curated_files = glob.glob(os.path.join(CURATED_FOLDER, "aligned_*.json"))

    if not curated_files:
        print("No curated files found to create a gold standard set.")
        return

    # For this example, we'll just tag the first 5 files as gold standard
    gold_standard_files = curated_files[:5]

    for file_path in gold_standard_files:
        with open(file_path, 'r+') as f:
            data = json.load(f)
            data["is_gold_standard"] = True
            f.seek(0)  # Rewind to the start of the file
            json.dump(data, f, indent=4)
            f.truncate()  # Remove any trailing data
        print(f"  Tagged {os.path.basename(file_path)} as gold standard.")


# --- DAG Definition ---
with DAG(
        dag_id="phase_4_curate_and_qc",
        start_date=pendulum.datetime(2025, 7, 23, tz="America/Los_Angeles"),
        schedule=None,  # Manual trigger for this final phase
        catchup=False,
        tags=["origami-ai", "phase-4", "curation"],
) as dag:
    find_records = PythonOperator(
        task_id="find_aligned_records_to_curate",
        python_callable=find_aligned_records_task,
    )

    # For each record found, run the QC check
    qc_check = PythonOperator.partial(
        task_id="run_quality_check",
        python_callable=run_quality_checks,
    ).expand(aligned_json_path=find_records.output)

    # For each record and its corresponding QC score, run the filter
    route_records = PythonOperator.partial(
        task_id="filter_and_route",
        python_callable=filter_and_route_task,
    ).expand(json_path=find_records.output, quality_scores=qc_check.output)

    # After all records are sorted, create the gold standard set
    export_gold = PythonOperator(
        task_id="export_gold_dataset",
        python_callable=export_gold_dataset_task,
    )

    find_records >> qc_check >> route_records >> export_gold
