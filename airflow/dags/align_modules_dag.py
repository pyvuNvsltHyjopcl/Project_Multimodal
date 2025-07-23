from __future__ import annotations
import pendulum
import os
import glob

from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator

# This relative import now points to our new, official mapper script.
from .utils.origami_mapper import enrich_data_for_ai_modules

# --- Configuration ---
DATA_LAKE_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
PROCESSED_DATA_FOLDER = os.path.join(DATA_LAKE_ROOT, 'processed')


def find_new_processed_records_task():
    """Finds new JSON records that need alignment."""
    print(f"Scanning for new records in: {PROCESSED_DATA_FOLDER}")
    records_to_align = glob.glob(os.path.join(PROCESSED_DATA_FOLDER, "aligned_*.json"))
    # Let's correct the logic to find UN-aligned files
    records_to_align = [f for f in glob.glob(os.path.join(PROCESSED_DATA_FOLDER, "*.json")) if
                        not os.path.basename(f).startswith('aligned_')]

    if not records_to_align:
        print("No new records found to align.")
        return []

    print(f"Found {len(records_to_align)} new records to align.")
    return records_to_align


def log_alignment_results_task(aligned_files):
    """Logs a summary of the alignment process."""
    successful_alignments = [path for path in aligned_files if path is not None]
    print("--- Alignment Summary ---")
    if successful_alignments:
        print(f"Successfully aligned {len(successful_alignments)} records.")
    else:
        print("No records were aligned in this run.")
    print("--- End of Summary ---")


with DAG(
        dag_id="phase_3_align_for_ai_modules",
        start_date=pendulum.datetime(2025, 7, 22, tz="America/Los_Angeles"),
        schedule="@weekly",
        catchup=False,
        tags=["origami-ai", "phase-3", "alignment"],
) as dag:
    find_new_records = PythonOperator(
        task_id="find_new_processed_records",
        python_callable=find_new_processed_records_task,
    )

    align_records = PythonOperator.partial(
        task_id="align_record",
        python_callable=enrich_data_for_ai_modules,
        op_kwargs={"output_folder": PROCESSED_DATA_FOLDER},
    ).expand(processed_json_path=find_new_records.output)

    log_results = PythonOperator(
        task_id="log_alignment_results",
        python_callable=log_alignment_results_task,
        op_args=[align_records.output],
        trigger_rule="all_done",
    )

    find_new_records >> align_records >> log_results
