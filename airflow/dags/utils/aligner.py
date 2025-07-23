from __future__ import annotations
import pendulum
import os
import glob

from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator

# Import the specialized 'tuner' function from our utils folder
from utils.aligner import enrich_data_for_ai_modules

# --- Configuration ---
# Define the base paths for our data lake.
# In a real production environment, these would be Airflow Variables.
DATA_LAKE_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
PROCESSED_DATA_FOLDER = os.path.join(DATA_LAKE_ROOT, 'processed')


# --- Task Functions ---

def find_new_processed_records_task(**kwargs):
    """
    Task 1: Scans the 'processed' folder for new JSON records that haven't been aligned yet.
    It finds all *.json files but excludes any that already start with 'aligned_'.
    """
    print(f"Scanning for new records in: {PROCESSED_DATA_FOLDER}")

    # Use glob to find all .json files
    all_json_files = glob.glob(os.path.join(PROCESSED_DATA_FOLDER, "*.json"))

    # Filter out files that have already been aligned
    records_to_align = [
        f for f in all_json_files if not os.path.basename(f).startswith('aligned_')
    ]

    if not records_to_align:
        print("No new records found to align.")
        return None  # Return None to signal no work to be done

    print(f"Found {len(records_to_align)} new records to align:")
    for record_path in records_to_align:
        print(f"  - {os.path.basename(record_path)}")

    # Push the list of file paths for the next task to use
    kwargs['ti'].xcom_push(key='records_to_process', value=records_to_align)
    return records_to_align


def log_alignment_results_task(**kwargs):
    """
    Final Task: Logs a summary of the alignment process.
    """
    ti = kwargs['ti']

    # Pull the list of file paths that were processed by the dynamic tasks
    aligned_files = ti.xcom_pull(task_ids='align_record_task', key='return_value')

    # Filter out any None values that might result from failed tasks
    successful_alignments = [path for path in aligned_files if path is not None]

    print("--- Alignment Summary ---")
    if successful_alignments:
        print(f"Successfully aligned {len(successful_alignments)} records:")
        for path in successful_alignments:
            print(f"  -> {os.path.basename(path)}")
    else:
        print("No records were aligned in this run.")
    print("--- End of Summary ---")


# --- DAG Definition ---
with DAG(
        dag_id="phase_3_align_for_ai_modules",
        start_date=pendulum.datetime(2025, 7, 22, tz="America/Los_Angeles"),
        schedule="@weekly",  # This pipeline runs weekly to process new data
        catchup=False,
        tags=["origami-ai", "phase-3", "alignment"],
        doc_md="""
    ### Phase 3: Origami Module Alignment DAG

    This DAG takes the clean, processed data from Phase 2 and enriches it
    with the specific fields required by Origami AI's core modules (like STUM and AMDC).

    **Workflow:**
    1.  **`find_new_processed_records`**: Scans the processed data lake for new records.
    2.  **`align_record_task`**: For each new record, this dynamically spawned task calls the
        `enrich_data_for_ai_modules` function to add uncertainty fields, structured prompts, etc.
    3.  **`log_alignment_results`**: Prints a final report summarizing the successful alignments.
    """,
) as dag:
    # Task 1: Find what needs to be processed
    find_new_records = PythonOperator(
        task_id="find_new_processed_records",
        python_callable=find_new_processed_records_task,
    )

    # Task 2: Dynamically align each record found by the first task.
    # The .expand() method creates a parallel copy of this task for each file path
    # returned by the 'find_new_records' task.
    align_records = PythonOperator(
        task_id="align_record_task",
        python_callable=enrich_data_for_ai_modules,
        op_kwargs={
            "output_folder": PROCESSED_DATA_FOLDER
        },
        # This is the dynamic part: it maps the 'processed_json_path' argument
        # to each item in the list returned by the upstream task.
        op_args=find_new_records.output,
    )

    # Task 3: Log the final results
    log_results = PythonOperator(
        task_id="log_alignment_results",
        python_callable=log_alignment_results_task,
        trigger_rule="all_done",  # This task runs even if some alignment tasks fail
    )

    # --- Task Dependencies ---
    find_new_records >> align_records >> log_results

