from __future__ import annotations

import pendulum

from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator

# Import the helper functions we created in our 'utils' folder
# Airflow knows to look in the 'dags' folder and subfolders.
from utils.schema_helpers import create_data_schema_json, create_metadata_guide
from utils.data_validators import validate_json_schema

# --- DAG Configuration ---

with DAG(
        dag_id="phase_1_json_schema_setup",
        start_date=pendulum.datetime(2025, 7, 20, tz="America/Los_Angeles"),
        schedule=None,  # This DAG is for one-time manual runs.
        catchup=False,
        tags=["origami-ai", "phase-1", "json-schema"],
        doc_md="""
    ### Phase 1: JSON Schema & Metadata Setup DAG

    This DAG orchestrates the creation and validation of the core JSON data schema
    for the Origami AI project.

    **Tasks:**
    1.  **`define_schema_task`**: Creates the `data_schema_v1.json` file.
    2.  **`create_metadata_template_task`**: Creates the `metadata_annotations_spec.md` guide.
    3.  **`validate_schema_task`**: Validates the `data_schema_v1.json` file to ensure it's correct.
    """,
) as dag:
    # --- Task Definitions ---

    # Task 1: Create the main JSON schema file.
    # This task calls the function from our schema_helpers.py script.
    define_schema_task = PythonOperator(
        task_id="define_schema_task",
        python_callable=create_data_schema_json,
    )

    # Task 2: Create the human-readable guide for the metadata.
    # This also calls a function from schema_helpers.py.
    create_metadata_template_task = PythonOperator(
        task_id="create_metadata_template_task",
        python_callable=create_metadata_guide,
    )

    # Task 3: Validate the schema file that was just created.
    # This task calls the function from our data_validators.py script.
    # We pass the file path that is returned by the first task.
    validate_schema_task = PythonOperator(
        task_id="validate_schema_task",
        python_callable=validate_json_schema,
        # We tell Airflow to get the output from the `define_schema_task` and
        # use it as the input argument `file_path` for our validation function.
        op_kwargs={"file_path": "{{ task_instance.xcom_pull(task_ids='define_schema_task') }}"}
    )

    # --- Task Dependencies ---
    # This tells Airflow the order of operations:
    # - The 'define_schema_task' and 'create_metadata_template_task' can run at the same time.
    # - The 'validate_schema_task' MUST wait for 'define_schema_task' to finish successfully.
    [define_schema_task, create_metadata_template_task] >> validate_schema_task

