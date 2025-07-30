# test_phase_1_flow.py

# Import the functions from our utils scripts
from airflow.dags.utils.schema_helpers import create_data_schema_json, create_metadata_guide
from airflow.dags.utils.data_validators import validate_json_schema

def main():
    """Simulates the entire Phase 1 workflow."""
    print("====== STARTING PHASE 1 WORKFLOW SIMULATION ======\n")

    # Step 1: Simulate the first two Airflow tasks
    print("--- Task 1: Running create_data_schema_json() ---")
    # This function returns the path to the file it creates
    created_schema_path = create_data_schema_json()
    print("\n--- Task 2: Running create_metadata_guide() ---")
    create_metadata_guide()

    print("\n--- Tasks 1 & 2 Complete ---\n")

    # Step 2: Simulate the validation task, using the output from Step 1
    print("--- Task 3: Running validate_json_schema() ---")
    if created_schema_path:
        validate_json_schema(created_schema_path)
    else:
        print("Validation skipped because schema file path was not found.")

    print("\n====== PHASE 1 WORKFLOW SIMULATION COMPLETE ======")

if __name__ == "__main__":
    main()