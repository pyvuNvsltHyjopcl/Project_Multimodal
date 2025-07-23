import os
import glob
import sys

# --- Setup for Testing ---
# This block adds the 'airflow' folder to the path, allowing us to import
# our utility functions for this test run.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'airflow'))

# This import now correctly points to our official mapper script.
from dags.utils.origami_mapper import enrich_data_for_ai_modules

# --- Configuration ---
PROCESSED_DATA_FOLDER = os.path.join(PROJECT_ROOT, 'data', 'processed')


def main():
    """Simulates the Phase 3 data alignment workflow using the official mapper."""
    print("====== STARTING FINAL PHASE 3 WORKFLOW SIMULATION ======\n")

    # Find records to process
    records_to_align = [f for f in glob.glob(os.path.join(PROCESSED_DATA_FOLDER, "*.json")) if
                        not os.path.basename(f).startswith('aligned_')]

    if not records_to_align:
        print("No records from Phase 2 found to process. Please run the Phase 2 test first.")
        return

    print(f"Found {len(records_to_align)} records to process...")

    for record_path in records_to_align:
        # Call the imported function from our official mapper
        enrich_data_for_ai_modules(
            processed_json_path=record_path,
            output_folder=PROCESSED_DATA_FOLDER
        )

    print("\n====== PHASE 3 WORKFLOW SIMULATION COMPLETE ======")
    print(f"\nCheck the '{PROCESSED_DATA_FOLDER}' folder for the new 'aligned_...' JSON file.")


if __name__ == "__main__":
    main()
