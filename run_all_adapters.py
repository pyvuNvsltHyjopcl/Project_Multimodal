# run_all_adapters.py
# ! i !
# This script runs all three dataset adapters in a single end-to-end test.
# It processes the Bridge, RoboNet, and RT-1 datasets using the universal translator
# and outputs the results to the specified processed data directories.
# ! i !
# This is the final integration test to ensure all adapters work together seamlessly.
# It assumes the universal translator is correctly implemented and can handle all three datasets.
import os
import sys

# --- Setup for Testing ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'airflow'))

from dags.utils.ingestion_adapters.universal_translator import process_dataset

# --- Configuration ---
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

BRIDGE_DIR = os.path.join(RAW_DATA_DIR, 'bridge', '1.0.0')
ROBONET_DIR = os.path.join(RAW_DATA_DIR, 'robonet', 'robonet', 'robonet_sample_64', '4.0.1')
RT1_DIR = os.path.join(RAW_DATA_DIR, 'rt1')

BRIDGE_OUTPUT_DIR = os.path.join(PROCESSED_DATA_DIR, 'bridge_processed')
ROBONET_OUTPUT_DIR = os.path.join(PROCESSED_DATA_DIR, 'robonet_processed')
RT1_OUTPUT_DIR = os.path.join(PROCESSED_DATA_DIR, 'rt1_processed')


def main():
    """Runs the universal translator on all three dataset samples."""
    print("====== STARTING FINAL END-TO-END TRANSLATION TEST ======\n")

    print("--- Processing Bridge Dataset Sample ---")
    if os.path.exists(BRIDGE_DIR):
        process_dataset(BRIDGE_DIR, BRIDGE_OUTPUT_DIR, "bridge")
    else:
        print(f"ERROR: Bridge data directory not found at {BRIDGE_DIR}")
    print("-" * 40 + "\n")

    print("--- Processing RoboNet Dataset Sample ---")
    if os.path.exists(ROBONET_DIR):
        process_dataset(ROBONET_DIR, ROBONET_OUTPUT_DIR, "robonet")
    else:
        print(f"ERROR: RoboNet data directory not found at {ROBONET_DIR}")
    print("-" * 40 + "\n")

    print("--- Processing RT-1 Dataset Sample ---")
    if os.path.exists(RT1_DIR):
        process_dataset(RT1_DIR, RT1_OUTPUT_DIR, "rt1")
    else:
        print(f"ERROR: RT-1 data directory not found at {RT1_DIR}")
    print("-" * 40 + "\n")

    print("====== TRANSLATION TEST COMPLETE ======")
    print(f"\nCheck the '{PROCESSED_DATA_DIR}' folder for the new processed subdirectories.")


if __name__ == "__main__":
    main()
################################################################################
