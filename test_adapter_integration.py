import os
import sys

# --- Setup for Testing ---
# This block adds the 'airflow' folder to the path, allowing us to import
# our utility functions for this test run.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'airflow'))

# Import our three specialized, functional adapters
# Note: We will rename bridge_tfrecord_adapter.py to bridge_adapter.py for consistency
from dags.utils.ingestion_adapters.bridge_adapter import translate_bridge_to_standard_json
from dags.utils.ingestion_adapters.robonet_adapter import translate_robonet_to_standard_json
from dags.utils.ingestion_adapters.rt1_adapter import translate_rt1_to_standard_json

# --- Configuration ---
# Define the paths to the raw data you have downloaded
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
BRIDGE_RAW_PATH = os.path.join(RAW_DATA_DIR, 'bridge', '1.0.0', 'bridge_orig_ep100-train.tfrecord-00000-of-00002')
ROBONET_RAW_PATH = os.path.join(RAW_DATA_DIR, 'robonet', 'robonet', 'robonet_sample_64', '4.0.1',
                                'robonet-train.tfrecord-00001-of-00002')
RT1_RAW_PATH = os.path.join(RAW_DATA_DIR, 'rt1',
                            'fractal_fractal_20220817_data_traj_transform_rt_1_without_filters_disable_episode_padding_seq_length_6_no_preprocessor-train.array_record-00016-of-01024')

# Define where the processed output will be saved
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
BRIDGE_OUTPUT_DIR = os.path.join(PROCESSED_DATA_DIR, 'bridge_processed')
ROBONET_OUTPUT_DIR = os.path.join(PROCESSED_DATA_DIR, 'robonet_processed')
RT1_OUTPUT_DIR = os.path.join(PROCESSED_DATA_DIR, 'rt1_processed')


def main():
    """Runs all dataset adapters on the real data samples."""
    print("====== STARTING END-TO-END ADAPTER TEST ======\n")

    # --- 1. Process the Bridge Dataset ---
    print("--- Processing Bridge Dataset Sample ---")
    if os.path.exists(BRIDGE_RAW_PATH):
        # We assume you have renamed bridge_tfrecord_adapter.py to bridge_adapter.py
        translate_bridge_to_standard_json(BRIDGE_RAW_PATH, BRIDGE_OUTPUT_DIR)
    else:
        print(f"ERROR: Bridge data not found at {BRIDGE_RAW_PATH}")
    print("-" * 40 + "\n")

    # --- 2. Process the RoboNet Dataset ---
    print("--- Processing RoboNet Dataset Sample ---")
    if os.path.exists(ROBONET_RAW_PATH):
        translate_robonet_to_standard_json(ROBONET_RAW_PATH, ROBONET_OUTPUT_DIR)
    else:
        print(f"ERROR: RoboNet data not found at {ROBONET_RAW_PATH}")
    print("-" * 40 + "\n")

    # --- 3. Process the RT-1 Dataset ---
    print("--- Processing RT-1 Dataset Sample ---")
    if os.path.exists(RT1_RAW_PATH):
        translate_rt1_to_standard_json(RT1_RAW_PATH, RT1_OUTPUT_DIR)
    else:
        print(f"ERROR: RT-1 data not found at {RT1_RAW_PATH}")
    print("-" * 40 + "\n")

    print("====== ADAPTER TEST COMPLETE ======")
    print(f"\nCheck the '{PROCESSED_DATA_DIR}' folder for the new processed subdirectories.")


if __name__ == "__main__":
    main()
