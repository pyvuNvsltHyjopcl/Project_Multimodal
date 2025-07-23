import json
from datetime import datetime
import os  # Make sure os is imported


def validate_json_schema(file_path):
    """
    Performs a basic validation check on a given JSON data file.

    Args:
        file_path (str): The path to the JSON file to validate.

    Returns:
        bool: True if validation passes, False otherwise.
    """
    print(f"--- Running Validation on: {file_path} ---")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Check 1: Ensure essential top-level keys exist
        required_keys = [
            "schema_version",
            "episode_id",
            "session_info",
            "task_context",
            "sensor_data",
            "cross_modality_tags"
        ]
        for key in required_keys:
            if key not in data:
                print(f"Validation FAILED: Missing required key '{key}'.")
                return False
        print("PASSED: All required top-level keys are present.")

        # Check 2: Validate the UTC timestamp format
        start_time_str = data["session_info"]["start_time_utc"]
        try:
            datetime.strptime(start_time_str, '%Y-%m-%dT%H:%M:%S.%f%z')
            print("PASSED: Timestamp format is valid UTC with milliseconds.")
        except ValueError:
            print(f"Validation FAILED: Timestamp '{start_time_str}' is not in the correct format.")
            return False

        # Check 3: Ensure sensor_data is a list and not empty
        if not isinstance(data["sensor_data"], list) or len(data["sensor_data"]) == 0:
            print("Validation FAILED: 'sensor_data' must be a non-empty list.")
            return False
        print("PASSED: 'sensor_data' is a non-empty list.")

        print("\n--- Validation Successful! ---")
        return True

    except Exception as e:
        print(f"An error occurred during validation: {e}")
        return False


# --- ADD THIS NEW FUNCTION ---
def perform_pre_flight_checks(session_folder):
    """
    Checks if a raw data session folder contains the necessary files before processing.

    Args:
        session_folder (str): The path to the session folder (e.g., 'data/raw/session_xyz').

    Returns:
        dict: A dictionary of file paths if all checks pass, otherwise None.
    """
    print(f"--- Performing Pre-flight Checks on: {session_folder} ---")

    # Define the expected files
    expected_files = {
        "video": "sample_video.mp4",
        "audio": "sample_audio.mp3",
        "metadata": "metadata.json"
    }

    file_paths = {}
    all_checks_passed = True

    for key, filename in expected_files.items():
        path = os.path.join(session_folder, filename)
        if not os.path.exists(path):
            print(f"  [CHECK] FAILED: File '{filename}' not found.")
            all_checks_passed = False
        elif os.path.getsize(path) == 0:
            print(f"  [CHECK] FAILED: File '{filename}' is empty.")
            all_checks_passed = False
        else:
            print(f"  [CHECK] PASSED: File '{filename}' found and is not empty.")
            file_paths[key] = path

    if all_checks_passed:
        print("--- Pre-flight Checks Successful ---")
        return file_paths
    else:
        print("--- Pre-flight Checks Failed ---")
        return None
