import json
import os

# Define a path to store our outputs. For simplicity, we'll create a new folder
# at the top level of our project to hold the deliverables.
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'deliverables')
SCHEMA_FILE_PATH = os.path.join(OUTPUT_DIR, 'data_schema_v1.json')
GUIDE_FILE_PATH = os.path.join(OUTPUT_DIR, 'metadata_annotations_spec.md')


def create_data_schema_json():
    """
    Defines and creates the master JSON schema for a single data episode.

    This JSON-centered approach stores all metadata in the JSON file,
    but for large binary data (like video), it stores a *path* to the file
    instead of the data itself.
    """

    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    schema = {
        "schema_version": "1.0",
        "episode_id": "robot01_20250720_145643",
        "session_info": {
            "start_time_utc": "2025-07-20T21:56:43.123Z",  # Millisecond precision UTC
            "end_time_utc": "2025-07-20T21:58:01.456Z"
        },
        "task_context": {
            "goal": "Pick up the red block from the table.",
            "language_prompt": "Robot, please get the red block.",
            "user_intent": "Object retrieval from a surface."
        },
        "sensor_data": [
            {
                "modality": "video",
                "source_id": "front_cam_SN12345",
                "file_path": "path/to/episode_data/video_front.mp4",  # Path to the binary file
                "calibration_info": {
                    "type": "camera_intrinsics",
                    "resolution": [1920, 1080],
                    "fx": 525.0,
                    "fy": 525.0,
                    "cx": 960.0,
                    "cy": 540.0
                }
            },
            {
                "modality": "imu",
                "source_id": "imu_bosch_SN9876",
                "file_path": "path/to/episode_data/imu_data.csv",  # Path to the numerical data file
                "calibration_info": {
                    "type": "imu_bias",
                    "accel_bias": [0.01, -0.005, 0.02],
                    "gyro_bias": [0.001, 0.001, -0.002]
                }
            }
        ],
        "cross_modality_tags": [
            {
                "tag_id": "grasp_action_001",
                "start_time": "2025-07-20T21:57:10.500Z",
                "end_time": "2025-07-20T21:57:12.800Z",
                "description": "Robot arm extends and grasps the red block.",
                "modalities": ["video", "imu", "joint_states"]
            }
        ]
    }

    print(f"Creating JSON schema at: {SCHEMA_FILE_PATH}")
    with open(SCHEMA_FILE_PATH, 'w') as f:
        json.dump(schema, f, indent=4)
    print("... JSON schema created successfully.")
    return SCHEMA_FILE_PATH


def create_metadata_guide():
    """
    Creates the human-readable markdown guide explaining the schema.
    """

    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    content = """
# Origami AI: Metadata and Annotations Guide (v1.0)

This document explains the structure of our `data_schema_v1.json` file. It ensures everyone understands what each piece of information means.

## Top-Level Fields

- **`schema_version`**: The version of this blueprint.
- **`episode_id`**: A unique name for this specific recording session (e.g., `robotName_Date_Time`).

## `session_info` Block

This section contains timing information for the entire recording.

- **`start_time_utc` / `end_time_utc`**: The exact start and end times of the recording. **They must be in UTC format with millisecond precision** (e.g., `YYYY-MM-DDTHH:MM:SS.sssZ`). This is crucial for time synchronization.

## `task_context` Block

This section describes the "why" behind the recording.

- **`goal`**: A simple, high-level description of the task.
- **`language_prompt`**: The exact command spoken or typed to the robot.
- **`user_intent`**: A more detailed explanation of what the user wanted to achieve.

## `sensor_data` List

This is a list where each item represents one sensor used in the recording.

- **`modality`**: The type of sensor (e.g., `video`, `audio`, `imu`).
- **`source_id`**: The unique serial number or name of the sensor.
- **`file_path`**: The location of the actual data file (e.g., the `.mp4` video). The JSON file only *points* to this data to keep the JSON small and fast.
- **`calibration_info`**: A block containing all calibration details for that sensor. This is vital for the **AMDC (Adaptive Multi-Modal Data Calibration)** module to work correctly.

## `cross_modality_tags` List

This is our system for linking events across different sensors. Each item in the list is a "tag" for a specific event.

- **`tag_id`**: A unique ID for the event tag.
- **`start_time` / `end_time`**: The precise time window when the event occurred.
- **`description`**: A human-readable description of what happened (e.g., "Grasping action").
- **`modalities`**: A list of the sensors that were important for this event. This helps our **HTD-IRL (Hierarchical Task Decomposition)** module learn complex actions.
"""

    print(f"Creating metadata guide at: {GUIDE_FILE_PATH}")
    with open(GUIDE_FILE_PATH, 'w') as f:
        f.write(content)
    print("... Metadata guide created successfully.")
    return GUIDE_FILE_PATH