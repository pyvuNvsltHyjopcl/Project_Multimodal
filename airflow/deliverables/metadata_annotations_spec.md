
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
