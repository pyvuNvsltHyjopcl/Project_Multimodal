import tensorflow_datasets as tfds
import json
import os
import cv2
import numpy as np
import platform
import argparse


def process_single_dataset(dataset_dir, output_dir_base, dataset_name):
    """
    Uses the TFDS library to intelligently read a dataset from a directory,
    adapting to its specific structure (RT-1, Bridge, or RoboNet), and
    translates episodes into our project's standard session format.

    Args:
        dataset_dir (str): Path to the directory containing the dataset files
                           (e.g., '.../bridge/1.0.0').
        output_dir_base (str): The base directory to save processed sessions
                               (e.g., '.../processed/bridge_processed').
        dataset_name (str): A short name like 'bridge', 'robonet', or 'rt1'.
    """
    print(f"--- [Universal Processor] Processing {dataset_name} from {dataset_dir} ---")

    # --- PLATFORM CHECK FOR RT-1 ---
    if dataset_name == 'rt1' and platform.system() == 'Windows':
        print("\n" + "=" * 60)
        print("INFO: The RT-1 dataset uses the ArrayRecord format, which is not")
        print("      compatible with Windows. Skipping this dataset.")
        print("=" * 60 + "\n")
        return False

    try:
        builder = tfds.builder_from_directory(dataset_dir)

        # RT-1 requires a different method to load the data source
        if dataset_name == 'rt1':
            dataset = builder.as_data_source(split='train')
        else:
            dataset = builder.as_dataset(split='train')

        # Process the first 2 episodes for our sample run
        for i, episode in enumerate(dataset.take(2)):
            episode_id = f"{dataset_name}_ep_{i}"
            output_dir = os.path.join(output_dir_base, episode_id)
            os.makedirs(output_dir, exist_ok=True)

            print(f"  [Processor] Extracting episode: {episode_id}")

            # --- ADAPTIVE MULTIMODAL EXTRACTION ---
            language_instruction = "N/A"
            image_sequence = None

            episode_keys = episode.keys()

            if 'steps' in episode_keys:  # Handles RT-1 and Bridge
                steps_data = list(episode['steps'])
                if not steps_data:
                    print(f"  [Processor] WARNING: Episode {episode_id} contains no steps. Skipping.")
                    continue

                first_step = steps_data[0]
                if 'language_instruction' in first_step:
                    language_instruction = first_step['language_instruction'].numpy().decode('utf-8')
                elif 'observation' in first_step and 'natural_language_instruction' in first_step['observation']:
                    language_instruction = first_step['observation']['natural_language_instruction'].numpy().decode(
                        'utf-8')

                if 'observation' in first_step and 'image' in first_step['observation']:
                    image_sequence = [s['observation']['image'] for s in steps_data]

            elif 'video' in episode_keys:  # Handles RoboNet
                image_sequence = episode['video']

            if not image_sequence:
                print(f"  [Processor] WARNING: Could not extract image sequence for episode {episode_id}. Skipping.")
                continue

            # --- Create Video from Image Sequence ---
            video_path = os.path.join(output_dir, "video.mp4")

            first_frame = image_sequence[0].numpy()
            height, width, _ = first_frame.shape
            video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))

            for frame_tensor in image_sequence:
                frame = frame_tensor.numpy()
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            video_writer.release()

            # --- Create Manifest ---
            manifest = {
                "goal": language_instruction,
                "language_prompt": language_instruction,
                "user_intent": f"Task from {dataset_name.upper()} Dataset."
            }
            with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
                json.dump(manifest, f, indent=4)

        print(f"--- [Universal Processor] Finished processing {dataset_name}. ---")
        return True

    except Exception as e:
        print(f"--- [Universal Processor] ERROR processing {dataset_name}: {e} ---")
        return False


def main():
    """Runs the universal translator on all three dataset samples."""
    print("====== STARTING FINAL END-TO-END DATA PROCESSING ======\n")

    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
    PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

    # Define the paths to the dataset *directories*
    BRIDGE_DIR = os.path.join(RAW_DATA_DIR, 'bridge', '1.0.0')
    ROBONET_DIR = os.path.join(RAW_DATA_DIR, 'robonet', 'robonet', 'robonet_sample_64', '4.0.1')
    RT1_DIR = os.path.join(RAW_DATA_DIR, 'rt1')

    # Define the output directories
    BRIDGE_OUTPUT_DIR = os.path.join(PROCESSED_DATA_DIR, 'bridge_processed')
    ROBONET_OUTPUT_DIR = os.path.join(PROCESSED_DATA_DIR, 'robonet_processed')
    RT1_OUTPUT_DIR = os.path.join(PROCESSED_DATA_DIR, 'rt1_processed')

    # --- 1. Process the Bridge Dataset ---
    print("--- Processing Bridge Dataset Sample ---")
    if os.path.exists(BRIDGE_DIR):
        process_single_dataset(BRIDGE_DIR, BRIDGE_OUTPUT_DIR, "bridge")
    else:
        print(f"ERROR: Bridge data directory not found at {BRIDGE_DIR}")
    print("-" * 40 + "\n")

    # --- 2. Process the RoboNet Dataset ---
    print("--- Processing RoboNet Dataset Sample ---")
    if os.path.exists(ROBONET_DIR):
        process_single_dataset(ROBONET_DIR, ROBONET_OUTPUT_DIR, "robonet")
    else:
        print(f"ERROR: RoboNet data directory not found at {ROBONET_DIR}")
    print("-" * 40 + "\n")

    # --- 3. Process the RT-1 Dataset ---
    print("--- Processing RT-1 Dataset Sample ---")
    if os.path.exists(RT1_DIR):
        process_single_dataset(RT1_DIR, RT1_OUTPUT_DIR, "rt1")
    else:
        print(f"ERROR: RT-1 data directory not found at {RT1_DIR}")
    print("-" * 40 + "\n")

    print("====== DATA PROCESSING COMPLETE ======")
    print(f"\nCheck the '{PROCESSED_DATA_DIR}' folder for the new processed subdirectories.")


if __name__ == "__main__":
    main()
