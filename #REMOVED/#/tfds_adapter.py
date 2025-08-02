import tensorflow_datasets as tfds
import json
import os
import cv2
import numpy as np


def translate_tfds_to_standard_json(dataset_dir, output_dir_base, dataset_name):
    """
    Uses the TFDS library to read a dataset from a directory and translate
    a few episodes into our project's standard session format.

    Args:
        dataset_dir (str): Path to the directory containing the dataset files
                           (e.g., '.../bridge/1.0.0').
        output_dir_base (str): The base directory to save processed sessions
                               (e.g., '.../processed/bridge_processed').
        dataset_name (str): A short name like 'bridge', 'robonet', or 'rt1'.
    """
    print(f"--- [TFDS Adapter] Processing {dataset_name} from {dataset_dir} ---")

    try:
        # Use the TFDS builder to load the dataset from its files
        builder = tfds.builder_from_directory(dataset_dir)
        dataset = builder.as_dataset(split='train')

        # Process the first 2 episodes for our sample run
        for i, episode in enumerate(dataset.take(2)):
            episode_id = f"{dataset_name}_ep_{i}"
            output_dir = os.path.join(output_dir_base, episode_id)
            os.makedirs(output_dir, exist_ok=True)

            print(f"  [TFDS Adapter] Extracting episode: {episode_id}")

            # --- Extract Multimodal Data ---
            steps = episode['steps']

            # Get language from the first step (if it exists)
            language_instruction = "N/A"  # Default for datasets like RoboNet
            if 'language_instruction' in steps:
                language_instruction = steps['language_instruction'][0].numpy().decode('utf-8')
            elif 'observation' in steps and 'natural_language_instruction' in steps['observation']:
                language_instruction = steps['observation']['natural_language_instruction'][0].numpy().decode('utf-8')

            # --- Create Video from Image Sequence ---
            video_path = os.path.join(output_dir, "video.mp4")

            # Determine the image source based on dataset structure
            if 'video' in steps:  # RoboNet case
                image_sequence = steps['video']
            elif 'observation' in steps and 'image' in steps['observation']:  # Bridge and RT-1 case
                image_sequence = steps['observation']['image']
            else:
                print(f"  [TFDS Adapter] WARNING: No image data found for episode {episode_id}")
                continue

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

        print(f"--- [TFDS Adapter] Finished processing {dataset_name}. ---")
        return True

    except Exception as e:
        print(f"--- [TFDS Adapter] ERROR processing {dataset_name}: {e} ---")
        return False
