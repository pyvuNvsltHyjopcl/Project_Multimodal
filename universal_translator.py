import tensorflow_datasets as tfds
import json
import os
import cv2
import numpy as np
import platform


def process_dataset(dataset_dir, output_dir_base, dataset_name):
    """
    Uses the TFDS library to intelligently read a dataset from a directory,
    adapting to its specific structure (RT-1, Bridge, or RoboNet), and
    translates episodes into our project's standard session format.
    """
    print(f"--- [Universal Translator] Processing {dataset_name} from {dataset_dir} ---")

    try:
        builder = tfds.builder_from_directory(dataset_dir)

        # --- DEFINITIVE FIX FOR RT-1 ---
        # The ArrayRecord format returns a list-like object that does not have a .take() method.
        # We must handle it differently from standard TFRecord datasets.
        if "array_record" in builder.info.file_format.name.lower():
            print("  [Info] Detected ArrayRecord format (for RT-1).")
            # This check ensures the script is running in the correct environment
            if platform.system() == 'Windows':
                print("\nERROR: RT-1 ArrayRecord format is not compatible with Windows. Please run this in WSL.\n")
                return False
            data_source = builder.as_data_source(split='train')
            # Treat it like a list: get the first 2 items for our sample run
            dataset_iterator = data_source[:2]
        else:
            # For standard TFRecord files (Bridge, RoboNet), we can use .take()
            dataset = builder.as_dataset(split='train')
            dataset_iterator = dataset.take(2)

        for i, episode in enumerate(dataset_iterator):
            episode_id = f"{dataset_name}_ep_{i}"
            output_dir = os.path.join(output_dir_base, episode_id)
            os.makedirs(output_dir, exist_ok=True)

            print(f"  [Translator] Extracting episode: {episode_id}")

            # --- ADAPTIVE MULTIMODAL EXTRACTION ---
            language_instruction = "N/A"
            image_sequence = None
            episode_keys = episode.keys()

            if 'steps' in episode_keys:  # Handles RT-1 and Bridge
                steps_data = list(episode['steps'])
                if not steps_data:
                    print(f"  [Translator] WARNING: Episode {episode_id} contains no steps. Skipping.")
                    continue

                first_step = steps_data[0]
                if 'observation' in first_step and 'natural_language_instruction' in first_step['observation']:
                    language_instruction = first_step['observation']['natural_language_instruction'].numpy().decode(
                        'utf-8')

                if 'observation' in first_step and 'image' in first_step['observation']:
                    image_sequence = [s['observation']['image'] for s in steps_data]

            elif 'video' in episode_keys:  # Handles RoboNet
                image_sequence = episode['video']

            if not image_sequence:
                print(f"  [Translator] WARNING: Could not extract image sequence for episode {episode_id}. Skipping.")
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
                "goal": language_instruction, "language_prompt": language_instruction,
                "user_intent": f"Task from {dataset_name.upper()} Dataset."
            }
            with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
                json.dump(manifest, f, indent=4)

        print(f"--- [Universal Translator] Finished processing {dataset_name}. ---")
        return True

    except Exception as e:
        print(f"--- [Universal Translator] ERROR processing {dataset_name}: {e} ---")
        return False
