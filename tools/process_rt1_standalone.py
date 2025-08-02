import tensorflow_datasets as tfds
import json
import os
import cv2
import numpy as np
import argparse
import platform
import sys


def process_rt1_dataset(dataset_dir, output_dir_base):
    """
    Uses the TFDS library in a Linux/WSL environment to correctly process the
    RT-1 ArrayRecord dataset and translate it into our standard format.
    """
    dataset_name = "rt1"
    print(f"--- [RT-1 Standalone Processor] Starting to process {dataset_name} from {dataset_dir} ---")

    # Final check to ensure this is not run on a native Windows environment
    if platform.system() == 'Windows':
        print("\n" + "=" * 70)
        print("FATAL ERROR: This script cannot run on a native Windows Python interpreter.")
        print("The RT-1 .array_record format is only compatible with Linux.")
        print("\nSOLUTION: Please run this script from your WSL (Linux) terminal.")
        print("=" * 70 + "\n")
        sys.exit(1)  # Exit with an error code

    # Verify that the necessary blueprint files exist before proceeding.
    required_files = ["dataset_info.json", "features.json"]
    for f in required_files:
        if not os.path.exists(os.path.join(dataset_dir, f)):
            print(f"\nFATAL ERROR: The required blueprint file '{f}' was not found in {dataset_dir}")
            print("Please place the correct metadata files alongside your .array_record file.\n")
            sys.exit(1)

    try:
        builder = tfds.builder_from_directory(dataset_dir)

        # --- DEFINITIVE FIX FOR RT-1 ---
        # The ArrayRecord format returns a list-like object that does not have a .take() method.
        # We must handle it differently from standard TFRecord datasets.
        print("  [Info] Detected ArrayRecord format. Loading as data source...")
        data_source = builder.as_data_source(split='train')

        # Treat it like a list: get the first 2 items for our sample run
        dataset_iterator = data_source[:2]
        print(f"Successfully loaded RT-1 data source. Processing {len(dataset_iterator)} episodes.")

        for i, episode in enumerate(dataset_iterator):
            episode_id = f"{dataset_name}_ep_{i}"
            output_dir = os.path.join(output_dir_base, episode_id)
            os.makedirs(output_dir, exist_ok=True)

            print(f"  [Processor] Extracting episode: {episode_id}")

            steps_data = list(episode['steps'])
            if not steps_data:
                print(f"  [Processor] WARNING: Episode {episode_id} contains no steps. Skipping.")
                continue

            first_step = steps_data[0]
            language_instruction = first_step['observation']['natural_language_instruction'].numpy().decode('utf-8')
            image_sequence = [s['observation']['image'] for s in steps_data]

            # --- Create Video from Image Sequence ---
            video_path = os.path.join(output_dir, "video.mp4")

            first_frame = image_sequence[0].numpy()
            height, width, _ = first_frame.shape
            video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))

            for frame_tensor in image_sequence:
                frame = frame_tensor.numpy()
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            video_writer.release()
            print(f"    ✓ Video created at {video_path}")

            # --- Create Manifest (Metadata) ---
            manifest = {
                "goal": language_instruction, "language_prompt": language_instruction,
                "user_intent": f"Task from {dataset_name.upper()} Dataset."
            }
            with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
                json.dump(manifest, f, indent=4)
            print(f"    ✓ Metadata created.")

        print(f"\n--- [RT-1 Standalone Processor] Finished processing {dataset_name}. ---")
        return True

    except Exception as e:
        print(f"\n--- [RT-1 Standalone Processor] ERROR processing {dataset_name}: {e} ---")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process RT-1 dataset in a Linux/WSL environment.")
    parser.add_argument("--rt1_dir", required=True, help="Path to the raw RT-1 data directory (e.g., 'data/raw/rt1').")
    parser.add_argument("--output_dir", required=True,
                        help="Path to the processed output directory (e.g., 'data/processed/rt1_processed').")
    args = parser.parse_args()

    process_rt1_dataset(args.rt1_dir, args.output_dir)
