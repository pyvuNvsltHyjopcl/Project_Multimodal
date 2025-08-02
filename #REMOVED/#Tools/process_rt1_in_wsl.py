# import tensorflow_datasets as tfds
# import json
# import os
# import cv2
# import numpy as np
# import argparse
#
#
# def process_rt1_dataset(dataset_dir, output_dir_base):
#     """
#     Uses the TFDS library in a Linux environment to correctly process the
#     RT-1 ArrayRecord dataset and translate it into our standard format.
#     """
#     dataset_name = "rt1"
#     print(f"--- [RT-1 WSL Processor] Processing {dataset_name} from {dataset_dir} ---")
#
#     try:
#         builder = tfds.builder_from_directory(dataset_dir)
#         # This is the command that only works on Linux/macOS
#         dataset = builder.as_data_source(split='train')
#
#         print("Successfully loaded RT-1 data source...")
#
#         # Process the first 2 episodes for our sample run
#         for i, episode in enumerate(dataset.take(2)):
#             episode_id = f"{dataset_name}_ep_{i}"
#             output_dir = os.path.join(output_dir_base, episode_id)
#             os.makedirs(output_dir, exist_ok=True)
#
#             print(f"  [Processor] Extracting episode: {episode_id}")
#
#             steps_data = list(episode['steps'])
#             if not steps_data:
#                 print(f"  [Processor] WARNING: Episode {episode_id} contains no steps. Skipping.")
#                 continue
#
#             first_step = steps_data[0]
#             language_instruction = first_step['observation']['natural_language_instruction'].numpy().decode('utf-8')
#             image_sequence = [s['observation']['image'] for s in steps_data]
#
#             # --- Create Video from Image Sequence ---
#             video_path = os.path.join(output_dir, "video.mp4")
#
#             first_frame = image_sequence[0].numpy()
#             height, width, _ = first_frame.shape
#             video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))
#
#             for frame_tensor in image_sequence:
#                 frame = frame_tensor.numpy()
#                 video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
#             video_writer.release()
#
#             # --- Create Manifest ---
#             manifest = {
#                 "goal": language_instruction, "language_prompt": language_instruction,
#                 "user_intent": f"Task from {dataset_name.upper()} Dataset."
#             }
#             with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
#                 json.dump(manifest, f, indent=4)
#
#         print(f"--- [RT-1 WSL Processor] Finished processing {dataset_name}. ---")
#         return True
#
#     except Exception as e:
#         print(f"--- [RT-1 WSL Processor] ERROR processing {dataset_name}: {e} ---")
#         return False
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Process RT-1 dataset in a Linux/WSL environment.")
#     parser.add_argument("--rt1_dir", required=True, help="Path to the raw RT-1 data directory.")
#     parser.add_argument("--output_dir", required=True, help="Path to the processed output directory.")
#     args = parser.parse_args()
#
#     process_rt1_dataset(args.rt1_dir, args.output_dir)
#################################################################################################
import tensorflow_datasets as tfds
import json
import os
import cv2
import numpy as np
import argparse
import platform


def process_rt1_dataset(dataset_dir, output_dir_base):
    """
    Uses the TFDS library in a Linux/WSL environment to correctly process the
    RT-1 ArrayRecord dataset and translate it into our standard format.
    """
    dataset_name = "rt1"
    print(f"--- [RT-1 WSL Processor] Starting to process {dataset_name} from {dataset_dir} ---")

    # Final check to ensure this is not run on a native Windows environment
    if platform.system() == 'Windows':
        print("\nFATAL ERROR: This script cannot run on a native Windows Python interpreter.")
        print("It must be run using a WSL (Windows Subsystem for Linux) interpreter.")
        print("Please configure PyCharm to use your WSL interpreter for this script.\n")
        return False

    try:
        builder = tfds.builder_from_directory(dataset_dir)
        # This is the command that only works on Linux/macOS
        dataset = builder.as_data_source(split='train')

        print("Successfully loaded RT-1 data source...")

        # Process the first 2 episodes for our sample run
        for i, episode in enumerate(dataset.take(2)):
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

            # --- Create Manifest ---
            manifest = {
                "goal": language_instruction, "language_prompt": language_instruction,
                "user_intent": f"Task from {dataset_name.upper()} Dataset."
            }
            with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
                json.dump(manifest, f, indent=4)

        print(f"--- [RT-1 WSL Processor] Finished processing {dataset_name}. ---")
        return True

    except Exception as e:
        print(f"--- [RT-1 WSL Processor] ERROR processing {dataset_name}: {e} ---")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process RT-1 dataset in a Linux/WSL environment.")
    parser.add_argument("--rt1_dir", required=True, help="Path to the raw RT-1 data directory.")
    parser.add_argument("--output_dir", required=True, help="Path to the processed output directory.")
    args = parser.parse_args()

    process_rt1_dataset(args.rt1_dir, args.output_dir)

