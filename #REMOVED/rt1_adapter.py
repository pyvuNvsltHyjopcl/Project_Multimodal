#rt1_adapter.py
# import tensorflow as tf
# import json
# import os
#
#
# def translate_rt1_to_standard_json(tfrecord_path, output_dir):
#     """
#     Reads a single trajectory from an RT-1 TFRecord file and translates it
#     into our project's standard JSON manifest format.
#
#     Args:
#         tfrecord_path (str): Path to the .tfrecord file.
#         output_dir (str): The 'session' directory to save the output files.
#
#     Returns:
#         str: Path to the newly created metadata.json file.
#     """
#     print(f"--- [RT-1 Adapter] Translating {os.path.basename(tfrecord_path)} ---")
#
#     # Create the output session directory
#     os.makedirs(output_dir, exist_ok=True)
#
#     # In a real implementation, you would save the video frames here.
#     # For this example, we'll just create a placeholder file.
#     video_placeholder_path = os.path.join(output_dir, "video_from_rt1.mp4")
#     with open(video_placeholder_path, 'w') as f:
#         f.write("Placeholder for video data from RT-1.")
#
#     # RT-1 data is complex; we'll simulate reading one step for the language prompt.
#     # A full implementation would iterate through the dataset.
#     raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
#
#     # For demonstration, we'll just extract a plausible language instruction.
#     # A real script would parse the features according to RT-1's spec.
#     language_instruction = "pick up the coke can"  # Simulated extraction
#
#     # Build the JSON manifest according to our schema
#     manifest = {
#         "goal": language_instruction,
#         "language_prompt": language_instruction,
#         "user_intent": "Object manipulation from RT-1 dataset."
#     }
#
#     manifest_path = os.path.join(output_dir, "metadata.json")
#     with open(manifest_path, 'w') as f:
#         json.dump(manifest, f, indent=4)
#
#     print(f"--- [RT-1 Adapter] Created manifest at {manifest_path} ---")
#     return manifest_path
import tensorflow as tf
import json
import os
import cv2
import numpy as np


def translate_rt1_to_standard_json(tfrecord_path, output_dir_base):
    """
    Reads episodes from an RT-1 TFRecord/ArrayRecord file. This adapter handles the
    complex, nested structure of RT-1 to extract key multimodal features
    like image and language instruction.
    """
    print(f"--- [RT-1 Adapter] Processing {os.path.basename(tfrecord_path)} ---")

    # Define the feature spec based on RT-1's features.json
    feature_description = {
        'steps': tf.io.FixedLenSequenceFeature([], dtype=tf.string, allow_missing=True)
    }
    step_feature_description = {
        'observation/image': tf.io.FixedLenFeature([], dtype=tf.string),
        'observation/natural_language_instruction': tf.io.FixedLenFeature([], dtype=tf.string),
    }

    def _parse_step_function(proto):
        return tf.io.parse_single_example(proto, step_feature_description)

    def _parse_episode_function(proto):
        parsed_features = tf.io.parse_single_example(proto, feature_description)
        # The 'steps' are a sequence of smaller records, so we create a new dataset from them
        steps_ds = tf.data.Dataset.from_tensor_slices(parsed_features['steps'])
        return steps_ds.map(_parse_step_function)

    # Use TFRecordDataset which can also read ArrayRecord files
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

    # Each item in the raw_dataset is one full episode
    for i, episode_steps_ds in enumerate(raw_dataset.map(_parse_episode_function)):
        if i >= 2:  # Limit to processing only 2 episodes for a sample run
            break

        # Check if the episode has any steps before proceeding, to avoid errors
        try:
            first_step = next(iter(episode_steps_ds))
        except StopIteration:
            print(f"  [RT-1 Adapter] Skipping empty episode.")
            continue

        episode_id = f"rt1_ep_{i}"
        output_dir = os.path.join(output_dir_base, episode_id)
        os.makedirs(output_dir, exist_ok=True)

        print(f"  [RT-1 Adapter] Extracting episode: {episode_id}")

        language_instruction = first_step['observation/natural_language_instruction'].numpy().decode('utf-8')

        # --- Create Video from Image Sequence ---
        video_path = os.path.join(output_dir, "video.mp4")
        height, width = 256, 320  # Dimensions from RT-1 features.json
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))

        # Add the first frame we already extracted
        image_tensor = tf.io.decode_jpeg(first_step['observation/image'])
        frame = image_tensor.numpy()
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Process the rest of the frames in the episode
        for step in episode_steps_ds:
            image_tensor = tf.io.decode_jpeg(step['observation/image'])
            frame = image_tensor.numpy()
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        video_writer.release()

        # --- Create Manifest File ---
        manifest = {
            "goal": language_instruction,
            "language_prompt": language_instruction,
            "user_intent": "Task from RT-1 Dataset."
        }
        with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
            json.dump(manifest, f, indent=4)

    print(f"--- [RT-1 Adapter] Finished processing. ---")
