import tensorflow as tf
import json
import os
import cv2
import numpy as np


def translate_bridge_to_standard_json(tfrecord_path, output_dir_base):
    """
    Reads episodes from a Bridge Dataset TFRecord file, extracts the multimodal
    data (image, state, action, language), and translates it into our
    project's standard session format.
    """
    print(f"--- [Bridge Adapter] Processing {os.path.basename(tfrecord_path)} ---")

    # Define the feature description based on Bridge's features.json
    feature_description = {
        'steps': tf.io.FixedLenSequenceFeature([], dtype=tf.string, allow_missing=True)
    }
    step_feature_description = {
        'observation/image': tf.io.FixedLenFeature([], dtype=tf.string),
        'observation/state': tf.io.FixedLenFeature([7], dtype=tf.float32),
        'action': tf.io.FixedLenFeature([7], dtype=tf.float32),
        'language_instruction': tf.io.FixedLenFeature([1], dtype=tf.string),
    }

    def _parse_step_function(proto):
        return tf.io.parse_single_example(proto, step_feature_description)

    def _parse_episode_function(proto):
        parsed_features = tf.io.parse_single_example(proto, feature_description)
        steps_ds = tf.data.Dataset.from_tensor_slices(parsed_features['steps'])
        return steps_ds.map(_parse_step_function)

    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

    # Each item in the dataset is a single episode
    for i, episode_steps_ds in enumerate(raw_dataset.map(_parse_episode_function)):
        if i >= 2:  # Limit to processing only 2 episodes for a sample run
            break

        # Check if the episode has any steps before proceeding
        try:
            first_step = next(iter(episode_steps_ds))
        except StopIteration:
            print(f"  [Bridge Adapter] Skipping empty episode.")
            continue

        episode_id = f"bridge_ep_{i}"
        output_dir = os.path.join(output_dir_base, episode_id)
        os.makedirs(output_dir, exist_ok=True)

        print(f"  [Bridge Adapter] Extracting episode: {episode_id}")

        language_instruction = first_step['language_instruction'][0].numpy().decode('utf-8')

        # --- Create Video from Image Sequence ---
        video_path = os.path.join(output_dir, "video.mp4")
        height, width = 256, 256  # From features.json
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))

        # Add the first frame we already extracted
        image_tensor = tf.io.decode_jpeg(first_step['observation/image'])
        frame = image_tensor.numpy()
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Process the rest of the frames
        for step in episode_steps_ds:
            image_tensor = tf.io.decode_jpeg(step['observation/image'])
            frame = image_tensor.numpy()
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        video_writer.release()

        # --- Create Manifest File ---
        manifest = {
            "goal": language_instruction,
            "language_prompt": language_instruction,
            "user_intent": "Task from Bridge Dataset."
        }
        with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
            json.dump(manifest, f, indent=4)

    print(f"--- [Bridge Adapter] Finished processing. ---")
