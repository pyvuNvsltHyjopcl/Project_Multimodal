# #robonet_adapter.py
# import h5py
# import json
# import os
# import cv2
# import numpy as np
#
#
# def translate_robonet_to_standard_json(hdf5_path, output_dir):
#     """
#     Reads a trajectory from a RoboNet HDF5 file, extracts the video and
#     metadata, and translates it into our project's standard format.
#
#     Args:
#         hdf5_path (str): Path to the RoboNet .hdf5 file.
#         output_dir (str): The 'session' directory to save the output files.
#
#     Returns:
#         str: Path to the newly created metadata.json file, or None on failure.
#     """
#     print(f"--- [RoboNet Adapter] Translating {os.path.basename(hdf5_path)} ---")
#
#     if not os.path.exists(hdf5_path):
#         print(f"--- [RoboNet Adapter] ERROR: HDF5 file not found at {hdf5_path} ---")
#         return None
#
#     os.makedirs(output_dir, exist_ok=True)
#
#     try:
#         with h5py.File(hdf5_path, 'r') as hf:
#             # --- 1. Extract Video Data ---
#             # RoboNet often stores video as compressed JPEG images. We need to
#             # decode them and re-encode them into a standard MP4 video.
#             if 'observations/images' not in hf:
#                 print("--- [RoboNet Adapter] ERROR: No image data found in HDF5 file. ---")
#                 return None
#
#             image_dataset = hf['observations/images']
#             num_frames = image_dataset.shape[0]
#
#             # Get video properties from the first frame
#             first_frame_jpg = image_dataset[0]
#             first_frame_decoded = cv2.imdecode(np.frombuffer(first_frame_jpg, np.uint8), cv2.IMREAD_COLOR)
#             height, width, _ = first_frame_decoded.shape
#
#             video_output_path = os.path.join(output_dir, "video_from_robonet.mp4")
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             # RoboNet videos are often 15 fps
#             video_writer = cv2.VideoWriter(video_output_path, fourcc, 15.0, (width, height))
#
#             print(f"  [RoboNet Adapter] Extracting {num_frames} frames into video...")
#             for i in range(num_frames):
#                 jpg_data = image_dataset[i]
#                 frame = cv2.imdecode(np.frombuffer(jpg_data, np.uint8), cv2.IMREAD_COLOR)
#                 video_writer.write(frame)
#
#             video_writer.release()
#             print(f"  [RoboNet Adapter] Saved video to {video_output_path}")
#
#             # --- 2. Extract Metadata ---
#             # We extract attributes stored on the HDF5 groups.
#             robot_type = hf.attrs.get('robot', 'unknown_robot')
#             task_description = hf.attrs.get('description', 'No description available.')
#
#             # --- 3. Create the Manifest File ---
#             manifest = {
#                 "goal": task_description,
#                 "language_prompt": task_description,  # Often the same in RoboNet
#                 "user_intent": f"Task from RoboNet dataset, performed by {robot_type}."
#             }
#
#             manifest_path = os.path.join(output_dir, "metadata.json")
#             with open(manifest_path, 'w') as f:
#                 json.dump(manifest, f, indent=4)
#
#             print(f"--- [RoboNet Adapter] Created manifest at {manifest_path} ---")
#             return manifest_path
#
#     except Exception as e:
#         print(f"--- [RoboNet Adapter] An error occurred: {e} ---")
#         return None
import tensorflow as tf
import json
import os
import cv2
import numpy as np


def translate_robonet_to_standard_json(tfrecord_path, output_dir_base):
    """
    Reads episodes from a RoboNet TFRecord file, extracts the video stream,
    and translates it into our project's standard session format.
    """
    print(f"--- [RoboNet Adapter] Processing {os.path.basename(tfrecord_path)} ---")

    # Define the feature description based on RoboNet's features.json
    feature_description = {
        'video': tf.io.FixedLenFeature([], dtype=tf.string),
        'actions': tf.io.FixedLenFeature([100, 5], dtype=tf.float32),
    }

    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = raw_dataset.map(_parse_function)

    for i, record in enumerate(parsed_dataset):
        if i >= 2:  # Limit to 2 episodes for a sample run
            break

        episode_id = f"robonet_ep_{i}"
        output_dir = os.path.join(output_dir_base, episode_id)
        os.makedirs(output_dir, exist_ok=True)

        print(f"  [RoboNet Adapter] Extracting episode: {episode_id}")

        # --- Create Video ---
        # In RoboNet TFRecords, the entire video is often one encoded string
        video_tensor = tf.io.decode_video(record['video'])
        video_frames = video_tensor.numpy()

        video_path = os.path.join(output_dir, "video.mp4")
        height, width, _ = video_frames[0].shape
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))
        for frame in video_frames:
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        video_writer.release()

        # --- Create Manifest File ---
        # RoboNet does not have language instructions
        manifest = {
            "goal": "RoboNet visual demonstration",
            "language_prompt": "N/A",
            "user_intent": "Task from RoboNet Dataset (visual imitation)."
        }
        with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
            json.dump(manifest, f, indent=4)

    print(f"--- [RoboNet Adapter] Finished processing. ---")

