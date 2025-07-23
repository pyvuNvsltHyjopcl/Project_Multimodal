import cv2
import os
from pydub import AudioSegment
import json


# --- Video Processing Machine ---
def normalize_video(source_path, output_folder, target_fps=30, target_resolution=(1280, 720)):
    """
    Reads a video, standardizes its FPS and resolution, and saves the new video.

    Args:
        source_path (str): Path to the raw video file.
        output_folder (str): Folder to save the normalized video in.
        target_fps (int): The desired frames per second.
        target_resolution (tuple): The desired (width, height).

    Returns:
        str: The path to the newly created normalized video file, or None on failure.
    """
    print(f"  [VIDEO] Normalizing: {os.path.basename(source_path)}")
    if not os.path.exists(source_path):
        print(f"  [VIDEO] ERROR: Source file not found at {source_path}")
        return None

    video_capture = cv2.VideoCapture(source_path)
    if not video_capture.isOpened():
        print(f"  [VIDEO] ERROR: Could not open video file {source_path}")
        return None

    # Define the output file path and the video writer
    output_filename = f"normalized_{os.path.basename(source_path)}"
    output_path = os.path.join(output_folder, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, target_fps, target_resolution)

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        # Resize the frame to the target resolution
        resized_frame = cv2.resize(frame, target_resolution)
        video_writer.write(resized_frame)

    video_capture.release()
    video_writer.release()
    print(f"  [VIDEO] Saved normalized video to: {output_path}")
    return output_path


# --- Audio Processing Machine ---
def normalize_audio(source_path, output_folder, target_sample_rate=22050):
    """
    Reads an audio file, standardizes its sample rate, and saves it as a WAV file.

    Args:
        source_path (str): Path to the raw audio file.
        output_folder (str): Folder to save the normalized audio in.
        target_sample_rate (int): The desired sample rate in Hz.

    Returns:
        str: The path to the newly created normalized audio file, or None on failure.
    """
    print(f"  [AUDIO] Normalizing: {os.path.basename(source_path)}")
    if not os.path.exists(source_path):
        print(f"  [AUDIO] ERROR: Source file not found at {source_path}")
        return None

    try:
        audio = AudioSegment.from_file(source_path)

        # Resample the audio to the target rate
        resampled_audio = audio.set_frame_rate(target_sample_rate)

        # Define the output path (WAV is a good standard format)
        output_filename = f"normalized_{os.path.splitext(os.path.basename(source_path))[0]}.wav"
        output_path = os.path.join(output_folder, output_filename)

        resampled_audio.export(output_path, format="wav")
        print(f"  [AUDIO] Saved normalized audio to: {output_path}")
        return output_path
    except Exception as e:
        print(f"  [AUDIO] ERROR: Failed to process audio file. {e}")
        return None


# --- Language Processing Machine ---
def process_language_file(source_path):
    """
    Reads the metadata JSON file and extracts the task context.

    Args:
        source_path (str): Path to the metadata.json file.

    Returns:
        dict: The task context dictionary, or None on failure.
    """
    print(f"  [LANG] Processing: {os.path.basename(source_path)}")
    if not os.path.exists(source_path):
        print(f"  [LANG] ERROR: Source file not found at {source_path}")
        return None

    try:
        with open(source_path, 'r') as f:
            data = json.load(f)
        print("  [LANG] Successfully extracted task context.")
        return data
    except Exception as e:
        print(f"  [LANG] ERROR: Failed to read or parse JSON. {e}")
        return None
