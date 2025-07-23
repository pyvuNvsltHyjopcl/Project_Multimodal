import cv2
import json
import os
import numpy as np


def calculate_video_blur_score(video_path):
    """
    Calculates a blurriness score for a video by averaging the Laplacian
    variance of its frames. A higher score means a sharper video.
    """
    if not os.path.exists(video_path):
        return 0.0

    video_capture = cv2.VideoCapture(video_path)
    total_variance = 0
    frame_count = 0

    while frame_count < 50:  # Analyze up to the first 50 frames for speed
        success, frame = video_capture.read()
        if not success:
            break

        # Convert to grayscale and calculate Laplacian variance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        total_variance += variance
        frame_count += 1

    video_capture.release()

    if frame_count == 0:
        return 0.0

    return total_variance / frame_count


def run_quality_checks(aligned_json_path):
    """
    Runs all quality checks on a given data record and returns the scores.
    """
    print(f"  [QC] Running checks on: {os.path.basename(aligned_json_path)}")

    if not os.path.exists(aligned_json_path):
        print("  [QC] ERROR: JSON file not found.")
        return None

    with open(aligned_json_path, 'r') as f:
        data = json.load(f)

    quality_scores = {}

    # Find the video and audio files from the JSON record
    video_path = None
    for sensor in data.get("sensor_data", []):
        if sensor.get("modality") == "video":
            # Construct the absolute path if it's relative
            video_path = sensor.get("file_path")
            if not os.path.isabs(video_path):
                video_path = os.path.join(os.path.dirname(aligned_json_path), '..', video_path)  # Adjust path if needed

    # Run checks if files are found
    if video_path:
        blur_score = calculate_video_blur_score(video_path)
        quality_scores["video_blur_score"] = round(blur_score, 2)
        print(f"    - Video Blur Score: {quality_scores['video_blur_score']}")

    # Add a placeholder for audio checks
    quality_scores["audio_rms"] = round(np.random.uniform(0.01, 0.1), 4)
    print(f"    - Audio RMS (simulated): {quality_scores['audio_rms']}")

    return quality_scores
