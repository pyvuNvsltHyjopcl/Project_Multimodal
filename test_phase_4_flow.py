import os
import glob
import json
import shutil
import cv2  # We need cv2 for the test

# --- Configuration for the Test Run ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_LAKE_ROOT = os.path.join(PROJECT_ROOT, 'data')
PROCESSED_FOLDER = os.path.join(DATA_LAKE_ROOT, 'processed')
CURATED_FOLDER = os.path.join(DATA_LAKE_ROOT, 'curated')
REJECTED_FOLDER = os.path.join(DATA_LAKE_ROOT, 'rejected')


# --- Self-Contained Functions for Testing ---
# We copy the logic from curator.py directly into our test script.
def calculate_video_blur_score(video_path):
    if not os.path.exists(video_path): return 0.0
    video_capture = cv2.VideoCapture(video_path)
    total_variance, frame_count = 0, 0
    while frame_count < 50:
        success, frame = video_capture.read()
        if not success: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        total_variance += cv2.Laplacian(gray, cv2.CV_64F).var()
        frame_count += 1
    video_capture.release()
    return (total_variance / frame_count) if frame_count > 0 else 0.0


def run_quality_checks(aligned_json_path):
    print(f"  [QC] Running checks on: {os.path.basename(aligned_json_path)}")
    with open(aligned_json_path, 'r') as f:
        data = json.load(f)
    scores = {}
    for sensor in data.get("sensor_data", []):
        if sensor.get("modality") == "video":
            video_path = sensor.get("file_path")
            if not os.path.isabs(video_path):
                video_path = os.path.abspath(os.path.join(os.path.dirname(aligned_json_path), '..', video_path))
            blur_score = calculate_video_blur_score(video_path)
            scores["video_blur_score"] = round(blur_score, 2)
            print(f"    - Video Blur Score: {scores['video_blur_score']}")
    return scores


def main():
    """Simulates the entire Phase 4 curation workflow."""
    print("====== STARTING PHASE 4 WORKFLOW SIMULATION ======\n")

    # Ensure output directories exist
    os.makedirs(CURATED_FOLDER, exist_ok=True)
    os.makedirs(REJECTED_FOLDER, exist_ok=True)

    # 1. Find records to process
    records_to_curate = glob.glob(os.path.join(PROCESSED_FOLDER, "aligned_*.json"))
    if not records_to_curate:
        print("No aligned records from Phase 3 found. Please run the Phase 3 test first.")
        return
    print(f"Found {len(records_to_curate)} records to curate.")

    # 2. Run QC and Filter for each record
    for json_path in records_to_curate:
        quality_scores = run_quality_checks(json_path)

        passes_blur_check = quality_scores.get("video_blur_score", 0) >= 100.0
        destination_folder = CURATED_FOLDER if passes_blur_check else REJECTED_FOLDER

        print(
            f"  [ROUTER] Decision: {'PASS' if passes_blur_check else 'FAIL'}. Moving to {os.path.basename(destination_folder)}.")

        # Move files
        shutil.move(json_path, os.path.join(destination_folder, os.path.basename(json_path)))
        media_folder_name = os.path.splitext(os.path.basename(json_path))[0].replace('aligned_', '')
        media_folder_path = os.path.join(PROCESSED_FOLDER, media_folder_name)
        if os.path.exists(media_folder_path):
            shutil.move(media_folder_path, os.path.join(destination_folder, media_folder_name))

    # 3. Export Gold Standard
    print("\n--- Exporting Gold Standard Dataset ---")
    curated_files = glob.glob(os.path.join(CURATED_FOLDER, "aligned_*.json"))
    if curated_files:
        with open(curated_files[0], 'r+') as f:
            data = json.load(f)
            data["is_gold_standard"] = True
            f.seek(0);
            json.dump(data, f, indent=4);
            f.truncate()
        print(f"  Tagged {os.path.basename(curated_files[0])} as gold standard.")

    print("\n====== PHASE 4 WORKFLOW SIMULATION COMPLETE ======")


if __name__ == "__main__":
    main()
