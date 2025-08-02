#!/usr/bin/env python3
# rt1_translator.py
# A dedicated adapter for the RT-1 dataset: analyze its metadata, extract episodes into per-file metadata
# and video files in standardized session format.

import os
import json
import platform
import argparse

import tensorflow_datasets as tfds
import cv2
import numpy as np


def analyze_metadata(dataset_dir: str) -> dict:
    """
    Load and return the RT-1 dataset_info.json content if present.
    """
    info_path = os.path.join(dataset_dir, 'dataset_info.json')
    if not os.path.exists(info_path):
        print(f"[Analyzer] No dataset_info.json at {info_path}. Skipping metadata analysis.")
        return {}
    with open(info_path, 'r') as f:
        info = json.load(f)
    # Report key fields
    name = info.get('name', 'unknown')
    version = info.get('version', 'unknown')
    splits = info.get('splits', [])
    print(f"[Analyzer] Dataset name: {name}")
    print(f"[Analyzer] Version: {version}")
    for split in splits:
        print(f"[Analyzer] Split '{split.get('name')}', shards: {len(split.get('shardLengths', []))}")
    return info


def process_rt1(dataset_dir: str, output_base: str, num_episodes: int = 2) -> bool:
    """
    Processes the RT-1 dataset by extracting each episode into its own folder,
    writing out a video.mp4 and metadata.json per episode.
    """
    print(f"--- Processing RT-1 dataset at {dataset_dir} ---")

    # RT-1 uses ArrayRecord; skip on Windows
    if platform.system() == 'Windows':
        print("[RT-1 Adapter] ArrayRecord format not supported on Windows. Aborting.")
        return False

    # Analyze metadata first
    metadata = analyze_metadata(dataset_dir)

    try:
        builder = tfds.builder_from_directory(dataset_dir)
        dataset = builder.as_data_source(split='train')

        for i, ep in enumerate(dataset.take(num_episodes)):
            ep_id = f"rt1_ep_{i}"
            out_dir = os.path.join(output_base, ep_id)
            os.makedirs(out_dir, exist_ok=True)
            print(f"[RT-1 Adapter] Extracting {ep_id}")

            # Steps extraction
            if 'steps' not in ep:
                print(f"[RT-1 Adapter] No 'steps' in episode {ep_id}, skipping.")
                continue
            steps = list(ep['steps'])
            if not steps:
                print(f"[RT-1 Adapter] Empty steps for {ep_id}, skipping.")
                continue

            # Extract instruction
            first = steps[0]
            instruction = ""
            if 'language_instruction' in first:
                instruction = first['language_instruction'].numpy().decode('utf-8')

            # Build video
            images = [s['observation']['image'].numpy() for s in steps if 'observation' in s and 'image' in s['observation']]
            if not images:
                print(f"[RT-1 Adapter] No images found in {ep_id}, skipping.")
                continue

            h, w, _ = images[0].shape
            video_path = os.path.join(out_dir, 'video.mp4')
            vw = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w, h))
            for frame in images:
                vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            vw.release()

            # Write metadata.json
            manifest = {
                "episode_id": ep_id,
                "instruction": instruction,
                "num_steps": len(steps),
                "dataset_name": builder.info.name,
                "dataset_version": builder.info.version
            }
            with open(os.path.join(out_dir, 'metadata.json'), 'w') as mf:
                json.dump(manifest, mf, indent=4)

        print(f"--- Finished processing RT-1. Outputs at {output_base} ---")
        return True

    except Exception as e:
        print(f"[RT-1 Adapter] ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='RT-1 Dataset Adapter')
    parser.add_argument('--input_dir', required=True, help='Path to RT-1 TFDS directory')
    parser.add_argument('--output_dir', required=True, help='Where to save processed episodes')
    parser.add_argument('--num_eps', type=int, default=2, help='Number of episodes to process')
    args = parser.parse_args()
    process_rt1(args.input_dir, args.output_dir, args.num_eps)


if __name__ == '__main__':
    main()
