#!/usr/bin/env python3
import os
import argparse
import logging
from google.cloud import storage                          # RT‑1 anonymous download :contentReference[oaicite:6]{index=6}
import tensorflow_datasets as tfds                        # RoboNet sample :contentReference[oaicite:7]{index=7}
from huggingface_hub import snapshot_download            # BridgeData snapshot :contentReference[oaicite:8]{index=8}
import requests                                           # HTTP streaming
from tqdm import tqdm                                     # Progress bars :contentReference[oaicite:9]{index=9}

# ----------------------------
# Logging configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)                                                        # basicConfig usage :contentReference[oaicite:10]{index=10}
logger = logging.getLogger(__name__)

# ----------------------------
# RT‑1: anonymous GCS download
# ----------------------------
def download_rt1(base_dir, num_episodes):
    rt1_dir = os.path.join(base_dir, "rt1")
    os.makedirs(rt1_dir, exist_ok=True)
    logger.info(f"RT‑1 → downloading {num_episodes} episodes to {rt1_dir}")

    client = storage.Client.create_anonymous_client()
    bucket = client.bucket("gresearch")
    blobs = bucket.list_blobs(prefix="rt-1-data-release/")

    count = 0
    for blob in blobs:
        name = os.path.basename(blob.name)
        if count >= num_episodes:
            break
        if name.lower().endswith(".txt"):
            continue
        dest = os.path.join(rt1_dir, name)
        blob.download_to_filename(dest)
        logger.info(f"  • fetched {name}")
        count += 1

# ----------------------------
# RoboNet: TensorFlow Datasets sample
# ----------------------------
def download_robonet_sample(base_dir, num_episodes):
    rn_dir = os.path.join(base_dir, "robonet")
    os.makedirs(rn_dir, exist_ok=True)
    logger.info(f"RoboNet → loading {num_episodes} sample episodes into {rn_dir}")

    tfds.load(
        "robonet/robonet_sample_64",                 # 64×64 RoboNet sample :contentReference[oaicite:11]{index=11}
        split=f"train[:{num_episodes}]",             # first N episodes :contentReference[oaicite:12]{index=12}
        data_dir=rn_dir,
        download=True
    )
    logger.info("  • RoboNet sample ready via TFDS")

# ----------------------------
# BridgeData: Hugging Face snapshot + limited extract
# ----------------------------
def download_bridge(base_dir, num_files):
    bd_dir = os.path.join(base_dir, "bridge")
    os.makedirs(bd_dir, exist_ok=True)
    logger.info(f"BridgeData → snapshot downloading dusty-nv/bridge_orig_ep100 to {bd_dir}")

    repo_dir = snapshot_download(
        repo_id="dusty-nv/bridge_orig_ep100",
        repo_type="dataset",
        local_dir=bd_dir,
        local_dir_use_symlinks=False
    )

    files = sorted(f for f in os.listdir(repo_dir) if f.endswith(".hdf5"))
    for i, fname in enumerate(files):
        if i >= num_files:
            break
        src = os.path.join(repo_dir, fname)
        dst = os.path.join(bd_dir, fname)
        os.replace(src, dst)
        logger.info(f"  • extracted {fname}")

# ----------------------------
# Main & Argument Parsing
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download RT‑1, sample RoboNet, and a subset of BridgeData"
    )
    parser.add_argument(
        "--base_dir",
        default=r"D:\Users\OLoca\PycharmProjects\Project_Multimodal\data\raw",
        help="Root folder for rt1/, robonet/, and bridge/ (default: %(default)s)"
    )                                                # argparse usage :contentReference[oaicite:13]{index=13}
    parser.add_argument(
        "--num_rt1", type=int, default=20,
        help="Number of RT‑1 episodes to fetch"
    )
    parser.add_argument(
        "--num_robonet", type=int, default=15,
        help="Number of RoboNet sample episodes to load"
    )
    parser.add_argument(
        "--num_bridge", type=int, default=5,
        help="Number of BridgeData files to extract"
    )
    args = parser.parse_args()

    logger.info(f"Saving all data into: {args.base_dir}")
    download_rt1(args.base_dir, args.num_rt1)
    download_robonet_sample(args.base_dir, args.num_robonet)
    download_bridge(args.base_dir, args.num_bridge)
    logger.info("All downloads and extractions complete.")

if __name__ == "__main__":
    main()
