#!/usr/bin/env python3
"""
download_rt1_fractal.py

Download the first N TFRecord shards of the public RT-1 fractal dataset
(frames + telemetry) from the gresearch bucket.
"""

import os
import argparse
import requests
from urllib.parse import quote_plus

# Base URLs for listing and downloading
JSON_API_BASE = "https://storage.googleapis.com/storage/v1"
RAW_BASE      = "https://storage.googleapis.com"

def list_shards(bucket: str, prefix: str):
    """
    List all object names under `prefix/` by paging through the JSON API,
    filtering for .tfrecord files.
    """
    url = f"{JSON_API_BASE}/b/{bucket}/o"
    params = {
        "prefix": prefix + "/",
        "fields": "nextPageToken,items/name",
    }
    while True:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("items", []):
            name = item["name"]
            if ".tfrecord" in name:
                yield name
        token = data.get("nextPageToken")
        if not token:
            break
        params["pageToken"] = token  # continue paging

def download_shard(bucket: str, shard_name: str, output_dir: str):
    """
    Stream-download a single shard via HTTP GET with alt=media.
    """
    url = f"{RAW_BASE}/{bucket}/{quote_plus(shard_name)}?alt=media"
    local_path = os.path.join(output_dir, os.path.relpath(shard_name, OUTPUT_PREFIX))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1<<20):
                if chunk:
                    f.write(chunk)
    print(f"Downloaded {shard_name} → {local_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Download first N TFRecord shards from RT-1 fractal dataset"
    )
    p.add_argument("--num_shards", type=int, default=2,
                   help="Number of shards to download")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to save .tfrecord files")
    args = p.parse_args()

    BUCKET = "gresearch"
    OUTPUT_PREFIX = "robotics/fractal20220817_data/0.1.0"

    # List and download
    shards = list(list_shards(BUCKET, OUTPUT_PREFIX))
    to_dl = shards[: args.num_shards]
    if not to_dl:
        raise RuntimeError(f"No TFRecord shards found under gs://{BUCKET}/{OUTPUT_PREFIX}")  # safegaurd
    for s in to_dl:
        download_shard(BUCKET, s, args.output_dir)
    print(f"Finished downloading {len(to_dl)} shard(s).")
