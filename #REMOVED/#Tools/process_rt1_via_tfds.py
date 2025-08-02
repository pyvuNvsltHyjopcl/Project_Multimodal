# process_rt1_via_tfds.py
#
# A Python script that:
# 1) Auto-detects the shard filename prefix in 'data_dir'.
# 2) Patches dataset_info.json to use that prefix for shards.
# 3) Patches features.json (converts numeric 'dimensions' fields to strings).
# 4) Loads RT-1 episodes via TFDS's as_data_source() (supports ArrayRecord).
# 5) Writes an MP4 and aligned metadata JSON for each episode.
#
# Usage example:
#   python process_rt1_via_tfds.py \
#     --data_dir ../data/raw/rt1 \
#     --out_dir ../data/processed/rt1_processed \
#     --max_episodes 10

import os
import json
import shutil
import argparse
import glob

import numpy as np
import imageio
import tensorflow as tf
import tensorflow_datasets as tfds


def detect_shard_prefix(data_dir):
    """
    Find one shard file in data_dir and return its prefix
    (the part before '-train.array_record-').
    """
    pattern = os.path.join(data_dir, '*-train.array_record-*')
    shards = glob.glob(pattern)
    if not shards:
        raise FileNotFoundError(f'No shards found with pattern {pattern}')
    fname = os.path.basename(shards[0])
    prefix = fname.split('-train.array_record-')[0]
    print(f"Detected shard prefix: '{prefix}'")
    return prefix


def patch_dataset_info(data_dir, prefix):
    """
    Update dataset_info.json's filepathTemplate to use the given prefix.
    """
    info_path = os.path.join(data_dir, 'dataset_info.json')
    bak = info_path + '.bak'
    if not os.path.exists(bak):
        shutil.copy(info_path, bak)
        print(f"Backed up dataset_info.json to {bak}")

    info = json.load(open(info_path))
    template = f"{prefix}-{{SPLIT}}.{{FILEFORMAT}}-{{SHARD_X_OF_Y}}"
    for split in info.get('splits', []):
        split['filepathTemplate'] = template
    json.dump(info, open(info_path, 'w'), indent=2)
    print(f"Patched dataset_info.json filepathTemplate to '{template}'\n")


def patch_features(data_dir):
    """
    Back up and patch features.json so all numeric 'dimensions' lists become strings.
    """
    feat_path = os.path.join(data_dir, 'features.json')
    bak = feat_path + '.bak'
    if not os.path.exists(bak):
        shutil.copy(feat_path, bak)
        print(f"Backed up features.json to {bak}")

    data = json.load(open(feat_path))
    def fix(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == 'dimensions' and isinstance(v, list):
                    o[k] = [str(x) for x in v]
                else:
                    fix(v)
        elif isinstance(o, list):
            for e in o:
                fix(e)
    fix(data)
    json.dump(data, open(feat_path, 'w'), indent=2)
    print("Patched numeric shapes in features.json\n")


def process(data_dir, out_dir, max_ep):
    # 1) Detect shard prefix
    prefix = detect_shard_prefix(data_dir)

    # 2) Patch dataset_info.json with that prefix
    patch_dataset_info(data_dir, prefix)

    # 3) Patch features.json
    patch_features(data_dir)

    # 4) Load dataset via TFDS
    builder = tfds.builder_from_directory(data_dir)
    source = builder.as_data_source()
    ds = source['train']

    total = len(ds)
    print(f"Total episodes available: {total}\n")
    os.makedirs(out_dir, exist_ok=True)

    # 5) Iterate and process
    for idx in range(total if not max_ep else min(total, max_ep)):
        ep = ds[idx]

        # Extract frames (nested or flat)
        try:
            frames = ep['steps']['observation']['image']
        except Exception:
            frames = ep['steps/observation/image']

        # Extract instruction
        try:
            instr = ep['steps']['observation']['natural_language_instruction'][0]
        except Exception:
            instr = ep['steps/observation/natural_language_instruction'][0]
        if isinstance(instr, (bytes, bytearray)):
            instr = instr.decode('utf-8')

        # Extract actions (nested or flat)
        actions = []
        if 'steps' in ep and 'action' in ep['steps']:
            act_container = ep['steps']['action']
            if isinstance(act_container, dict):
                T = frames.shape[0]
                for t in range(T):
                    vec = []
                    for v in act_container.values():
                        vec.extend(v[t].tolist())
                    actions.append(vec)
            else:
                actions = act_container.tolist()
        else:
            action_keys = sorted(k for k in ep.keys() if k.startswith('steps/action/'))
            if action_keys:
                T = frames.shape[0]
                for t in range(T):
                    vec = []
                    for k in action_keys:
                        vec.extend(ep[k][t].tolist())
                    actions.append(vec)

        # Episode-level success flag
        try:
            succ = bool(ep['aspects']['success'])
        except Exception:
            succ = bool(ep.get('aspects/success', False))

        # Write video
        vid_path = os.path.join(out_dir, f"rt1_ep_{idx:05d}.mp4")
        imageio.mimsave(vid_path, frames, fps=10)

        # Write metadata
        meta = {
            'instruction': instr,
            'num_steps': frames.shape[0],
            'success': succ,
            'actions': actions
        }
        with open(os.path.join(out_dir, f"rt1_ep_{idx:05d}_metadata.json"), 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"[{idx+1:4d}/{total}] Saved {vid_path} (+{frames.shape[0]} frames)")

    print("Processing complete.")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Process RT-1 with TFDS pipeline')
    p.add_argument('--data_dir', required=True, help='Path to data/raw/rt1')
    p.add_argument('--out_dir', required=True, help='Path to write processed files')
    p.add_argument('--max_episodes', type=int, default=0, help='Max episodes to process')
    args = p.parse_args()
    process(args.data_dir, args.out_dir, args.max_episodes)
