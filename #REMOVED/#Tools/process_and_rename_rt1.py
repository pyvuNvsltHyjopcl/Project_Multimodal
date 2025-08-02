# process_rt1_via_tfds.py
#
# A Python script to process RT-1 episodes by:
# 1) Patching dataset_info.json to use the fractal prefix for shards.
# 2) Patching features.json so TFDS can parse numeric dimensions.
# 3) Loading episodes via TFDS's as_data_source() (ArrayRecord support).
# 4) Writing an MP4 and aligned metadata JSON per episode.
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

import numpy as np
import imageio
import tensorflow_datasets as tfds
import tensorflow as tf


def patch_dataset_info(data_dir):
    """
    Update dataset_info.json's filepathTemplate to match the fractal prefix of shard files.
    """
    info_path = os.path.join(data_dir, 'dataset_info.json')
    bak = info_path + '.bak'
    if not os.path.exists(bak):
        shutil.copy(info_path, bak)
        print(f"Backed up dataset_info.json to {bak}")

    info = json.load(open(info_path))
    fractal_prefix = (
        "fractal_fractal_20220817_data_traj_transform_rt_1_"
        "without_filters_disable_episode_padding_seq_length_6_no_preprocessor"
    )
    # Override the template
    for split in info.get('splits', []):
        split['filepathTemplate'] = f"{fractal_prefix}-{{SPLIT}}.{{FILEFORMAT}}-{{SHARD_X_OF_Y}}"
    json.dump(info, open(info_path, 'w'), indent=2)
    print("Patched dataset_info.json filepathTemplate to fractal prefix\n")


def patch_features(data_dir):
    """
    Back up and patch features.json so numeric 'dimensions' become strings.
    """
    feat_path = os.path.join(data_dir, 'features.json')
    bak = feat_path + '.bak'
    if not os.path.exists(bak):
        shutil.copy(feat_path, bak)
        print(f"Backed up features.json to {bak}")

    data = json.load(open(feat_path))
    def fix(o):
        if isinstance(o, dict):
            for k, v in list(o.items()):
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
    # 1) Patch dataset_info.json to use fractal shard prefix
    patch_dataset_info(data_dir)
    # 2) Patch features.json for TFDS parsing
    patch_features(data_dir)

    # 3) Load via TFDS
    builder = tfds.builder_from_directory(data_dir)
    ds = builder.as_data_source()['train']

    os.makedirs(out_dir, exist_ok=True)
    total = len(ds)
    print(f"Total episodes available: {total}\n")

    for idx in range(total if not max_ep else min(total, max_ep)):
        ep = ds[idx]
        frames = ep['steps']['observation']['image']
        instr = ep['steps']['observation']['natural_language_instruction'][0]
        if isinstance(instr, (bytes, bytearray)):
            instr = instr.decode('utf-8')

        act = ep['steps']['action']
        if isinstance(act, dict):
            T = frames.shape[0]
            actions = []
            for t in range(T):
                vec = []
                for k in sorted(act.keys()):
                    vec.extend(act[k][t].tolist())
                actions.append(vec)
        else:
            actions = act.tolist()

        succ = bool(ep.get('aspects', {}).get('success', False))

        # Write MP4
        vid_path = os.path.join(out_dir, f"rt1_ep_{idx:05d}.mp4")
        imageio.mimsave(vid_path, frames, fps=10)

        # Write metadata JSON
        meta = {
            'instruction': instr,
            'num_steps': frames.shape[0],
            'success': succ,
            'actions': actions
        }
        meta_path = os.path.join(out_dir, f"rt1_ep_{idx:05d}_metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"[{idx+1:4d}/{total}] Saved {vid_path} (+{frames.shape[0]} frames)")

    print("\nProcessing complete.")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Process RT-1 via TFDS pipeline')
    p.add_argument('--data_dir',  required=True, help='Path to data/raw/rt1')
    p.add_argument('--out_dir',   required=True, help='Output processed folder')
    p.add_argument('--max_episodes', type=int, default=0,
                   help='Max episodes to process (0=all)')
    args = p.parse_args()
    process(args.data_dir, args.out_dir, args.max_episodes)
