# process_rt1_manual.py
#
# A Python script that:
# 1) Auto-detects the RT-1 shard filename prefix in 'raw_dir'.
# 2) Patches dataset_info.json and features.json.
# 3) Processes RT-1 episodes directly from ArrayRecord shards using TensorFlow SequenceExample parsing.
# 4) Writes an MP4 and aligned metadata JSON for each episode.
#
# Usage example:
#   python process_rt1_manual.py \
#     --raw_dir ../data/raw/rt1 \
#     --out_dir ../data/processed/rt1_processed \
#     --max_episodes 10

import os
import glob
import json
import shutil
import argparse

import numpy as np
import imageio
import tensorflow as tf
from array_record.python.array_record_data_source import ArrayRecordDataSource


def detect_shard_prefix(raw_dir):
    pattern = os.path.join(raw_dir, '*-train.array_record-*')
    shards = glob.glob(pattern)
    if not shards:
        raise FileNotFoundError(f"No shards found in {raw_dir}")
    # take first filename
    fname = os.path.basename(shards[0])
    prefix = fname.split('-train.array_record-')[0]
    print(f"Detected shard prefix: '{prefix}'")
    return prefix


def patch_dataset_info(raw_dir, prefix):
    path = os.path.join(raw_dir, 'dataset_info.json')
    bak = path + '.bak'
    if not os.path.exists(bak):
        shutil.copy(path, bak)
        print(f"Backed up dataset_info.json to {bak}")
    info = json.load(open(path))
    # set filepathTemplate
    for split in info.get('splits', []):
        split['filepathTemplate'] = f"{prefix}-{{SPLIT}}.{{FILEFORMAT}}-{{SHARD_X_OF_Y}}"
    json.dump(info, open(path, 'w'), indent=2)
    print("Patched dataset_info.json template\n")


def patch_features(raw_dir):
    path = os.path.join(raw_dir, 'features.json')
    bak = path + '.bak'
    if not os.path.exists(bak):
        shutil.copy(path, bak)
        print(f"Backed up features.json to {bak}")
    data = json.load(open(path))
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
    json.dump(data, open(path, 'w'), indent=2)
    print("Patched features.json shapes\n")


def parse_sequence_example(raw_bytes):
    seq = tf.train.SequenceExample()
    seq.ParseFromString(raw_bytes)
    fl = seq.feature_lists.feature_list
    # frames
    feats = fl['steps/observation/image'].feature
    frames = np.stack([tf.io.decode_jpeg(f.bytes_list.value[0]).numpy() for f in feats])
    # instruction
    instr_b = fl['steps/observation/natural_language_instruction'].feature[0].bytes_list.value[0]
    instruction = instr_b.decode('utf-8')
    # actions
    keys = sorted(k for k in fl if k.startswith('steps/action/'))
    T = frames.shape[0]
    actions = []
    for t in range(T):
        vec = []
        for k in keys:
            vec.extend(fl[k].feature[t].float_list.value)
        actions.append(vec)
    # success
    ctx = seq.context.feature
    success = False
    if 'aspects/success' in ctx:
        v = ctx['aspects/success'].int64_list.value
        success = bool(v[0]) if v else False
    return frames, instruction, actions, success


def process(raw_dir, out_dir, max_ep):
    prefix = detect_shard_prefix(raw_dir)
    patch_dataset_info(raw_dir, prefix)
    patch_features(raw_dir)
    # collect shards
    pattern = os.path.join(raw_dir, f'{prefix}-train.array_record-*')
    shards = sorted(glob.glob(pattern))
    print(f"Found {len(shards)} shards to read.")
    ds = ArrayRecordDataSource(shards)
    total = len(ds)
    print(f"Total episodes: {total}\n")
    os.makedirs(out_dir, exist_ok=True)
    # iterate
    for idx, raw in enumerate(ds):
        if max_ep and idx >= max_ep:
            break
        try:
            frames, instr, acts, succ = parse_sequence_example(raw)
        except Exception as e:
            print(f"Skip {idx}: {e}")
            continue
        # save video
        vid = os.path.join(out_dir, f"rt1_ep_{idx:05d}.mp4")
        imageio.mimsave(vid, frames, fps=10)
        # save metadata
        meta = {
            'instruction': instr,
            'num_steps': frames.shape[0],
            'success': succ,
            'actions': acts
        }
        with open(os.path.join(out_dir, f"rt1_ep_{idx:05d}_metadata.json"), 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"[{idx+1:4d}] Saved {vid} ({frames.shape[0]} frames)")
    print("Processing complete.")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--max_episodes', type=int, default=0)
    args = p.parse_args()
    process(args.raw_dir, args.out_dir, args.max_episodes)
