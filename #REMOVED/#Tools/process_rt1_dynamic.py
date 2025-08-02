# process_rt1_dynamic.py
#
# A Python script that:
# 1) Detects the RT-1 shard filename prefix.
# 2) Patches dataset_info.json and features.json for TFDS compatibility.
# 3) Reads ArrayRecord shards and parses each episode dynamically.
# 4) Auto-detects sequence feature-list keys for images, instructions, and actions.
# 5) Saves an MP4 and metadata JSON per episode.
#
# Usage:
#   python process_rt1_dynamic.py \
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
    shards = glob.glob(os.path.join(raw_dir, '*-train.array_record-*'))
    if not shards:
        raise FileNotFoundError(f"No shards found in {raw_dir}")
    prefix = os.path.basename(shards[0]).split('-train.array_record-')[0]
    print(f"Detected shard prefix: '{prefix}'")
    return prefix


def patch_dataset_info(raw_dir, prefix):
    info_path = os.path.join(raw_dir, 'dataset_info.json')
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


def patch_features(raw_dir):
    feat_path = os.path.join(raw_dir, 'features.json')
    bak = feat_path + '.bak'
    if not os.path.exists(bak):
        shutil.copy(feat_path, bak)
        print(f"Backed up features.json to {bak}")
    data = json.load(open(feat_path))
    def fix_dims(o):
        if isinstance(o, dict):
            for k,v in o.items():
                if k=='dimensions' and isinstance(v,list):
                    o[k] = [str(x) for x in v]
                else:
                    fix_dims(v)
        elif isinstance(o,list):
            for e in o:
                fix_dims(e)
    fix_dims(data)
    json.dump(data, open(feat_path, 'w'), indent=2)
    print("Patched features.json dimensions to strings\n")


def find_seq_key(feature_list, substring):
    """
    Return first key in feature_list containing substring with non-empty entries.
    """
    for k, fl in feature_list.items():
        if substring in k and len(fl.feature) > 0:
            return k
    raise KeyError(f"No non-empty sequence key containing '{substring}'")


def parse_sequence_example(raw_bytes):
    seq = tf.train.SequenceExample()
    seq.ParseFromString(raw_bytes)
    fl = seq.feature_lists.feature_list

    # Dynamic keys
    img_key  = find_seq_key(fl, 'image')
    inst_key = find_seq_key(fl, 'natural_language_instruction')
    act_keys = [k for k in fl if 'action' in k]

    # Decode frames
    feats = fl[img_key].feature
    frames = np.stack([tf.io.decode_jpeg(f.bytes_list.value[0]).numpy() for f in feats])

    # Decode instruction
    instr_b = fl[inst_key].feature[0].bytes_list.value[0]
    instruction = instr_b.decode('utf-8')

    # Decode actions
    T = frames.shape[0]
    actions = []
    for t in range(T):
        vec = []
        for k in sorted(act_keys):
            vec.extend(fl[k].feature[t].float_list.value)
        actions.append(vec)

    # Context success flag
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

    shards = sorted(glob.glob(os.path.join(raw_dir, f'{prefix}-train.array_record-*')))
    print(f"Found {len(shards)} shards to process.")
    ds = ArrayRecordDataSource(shards)

    os.makedirs(out_dir, exist_ok=True)
    for idx, raw in enumerate(ds):
        if max_ep and idx >= max_ep:
            break
        try:
            frames, instr, acts, succ = parse_sequence_example(raw)
        except Exception as e:
            print(f"Skip {idx}: {e}")
            continue

        # Save video
        vid = os.path.join(out_dir, f"rt1_ep_{idx:05d}.mp4")
        imageio.mimsave(vid, frames, fps=10)
        # Save metadata
        meta = {'instruction': instr,
                'num_steps': frames.shape[0],
                'success': succ,
                'actions': acts}
        with open(os.path.join(out_dir, f"rt1_ep_{idx:05d}_metadata.json"), 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"[{idx+1:4d}] Saved {vid} ({frames.shape[0]} frames)")

    print("Processing complete.")

if __name__=='__main__':
    p = argparse.ArgumentParser(description='Dynamic RT-1 processor')
    p.add_argument('--rt1_dir',     required=True, help='Path to raw RT-1 shards (data/raw/rt1)')
    p.add_argument('--output_dir',  required=True, help='Output directory for processed episodes')
    p.add_argument('--max_episodes',type=int, default=0, help='Max episodes to process (0 = all)')
    args = p.parse_args()
    process(args.rt1_dir, args.output_dir, args.max_episodes)
