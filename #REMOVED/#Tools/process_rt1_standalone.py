#!/usr/bin/env python3
"""
process_rt1_standalone.py

Reads RT-1 episodes from ArrayRecord shards under --input_dir,
and for each:
  • Parses as tf.train.SequenceExample
  • Compiles camera frames into an MP4
  • Dumps a metadata JSON with instruction, actions, and success

Usage:
  python process_rt1_standalone.py \
    --input_dir ../data/raw/rt1 \
    --output_dir ../data/processed/rt1_processed \
    [--max_episodes 100]
"""

import os, glob, json, argparse
import numpy as np
import imageio
import tensorflow as tf
from array_record.python.array_record_data_source import ArrayRecordDataSource

def parse_episode(raw_bytes):
    # Parse raw bytes as a SequenceExample
    seq = tf.train.SequenceExample()
    seq.ParseFromString(raw_bytes)

    # 1) Extract frame bytes from the image feature-list
    img_flist = seq.feature_lists.feature_list['steps/observation/image']
    frames = []
    for feat in img_flist.feature:
        img_bytes = feat.bytes_list.value[0]
        # Decode JPEG→uint8 array
        arr = tf.io.decode_jpeg(img_bytes).numpy()
        frames.append(arr)
    frames = np.stack(frames)  # shape (T, H, W, 3)

    # 2) Instruction (take first step's instruction text)
    instr_flist = seq.feature_lists.feature_list[
        'steps/observation/natural_language_instruction'
    ]
    instr_bytes = instr_flist.feature[0].bytes_list.value[0]
    instruction = instr_bytes.decode('utf-8')

    # 3) Actions: collect all 'steps/action/...' feature-lists,
    #    concatenate each step's floats into one vector
    act_keys = [k for k in seq.feature_lists.feature_list
                if k.startswith('steps/action/')]
    T = len(frames)
    actions = []
    for t in range(T):
        step_vec = []
        for key in sorted(act_keys):
            feat = seq.feature_lists.feature_list[key].feature[t]
            step_vec.extend(feat.float_list.value)
        actions.append(step_vec)

    # 4) Success flag (episode-level context feature, if present)
    success = False
    ctx = seq.context.feature
    if 'aspects/success' in ctx:
        success = bool(ctx['aspects/success'].int64_list.value[0])

    return frames, instruction, actions, success

def main(input_dir, output_dir, max_ep):
    shards = sorted(glob.glob(os.path.join(input_dir, '*-train.array_record-*')))
    if not shards:
        raise RuntimeError("No shards found in " + input_dir)
    print(f"Found {len(shards)} shards.")

    ds = ArrayRecordDataSource(shards)
    os.makedirs(output_dir, exist_ok=True)

    for idx, raw in enumerate(ds):
        if max_ep and idx >= max_ep:
            break

        frames, inst, acts, succ = parse_episode(raw)

        # Save video
        vid_path = os.path.join(output_dir, f"rt1_ep_{idx:05d}.mp4")
        imageio.mimsave(vid_path, frames, fps=10)

        # Save metadata
        meta = {
            "instruction": inst,
            "num_steps": len(frames),
            "success": succ,
            "actions": acts
        }
        meta_path = os.path.join(output_dir, f"rt1_ep_{idx:05d}_metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"[{idx+1:4d}] → {vid_path} (+{len(frames)} frames)")

    print("All done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",  required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_episodes", type=int, default=0,
                   help="Only process this many episodes (0 ⇒ all)")
    args = p.parse_args()
    main(args.input_dir, args.output_dir, args.max_episodes)
