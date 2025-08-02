#path: tools/patch_and_process_rt1.py
#!/usr/bin/env python3
"""
patch_and_process_rt1.py

1) Patches data/raw/rt1/features.json so TFDS will accept numeric shapes.
2) Uses TFDS’s .as_data_source() to load RT-1 directly from ArrayRecord.
3) For each episode (up to --max_episodes):
     • Saves an MP4 of the camera frames.
     • Dumps a metadata.json with instruction, action vectors, and success flag.

Usage:
  python patch_and_process_rt1.py \
    --input_dir ../data/raw/rt1 \
    --output_dir ../data/processed/rt1_processed \
    --max_episodes 10
"""

import os, json, shutil, argparse
import numpy as np
import imageio
import tensorflow as tf
import tensorflow_datasets as tfds

def patch_features(json_path):
    bak = json_path + ".bak"
    if not os.path.exists(bak):
        shutil.copy(json_path, bak)
    data = json.load(open(json_path))
    def fix(o):
        if isinstance(o, dict):
            for k,v in o.items():
                if k=="dimensions" and isinstance(v,list):
                    o[k] = [str(x) for x in v]
                else:
                    fix(v)
        elif isinstance(o,list):
            for e in o: fix(e)
    fix(data)
    json.dump(data, open(json_path,"w"), indent=2)

def process(rt1_dir, out_dir, max_ep):
    # 1) patch shapes → TFDS can read
    feat = os.path.join(rt1_dir, "features.json")
    patch_features(feat)

    # 2) build and use as_data_source()
    builder = tfds.builder_from_directory(rt1_dir)
    source = builder.as_data_source()
    train_ds = source["train"]  # an indexable, iterable data source of NumPy dicts

    os.makedirs(out_dir, exist_ok=True)
    for idx in range(min(len(train_ds), max_ep) if max_ep else len(train_ds)):
        ep = train_ds[idx]  # a dict where ep["steps"]["observation"]["image"] is an (T,H,W,3) array

        # a) frames & instruction
        frames = ep["steps"]["observation"]["image"]
        instr  = ep["steps"]["observation"]["natural_language_instruction"][0]
        if isinstance(instr, bytes): instr = instr.decode()

        # b) actions
        act = ep["steps"]["action"]
        if isinstance(act, dict):
            T = frames.shape[0]
            actions = []
            for t in range(T):
                vec = []
                for sub in sorted(act.values(), key=lambda x: x.shape):  # preserve key order
                    vec.extend(sub[t].tolist())
                actions.append(vec)
        else:
            actions = act.tolist()

        # c) success flag
        success = bool(ep.get("aspects",{}).get("success", False))

        # d) save video
        vid = os.path.join(out_dir, f"rt1_ep_{idx:05d}.mp4")
        imageio.mimsave(vid, frames, fps=10)

        # e) save metadata
        meta = {
            "instruction": instr,
            "num_steps": frames.shape[0],
            "success": success,
            "actions": actions
        }
        with open(os.path.join(out_dir, f"rt1_ep_{idx:05d}_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[{idx+1:4d}] → {vid} (+{frames.shape[0]} frames)")

    print("Done.")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",  required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_episodes", type=int, default=0)
    args = p.parse_args()
    process(args.input_dir, args.output_dir, args.max_episodes)
