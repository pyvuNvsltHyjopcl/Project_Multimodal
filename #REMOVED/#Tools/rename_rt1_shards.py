# rename_rt1_shards.py
#
# A Python script to rename RT-1 array_record shards from the long fractal prefix
# to the TFDS-expected RT_1_paper_release prefix, matching the dataset_info.json.
#
# Usage:
#   python rename_rt1_shards.py --raw_dir /path/to/data/raw/rt1
#
import os
import argparse


def main(raw_dir):
    # The original long prefix in your filenames
    old_prefix = (
        "fractal_fractal_20220817_data_traj_transform_rt_1_"
        "without_filters_disable_episode_padding_seq_length_6_no_preprocessor"
    )
    # The new prefix expected by TFDS
    new_prefix = "RT_1_paper_release"

    # List all matching shard files
    pattern = f"{old_prefix}-train.array_record-*-of-*.array_record"
    # Actually your files end with "-of-01024" etc.
    # We'll glob for the generic pattern:
    files = [f for f in os.listdir(raw_dir)
             if f.startswith(old_prefix + "-train.array_record-")]

    if not files:
        print(f"No files found with prefix '{old_prefix}' in {raw_dir}")
        return

    for fname in sorted(files):
        # Extract shard suffix, e.g. "00000-of-01024"
        parts = fname.split("-train.array_record-")
        if len(parts) != 2:
            continue
        suffix = parts[1]
        new_name = f"{new_prefix}-train.array_record-{suffix}"
        src = os.path.join(raw_dir, fname)
        dst = os.path.join(raw_dir, new_name)
        print(f"Renaming '{fname}' → '{new_name}'")
        os.rename(src, dst)

    print("Rename complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Rename RT-1 shards to TFDS expected filenames"
    )
    parser.add_argument(
        "--raw_dir", required=True,
        help="Path to the raw RT-1 shard directory (e.g. data/raw/rt1)"
    )
    args = parser.parse_args()
    main(args.raw_dir)
