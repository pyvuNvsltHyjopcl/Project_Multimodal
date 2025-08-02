# #!/usr/bin/env python3
# import os
# import json
# import cv2
# from array_record import ArrayRecord  # reader for .array_record Riegeli files :contentReference[oaicite:2]{index=2}
# 
# def process_rt1_direct(dataset_dir, output_dir, max_eps=2):
#     """
#     Directly reads RT‑1 ArrayRecord shards and writes video.mp4 + metadata.json
#     for up to max_eps episodes.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     # List and sort all .array_record shard files
#     shards = sorted(f for f in os.listdir(dataset_dir) if f.endswith(".array_record"))
#     ep_count = 0
# 
#     for shard in shards:
#         shard_path = os.path.join(dataset_dir, shard)
#         # Open the shard for random access
#         arr = ArrayRecord(shard_path)  # ArrayRecord supports __getitem__ and len :contentReference[oaicite:3]{index=3}
#         num_rec = len(arr)
# 
#         for idx in range(num_rec):
#             if ep_count >= max_eps:
#                 return
#             data = arr[idx]  # a dict mapping feature names → NumPy arrays
#             ep_id = f"rt1_ep_{ep_count}"
#             ep_dir = os.path.join(output_dir, ep_id)
#             os.makedirs(ep_dir, exist_ok=True)
# 
#             # Extract language (if present) and image sequence
#             lang = data.get("language_instruction", b"N/A")
#             if isinstance(lang, (bytes, bytearray)):
#                 try:
#                     lang = lang.decode("utf-8")
#                 except Exception:
#                     lang = "N/A"
# 
#             imgs = data.get("observation/image") or data.get("image") or []
#             if not len(imgs):
#                 print(f"[WARN] No images in {shard}[{idx}]—skipping.")
#                 continue
# 
#             # Write video.mp4 at 30fps
#             first = imgs[0]
#             h, w, _ = first.shape
#             writer = cv2.VideoWriter(
#                 os.path.join(ep_dir, "video.mp4"),
#                 cv2.VideoWriter_fourcc(*"mp4v"),
#                 30.0, (w, h)
#             )
#             for frame in imgs:
#                 writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
#             writer.release()
# 
#             # Write metadata.json
#             manifest = {
#                 "goal": lang,
#                 "language_prompt": lang,
#                 "user_intent": "Task from RT-1 Dataset"
#             }
#             with open(os.path.join(ep_dir, "metadata.json"), "w") as mf:
#                 json.dump(manifest, mf, indent=4)
# 
#             print(f"[INFO] Wrote {ep_id} → {len(imgs)} frames + metadata.")
#             ep_count += 1
# 
# if __name__ == "__main__":
#     import argparse
#     p = argparse.ArgumentParser()
#     p.add_argument("--dataset_dir", required=True,
#                    help="Path to RT-1 folder containing .array_record shards")
#     p.add_argument("--output_dir",  required=True,
#                    help="Where to write video/mp4 and metadata.json")
#     p.add_argument("--max_eps",  type=int, default=2,
#                    help="How many episodes to process")
#     args = p.parse_args()
# 
#     process_rt1_direct(args.dataset_dir, args.output_dir, args.max_eps)
################################################################################



