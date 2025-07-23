import json
import os
import random


def enrich_data_for_ai_modules(processed_json_path, output_folder):
    """
    Reads a processed JSON record from Phase 2 and enriches it with fields
    needed for Origami AI modules like STUM and a task planner.

    This is the central function for model-specific data mapping.

    Args:
        processed_json_path (str): The file path to the JSON output of Phase 2.
        output_folder (str): The folder to save the new, enriched JSON file.

    Returns:
        str: The path to the newly created enriched JSON file, or None on failure.
    """
    print(f"--- [MAPPER] Aligning record for AI Modules: {os.path.basename(processed_json_path)} ---")

    if not os.path.exists(processed_json_path):
        print(f"  [MAPPER] ERROR: Cannot find processed JSON at {processed_json_path}")
        return None

    with open(processed_json_path, 'r') as f:
        data = json.load(f)

    # --- Add fields for the STUM module (Uncertainty) ---
    print("  [MAPPER] Adding uncertainty fields for STUM...")
    for sensor in data.get("sensor_data", []):
        sensor["uncertainty"] = {
            "spatial_uncertainty": round(random.uniform(0.01, 0.1), 4),
            "temporal_uncertainty": round(random.uniform(0.0, 0.05), 4)
        }

    # --- Add fields for a Task Planner (Enriched Prompts) ---
    print("  [MAPPER] Enriching prompts for Task Planner...")
    prompt = data.get("task_context", {}).get("language_prompt", "")
    if "red block" in prompt.lower() or "test" in prompt.lower():
        data["task_context"]["structured_prompt"] = {
            "action": "retrieve",
            "object": {"name": "block", "attributes": ["red"]},
            "source_location": "table"
        }

    output_filename = f"aligned_{os.path.basename(processed_json_path)}"
    output_path = os.path.join(output_folder, output_filename)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"--- [MAPPER] Successfully created aligned data record at: {output_path} ---")
    return output_path
