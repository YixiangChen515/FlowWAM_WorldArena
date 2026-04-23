#!/usr/bin/env python3
"""Generate summary.json for WorldArena evaluation from an eval output directory.

Usage:
    python generate_summary.py \
        --eval_dir /path/to/FlowWAM_eval \
        --test_dataset_dir /path/to/test_dataset

Scans {eval_dir}/*_test/ for episode*.mp4 files and builds summary.json
with gt_path, first-frame image path, and original instruction text.

The output summary.json is placed in {eval_dir}/summary.json, which can
then be passed to WorldArena evaluation scripts:

    # Standard metrics (8 perception metrics)
    bash run_evaluation.sh MODEL_NAME {eval_dir}/{model}_test {eval_dir}/summary.json "image_quality,..."

    # VLM judge (Interaction Quality, Perspectivity, Instruction Following)
    bash run_VLM_judge.sh MODEL_NAME {eval_dir}/{model}_test {eval_dir}/summary.json

    # Action Following (needs _test, _test_1, _test_2)
    bash run_action_following.sh MODEL_NAME {eval_dir}/{model}_test {eval_dir}/summary.json
"""

import argparse
import json
import os
import re
import glob


def natural_sort_key(name):
    parts = re.split(r'(\d+)', name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def detect_model_name(eval_dir):
    """Auto-detect model name from the *_test directory inside eval_dir."""
    for entry in os.listdir(eval_dir):
        full = os.path.join(eval_dir, entry)
        if os.path.isdir(full) and entry.endswith("_test") and not re.search(r"_test_\d+$", entry):
            return entry[:-len("_test")]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate summary.json for WorldArena evaluation.")
    parser.add_argument("--eval_dir", type=str, required=True,
                        help="Eval output directory (e.g. .../FlowWAM_eval)")
    parser.add_argument("--test_dataset_dir", type=str, required=True,
                        help="Path to WorldArena test_dataset/ directory")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name (auto-detected from eval_dir if omitted)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: {eval_dir}/summary.json)")
    args = parser.parse_args()

    eval_dir = os.path.abspath(args.eval_dir)

    model_name = args.model_name or detect_model_name(eval_dir)
    if model_name is None:
        print("ERROR: cannot detect model_name. Provide --model_name explicitly.")
        return

    video_dir = os.path.join(eval_dir, f"{model_name}_test")
    video_dir_1 = os.path.join(eval_dir, f"{model_name}_test_1")
    video_dir_2 = os.path.join(eval_dir, f"{model_name}_test_2")

    if not os.path.isdir(video_dir):
        print(f"ERROR: video directory not found: {video_dir}")
        return

    has_action_following = os.path.isdir(video_dir_1) and os.path.isdir(video_dir_2)

    first_frame_dir = os.path.join(
        args.test_dataset_dir, "first_frame", "fixed_scene_task")
    instruction_dir = os.path.join(
        args.test_dataset_dir, "instructions", "fixed_scene_task")
    gt_video_dir = os.path.join(
        args.test_dataset_dir, "data", "fixed_scene_task", "gt_video")

    video_files = sorted(
        glob.glob(os.path.join(video_dir, "episode*.mp4")),
        key=lambda p: natural_sort_key(os.path.basename(p)))

    summary = []
    missing_frame = 0
    missing_instr = 0

    for video_path in video_files:
        ep_name = os.path.splitext(os.path.basename(video_path))[0]

        first_frame_path = os.path.join(first_frame_dir, f"{ep_name}.png")
        if not os.path.exists(first_frame_path):
            first_frame_path = os.path.join(first_frame_dir, f"{ep_name}.jpg")
        if not os.path.exists(first_frame_path):
            missing_frame += 1
            first_frame_path = ""

        instruction_path = os.path.join(instruction_dir, f"{ep_name}.json")
        prompt = ""
        if os.path.exists(instruction_path):
            with open(instruction_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            prompt = data.get("instruction", "")
        else:
            missing_instr += 1

        gt_path = os.path.join(gt_video_dir, f"{ep_name}.mp4")

        summary.append({
            "gt_path": os.path.abspath(gt_path),
            "image": os.path.abspath(first_frame_path) if first_frame_path else "",
            "prompt": [prompt] if prompt else [""],
        })

    output_path = args.output or os.path.join(eval_dir, "summary.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    n_v1 = len(glob.glob(os.path.join(video_dir_1, "episode*.mp4"))) if has_action_following else 0
    n_v2 = len(glob.glob(os.path.join(video_dir_2, "episode*.mp4"))) if has_action_following else 0

    print(f"=== Summary Generation Complete ===")
    print(f"  model_name:    {model_name}")
    print(f"  eval_dir:      {eval_dir}")
    print(f"  output:        {output_path}")
    print(f"  episodes:      {len(summary)}")
    if missing_frame:
        print(f"  WARNING:       {missing_frame} episodes missing first_frame")
    if missing_instr:
        print(f"  WARNING:       {missing_instr} episodes missing instruction")

    print(f"\n=== Variant Status ===")
    print(f"  {model_name}_test:    {len(video_files)} videos")
    print(f"  {model_name}_test_1:  {n_v1} videos {'OK' if n_v1 else 'MISSING'}")
    print(f"  {model_name}_test_2:  {n_v2} videos {'OK' if n_v2 else 'MISSING'}")

    print(f"\n=== Available Evaluations ===")
    print(f"  Standard metrics (no GT needed):")
    print(f"    cd /path/to/WorldArena/video_quality")
    print(f"    bash run_evaluation.sh {model_name} {video_dir} {output_path} \\")
    print(f'      "image_quality,aesthetic_quality,subject_consistency,background_consistency,'
          f'dynamic_degree,flow_score,photometric_smoothness,motion_smoothness"')
    print()
    print(f"  VLM Judge (Interaction Quality, Perspectivity, Instruction Following):")
    print(f"    bash run_VLM_judge.sh {model_name} {video_dir} {output_path}")

    if has_action_following:
        print()
        print(f"  Action Following (3 variants detected):")
        print(f"    bash run_action_following.sh {model_name} {video_dir} {output_path}")
    else:
        print()
        print(f"  Action Following: UNAVAILABLE (need _test_1 and _test_2)")

    print()
    print(f"  Metrics needing GT video (Depth/Trajectory/Semantic/JEPA): "
          f"require gt_video files at {gt_video_dir}")


if __name__ == "__main__":
    main()
