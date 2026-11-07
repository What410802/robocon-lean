#!/usr/bin/env python3
"""Compare the deployment-side depth-policy inference with the same policy in Python.

rl_sim with RL_DEPTH_DEBUG=1 writes /tmp/rl_depth_debug.csv:
    last_control_sim_time ; depth_frame_sim_time ; policy_input ; raw_action ; raw_metric_depth
where policy_input is the exact history vector fed to TorchScript, raw_action is the
model output before action clipping, and the depth column is the metric frame that
DepthInput() normalizes.  This script rebuilds the depth tensor from that raw frame
(the training/eval contract) and calls the same TorchScript from Python, then reports
the largest action difference.  Same maths as eval policy_runner.render_depth.

Usage (repo root, demo or eval env):
  RL_DEPTH_DEBUG=1 RL_DEPTH_DEBUG_MAX=4 <run a depth policy briefly>
  pixi run -e demo python demo/tests/check_depth_inference.py \
      --policy demo/rl_sar/policy/blackW/plum_depth_final_v1
"""
import argparse
import json
import sys

import numpy as np
import torch
import yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True,
                    help="policy directory (config.yaml + depth.json + policy.pt)")
    ap.add_argument("--debug", default="/tmp/rl_depth_debug.csv")
    ap.add_argument("--tolerance", type=float, default=1e-5,
                    help="max |C++ action - Python action| (float32 round trip)")
    args = ap.parse_args()

    with open(f"{args.policy}/depth.json") as f:
        depth = json.load(f)
    with open(f"{args.policy}/config.yaml") as f:
        config = yaml.safe_load(f)
    # Policy configs are keyed by "<robot>/<policy>"; pick that block.
    if len(config) == 1 and isinstance(next(iter(config.values())), dict):
        config = next(iter(config.values()))
    width, height = int(depth["width"]), int(depth["height"])
    cutoff = float(depth["cutoff"])
    min_depth = float(depth["min_depth"])
    action_size = int(config["num_of_dofs"])
    clip_upper = np.asarray(config.get("clip_actions_upper", [100.0] * action_size), dtype=np.float32)
    clip_lower = np.asarray(config.get("clip_actions_lower", [-100.0] * action_size), dtype=np.float32)

    model = torch.jit.load(f"{args.policy}/policy.pt")
    model.eval()

    rows = [line.split(";") for line in open(args.debug).read().splitlines() if line]
    if not rows:
        print(f"{args.debug} is empty; run rl_sim with RL_DEPTH_DEBUG=1 first")
        return 2

    worst = 0.0
    for row_index, row in enumerate(rows):
        if len(row) != 5:
            print(f"row {row_index}: expected 5 ';'-separated columns, got {len(row)}")
            return 2
        obs = np.asarray([float(v) for v in row[2].split(",")], dtype=np.float32)
        cpp_action = np.asarray([float(v) for v in row[3].split(",")], dtype=np.float32)
        raw_depth = np.asarray([float(v) for v in row[4].split(",")], dtype=np.float32)
        if raw_depth.size != width * height:
            print(f"row {row_index}: depth has {raw_depth.size} values, contract is {width}x{height}")
            return 2
        metric = raw_depth.reshape(height, width)
        normalized = np.where(np.isfinite(metric) & (metric > 0.0), metric, cutoff)
        normalized = np.clip(normalized, min_depth, cutoff) / cutoff
        with torch.no_grad():
            depth_tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
            python_action = model(torch.from_numpy(obs).unsqueeze(0),
                                  [depth_tensor]).squeeze(0).numpy()
        diff = np.abs(python_action - cpp_action)
        worst = max(worst, float(diff.max()))
        clipped = np.clip(python_action, clip_lower, clip_upper)
        print(f"row {row_index:2d} sim_time={float(row[0]):8.3f} depth_time={float(row[1]):8.3f} "
              f"obs={obs.size} action={python_action.size} max_abs_diff={diff.max():.3e} "
              f"clipped_diff={np.abs(clipped - np.clip(cpp_action, clip_lower, clip_upper)).max():.3e}")
    print(f"frames={len(rows)} worst max_abs_diff={worst:.3e} tolerance={args.tolerance:.1e}")
    return 0 if worst <= args.tolerance else 1


if __name__ == "__main__":
    raise SystemExit(main())
