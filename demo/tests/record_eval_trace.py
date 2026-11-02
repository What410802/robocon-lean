#!/usr/bin/env python3
"""Record the demo `/eval_state` JSON topic as an eval-trace JSONL (P3-09).

Run from the repo root inside the demo env after sourcing the built workspace, alongside a
headless launch that samples every 2 ms physics frame (judged rate):

  source demo/install/setup.bash
  ros2 launch rl_sar rl_sim_all.launch.py ... viewer:=false realtime:=true \
      eval_state_every:=1 publish_every:=1
  python demo/tests/record_eval_trace.py --part 1 --robot blackW \
      --scene $PWD/demo/robots/blackW/scene_part1.xml --seconds 30 --out artifacts/traces/part1.jsonl

Pass `eval_state_every:=10` only for IO-light debug runs: the judge replay expects 2 ms frames.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String

LEGS = {"FL_foot": "FL", "FR_foot": "FR", "RL_foot": "RL", "RR_foot": "RR"}
ROOT = Path(__file__).resolve().parents[1]  # .../demo
DEFAULT_LEVELS = ROOT / "scenes/part2/part2_official_levels.json"


class Recorder(Node):
    def __init__(self, args) -> None:
        super().__init__("eval_trace_recorder")
        self.args = args
        self.frames = 0
        self.out = open(args.out, "w")
        meta = {"kind": "meta", "schema": 1, "part": args.part, "robot": args.robot,
                "scene": args.scene, "policy_sha256": args.policy_sha256,
                # sim_node /eval_state reports wheel body centres; the v5 replay
                # subtracts WHEEL_RADIUS for this legacy frame.
                "support_frame": "wheel_center"}
        if args.part == 2 and args.level:
            levels = json.loads(Path(args.levels).read_text())["levels"]
            entry = next((e for e in levels if e.get("key") == args.level), None)
            if entry is None:
                raise SystemExit(f"unknown Part2 level {args.level!r}")
            meta["level"] = entry
            meta["level_entry"] = entry
        self.out.write(json.dumps(meta) + "\n")
        self.create_subscription(String, "/eval_state", self._cb, qos_profile_sensor_data)

    def _cb(self, msg) -> None:
        d = json.loads(msg.data)
        support = {LEGS[k]: v for k, v in (d.get("feet") or {}).items() if k in LEGS and v}
        frame = {
            "kind": "frame",
            "t": d["t"],
            "base": d["base"],
            # sim_node /eval_state publishes the base quaternion (w,x,y,z); Part2
            # replay derives fall tilt from it.  Keep it so judged traces carry
            # the same fall evidence as in-process runs.
            "quat": d.get("quat"),
            "yaw": 0.0,
            "foot_tags": {leg: [] for leg in LEGS.values()},
            "ground_feet": [],
            "bar_z": d.get("bar_z"),
            "support": support,
            "contacts": d.get("contacts") or [],
            "steps": d.get("steps"), "cmd_step": d.get("cmd_step"), "cmd": d.get("cmd"), "cmd_stale": d.get("cmd_stale"),
            "q": d.get("q"), "dq": d.get("dq"), "tau": d.get("tau"),
        }
        self.out.write(json.dumps(frame) + "\n")
        self.out.flush()
        self.frames += 1

    def close(self) -> None:
        self.out.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, required=True)
    ap.add_argument("--robot", required=True)
    ap.add_argument("--scene", default="")
    ap.add_argument("--policy-sha256", default=None)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--level", default=None, help="Part2 official level key")
    ap.add_argument("--levels", default=str(DEFAULT_LEVELS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rclpy.init()
    node = Recorder(args)
    deadline = time.monotonic() + args.seconds
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.close()
    print(f"recorded {node.frames} eval frames -> {args.out}")
    node.destroy_node()
    rclpy.shutdown()
    return 0 if node.frames else 1


if __name__ == "__main__":
    raise SystemExit(main())
