#!/usr/bin/env python3
"""Record /motor_state into a CSV trace (runtime regression / motor-state checks).

Usage (inside the demo env, after sourcing the workspace):
  python demo/tests/record_motor_state.py --seconds 6 --out artifacts/traces/motor_state.csv
"""

from __future__ import annotations

import argparse
import csv
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from quadruped_ros2.msg import MotorState


class Recorder(Node):
    def __init__(self) -> None:
        super().__init__("motor_state_recorder")
        self.rows: list[list[float]] = []
        self.create_subscription(MotorState, "/motor_state", self._cb, qos_profile_sensor_data)

    def _cb(self, msg: MotorState) -> None:
        self.rows.append([float(msg.sim_time), *map(float, msg.q), *map(float, msg.dq),
                          *map(float, msg.tau)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rclpy.init()
    node = Recorder()
    deadline = time.monotonic() + args.seconds
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    n_dof = len(node.rows[0]) // 3 if node.rows else 0
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sim_time"] + [f"q{i}" for i in range(n_dof)]
                   + [f"dq{i}" for i in range(n_dof)] + [f"tau{i}" for i in range(n_dof)])
        w.writerows(node.rows)
    node.destroy_node()
    rclpy.shutdown()
    print(f"recorded {len(node.rows)} samples / {n_dof} dof -> {args.out}")
    return 0 if node.rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
