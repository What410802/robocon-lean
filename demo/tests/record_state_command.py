#!/usr/bin/env python3
"""Record /motor_state plus the latest /mit_command as a CSV trace (state/command attribution).

Each state row carries the last command seen before it (the command that drove that state),
so two runs can be compared on the same sim_time grid without wall-clock alignment.

Usage (inside the demo env, after sourcing the workspace):
  python demo/tests/record_state_command.py --seconds 6 --out artifacts/traces/state_cmd.csv
"""
from __future__ import annotations

import argparse
import csv
import importlib
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


class Recorder(Node):
    def __init__(self, msg_package: str) -> None:
        super().__init__("state_command_recorder")
        msgs = importlib.import_module(f"{msg_package}.msg")
        self.motor_state_type = msgs.MotorState
        self.mit_command_type = msgs.MitCommand
        self.rows: list[list[float]] = []
        self.last_cmd = None
        self.n_dof = 0
        self.create_subscription(self.motor_state_type, "/motor_state", self._state,
                                 qos_profile_sensor_data)
        self.create_subscription(self.mit_command_type, "/mit_command", self._cmd,
                                 qos_profile_sensor_data)

    def _cmd(self, msg) -> None:
        self.last_cmd = msg

    def _state(self, msg) -> None:
        with_cmd = self.last_cmd is not None
        cmd = self.last_cmd
        self.n_dof = len(msg.q)
        row = [float(msg.sim_time), *map(float, msg.q), *map(float, msg.dq),
               *map(float, msg.tau)]
        row += [1.0 if with_cmd else 0.0]
        for arr in ((cmd.q, cmd.w, cmd.kp, cmd.kd, cmd.tau) if with_cmd else [()] * 5):
            row += list(map(float, arr)) if with_cmd else [float("nan")] * self.n_dof
        self.rows.append(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg-package", default="quadruped_ros2",
                    help="ROS 2 package that defines MitCommand/MotorState")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rclpy.init()
    node = Recorder(args.msg_package)
    deadline = time.monotonic() + args.seconds
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    n = node.n_dof
    header = (["sim_time"] + [f"q{i}" for i in range(n)] + [f"dq{i}" for i in range(n)]
              + [f"tau{i}" for i in range(n)] + ["has_cmd"]
              + [f"cq{i}" for i in range(n)] + [f"cw{i}" for i in range(n)]
              + [f"ckp{i}" for i in range(n)] + [f"ckd{i}" for i in range(n)]
              + [f"ctau{i}" for i in range(n)])
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(node.rows)
    print(f"recorded {len(node.rows)} samples ({n} dof, {args.msg_package}) -> {args.out}")
    node.destroy_node()
    rclpy.shutdown()
    return 0 if node.rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
