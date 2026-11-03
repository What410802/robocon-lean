#!/usr/bin/env python3
"""Kinematic "ideal velocity follower" stand-in for the MuJoCo sim_node.

Purpose: test the Part1 route logic independently of the connector policy.  The
node subscribes to the route's virtual gamepad /joy stream, integrates the
commanded body-frame velocity exactly (no dynamics, no slip), samples the real
Part1 scene geometry with MuJoCo ray casts for standing height / slope attitude,
and publishes the same /eval_state JSON as sim_node.

It is intentionally *not* a physically valid simulator: supports are synthetic
nominal footprints, there is no contact force, and obstacles are climbed
kinematically.  Use it only with ``PART1_ROUTE_JUDGE=0`` to expose path / leg
sequencing bugs; the current policy must still be validated in the real sim.

Usage (after `source demo/install/setup.bash`):
  .pixi/envs/demo/bin/python tools/route/fake_kinematic_route_sim.py \
      --scene demo/robots/blackW/scene_part1.xml --rate 50
The route node is started separately with PART1_ROUTE_STEPLOCK=1 so the two
nodes advance one route tick per /eval_state frame.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Joy
from std_msgs.msg import String

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE = ROOT / "demo/robots/blackW/scene_part1.xml"
JOY_COMMAND_SCALE = 1.5
STAND_HEIGHT = 0.530  # fake base height above ground (long-slope z<0.54 gate)
WHEEL_RADIUS = 0.103
# Nominal wheel-centre offsets in the trunk yaw frame (default kinematics).
FOOT_XY = {
    "FL": (0.2006, 0.2277),
    "FR": (0.2006, -0.2277),
    "RL": (-0.2246, 0.2277),
    "RR": (-0.2246, -0.2277),
}
DEFAULT_DOF_POS = {
    "FL_hip_joint": 0.05, "FL_thigh_joint": 0.8014, "FL_calf_joint": -1.527,
    "FL_wheel_joint": 0.0,
    "FR_hip_joint": -0.05, "FR_thigh_joint": -0.8014, "FR_calf_joint": 1.527,
    "FR_wheel_joint": 0.0,
    "RL_hip_joint": -0.05, "RL_thigh_joint": 0.8014, "RL_calf_joint": -1.527,
    "RL_wheel_joint": 0.0,
    "RR_hip_joint": 0.05, "RR_thigh_joint": -0.8014, "RR_calf_joint": 1.527,
    "RR_wheel_joint": 0.0,
}
ROBOT_PREFIXES = ("trunk", "FL_", "FR_", "RL_", "RR_", "imu_link", "d435_link")


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def euler_quat(roll: float, pitch: float, yaw: float) -> list[float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    # R = Rz(yaw) Ry(pitch) Rx(roll), w,x,y,z
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = cy * sp * cr + sy * cp * sr
    z = sy * cp * cr - cy * sp * sr
    return [w, x, y, z]


class Terrain:
    def __init__(self, scene: Path) -> None:
        self.m = mujoco.MjModel.from_xml_path(str(scene))
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i)
                 for i in range(self.m.nbody)]
        robot_bodies = {i for i, n in enumerate(names)
                        if n and any(n.startswith(p) for p in ROBOT_PREFIXES)}
        for gi in range(self.m.ngeom):
            if int(self.m.geom_bodyid[gi]) in robot_bodies:
                self.m.geom_group[gi] = 1  # exclude robot from ray casts
        self.groups = np.zeros(6, dtype=np.uint8)
        self.groups[0] = 1
        self.wheel_geoms = {}
        for gi in range(self.m.ngeom):
            n = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, gi)
            if n and n.endswith("_wheel_collision"):
                self.wheel_geoms[n.split("_")[0]] = n
        self.bar_body = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_BODY, "height_bar_beam_body")

    def height(self, x: float, y: float) -> float:
        pnt = np.array([x, y, 3.0], dtype=np.float64)
        vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        gid = np.array([-1], dtype=np.int32)
        dist = mujoco.mj_ray(self.m, self.d, pnt, vec, self.groups, 1, -1, gid)
        if dist < 0.0:
            return 0.0
        return 3.0 - float(dist)


class FakeRouteSim(Node):
    def __init__(self, scene: Path, rate: float, spawn=(3.7, -9.0, 0.0)) -> None:
        super().__init__("fake_route_sim")
        self.terrain = Terrain(scene)
        self.rate = rate
        self.dt = 1.0 / rate
        self.x, self.y, self.yaw = (float(v) for v in spawn)
        self.t = 0.0
        self.steps = 0
        self.axes = [0.0] * 8
        self.buttons = [0] * 11
        self.last_joy_wall = 0.0
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Joy, "/joy", self._on_joy, qos)
        self.pub = self.create_publisher(String, "/eval_state", qos)
        self.create_timer(self.dt, self._step)

    def _on_joy(self, msg: Joy) -> None:
        self.axes = list(msg.axes) + [0.0] * max(0, 8 - len(msg.axes))
        self.buttons = list(msg.buttons) + [0] * max(0, 11 - len(msg.buttons))
        self.axes = self.axes[:8]
        self.buttons = self.buttons[:11]
        self.last_joy_wall = time.monotonic()

    def _cmd(self) -> tuple[float, float, float]:
        # Route command(): vx in axes[1], vy in axes[0], wz in axes[3].
        vx = max(-1.0, min(1.0, self.axes[1])) * JOY_COMMAND_SCALE
        vy = max(-1.0, min(1.0, self.axes[0])) * JOY_COMMAND_SCALE
        wz = max(-1.0, min(1.0, self.axes[3])) * JOY_COMMAND_SCALE
        return vx, vy, wz

    def _step(self) -> None:
        vx, vy, wz = self._cmd()
        self.x += (vx * math.cos(self.yaw) - vy * math.sin(self.yaw)) * self.dt
        self.y += (vx * math.sin(self.yaw) + vy * math.cos(self.yaw)) * self.dt
        self.yaw = wrap(self.yaw + wz * self.dt)
        self.t += self.dt
        self.steps += 1

        c, s = math.cos(self.yaw), math.sin(self.yaw)
        feet = {}
        grounds = {}
        for leg, (fx, fy) in FOOT_XY.items():
            wx = self.x + c * fx - s * fy
            wy = self.y + s * fx + c * fy
            gz = self.terrain.height(wx, wy)
            grounds[leg] = gz
            feet[f"{leg}_foot"] = [wx, wy, gz + WHEEL_RADIUS]
        front = 0.5 * (grounds["FL"] + grounds["FR"])
        rear = 0.5 * (grounds["RL"] + grounds["RR"])
        left = 0.5 * (grounds["FL"] + grounds["RL"])
        right = 0.5 * (grounds["FR"] + grounds["RR"])
        pitch = math.atan2(front - rear, 0.4252)
        roll = math.atan2(left - right, 0.4554)
        z = 0.25 * sum(grounds.values()) + STAND_HEIGHT
        quat = euler_quat(roll, pitch, self.yaw)
        contacts = []
        for leg, gname in self.terrain.wheel_geoms.items():
            contacts.append([gname, "floor"])
        bar_z = None
        if self.terrain.bar_body >= 0:
            bar_z = float(self.terrain.d.xpos[self.terrain.bar_body, 2])
        # q in actuator order (replay video reads it as actuator qpos order).
        m = self.terrain.m
        q = []
        for ai in range(m.nu):
            jid = int(m.actuator_trnid[ai, 0])
            jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
            q.append(DEFAULT_DOF_POS.get(jname, 0.0))  # type: ignore[arg-type]
        d = {
            "t": round(self.t, 6),
            "base": [round(self.x, 9), round(self.y, 9), round(z, 9)],
            "quat": [round(v, 9) for v in quat],
            "feet": {k: [round(v, 9) for v in p] for k, p in feet.items()},
            "bar_z": bar_z,
            "contacts": contacts,
            "cmd_step": self.steps,
            "steps": self.steps,
            "cmd_stale": False,
            "cmd": [[0.0, 0.0, 0.0, qq, 0.0] for qq in q],
            "q": q,
            "dq": [0.0] * m.nu,
            "tau": [0.0] * m.nu,
        }
        self.pub.publish(String(data=json.dumps(d)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=str(DEFAULT_SCENE))
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--spawn-x", type=float, default=3.7)
    ap.add_argument("--spawn-y", type=float, default=-9.0)
    ap.add_argument("--spawn-yaw", type=float, default=0.0)
    args = ap.parse_args()
    rclpy.init()
    node = FakeRouteSim(Path(args.scene), args.rate,
                        (args.spawn_x, args.spawn_y, args.spawn_yaw))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
