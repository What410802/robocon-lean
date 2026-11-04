#!/usr/bin/env python3
"""blackW three-axis velocity-command response probe and F1-F7 gate.

Implements the eval side of
``docs/agent-docs/velocity-tracking-plan-20261031.md`` (section 2 gate, section 5
probe design).  The probe reuses the deployment-contract MuJoCo runner
``eval/robocon_eval/policy_runner.PolicyRunner`` (50 Hz policy, 5 ms PD, command
limiter, blackW weights) so what it measures is what the deployment stack does,
on the flat scene ``demo/robots/blackW/smoke_flat.xml`` (CPU MuJoCo).

Measurement contract
--------------------
* Every scenario starts from a fresh ``reset()`` + fixed-gain settle
  (default 0.6 s), then holds its command(s) for 2.4 s.
* Steady-state means/stds are taken over the **last 1.2 s** of the hold.  A
  multi-step scenario (stop / step-up) is one continuous run; the command
  changes are step inputs.
* Body-frame velocities come from the free-joint state at every 20 ms policy
  tick::

      R    = _quat_to_mat(data.qpos[3:7])
      v_b  = R.T @ data.qvel[0:3]        # world -> body (linear is world-frame)
      w_b  = data.qvel[3:6]              # ALREADY body-frame for a free joint

  MuJoCo stores free-joint angular velocity in the body frame (verified against
  ``mj_objectVelocity(flg_local=1)``), so an extra ``R.T`` would double-rotate
  it.  ``PolicyRunner.frame()`` is not used: it has no linear velocity.

Gate F1-F7 (see the plan table)
-------------------------------
F1 walk 1.2 s @ 0.5 then cmd=0 2.4 s: steady |v_xy| < 0.05 m/s and no backward
   body-frame displacement > 0.08 m within 1.5 s of the step.
F2 vx sweep {-0.3,-0.1,0,0.1,0.2,...,1.0}: for cmd >= 0.2 |vx-cmd| <= 0.12;
   sign correct over the whole range; monotonic; no dead zone (adjacent 0.1 m/s
   command steps change the measured speed by < 0.2 m/s, near-zero continuous).
F3 reverse cmd=-0.2: steady error <= 0.12 m/s, no fall, no illegal contact
   (ground touching anything but a wheel cylinder).
F4 (0.4,+-0.2,0) and (0.5,+-0.2,0): |vy-cmd_vy| <= 0.08 m/s, lateral direction
   correct.
F5 (0.4,0,+-0.5) and (0.5,0,+-0.5): |wz-cmd_wz| <= 0.15 rad/s, or 1 s integrated
   yaw >= 60 % of the commanded yaw.
F6 0.5 -> 0 -> 0.5 steps (1 s each): stop segment satisfies F1; the resume
   segment is back inside the F2 error band within 0.8 s.
F7 all gate scenarios: no fall, tilt < 30 deg, base_z normal (0.30-0.85 m),
   finite state.

Outputs
-------
* human-readable table on stdout,
* ``--out`` JSON with per-scenario command / mean / std / displacement / tilt /
  fall / illegal contacts, plus the gate check list,
* ``--gate`` exit code: 0 = all F1-F7 pass, 1 = gate failed, 2 = contract/usage
  error.

Determinism: the runner is CPU MuJoCo with no RNG in the control path, so two
runs with the same policy/scene/settle produce identical JSON (except the
``created_utc`` stamp).

Usage
-----
::

    cd ~/robocon-lean
    MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 \\
      .pixi/envs/eval/bin/python tools/tracking/command_probe.py \\
      --policy row2p_p40w40_v1 --part 1 --gate \\
      --out artifacts/velocity-tracking/baseline/row2p_p40w40_v1.json

    # Part-2 policy (level is only used for its declared spawn/flat context)
    ... --policy s1d_trench_reward_v75_m100 --part 2 --level trench_30cm ...

    # full sweep (all scenarios, no gate)
    ... --policy flat --part 1 --out /tmp/flat_sweep.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "eval"))

from robocon_eval.policy_runner import (  # noqa: E402
    ContractError,
    PolicyRunner,
    _quat_to_mat,
)

DEFAULT_SCENE = BASE / "demo/robots/blackW/smoke_flat.xml"
DEFAULT_SETTLE_S = 0.6
DEFAULT_HOLD_S = 2.4
DEFAULT_TAIL_S = 1.2

# F7 / fall thresholds.  Nominal standing base height on this scene is ~0.52 m;
# 0.30 m is below any recoverable stance and 0.85 m is well above a rear-up.
FALL_TILT_DEG = 30.0
NORMAL_BASE_Z = (0.30, 0.85)

VX_CMDS = (-0.3, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
F4_CMDS = ((0.4, -0.2, 0.0), (0.4, 0.2, 0.0), (0.5, -0.2, 0.0), (0.5, 0.2, 0.0))
F5_CMDS = ((0.4, 0.0, -0.5), (0.4, 0.0, 0.5), (0.5, 0.0, -0.5), (0.5, 0.0, 0.5))

# F2 pass/fail tolerances, from the plan table.
F2_TRACK_TOL = 0.12        # |vx - cmd| for cmd >= 0.2
F2_ZERO_TOL = 0.10         # |vx| for |cmd| < 0.05 (continuity through zero)
F2_MONOTONIC_SLACK = 0.02  # measurement noise allowance
F2_DEADZONE_STEP = 0.20    # adjacent 0.1 m/s command pairs
F1_STEADY_XY = 0.05        # m/s
F1_BACKWARD = 0.08         # m
F6_RESUME_WINDOW = (0.6, 0.8)  # seconds after the step-up
F6_RESUME_TOL = F2_TRACK_TOL

FMT_CMD = "({:5.2f},{:5.2f},{:5.2f})"


def _yaw_from_quat(quat: np.ndarray) -> float:
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _fmt3(values: Iterable[float], width: int = 5, prec: int = 2) -> str:
    return "(" + ",".join(f"{float(v):{width}.{prec}f}" for v in values) + ")"


def _round(values: Iterable[float], digits: int = 4) -> list[float]:
    return [round(float(v), digits) for v in values]


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
@dataclass
class Segment:
    """One constant-command hold inside a scenario (per policy tick arrays)."""

    label: str
    seconds: float
    dt: float
    cmd: np.ndarray          # (n, 3) command actually fed to the policy
    t: np.ndarray            # (n,) seconds since segment start
    v: np.ndarray            # (n, 3) body-frame linear velocity
    w: np.ndarray            # (n, 3) body-frame angular velocity
    yaw: np.ndarray          # (n,) yaw
    tilt: np.ndarray         # (n,) degrees
    base: np.ndarray         # (n, 3) world base position
    R0: np.ndarray           # (3, 3) body orientation at segment start
    illegal: np.ndarray      # (n,) bool: non-wheel robot geom touching ground
    illegal_names: dict[str, int] = field(default_factory=dict)
    finite: bool = True
    settle_drift: list[float] = field(default_factory=list)

    def tail(self, tail_s: float) -> slice:
        n = len(self.t)
        k = max(1, min(n, int(round(tail_s / self.dt))))
        return slice(n - k, n)

    def tail_speed_xy(self, tail_s: float) -> float:
        v = self.v[self.tail(tail_s)]
        return float(np.hypot(v[:, 0], v[:, 1]).mean())

    def tail_mean(self, tail_s: float) -> np.ndarray:
        return self.v[self.tail(tail_s)].mean(axis=0)

    def tail_std(self, tail_s: float) -> np.ndarray:
        return self.v[self.tail(tail_s)].std(axis=0)

    def tail_mean_w(self, tail_s: float) -> np.ndarray:
        return self.w[self.tail(tail_s)].mean(axis=0)

    def tail_std_w(self, tail_s: float) -> np.ndarray:
        return self.w[self.tail(tail_s)].std(axis=0)

    def displacement(self) -> tuple[np.ndarray, np.ndarray]:
        delta = self.base[-1] - self.base[0]
        return delta, self.R0.T @ delta

    def backward_excursion(self, window_s: float = 1.5) -> float:
        """Largest *net* backward (body -x) excursion inside ``window_s``.

        Position based: how far behind the step-start point the base got.  A
        policy that first coasts forward on old momentum and only then starts
        reversing can keep this positive for a while, so the gate also checks
        :meth:`backward_distance`.
        """
        n = min(len(self.t), int(round(window_s / self.dt)))
        local = self.base[:n] - self.base[0]
        body_x = local @ self.R0[:, 0]  # dot with initial body x axis
        return max(0.0, float(-body_x.min())) if n > 0 else 0.0

    def backward_distance(self, window_s: float = 1.5) -> float:
        """Cumulative distance travelled backwards (integral of max(0, -vx_b)).

        This is the measurement the F1/F6 stop checks use: a policy that keeps
        commanding reverse speed after the stop step accumulates distance even
        if it first coasts forward and the net displacement stays positive.
        """
        n = min(len(self.t), int(round(window_s / self.dt)))
        if n <= 0:
            return 0.0
        return float(np.maximum(0.0, -self.v[:n, 0]).sum() * self.dt)

    def yaw_delta(self, t0: float, t1: float) -> float:
        i0 = int(round(t0 / self.dt))
        i1 = int(round(t1 / self.dt))
        i0 = max(0, min(i0, len(self.yaw) - 1))
        i1 = max(i0 + 1, min(i1, len(self.yaw)))
        d = self.yaw[i1 - 1] - self.yaw[i0]
        return float(math.atan2(math.sin(d), math.cos(d)))

    def window_mean_v(self, t0: float, t1: float) -> np.ndarray:
        i0 = int(round(t0 / self.dt))
        i1 = int(round(t1 / self.dt))
        i0 = max(0, min(i0, len(self.v) - 1))
        i1 = max(i0 + 1, min(i1, len(self.v)))
        return self.v[i0:i1].mean(axis=0)

    @property
    def tilt_max(self) -> float:
        return float(self.tilt.max())

    @property
    def z_min(self) -> float:
        return float(self.base[:, 2].min())

    @property
    def z_max(self) -> float:
        return float(self.base[:, 2].max())

    @property
    def illegal_frames(self) -> int:
        return int(self.illegal.sum())

    @property
    def fall(self) -> bool:
        return bool(
            not self.finite
            or self.tilt_max >= FALL_TILT_DEG
            or self.z_min < NORMAL_BASE_Z[0]
            or self.z_max > NORMAL_BASE_Z[1]
        )

    def summary(self, tail_s: float) -> dict[str, Any]:
        disp_world, disp_body = self.displacement()
        mean = self.tail_mean(tail_s)
        std = self.tail_std(tail_s)
        return {
            "label": self.label,
            "seconds": self.seconds,
            "cmd": _round(self.cmd.mean(axis=0)),
            "cmd_std": _round(self.cmd.std(axis=0)),
            "mean": _round(mean),
            "std": _round(std),
            "mean_w": _round(self.tail_mean_w(tail_s)),
            "std_w": _round(self.tail_std_w(tail_s)),
            "mean_full": _round(self.v.mean(axis=0)),
            "tail_speed_xy": round(self.tail_speed_xy(tail_s), 4),
            "displacement_world": _round(disp_world),
            "displacement_body": _round(disp_body),
            "backward_distance_1p5s": round(self.backward_distance(), 4),
            "backward_excursion_1p5s": round(self.backward_excursion(), 4),
            "delta_yaw_tail_1s": round(self.yaw_delta(tail_s, tail_s + 1.0), 4),
            "tilt_max_deg": round(self.tilt_max, 2),
            "z_min": round(self.z_min, 4),
            "z_max": round(self.z_max, 4),
            "fall": self.fall,
            "illegal_contact_frames": self.illegal_frames,
            "illegal_contact_geoms": dict(self.illegal_names),
            "finite": bool(self.finite),
        }


@dataclass
class ScenarioRun:
    key: str
    kind: str
    steps: list[Step]
    segments: list[Segment]
    settle_drift: list[float]
    elapsed_s: float
    informational: bool = False

    @property
    def tilt_max(self) -> float:
        return max(s.tilt_max for s in self.segments)

    @property
    def z_min(self) -> float:
        return min(s.z_min for s in self.segments)

    @property
    def z_max(self) -> float:
        return max(s.z_max for s in self.segments)

    @property
    def illegal_frames(self) -> int:
        return sum(s.illegal_frames for s in self.segments)

    @property
    def fall(self) -> bool:
        return any(s.fall for s in self.segments)

    def summary(self, tail_s: float) -> dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "informational": self.informational,
            "settle_drift": self.settle_drift,
            "elapsed_s": round(self.elapsed_s, 4),
            "fall": self.fall,
            "tilt_max_deg": round(self.tilt_max, 2),
            "z_min": round(self.z_min, 4),
            "z_max": round(self.z_max, 4),
            "illegal_contact_frames": self.illegal_frames,
            "segments": [s.summary(tail_s) for s in self.segments],
        }


@dataclass
class Step:
    """One constant-command segment of a scenario."""

    label: str
    cmd: tuple[float, float, float]
    seconds: float


@dataclass
class ScenarioSpec:
    key: str
    kind: str
    steps: list[Step]
    informational: bool = False


# ---------------------------------------------------------------------------
# runner wrapper
# ---------------------------------------------------------------------------
class ProbeSession:
    """Thin probe layer on top of ``PolicyRunner``."""

    def __init__(self, runner: PolicyRunner, settle_s: float):
        self.runner = runner
        self.settle_s = settle_s
        model = runner.model
        # Ground = geoms rigidly attached to the world body (flat floor plane).
        self.ground_geoms = {
            g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == 0
        }
        robot_bodies = self._robot_body_ids()
        self.robot_geoms = {
            g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in robot_bodies
        }
        self.wheel_geoms = set(runner.wheel_geoms.values())
        self.settle_drift: list[float] = []

    def _robot_body_ids(self) -> set[int]:
        trunk = self.runner.trunk
        ids: set[int] = set()
        for bid in range(self.runner.model.nbody):
            parent = bid
            while parent != 0:
                if parent == trunk:
                    ids.add(bid)
                    break
                parent = int(self.runner.model.body_parentid[parent])
        return ids

    def prepare(self) -> list[float]:
        """Fresh reset + fixed-gain settle into the deployment entry state."""
        self.runner.reset()
        drift = self.runner.settle(self.settle_s).get("drift") or [0.0, 0.0, 0.0]
        self.settle_drift = [round(float(v), 6) for v in drift]
        return self.settle_drift

    def _illegal_contacts(self) -> tuple[int, dict[str, int]]:
        data = self.runner.data
        names: dict[str, int] = {}
        for i in range(data.ncon):
            contact = data.contact[i]
            g1, g2 = int(contact.geom[0]), int(contact.geom[1])
            if g1 in self.ground_geoms and g2 in self.robot_geoms:
                other = g2
            elif g2 in self.ground_geoms and g1 in self.robot_geoms:
                other = g1
            else:
                continue
            if other in self.wheel_geoms:
                continue
            name = mujoco.mj_id2name(
                self.runner.model, mujoco.mjtObj.mjOBJ_GEOM, int(other)) or f"geom{other}"
            names[name] = names.get(name, 0) + 1
        return sum(names.values()), names

    def run_segment(
        self,
        label: str,
        cmd: np.ndarray | tuple[float, float, float],
        seconds: float,
        *,
        adaptive=None,
        nominal: np.ndarray | None = None,
    ) -> Segment:
        runner = self.runner
        cmd_arr = np.asarray(cmd, dtype=np.float64).reshape(3)
        n = max(1, int(round(seconds / runner.policy_dt)))
        q0 = np.array(runner.data.qpos, copy=True)
        R0 = _quat_to_mat(q0[3:7])
        cmds = np.empty((n, 3), dtype=np.float64)
        v = np.empty((n, 3), dtype=np.float64)
        w = np.empty((n, 3), dtype=np.float64)
        yaw = np.empty(n, dtype=np.float64)
        tilt = np.empty(n, dtype=np.float64)
        base = np.empty((n, 3), dtype=np.float64)
        illegal = np.zeros(n, dtype=bool)
        illegal_names: dict[str, int] = {}
        finite = True
        adaptive_cmd = adaptive
        for i in range(n):
            if adaptive_cmd is not None:
                fed = np.asarray(
                    adaptive_cmd(runner, i, nominal if nominal is not None
                                 else np.zeros(3)), dtype=np.float64)
            else:
                fed = cmd_arr
            runner.step(fed)
            data = runner.data
            quat = np.asarray(data.qpos[3:7], dtype=np.float64)
            R = _quat_to_mat(quat)
            cmds[i] = fed
            v[i] = R.T @ np.asarray(data.qvel[0:3], dtype=np.float64)
            # MuJoCo free-joint angular velocity is already in the body frame.
            w[i] = np.asarray(data.qvel[3:6], dtype=np.float64)
            yaw[i] = _yaw_from_quat(quat)
            tilt[i] = math.degrees(math.acos(max(-1.0, min(1.0, float(R[2, 2])))))
            base[i] = np.asarray(data.qpos[0:3], dtype=np.float64)
            n_illegal, names = self._illegal_contacts()
            if n_illegal:
                illegal[i] = True
                for name, count in names.items():
                    illegal_names[name] = illegal_names.get(name, 0) + count
            if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
                finite = False
        return Segment(
            label=label,
            seconds=float(seconds),
            dt=float(runner.policy_dt),
            cmd=cmds,
            t=np.arange(n, dtype=np.float64) * runner.policy_dt,
            v=v,
            w=w,
            yaw=yaw,
            tilt=tilt,
            base=base,
            R0=R0,
            illegal=illegal,
            illegal_names=illegal_names,
            finite=finite,
            settle_drift=list(self.settle_drift),
        )

    def run_scenario(self, spec: ScenarioSpec, tail_s: float) -> ScenarioRun:
        start = time.time()
        self.prepare()
        segments: list[Segment] = []
        for step in spec.steps:
            segments.append(self.run_segment(step.label, step.cmd, step.seconds))
        return ScenarioRun(
            key=spec.key,
            kind=spec.kind,
            steps=list(spec.steps),
            segments=segments,
            settle_drift=list(self.settle_drift),
            elapsed_s=time.time() - start,
            informational=spec.informational,
        )


# ---------------------------------------------------------------------------
# scenario definitions
# ---------------------------------------------------------------------------
def hold(label: str, cmd: tuple[float, float, float], hold_s: float,
         key: str | None = None) -> ScenarioSpec:
    return ScenarioSpec(
        key=key or label,
        kind="hold",
        steps=[Step(label=label, cmd=cmd, seconds=hold_s)],
    )


def gate_scenarios(hold_s: float) -> list[ScenarioSpec]:
    specs: list[ScenarioSpec] = [
        ScenarioSpec(
            key="F1_walk_then_zero",
            kind="sequence",
            steps=[
                Step("walk", (0.5, 0.0, 0.0), 1.2),
                Step("zero", (0.0, 0.0, 0.0), hold_s),
            ],
        )
    ]
    for cmd in VX_CMDS:
        specs.append(hold(f"F2_vx_{cmd:+.1f}", (cmd, 0.0, 0.0), hold_s))
    specs.append(hold("F3_reverse", (-0.2, 0.0, 0.0), hold_s))
    for vx, vy, wz in F4_CMDS:
        specs.append(hold(f"F4_vy_{vy:+.1f}_vx{vx:.1f}", (vx, vy, wz), hold_s))
    for vx, vy, wz in F5_CMDS:
        specs.append(hold(f"F5_wz_{wz:+.2f}_vx{vx:.1f}", (vx, vy, wz), hold_s))
    specs.append(
        ScenarioSpec(
            key="F6_step_0.5_0_0.5",
            kind="sequence",
            steps=[
                Step("0.5", (0.5, 0.0, 0.0), 1.0),
                Step("zero", (0.0, 0.0, 0.0), 1.0),
                Step("0.5_again", (0.5, 0.0, 0.0), 1.0),
            ],
        )
    )
    # Plan section 5 also lists a zero hold from standstill; the plan's F gate
    # only needs the after-walking F1 variant, so this one is informational and
    # does not affect the exit code.
    specs.append(hold("zero_hold", (0.0, 0.0, 0.0), hold_s))
    specs[-1].informational = True
    return specs


def sweep_scenarios(hold_s: float, adaptive: bool) -> list[ScenarioSpec]:
    specs: list[ScenarioSpec] = []
    for cmd in VX_CMDS:
        specs.append(hold(f"vx_{cmd:+.1f}", (cmd, 0.0, 0.0), hold_s))
    for vy in (-0.2, -0.1, 0.0, 0.1, 0.2):
        specs.append(hold(f"vy_{vy:+.1f}", (0.5, vy, 0.0), hold_s))
    for wz in (-0.5, -0.25, 0.0, 0.25, 0.5):
        specs.append(hold(f"wz_{wz:+.2f}", (0.5, 0.0, wz), hold_s))
    specs.append(hold("zero_hold", (0.0, 0.0, 0.0), hold_s))
    specs.append(hold("reverse", (-0.2, 0.0, 0.0), hold_s))
    specs.append(
        ScenarioSpec(
            key="walk_then_zero",
            kind="sequence",
            steps=[
                Step("walk", (0.5, 0.0, 0.0), 1.2),
                Step("zero", (0.0, 0.0, 0.0), hold_s),
            ],
        )
    )
    specs.append(
        ScenarioSpec(
            key="step_0.5_0_0.5",
            kind="sequence",
            steps=[
                Step("0.5", (0.5, 0.0, 0.0), 1.0),
                Step("zero", (0.0, 0.0, 0.0), 1.0),
                Step("0.5_again", (0.5, 0.0, 0.0), 1.0),
            ],
        )
    )
    if adaptive:
        specs.append(
            ScenarioSpec(
                key="adaptive_forward_0.5",
                kind="adaptive",
                steps=[Step("adaptive", (0.5, 0.0, 0.0), 4.0)],
            )
        )
    return specs


# ---------------------------------------------------------------------------
# gate evaluation
# ---------------------------------------------------------------------------
def _check(check_id: str, passed: bool, detail: str,
           failures: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": check_id,
        "passed": bool(passed),
        "detail": detail,
        "failures": list(failures or []),
    }


def _tail_mean_vx(run: ScenarioRun, tail_s: float) -> float:
    return float(run.segments[-1].tail_mean(tail_s)[0])


def evaluate_gate(runs: dict[str, ScenarioRun], tail_s: float,
                  vx_curve: list[tuple[float, float]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    # ---- F1: walk then stop ------------------------------------------------
    f1 = runs["F1_walk_then_zero"]
    stop = f1.segments[1]
    steady = stop.tail_speed_xy(tail_s)
    backward = stop.backward_distance(1.5)
    failures = []
    if not steady < F1_STEADY_XY:
        failures.append(
            f"steady |v_xy|={steady:.3f} m/s >= {F1_STEADY_XY:.2f} m/s")
    if not backward < F1_BACKWARD:
        failures.append(
            f"backward distance={backward:.3f} m >= {F1_BACKWARD:.2f} m "
            "(within 1.5 s of the zero step)")
    checks.append(_check(
        "F1", not failures,
        f"steady |v_xy|={steady:.3f} m/s (limit {F1_STEADY_XY:.2f}); "
        f"backward distance={backward:.3f} m (limit {F1_BACKWARD:.2f})",
        failures))

    # ---- F2: vx sweep ------------------------------------------------------
    failures = []
    measured = {cmd: meas for cmd, meas in vx_curve}
    tracking_failures = []
    for cmd, meas in vx_curve:
        if cmd >= 0.2 - 1e-9 and abs(meas - cmd) > F2_TRACK_TOL + 1e-9:
            tracking_failures.append(
                f"vx={cmd:+.1f}: |{meas:+.3f}-{cmd:+.1f}|="
                f"{abs(meas - cmd):.3f} > {F2_TRACK_TOL:.2f}")
    failures.extend(tracking_failures)
    sign_failures = []
    for cmd, meas in vx_curve:
        if cmd >= 0.05:
            if not meas > 0.0:
                sign_failures.append(f"vx={cmd:+.1f}: measured {meas:+.3f} not positive")
        elif cmd <= -0.05:
            if not meas < 0.0:
                sign_failures.append(f"vx={cmd:+.1f}: measured {meas:+.3f} not negative")
        else:
            if abs(meas) > F2_ZERO_TOL + 1e-9:
                sign_failures.append(
                    f"vx={cmd:+.1f}: near-zero residual {meas:+.3f} "
                    f"> {F2_ZERO_TOL:.2f}")
    failures.extend(sign_failures)
    mono_failures = []
    for (cmd_a, meas_a), (cmd_b, meas_b) in zip(vx_curve, vx_curve[1:]):
        if meas_b < meas_a - F2_MONOTONIC_SLACK:
            mono_failures.append(
                f"not monotonic: vx={cmd_b:+.1f} measured {meas_b:+.3f} < "
                f"vx={cmd_a:+.1f} measured {meas_a:+.3f} - slack "
                f"{F2_MONOTONIC_SLACK:.2f}")
    failures.extend(mono_failures)
    deadzone_failures = []
    for (cmd_a, meas_a), (cmd_b, meas_b) in zip(vx_curve, vx_curve[1:]):
        if abs((cmd_b - cmd_a) - 0.1) < 1e-9:
            jump = abs(meas_b - meas_a)
            if jump >= F2_DEADZONE_STEP:
                deadzone_failures.append(
                    f"dead zone: cmd {cmd_a:+.1f}->{cmd_b:+.1f} changes measured "
                    f"speed by {jump:.3f} m/s >= {F2_DEADZONE_STEP:.2f}")
    failures.extend(deadzone_failures)
    checks.append(_check(
        "F2", not failures,
        f"vx sweep {len(vx_curve)} points; tracking(fail)={len(tracking_failures)}, "
        f"sign(fail)={len(sign_failures)}, monotonic(fail)={len(mono_failures)}, "
        f"dead-zone(fail)={len(deadzone_failures)}",
        failures))

    # ---- F3: reverse -------------------------------------------------------
    f3 = runs["F3_reverse"].segments[0]
    err3 = abs(float(f3.tail_mean(tail_s)[0]) + 0.2)
    failures = []
    if err3 > 0.12 + 1e-9:
        failures.append(f"steady vx error={err3:.3f} m/s > 0.12 m/s")
    if f3.fall:
        failures.append(
            f"fall: tilt_max={f3.tilt_max:.1f} deg, z in "
            f"[{f3.z_min:.3f},{f3.z_max:.3f}] m")
    if f3.illegal_frames:
        failures.append(
            f"illegal contact on {f3.illegal_frames} frames "
            f"({f3.illegal_names})")
    checks.append(_check(
        "F3", not failures,
        f"reverse vx=-0.2: steady error={err3:.3f} m/s (limit 0.12); "
        f"tilt_max={f3.tilt_max:.1f} deg; illegal frames={f3.illegal_frames}",
        failures))

    # ---- F4: lateral -------------------------------------------------------
    failures = []
    for vx, vy, _wz in F4_CMDS:
        seg = runs[f"F4_vy_{vy:+.1f}_vx{vx:.1f}"].segments[0]
        meas_vy = float(seg.tail_mean(tail_s)[1])
        if abs(meas_vy - vy) > 0.08 + 1e-9:
            failures.append(
                f"cmd=({vx:.1f},{vy:+.1f},0): |vy {meas_vy:+.3f}-{vy:+.1f}|="
                f"{abs(meas_vy - vy):.3f} > 0.08")
        elif meas_vy * vy <= 0.0:
            failures.append(
                f"cmd=({vx:.1f},{vy:+.1f},0): measured vy {meas_vy:+.3f} "
                "is not in the commanded lateral direction")
    checks.append(_check(
        "F4", not failures,
        f"{len(F4_CMDS)} lateral scenarios, tolerance 0.08 m/s", failures))

    # ---- F5: yaw -----------------------------------------------------------
    failures = []
    details = []
    for vx, _vy, wz in F5_CMDS:
        seg = runs[f"F5_wz_{wz:+.2f}_vx{vx:.1f}"].segments[0]
        meas_wz = float(seg.tail_mean_w(tail_s)[2])
        yaw_delta = seg.yaw_delta(tail_s, tail_s + 1.0)
        rate_ok = abs(meas_wz - wz) <= 0.15 + 1e-9
        integral_ok = (yaw_delta * wz > 0.0
                       and abs(yaw_delta) >= 0.6 * abs(wz) * 1.0 - 1e-9)
        details.append(
            f"({vx:.1f},0,{wz:+.2f}): wz={meas_wz:+.3f}, dyaw_1s={yaw_delta:+.3f}")
        if not (rate_ok or integral_ok):
            failures.append(
                f"cmd=({vx:.1f},0,{wz:+.2f}): |wz {meas_wz:+.3f}-{wz:+.2f}|="
                f"{abs(meas_wz - wz):.3f} > 0.15 and 1 s yaw {yaw_delta:+.3f} rad "
                f"< 60 % of {wz:+.2f} rad")
    checks.append(_check(
        "F5", not failures,
        f"{len(F5_CMDS)} yaw scenarios; tolerance 0.15 rad/s or 60% yaw integral; "
        + "; ".join(details), failures))

    # ---- F6: step 0.5 -> 0 -> 0.5 -----------------------------------------
    f6 = runs["F6_step_0.5_0_0.5"]
    stop = f6.segments[1]
    resume = f6.segments[2]
    failures = []
    stop_steady = stop.tail_speed_xy(0.5)  # tail = last half of the 1 s stop
    stop_back = stop.backward_distance(1.0)  # segment is shorter than 1.5 s
    if not stop_steady < F1_STEADY_XY:
        failures.append(
            f"stop segment: steady |v_xy|={stop_steady:.3f} m/s >= "
            f"{F1_STEADY_XY:.2f} m/s")
    if not stop_back < F1_BACKWARD:
        failures.append(
            f"stop segment: backward distance={stop_back:.3f} m >= "
            f"{F1_BACKWARD:.2f} m")
    resume_v = resume.window_mean_v(*F6_RESUME_WINDOW)
    resume_err = abs(float(resume_v[0]) - 0.5)
    if resume_err > F6_RESUME_TOL + 1e-9:
        failures.append(
            f"resume segment: mean vx({F6_RESUME_WINDOW[0]:.1f}-"
            f"{F6_RESUME_WINDOW[1]:.1f} s)={resume_v[0]:+.3f} m/s, "
            f"|{resume_v[0]:+.3f}-0.500|={resume_err:.3f} > {F6_RESUME_TOL:.2f}")
    checks.append(_check(
        "F6", not failures,
        f"stop steady |v_xy|={stop_steady:.3f}, backward={stop_back:.3f}; "
        f"resume vx@{F6_RESUME_WINDOW[0]:.1f}-{F6_RESUME_WINDOW[1]:.1f}s="
        f"{resume_v[0]:+.3f} (err {resume_err:.3f}, limit {F6_RESUME_TOL:.2f})",
        failures))

    # ---- F7: no fall / tilt / base height over the gate -------------------
    failures = []
    worst_tilt = 0.0
    worst_tilt_key = ""
    z_lo, z_hi = float("inf"), float("-inf")
    z_lo_key = z_hi_key = ""
    for key, run in runs.items():
        if run.informational:
            continue
        for seg in run.segments:
            if seg.tilt_max > worst_tilt:
                worst_tilt, worst_tilt_key = seg.tilt_max, key
            if seg.z_min < z_lo:
                z_lo, z_lo_key = seg.z_min, key
            if seg.z_max > z_hi:
                z_hi, z_hi_key = seg.z_max, key
            if not seg.finite:
                failures.append(f"{key}: non-finite MuJoCo state")
            if seg.fall:
                reasons = []
                if seg.tilt_max >= FALL_TILT_DEG:
                    reasons.append(f"tilt {seg.tilt_max:.1f} deg")
                if seg.z_min < NORMAL_BASE_Z[0]:
                    reasons.append(f"base_z {seg.z_min:.3f} m")
                if seg.z_max > NORMAL_BASE_Z[1]:
                    reasons.append(f"base_z {seg.z_max:.3f} m")
                failures.append(f"{key}/{seg.label}: fall ({', '.join(reasons)})")
    if worst_tilt >= FALL_TILT_DEG:
        failures.append(
            f"max tilt {worst_tilt:.1f} deg >= {FALL_TILT_DEG:.0f} deg "
            f"({worst_tilt_key})")
    checks.append(_check(
        "F7", not failures,
        f"max tilt={worst_tilt:.1f} deg ({worst_tilt_key}); "
        f"base_z range [{z_lo:.3f} ({z_lo_key}), {z_hi:.3f} ({z_hi_key})]; "
        f"normal band [{NORMAL_BASE_Z[0]:.2f},{NORMAL_BASE_Z[1]:.2f}] m",
        failures))

    passed = sum(1 for c in checks if c["passed"])
    return {
        "passed": passed == len(checks),
        "passed_count": passed,
        "total": len(checks),
        "checks": checks,
        "failures": [f"{c['id']}: {item}" for c in checks if not c["passed"]
                     for item in (c["failures"] or [c["detail"]])],
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def scenario_table_rows(runs: list[ScenarioRun], tail_s: float) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        for seg, seg_summary in zip(run.segments, run.summary(tail_s)["segments"]):
            rows.append({
                "scenario": run.key,
                "label": seg.label if len(run.segments) > 1 else "",
                "cmd": seg_summary["cmd"],
                "mean": seg_summary["mean"],
                "std": seg_summary["std"],
                "mean_w": seg_summary["mean_w"],
                "tilt": seg_summary["tilt_max_deg"],
                "z": (seg_summary["z_min"], seg_summary["z_max"]),
                "fall": seg_summary["fall"],
                "illegal": seg_summary["illegal_contact_frames"],
                "info": run.informational,
            })
    return rows


def print_table(runs: list[ScenarioRun], tail_s: float) -> None:
    header = (
        f"{'scenario':36s} {'seg':6s} {'cmd (vx,vy,wz)':19s} "
        f"{'mean v body':21s} {'std v':19s} {'mean w body':21s} "
        f"{'tilt':>6s} {'z range':>15s} {'fall':>4s} {'ill':>3s}"
    )
    print(header)
    print("-" * len(header))
    for row in scenario_table_rows(runs, tail_s):
        cmd = _fmt3(row["cmd"])
        mean = _fmt3(row["mean"])
        std = _fmt3(row["std"], width=4)
        mean_w = _fmt3(row["mean_w"], width=5, prec=3)
        zlo, zhi = row["z"]
        print(
            f"{row['scenario']:36s} {row['label']:6s} {cmd:19s} "
            f"{mean:21s} {std:19s} {mean_w:21s} "
            f"{row['tilt']:6.1f} {zlo:7.3f}-{zhi:6.3f} "
            f"{'yes' if row['fall'] else 'no':>4s} {row['illegal']:3d}"
        )


def print_gate(gate: dict[str, Any]) -> None:
    print()
    print("gate checks (F1-F7):")
    for check in gate["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  {status}  {check['id']}  {check['detail']}")
        for failure in check["failures"]:
            print(f"        - {failure}")
    print()
    print(f"GATE: {'PASS' if gate['passed'] else 'FAIL'} "
          f"({gate['passed_count']}/{gate['total']} checks)")
    if not gate["passed"]:
        print("failed checks: "
              + ", ".join(c["id"] for c in gate["checks"] if not c["passed"]))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="blackW three-axis velocity-command probe + F1-F7 gate "
                    "(docs/training/VT-RECIPE.md)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy", required=True,
                        help="policy directory name under demo/rl_sar/policy/<robot>/, "
                             "or an exported checkpoint directory")
    parser.add_argument("--robot", default="blackW", help="robot manifest name")
    parser.add_argument("--part", type=int, default=1, choices=(1, 2),
                        help="deployment part (part 2 needs --level)")
    parser.add_argument("--level", default=None,
                        help="Part2 official level key (used for spawn/context only; "
                             "--scene overrides the terrain)")
    parser.add_argument("--scene", default=str(DEFAULT_SCENE),
                        help="MuJoCo scene XML (default: flat smoke scene)")
    parser.add_argument("--settle", type=float, default=DEFAULT_SETTLE_S,
                        help="fixed-gain settle before each scenario (s)")
    parser.add_argument("--hold", type=float, default=DEFAULT_HOLD_S,
                        help="command hold per scenario (s)")
    parser.add_argument("--tail", type=float, default=DEFAULT_TAIL_S,
                        help="steady-state averaging window at the end (s)")
    parser.add_argument("--gate", action="store_true",
                        help="run the F1-F7 gate and return non-zero on failure")
    parser.add_argument("--adaptive", action="store_true",
                        help="also run a closed-loop adaptive-command scenario "
                             "(full sweep mode only)")
    parser.add_argument("--out", default=None,
                        help="write JSON report to this path (directories created)")
    parser.add_argument("--seed", type=int, default=0,
                        help="recorded in the report (the runner path is deterministic)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scene = Path(args.scene)
    if not scene.is_absolute():
        scene = BASE / scene
    if not scene.is_file():
        print(f"error: scene not found: {scene}", file=sys.stderr)
        return 2
    if args.part == 2 and not args.level:
        print("error: --part 2 requires --level (Part2 official level key)",
              file=sys.stderr)
        return 2
    if args.hold <= 0 or args.settle < 0 or args.tail <= 0:
        print("error: --hold/--tail must be > 0 and --settle >= 0", file=sys.stderr)
        return 2
    tail_s = min(args.tail, args.hold)
    mode = "gate" if args.gate else "sweep"
    started = time.time()
    print(f"=== command_probe [{mode}] robot={args.robot} part={args.part} "
          f"policy={args.policy} ===")
    print(f"scene: {scene}")
    print(f"settle={args.settle:.2f}s hold={args.hold:.2f}s tail={tail_s:.2f}s "
          f"seed={args.seed}")
    if args.part == 2:
        print(f"level: {args.level} (spawn/context only; flat scene override)")

    runner = None
    try:
        runner = PolicyRunner(
            args.robot, args.part, args.policy, args.level,
            scene=scene, command=(0.0, 0.0, 0.0), settle_s=args.settle,
            seed=args.seed,
        )
        session = ProbeSession(runner, args.settle)
        print(f"policy_dir: {_rel(runner.policy_dir)}")
        print(f"hashes: policy={runner.policy_sha256[:16]}... "
              f"config={runner.config_sha256[:16]}... "
              f"scene={runner.scene_sha256[:16]}...")
        print(f"contract: control_dt={runner.control_dt*1e3:.1f}ms "
              f"policy_dt={runner.policy_dt*1e3:.1f}ms "
              f"obs={runner.obs_dim} history={runner.history_len} "
              f"warnings={len(runner.warnings)}")
        for warning in runner.warnings:
            print(f"  runner warning: {warning}")

        if args.gate:
            specs = gate_scenarios(args.hold)
        else:
            specs = sweep_scenarios(args.hold, args.adaptive)
        runs: dict[str, ScenarioRun] = {}
        ordered: list[ScenarioRun] = []
        for spec in specs:
            if spec.kind == "adaptive":
                adaptive = _make_adaptive(session)
                scenario_run = _run_adaptive(session, spec, adaptive)
            else:
                scenario_run = session.run_scenario(spec, tail_s)
            runs[spec.key] = scenario_run
            ordered.append(scenario_run)
        elapsed = time.time() - started

        print()
        print_table(ordered, tail_s)

        gate = None
        vx_curve = [(cmd, _tail_mean_vx(runs[f"F2_vx_{cmd:+.1f}"], tail_s))
                    for cmd in VX_CMDS] if args.gate else None
        if args.gate:
            gate = evaluate_gate(runs, tail_s, vx_curve)
            print_gate(gate)

        report = {
            "schema_version": 1,
            "kind": "blackW_velocity_command_probe",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": mode,
            "robot": args.robot,
            "part": args.part,
            "level": args.level,
            "policy": args.policy,
            "policy_dir": _rel(runner.policy_dir),
            "scene": _rel(scene),
            "settle_s": args.settle,
            "hold_s": args.hold,
            "tail_s": tail_s,
            "seed": args.seed,
            "elapsed_s": round(elapsed, 3),
            "thresholds": {
                "fall_tilt_deg": FALL_TILT_DEG,
                "normal_base_z": list(NORMAL_BASE_Z),
                "f1_steady_xy": F1_STEADY_XY,
                "f1_backward_m": F1_BACKWARD,
                "f2_track_tol": F2_TRACK_TOL,
                "f2_zero_tol": F2_ZERO_TOL,
                "f2_monotonic_slack": F2_MONOTONIC_SLACK,
                "f2_deadzone_step": F2_DEADZONE_STEP,
                "f4_vy_tol": 0.08,
                "f5_wz_tol": 0.15,
                "f6_resume_window_s": list(F6_RESUME_WINDOW),
                "f6_resume_tol": F6_RESUME_TOL,
            },
            "contract": {
                "policy_sha256": runner.policy_sha256,
                "config_sha256": runner.config_sha256,
                "scene_sha256": runner.scene_sha256,
                "robot_sha256": runner.robot_sha256,
                "control_dt_s": runner.control_dt,
                "policy_dt_s": runner.policy_dt,
                "num_observations": runner.obs_dim,
                "history_len": runner.history_len,
                "joint_mapping": runner.mapping.tolist(),
                "warnings": list(runner.warnings),
                "versions": {
                    "mujoco": mujoco.__version__,
                    "numpy": np.__version__,
                    "torch": _torch_version(),
                    "python": sys.version.split()[0],
                },
            },
            "scenarios": [run.summary(tail_s) for run in ordered],
            "vx_curve": vx_curve,
            "gate": gate,
        }
        if args.out:
            out = Path(args.out)
            if not out.is_absolute():
                out = BASE / out
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
            print(f"\njson: {_rel(out)}")
            if args.gate:
                assert gate is not None
                baseline_dir = BASE / "artifacts/velocity-tracking/baseline"
                if out.parent.resolve() == baseline_dir.resolve():
                    print("saved as baseline artifact (artifacts/ is gitignored)")
        if args.gate and gate is not None and not gate["passed"]:
            return 1
        return 0
    except ContractError as exc:
        print(f"error: policy/robot/scene contract violation: {exc}", file=sys.stderr)
        return 2
    finally:
        if runner is not None:
            runner.close()


def _make_adaptive(session: ProbeSession):  # -> AdaptiveCommand
    """Course-following closed loop: hold vx, centre on the spawn lateral line.

    ``AdaptiveCommand`` centres on ``center_y`` and heading 0; blackW Part1
    spawns at y=-9 m, so the default centre (0) would saturate vy/wz from the
    first tick.  Use the declared spawn as the reference line instead.
    """
    from robocon_eval.policy_runner import AdaptiveCommand

    spawn = session.runner.spawn
    return AdaptiveCommand(vx=0.5, center_y=float(spawn[1]))


def _run_adaptive(session: ProbeSession, spec: ScenarioSpec,
                  adaptive) -> ScenarioRun:
    start = time.time()
    session.prepare()
    step = spec.steps[0]
    seg = session.run_segment(
        step.label, step.cmd, step.seconds,
        adaptive=adaptive, nominal=np.asarray(step.cmd, dtype=np.float64))
    return ScenarioRun(
        key=spec.key, kind=spec.kind, steps=list(spec.steps),
        segments=[seg], settle_drift=list(session.settle_drift),
        elapsed_s=time.time() - start, informational=True)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(BASE))
    except ValueError:
        return str(path)


def _torch_version() -> str:
    import torch

    return torch.__version__


if __name__ == "__main__":
    raise SystemExit(main())
