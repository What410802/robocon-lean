#!/usr/bin/env python3
"""GO/NO-GO gates for the plum teacher and student.

Two deliberately DIFFERENT verdicts (do not conflate them):

* ``teacher`` -- in-sim judge-mirrored success predicate at the OFFICIAL field
  (curriculum difficulty 1.0, i.e. the real 0.2 m cubes — difficulty 0.0 is the
  easy contiguous plateau and would flatter the number).  The teacher consumes
  a privileged fine scan that the eval runner cannot provide, so this is a cheap
  capability pre-screen, NOT an eval-judge score.  Reuses the repo's existing
  ``check_tier_success.py`` (counts real completed episodes).
* ``student`` -- real eval-side judge via ``robocon_eval.cli run`` with the
  exported depth policy.  This is the scoring verdict.

Both report success_rate + Wilson 95% CI.  A seed sweep on the deterministic
judge scene is NOT independent spawn randomisation; use tools/campaign for a
full spawn-offset robustness matrix before claiming "0 failures".
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "train" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "train" / "src"))

from alldog_mjlab.tasks.velocity.profiles.plum.teacher import COMMAND_VX  # noqa: E402

#: Midpoint of the teacher's fixed command range; the gate must use it.
TEACHER_GATE_VX = 0.5 * (COMMAND_VX[0] + COMMAND_VX[1])
TRAIN_PY = ROOT / ".pixi" / "envs" / "train" / "bin" / "python"
EVAL_PY = ROOT / ".pixi" / "envs" / "eval" / "bin" / "python"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def _run(cmd: list[str], env: dict) -> str:
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if proc.returncode != 0 and not proc.stdout.strip():
        print(proc.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"gate subprocess failed (rc={proc.returncode}): {' '.join(cmd)}")
    return proc.stdout


def cmd_teacher(args) -> int:
    env = {**os.environ, "MUJOCO_GL": "egl", "PYTHONPATH": str(ROOT / "train" / "src")}
    cmd = [
        str(TRAIN_PY), "train/scripts/checks/check_tier_success.py",
        "--task", "blackw-plum-teacher", "--checkpoint", args.checkpoint,
        "--envs", str(args.envs), "--episodes", str(args.episodes),
        "--clean-obs", "--no-dr", "--seed", str(args.seed),
        # The teacher trains on a shrinking-cube/step curriculum; difficulty 0.0 is
        # the easy plateau, NOT the official field.  Gate at 1.0 always.
        "--difficulty", "1.0",
        # ...and at the command it was trained for (0.65-0.80 m/s).  The checker
        # defaults to 0.3, which is far off-distribution and reads as a failure
        # even when the policy is fine (measured: v3c 0/295 at 0.3 vs its own
        # training range).
        "--vx", str(TEACHER_GATE_VX),
    ]
    out = _run(cmd, env)
    line = next((l for l in reversed(out.strip().splitlines()) if l.startswith("{")), None)
    if line is None:
        raise SystemExit("check_tier_success produced no JSON summary")
    data = json.loads(line)
    n, k = data["episodes"], data["successes"]
    lo, hi = wilson(k, n)
    print(f"[gate/teacher] IN-SIM judge-mirrored success: {k}/{n} = {data['success_rate']:.3f} "
          f"Wilson95=[{lo:.3f},{hi:.3f}]")
    print("[gate/teacher] term_counts:", data.get("term_counts"))
    print("[gate/teacher] CAVEAT: this is the training predicate, NOT the eval judge.")
    go = data["success_rate"] >= args.threshold
    print(f"[gate/teacher] {'GO' if go else 'NO-GO'} (threshold {args.threshold})")
    return 0 if go else 1


def cmd_student(args) -> int:
    env = {**os.environ, "MUJOCO_GL": "egl", "PYTHONPATH": str(ROOT / "eval")}
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    statuses = []
    for seed in range(args.trials):
        result_path = results_dir / f"trial_{seed:04d}.json"
        cmd = [
            str(EVAL_PY), "-m", "robocon_eval.cli", "run",
            "--robot", "blackW", "--part", "2", "--level", args.level,
            "--policy", args.policy, "--profile", "official",
            "--depth-contract", str(Path(args.policy) / "depth.json"),
            "--duration", str(args.duration),
            "--seed", str(seed),
            "--out", str(results_dir / f"trace_{seed:04d}.jsonl"),
            "--result", str(result_path),
        ]
        _run(cmd, env)
        data = json.loads(result_path.read_text(encoding="utf-8"))
        statuses.append(data.get("status"))
    k = sum(s == "success" for s in statuses)
    n = len(statuses)
    lo, hi = wilson(k, n)
    from collections import Counter
    print(f"[gate/student] EVAL judge success: {k}/{n} = {k/n:.3f} "
          f"Wilson95=[{lo:.3f},{hi:.3f}]")
    print(f"[gate/student] status breakdown: {dict(Counter(statuses))}")
    print("[gate/student] CAVEAT: seed sweep is deterministic replay, NOT independent "
          "spawn randomisation; use tools/campaign for the robustness matrix.")
    go = (k / n) >= args.threshold
    print(f"[gate/student] {'GO' if go else 'NO-GO'} (threshold {args.threshold})")
    return 0 if go else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    t = sub.add_parser("teacher", help="in-sim teacher capability pre-screen")
    t.add_argument("--checkpoint", required=True)
    t.add_argument("--envs", type=int, default=64)
    t.add_argument("--episodes", type=int, default=4)
    t.add_argument("--threshold", type=float, default=0.25)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=cmd_teacher)

    s = sub.add_parser("student", help="real eval-side judge on the depth policy")
    s.add_argument("--policy", required=True, help="exported policy directory")
    s.add_argument("--level", default="plum_1")
    s.add_argument("--trials", type=int, default=100)
    s.add_argument("--duration", type=float, default=24.0)
    s.add_argument("--threshold", type=float, default=0.90)
    s.add_argument("--results", default="artifacts/plum/gate")
    s.set_defaults(func=cmd_student)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
