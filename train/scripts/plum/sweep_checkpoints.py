#!/usr/bin/env python3
"""Success rate of each checkpoint at the OFFICIAL field (difficulty 1.0).

``gate.py teacher`` evaluates one checkpoint; this sweeps a run's saved
checkpoints so the best one is picked by measurement.  Success on a fixed
knife-edge field is noisy across training (the old repo saw 1.3-2.8 terminations
per batch over time), so the final checkpoint is not reliably the best.

    MUJOCO_GL=egl pixi run -e train python train/scripts/plum/sweep_checkpoints.py \
        --run logs/rsl_rl/blackw_plum_teacher/<run> --envs 64 --episodes 2

Prints one line per checkpoint: successes/episodes and the termination counts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRAIN_PY = ROOT / ".pixi" / "envs" / "train" / "bin" / "python"
CHECKER = ROOT / "train" / "scripts" / "checks" / "check_tier_success.py"
sys.path.insert(0, str(ROOT / "train" / "src"))
from alldog_mjlab.tasks.velocity.profiles.plum.teacher import COMMAND_VX  # noqa: E402

#: The teacher trains against a fixed forward command; gate at its midpoint
#: (same value as gate.py).  The checker's own default, 0.3, is off-distribution.
TEACHER_GATE_VX = 0.5 * (COMMAND_VX[0] + COMMAND_VX[1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--difficulty", type=float, default=1.0)
    ap.add_argument("--vx", type=float, default=TEACHER_GATE_VX)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = Path(args.run)
    ckpts = sorted(run.glob("model_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    if not ckpts:
        raise SystemExit(f"no checkpoints in {run}")
    keep = [p for p in ckpts
            if int(p.stem.split("_")[-1]) % args.every == 0 or p == ckpts[-1]]
    print(f"sweeping {len(keep)} of {len(ckpts)} checkpoints at "
          f"difficulty {args.difficulty}, vx={args.vx}")

    import os
    env = {**os.environ, "MUJOCO_GL": "egl", "PYTHONPATH": str(ROOT / "train" / "src")}
    results = []
    for ckpt in keep:
        cmd = [
            str(TRAIN_PY), str(CHECKER), "--task", "blackw-plum-teacher",
            "--checkpoint", str(ckpt), "--envs", str(args.envs),
            "--episodes", str(args.episodes), "--clean-obs", "--no-dr",
            "--difficulty", str(args.difficulty), "--vx", str(args.vx),
            "--seed", str(args.seed),
        ]
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        line = next((ln for ln in reversed(proc.stdout.strip().splitlines())
                     if ln.startswith("{")), None)
        if line is None:
            print(f"  {ckpt.name:20s} FAILED TO EVAL")
            continue
        data = json.loads(line)
        results.append((data["success_rate"], ckpt.name, data))
        print(f"  {ckpt.name:20s} {data['successes']:3d}/{data['episodes']:3d} "
              f"= {data['success_rate']:.3f}  "
              f"terms={ {k: v for k, v in data['term_counts'].items() if v} }")

    if results:
        best = max(results)
        print(f"\nbest: {best[1]} at {best[0]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
