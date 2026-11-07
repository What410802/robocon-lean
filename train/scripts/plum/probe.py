#!/usr/bin/env python3
"""Measure throughput and VRAM for a candidate plum-teacher batch size.

The GPU is a single 16 GB card.  Training time scales with env count, and a
running out of memory mid-run wastes hours, so sample real steps first.

    MUJOCO_GL=egl pixi run -e train python train/scripts/plum/probe.py --envs 1024 512 256
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "train" / "src"))

import torch  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

from alldog_mjlab.tasks.velocity.profiles.plum.teacher import plum_teacher_env_cfg  # noqa: E402


def probe(num_envs: int, steps: int) -> dict:
    cfg = plum_teacher_env_cfg(stage=0)
    cfg.scene.num_envs = num_envs
    try:
        env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    except Exception as exc:  # noqa: BLE001 - OOM is the expected failure mode
        return {"envs": num_envs, "ok": False, "error": type(exc).__name__}
    env.reset()
    action = torch.zeros(num_envs, 16, device="cuda:0")
    # Warm up the kernels/RNG before timing.
    for _ in range(3):
        env.step(action)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(action)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    env_steps_per_s = num_envs * steps / dt
    peak = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30
    env.close()
    torch.cuda.reset_peak_memory_stats()
    return {"envs": num_envs, "ok": True, "env_steps_per_s": env_steps_per_s,
            "peak_gib": peak, "reserved_gib": reserved}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--envs", type=int, nargs="+", default=[1024, 512, 256])
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()
    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
    for n in args.envs:
        r = probe(n, args.steps)
        if r["ok"]:
            per_iter = 64 / r["env_steps_per_s"] * 1000  # s per 64-step iteration
            print(f"  envs={r['envs']:5d} ok  {r['env_steps_per_s']:8.0f} env-steps/s  "
                  f"peak={r['peak_gib']:.1f} GiB reserved={r['reserved_gib']:.1f} GiB  "
                  f"~{per_iter / 60:.2f} min / 64-step iter")
        else:
            print(f"  envs={r['envs']:5d} FAILED ({r['error']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
