#!/usr/bin/env python3
"""Export a blackW HIM checkpoint to the official HIM TorchScript policy contract.

Input: canonical 6x57 history, frame-major newest -> oldest, flattened to 342.
Output: 16 raw actions (FL,FR,RL,RR x hip,thigh,calf,wheel). The judge applies the
official HIM signed wheel action scales (FL/RL +10, FR/RR -10).

Usage:
  MUJOCO_GL=egl .venv/bin/python train/scripts/export/export_him_policy.py \
    --task blackw-official-him --checkpoint logs/.../model_0.pt \
    --out ~/rc_work/models/him_model0
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="blackw-official-him")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    configure_torch_backends()
    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(args.task).clip_actions)
    runner_cls = load_runner_cls(args.task)
    runner = runner_cls(env, asdict(load_rl_cfg(args.task)), device=args.device)
    from alldog_mjlab.utils.action_contract import check_checkpoint_contract

    declared = check_checkpoint_contract(args.checkpoint, expected="signed")
    if declared is None:
        print("[contract] WARNING: checkpoint has no action_contract tag; "
              "assuming signed HIM contract", flush=True)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device)
    actor = runner.alg.actor
    spec = runner.alg.spec
    jit_model = actor.as_jit().to("cpu").eval()
    traced = torch.jit.script(jit_model)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    policy_path = out_dir / "policy.pt"
    traced.save(str(policy_path))

    dummy = torch.zeros(1, spec.history_dim, dtype=torch.float32)
    with torch.no_grad():
        action = traced(dummy)
    if tuple(action.shape) != (1, spec.action_dim):
        raise RuntimeError(f"unexpected action shape {tuple(action.shape)}")
    contract = {
        "task": args.task,
        "checkpoint": str(args.checkpoint),
        "policy": str(policy_path),
        "input": "canonical history [B, history_length*single_frame_dim], frame-major newest->oldest",
        "history_length": spec.history_length,
        "single_frame_dim": spec.single_frame_dim,
        "input_dim": spec.history_dim,
        "action_dim": spec.action_dim,
        "action_contract": "signed",
        "action_order": "FL,FR,RL,RR x hip,thigh,calf,wheel",
        "action_scale": {
            "legs": [0.25, 0.25, 0.25, 0.25],
            "wheel": {"FL": 10.0, "FR": -10.0, "RL": 10.0, "RR": -10.0},
            "note": "official HIM signed wheel scale; do NOT use PPO trained-blackw unsigned +10 contract",
        },
        "dummy_action": action[0].tolist(),
    }
    policy_sha = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "robot": "blackW",
        "policy": out_dir.name,
        "weight_file": "policy.pt",
        "policy_sha256": policy_sha,
        "checkpoint": str(args.checkpoint),
        "contract": {
            "num_dofs": int(spec.action_dim),
            "num_observations": int(spec.single_frame_dim),
            "history": int(spec.history_length),
            "support_kind": "wheel",
            "action_contract": "signed",
        },
        "export": contract,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote", out_dir / "manifest.json")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
