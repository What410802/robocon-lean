#!/usr/bin/env python3
"""Export a native-history policy to TorchScript with the HIM deployment layout.

MjLab native history (``ObservationGroupCfg.history_length=6``) is **term-major,
oldest -> newest**:

    [command x6, ang_vel x6, gravity x6, joint_pos x6, joint_vel x6, actions x6]

whereas the deployment contract (``rl_sar`` HIM / ``best.pt`` / the local judge)
is **frame-major, newest -> oldest**, i.e. 6 x 45 with frame 0 newest.

This script permutes the actor's first Linear input columns from the native
layout to the deployment layout (no retraining), then exports the actor JIT
module.  Works for both blind MLP actors (45*k input) and depth CNN actors
(45*k + 64 CNN latent input; the CNN columns are not permuted).

Usage (remote training repo, GPU):
    MUJOCO_GL=egl .venv/bin/python train/scripts/export/export_history_policy.py \
        --task black-competition-hist \
        --checkpoint logs/rsl_rl/black_velocity/<run>/model_xxx.pt \
        --out ~/models/competition_hist/policy.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from alldog_mjlab.utils.observation_contract import (
    BLACKW_DEPLOY_LAYOUT, BLACKW_FRAME_DIM, BLACKW_TERM_DIMS, BLACKW_TERMS,
)

ROBOTS = {
    # robot -> (single frame dim, deployment term layout)
    "black": (45, (
        (3, 0),    # command
        (3, 3),    # base_ang_vel
        (3, 6),    # projected_gravity
        (12, 9),   # joint_pos
        (12, 21),  # joint_vel
        (12, 33),  # actions
    )),
    "blackW": (57, (
        (3, 0),    # command
        (3, 3),    # base_ang_vel
        (3, 6),    # projected_gravity
        (16, 9),   # joint_pos (incl. 4 wheel positions, zeroed by the task)
        (16, 25),  # joint_vel (incl. 4 wheel velocities)
        (16, 41),  # previous actions (3 position residual + 1 wheel velocity per leg)
    )),
}


def build_permutation(history_length: int, cnn_dim: int, frame_dim: int = 45,
                      term_layout=()) -> np.ndarray:
    """Permutation p with target[j] = native[p[j]] for the first frame_dim*k columns."""
    frames = history_length
    target_dim = frame_dim * frames
    p = np.zeros(target_dim + cnn_dim, dtype=np.int64)
    native_block_offset = 0
    for term_dim, frame_offset in term_layout:
        for frame in range(frames):  # target frame 0 = newest
            native_frame = frames - 1 - frame  # native index 0 = oldest
            for d in range(term_dim):
                target_index = frame * frame_dim + frame_offset + d
                native_index = native_block_offset + native_frame * term_dim + d
                p[target_index] = native_index
        native_block_offset += term_dim * frames
    for j in range(cnn_dim):
        p[target_dim + j] = target_dim + j
    return p


def first_linear_module(actor):
    for name, module in actor.named_modules():
        if isinstance(module, torch.nn.Linear):
            return module
    raise ValueError("history export: actor has no nn.Linear layer")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", required=True)
    ap.add_argument("--robot", choices=tuple(ROBOTS), default=None,
                    help="deployment frame layout; inferred from --task when omitted")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    robot = args.robot or ("blackW" if args.task.startswith("blackw-") else "black")
    frame_dim, term_layout = ROBOTS[robot]
    if robot == "blackW":
        if frame_dim != BLACKW_FRAME_DIM or tuple(term_layout) != BLACKW_DEPLOY_LAYOUT:
            raise SystemExit("blackW export layout diverges from observation_contract.py")
    configure_torch_backends()
    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = 1
    for sensor in env_cfg.scene.sensors or ():
        if hasattr(sensor, "debug_vis"):
            sensor.debug_vis = False
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=None)
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(load_rl_cfg(args.task)), device=args.device)
    from alldog_mjlab.utils.action_contract import (
        current_training_contract, check_checkpoint_contract,
    )

    expected_contract = current_training_contract()
    declared = check_checkpoint_contract(args.checkpoint, expected=expected_contract)
    if declared is None:
        print(f"[contract] WARNING: checkpoint has no action_contract tag; "
              f"assuming env contract {expected_contract!r}", flush=True)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True,
                map_location=args.device)

    actor = runner.alg.get_policy()
    obs_manager = env.unwrapped.observation_manager
    actor_shape = tuple(int(v) for v in obs_manager.group_obs_dim["actor"])
    if len(actor_shape) != 1:
        raise ValueError(f"history export: actor group is not flat: {actor_shape}")
    actor_dim = actor_shape[0]
    if actor_dim % frame_dim != 0:
        raise ValueError(f"history export: actor dim {actor_dim} is not {frame_dim}*k for robot {robot}")
    history_length = actor_dim // frame_dim
    camera_shape = None
    if "camera" in obs_manager.group_obs_dim:
        camera_shape = tuple(int(v) for v in obs_manager.group_obs_dim["camera"])
    linear = first_linear_module(actor)
    in_features = int(linear.weight.shape[1])
    # The camera group is CNN-encoded: the extra input columns are the CNN *latent*
    # (e.g. 64), not raw pixels (prod(shape) = 1*48*64).  Derive it from the actor.
    cnn_latent_dim = in_features - actor_dim
    if cnn_latent_dim < 0:
        raise ValueError(
            f"history export: first Linear in={in_features} < actor dim {actor_dim}")
    if camera_shape is not None and cnn_latent_dim == 0:
        raise ValueError("history export: camera group present but actor has no CNN-latent columns")
    perm = build_permutation(history_length, cnn_latent_dim, frame_dim, term_layout)
    old_weight = linear.weight.data.clone()
    linear.weight.data = old_weight[:, perm].contiguous()
    print(f"permuted first Linear columns: {in_features} "
          f"(history={history_length}, cnn-latent={cnn_latent_dim})")

    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    runner.export_policy_to_jit(str(out_dir))
    policy_path = out_dir / "policy.pt"
    policy = torch.jit.load(str(policy_path), map_location="cpu").eval()

    # Numerical check: reordered input batch must reproduce the original first-layer output.
    device = linear.weight.device
    x_native = torch.randn(1, in_features, device=device)
    x_target = x_native[:, torch.from_numpy(perm).to(device)]
    expected = x_native @ old_weight.T + linear.bias.data
    got = x_target @ linear.weight.data.T + linear.bias.data
    assert torch.allclose(expected, got, atol=1e-5), "permutation check failed"

    dummy_1d = torch.zeros(1, actor_dim)
    if cnn_latent_dim:
        dummy_2d = [torch.zeros(1, *camera_shape)]
        with torch.no_grad():
            action = policy(dummy_1d, dummy_2d)
    else:
        with torch.no_grad():
            action = policy(dummy_1d)
    print(f"exported -> {policy_path} sha256={sha256(policy_path)} size={policy_path.stat().st_size}")
    print(f"dummy forward action_shape={tuple(action.shape)}")
    from alldog_mjlab.tasks.velocity.common.config import BLACK_CONFIG as C
    ckpt = Path(args.checkpoint)
    meta = {
        "schema_version": 1,
        "robot": robot,
        "task": args.task,
        "frequency": {
            "physics_dt": C.control.physics_dt,
            "control_period": C.control.control_period,
            "policy_period": C.control.policy_period,
            "sim_decimation": C.control.sim_decimation,
            "deployment_decimation": C.control.deployment_decimation,
        },
        "checkpoint_sha256": sha256(ckpt),
        "checkpoint": str(args.checkpoint),
        "policy": str(policy_path),
        "sha256": sha256(policy_path),
        "layout": "frame-major newest->oldest (HIM deployment)",
        "history_length": history_length,
        "actor_dim": actor_dim,
        "cnn_latent_dim": cnn_latent_dim,
        "camera_shape": list(camera_shape) if camera_shape else None,
        "action_contract": expected_contract,
        "observation_layout": {
            "terms": list(BLACKW_TERMS),
            "term_dims": list(BLACKW_TERM_DIMS),
            "frame_dim": BLACKW_FRAME_DIM,
            "history": history_length,
            "deploy_layout": [list(x) for x in BLACKW_DEPLOY_LAYOUT],
        },
        "action_shape": list(action.shape),
    }
    manifest = {
        "schema_version": 1,
        "robot": robot,
        "policy": out_dir.name,
        "weight_file": "policy.pt",
        "policy_sha256": meta["sha256"],
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "contract": {
            "num_dofs": int(action.shape[-1]),
            "num_observations": int(actor_dim),
            "history": int(history_length),
            "support_kind": "wheel" if robot == "blackW" else "foot",
            "action_contract": expected_contract,
        },
        "export": meta,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", out_dir / "manifest.json")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
