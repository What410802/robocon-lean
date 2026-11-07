#!/usr/bin/env python3
"""Export a depth-actor (rsl_rl ``CNNModel``) checkpoint as a deployable policy dir.

Writes ``policy.pt`` (TorchScript), ``depth.json`` (the camera contract verbatim),
``config.yaml`` (deployment frame layout, command dropped) and ``manifest.json``.
The actor's first Linear columns are permuted from MjLab's native term-major /
oldest-first history to the deployment frame-major / newest-first layout; the CNN
latent columns are left in place. Unlike ``export_history_policy.py`` this uses
the plum 54-dim frame (no command), not the 57-dim blackW frame.

    MUJOCO_GL=egl PYTHONPATH=train/src .pixi/envs/train/bin/python \
      train/scripts/plum/export_depth_policy.py \
      --checkpoint <model_x.pt> --out demo/rl_sar/policy/blackW/plum_depth_v1 \
      [--metrics artifacts/plum/eval/xxx.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "train" / "src"))

import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from rsl_rl.models.cnn_model import CNNModel  # noqa: E402

from alldog_mjlab.tasks.velocity.common.config import BLACK_CONFIG  # noqa: E402
from alldog_mjlab.tasks.velocity.profiles.plum.depth_camera import (  # noqa: E402
    CNN_LATENT_DIM, DEPTH_CNN_CFG, DEPTH_CONTRACT,
)
import numpy as np  # noqa: E402

from alldog_mjlab.tasks.velocity.profiles.plum.student import (  # noqa: E402
    _deploy_config, build_permutation,
)
from alldog_mjlab.tasks.velocity.profiles.plum.teacher import (  # noqa: E402
    ACTOR_HISTORY, NO_COMMAND_FRAME_DIM, PROPRIO_DIM,
)

#: Deployment frame layout: (offset, dim) of the blocks whose JOINT order differs.
#: ``build_permutation`` reorders terms and history but assumes identical joint
#: order on both sides.  It is not identical: mjlab's asset orders joints
#: FL, FR, RR, RL, while ``robot.yaml`` / the eval build them FL, FR, RL, RR.
#: The ``actions`` block is in ACTION-TERM order (FL, FR, RL, RR) on both sides,
#: so only joint_pos and joint_vel need realignment.
_JOINT_BLOCKS = ((6, 16), (22, 16))


def _joint_order_swap() -> np.ndarray:
    """Native (FL,FR,RR,RL) -> deployment (FL,FR,RL,RR) joint index map."""
    s = np.arange(16)
    s[8:12], s[12:16] = np.arange(12, 16), np.arange(8, 12)
    return s


def align_joint_order(perm: np.ndarray) -> np.ndarray:
    """Rewrite a frame permutation so joint blocks are read in training order.

    ``perm`` satisfies ``deployed[j] = native[perm[j]]``.  A deployment vector
    ``E`` built by the eval holds joint blocks in policy order, i.e.
    ``E[f, joint_pos, i] == native[..., swap(i)]``.  Reading ``E[perm]`` would
    therefore feed the actor its two rear legs swapped; this rewrites the joint
    columns of ``perm`` so ``E[q]`` reproduces the native vector.
    """
    q = perm.copy()
    swap = _joint_order_swap()
    for f in range(ACTOR_HISTORY):
        for off, dim in _JOINT_BLOCKS:
            for i in range(dim):
                q[f * NO_COMMAND_FRAME_DIM + off + i] = (
                    f * NO_COMMAND_FRAME_DIM + off + int(swap[i]))
    return q


def build_eval_layout(native: torch.Tensor) -> torch.Tensor:
    """Reference: the vector the EVAL builds from the same physical state."""
    e = torch.zeros_like(native)
    swap = _joint_order_swap()
    blocks = ((0, 3), (3, 3), (6, 16), (22, 16), (38, 16))
    offsets, off = {}, 0
    for name, (off_d, dim) in zip(("ang", "grav", "jp", "jv", "act"), blocks):
        offsets[name] = off
        off += dim * ACTOR_HISTORY
    for f in range(ACTOR_HISTORY):
        nf = ACTOR_HISTORY - 1 - f
        for name, (off_d, dim) in zip(("ang", "grav", "jp", "jv", "act"), blocks):
            base_n = offsets[name] + nf * dim
            base_d = f * NO_COMMAND_FRAME_DIM + off_d
            if name in ("jp", "jv"):
                for i in range(dim):
                    e[base_d + i] = native[base_n + int(swap[i])]
            else:
                e[base_d:base_d + dim] = native[base_n:base_n + dim]
    return e


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", default=None, help="policy name (default: out dir name)")
    ap.add_argument("--metrics", type=Path, default=None, help="optional eval JSON")
    args = ap.parse_args()
    name = args.name or args.out.name

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("actor_state_dict", ckpt)
    sample = TensorDict(
        {"actor": torch.zeros(2, PROPRIO_DIM),
         "camera": torch.zeros(2, 1, DEPTH_CONTRACT["height"], DEPTH_CONTRACT["width"])},
        batch_size=[2],
    )
    actor = CNNModel(
        sample, {"actor": ("actor", "camera"), "critic": ("critic",)},
        obs_set="actor", output_dim=16,
        hidden_dims=tuple(BLACK_CONFIG.policy.actor_hidden_dims),
        activation=BLACK_CONFIG.policy.activation,
        obs_normalization=False,
        distribution_cfg=None,
        cnn_cfg=dict(DEPTH_CNN_CFG),
    )
    # The distribution head is training-only; deployment reads the mean.
    state = {k: v for k, v in state.items() if not k.startswith("distribution.")}
    actor.load_state_dict(state, strict=True)
    actor.eval()

    perm = build_permutation(CNN_LATENT_DIM)  # 324 + CNN latent (frames only)
    linear = actor.mlp[0]
    if linear.in_features != len(perm):
        raise SystemExit(
            f"actor first Linear in={linear.in_features} != perm {len(perm)}")
    old = linear.weight.data.clone()
    linear.weight.data = old[:, torch.from_numpy(perm)].contiguous()

    # Numerical check: the same native input must produce the same action.
    # Layout parity beyond this (joint order, camera) is asserted by the
    # training-side observation dump: joint_pos/joint_vel declare
    # FL,FR,RL,RR with preserve_order=True, exactly the deploy policy order.
    x_native = torch.randn(1, len(perm))
    x_deploy = x_native[:, torch.from_numpy(perm)]
    expected = x_native @ old.T + linear.bias.data
    got = x_deploy @ linear.weight.data.T + linear.bias.data
    if not torch.allclose(expected, got, atol=1e-5):
        raise SystemExit("history permutation check failed")

    jit = torch.jit.script(actor.as_jit().cpu().eval())
    args.out.mkdir(parents=True, exist_ok=True)
    jit.save(str(args.out / "policy.pt"))
    policy = torch.jit.load(str(args.out / "policy.pt")).eval()
    depth = torch.rand(1, 1, DEPTH_CONTRACT["height"], DEPTH_CONTRACT["width"])
    action = policy(torch.randn(1, PROPRIO_DIM), [depth])
    if tuple(action.shape) != (1, 16):
        raise SystemExit(f"scripted forward returned {tuple(action.shape)}")

    (args.out / "depth.json").write_text(
        json.dumps(DEPTH_CONTRACT, indent=1) + "\n", encoding="utf-8")
    (args.out / "config.yaml").write_text(
        __import__("yaml").safe_dump(_deploy_config(name, depth=DEPTH_CONTRACT),
                                     sort_keys=False),
        encoding="utf-8")
    metrics = {}
    if args.metrics is not None:
        metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "robot": "blackW",
        "part": 2,
        "level": "plum",
        "policy": name,
        "route": "depth CNN (64-D latent) + 324-D proprio history -> MLP (no fine scan)",
        "weight_file": "policy.pt",
        "policy_sha256": sha256(args.out / "policy.pt"),
        "checkpoint": str(args.checkpoint),
        "contract": {
            "num_observations": 54,
            "history": 6,
            "num_dofs": 16,
            "support_kind": "wheel",
            "obs_terms": ["ang_vel_body", "gravity_vec", "dof_pos", "dof_vel", "actions"],
            "policy_call": "policy(obs_1d_324, [depth])",
            "depth_input": "1x48x64 normalized depth in [0,1], cutoff 3.0 m",
        },
        "metrics": metrics,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"[plum/export-depth] wrote {args.out} "
          f"(policy.pt, depth.json, config.yaml, manifest.json)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
