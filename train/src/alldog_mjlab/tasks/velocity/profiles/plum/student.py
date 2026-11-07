"""Depth-CNN student and deployment export.

Architecture (identical to the old repo's successful plum student):

    depth (1x48x64) -> CNN -> 64-D latent -> Linear -> predicted fine scan (1764)
    action = FROZEN_TEACHER_MLP([proprio_history(324), predicted_scan])

Only the CNN + scan head are trained.  The teacher MLP is frozen and detached,
so the student can never exceed the teacher; it merely replaces privileged
geometry with a camera.  Loss = action MSE + 0.1 * scan MSE on teacher rollouts.

Export rewrites the teacher's first Linear columns from MjLab's native
term-major / oldest-first history layout to the deployment frame-major /
newest-first layout, then scripts ``policy(obs_324, [depth])``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from rsl_rl.models.cnn_model import CNN

from .depth_camera import CNN_LATENT_DIM, DEPTH_CNN_CFG, DEPTH_CONTRACT
from .fine_scan import fine_scan_dim
from .teacher import (
    ACTOR_HISTORY,
    NO_COMMAND_DEPLOY_LAYOUT,
    NO_COMMAND_FRAME_DIM,
    PROPRIO_DIM,
)

TEACHER_LAYERS = ("mlp.0", "mlp.2", "mlp.4", "mlp.6")
REPO_ROOT = Path(__file__).resolve().parents[7]
_TEMPLATE_CONFIG = (
    REPO_ROOT / "demo" / "rl_sar" / "policy" / "blackW" /
    "row2p_p40w40_v1" / "config.yaml"
)


# ---------------------------------------------------------------------------
# Frozen teacher / trainable student modules
# ---------------------------------------------------------------------------


def build_teacher_mlp(actor_state_dict: dict, device: str = "cpu") -> nn.Sequential:
    """Reconstruct the actor MLP (2088 -> 512 -> 256 -> 128 -> 16).

    ``actor_obs_normalization`` is False in BLACK_CONFIG, so the actor is a bare
    MLP; no running-normaliser state needs restoring (verified: checkpoints
    contain only ``mlp.*`` and ``distribution.std_param``).
    """
    layers: list[nn.Module] = []
    for i, key in enumerate(TEACHER_LAYERS):
        w, b = actor_state_dict[f"{key}.weight"], actor_state_dict[f"{key}.bias"]
        layers.append(nn.Linear(w.shape[1], w.shape[0]))
        if i < len(TEACHER_LAYERS) - 1:
            layers.append(nn.ELU())
    mlp = nn.Sequential(*layers).to(device)
    with torch.no_grad():
        for i, key in enumerate(TEACHER_LAYERS):
            lin = mlp[2 * i]
            lin.weight.copy_(actor_state_dict[f"{key}.weight"])
            lin.bias.copy_(actor_state_dict[f"{key}.bias"])
    for p in mlp.parameters():
        p.requires_grad_(False)
    return mlp.eval()


def build_scan_cnn(scan_dim: int, device: str = "cpu") -> tuple[CNN, nn.Linear]:
    cnn = CNN(
        input_dim=(DEPTH_CONTRACT["height"], DEPTH_CONTRACT["width"]),
        input_channels=1,
        **DEPTH_CNN_CFG,
    ).to(device)
    head = nn.Linear(CNN_LATENT_DIM, scan_dim).to(device)
    return cnn, head


def student_action(
    teacher: nn.Module, cnn: CNN, head: nn.Linear,
    proprio: torch.Tensor, depth: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (student action, predicted scan).  ``proprio`` is native 324-D."""
    pred = head(cnn(depth))
    return teacher(torch.cat([proprio, pred], dim=-1)), pred


def distillation_loss(
    teacher: nn.Module, cnn: CNN, head: nn.Linear,
    proprio: torch.Tensor, depth: torch.Tensor,
    gt_scan: torch.Tensor, gt_action: torch.Tensor,
    action_weight: float = 1.0, scan_weight: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        teacher_action = teacher(torch.cat([proprio, gt_scan], dim=-1))
    student_act, pred = student_action(teacher, cnn, head, proprio, depth)
    l_act = F.mse_loss(student_act, teacher_action)
    l_scan = F.mse_loss(pred, gt_scan)
    return action_weight * l_act + scan_weight * l_scan, l_act, l_scan


# ---------------------------------------------------------------------------
# Deployment export
# ---------------------------------------------------------------------------


def build_permutation(
    extra_dim: int,
    frame_dim: int = NO_COMMAND_FRAME_DIM,
    history: int = ACTOR_HISTORY,
    term_layout=NO_COMMAND_DEPLOY_LAYOUT,
) -> np.ndarray:
    """Column permutation p with ``deploy[j] = native[p[j]]``.

    The first ``frame_dim*history`` columns are the proprio history (permuted
    from term-major/oldest-first to frame-major/newest-first); the trailing
    ``extra_dim`` scan columns stay in place.
    """
    target_dim = frame_dim * history
    p = np.zeros(target_dim + extra_dim, dtype=np.int64)
    native_block_offset = 0
    for term_dim, frame_offset in term_layout:
        for frame in range(history):            # target frame 0 = newest
            native_frame = history - 1 - frame  # native index 0 = oldest
            for d in range(term_dim):
                p[frame * frame_dim + frame_offset + d] = (
                    native_block_offset + native_frame * term_dim + d)
        native_block_offset += term_dim * history
    for j in range(extra_dim):
        p[target_dim + j] = target_dim + j
    return p


class StudentDeploy(nn.Module):
    """TorchScript policy: ``forward(obs_1d_324, [depth_1x1x48x64]) -> 16``."""

    def __init__(self, cnn: CNN, scan_head: nn.Linear, teacher: nn.Sequential):
        super().__init__()
        self.cnn = cnn
        self.scan_head = scan_head
        self.teacher = teacher

    def forward(self, obs_1d: torch.Tensor, obs_2d: list[torch.Tensor]) -> torch.Tensor:
        pred = self.scan_head(self.cnn(obs_2d[0]))
        return self.teacher(torch.cat([obs_1d, pred], dim=-1))

    @torch.jit.export
    def reset(self) -> None:
        pass


def _deploy_config(policy_name: str, depth: dict | None = None) -> dict:
    """Deployment config: same PD/action contract, minus the 3-D command obs.

    ``depth`` (the ``DEPTH_CONTRACT``) marks the policy as depth-conditioned so
    the C++ runtime loads it as ``forward(obs_1d, [image])`` instead of
    ``forward(obs_1d)``; the blind policies never carry the block.
    """
    raw = yaml.safe_load(_TEMPLATE_CONFIG.read_text(encoding="utf-8"))
    cfg = dict(next(iter(raw.values())))
    cfg["num_observations"] = NO_COMMAND_FRAME_DIM
    cfg["observations"] = [t for t in cfg["observations"] if t != "commands"]
    if depth is not None:
        cfg["depth"] = {
            "enabled": True,
            "camera_name": depth["camera_name"],
            "width": depth["width"],
            "height": depth["height"],
            "cutoff": depth["cutoff"],
            "min_depth": depth["min_depth"],
            "visible_geom_groups": list(depth["visible_geom_groups"]),
        }
    return {f"blackW/{policy_name}": cfg}


def export_policy_dir(
    out_dir: Path,
    policy_name: str,
    cnn: CNN,
    head: nn.Linear,
    teacher: nn.Sequential,
    teacher_checkpoint: str,
    metrics: dict,
) -> Path:
    """Write policy.pt + depth.json + config.yaml + manifest.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    scan_dim = fine_scan_dim()

    # Permute the frozen teacher's first layer in place (deployment layout).
    perm = build_permutation(scan_dim)
    assert teacher[0].in_features == len(perm), (
        teacher[0].in_features, len(perm))
    with torch.no_grad():
        teacher[0].weight.copy_(teacher[0].weight[:, torch.from_numpy(perm)])

    deploy = StudentDeploy(cnn, head, teacher).cpu().eval()
    dummy_obs = torch.randn(1, PROPRIO_DIM)
    dummy_depth = torch.rand(1, 1, DEPTH_CONTRACT["height"], DEPTH_CONTRACT["width"])
    with torch.no_grad():
        out = deploy(dummy_obs, [dummy_depth])
    assert tuple(out.shape) == (1, 16), tuple(out.shape)
    torch.jit.script(deploy).save(str(out_dir / "policy.pt"))

    (out_dir / "depth.json").write_text(
        json.dumps(DEPTH_CONTRACT, indent=1) + "\n", encoding="utf-8")
    (out_dir / "config.yaml").write_text(
        yaml.safe_dump(_deploy_config(policy_name), sort_keys=False), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "robot": "blackW",
        "part": 2,
        "level": "plum",
        "policy": policy_name,
        "route": "depth CNN -> predicted 1764-D fine scan -> frozen privileged teacher MLP",
        "source_teacher": teacher_checkpoint,
        "weight_file": "policy.pt",
        "contract": {
            "num_observations": NO_COMMAND_FRAME_DIM,
            "history": ACTOR_HISTORY,
            "num_dofs": 16,
            "support_kind": "wheel",
            "obs_terms": ["ang_vel_body", "gravity_vec", "dof_pos", "dof_vel", "actions"],
            "policy_call": "policy(obs_1d_324, [depth])",
            "depth_input": "1x48x64 normalized depth in [0,1], cutoff 3.0 m",
        },
        "metrics": metrics,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"[plum/export] wrote {out_dir} (policy.pt, depth.json, config.yaml, "
          "manifest.json)", flush=True)
    return out_dir
