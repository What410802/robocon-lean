"""Mixed wheel-leg action term for blackW.

One action term per leg, four raw actions per leg in the policy order
hip -> thigh -> calf -> wheel. The first three are position residuals around the
default joint pose (scale 0.25); the wheel action is a velocity target (scale 10).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_SANITIZE_COUNTS: dict[str, int] = {'action': 0}
_SANITIZE_WARNED: set[str] = set()


def _warn_sanitize(kind: str) -> None:
    count = _SANITIZE_COUNTS[kind]
    if kind not in _SANITIZE_WARNED or count % 100000 < count % 1000:
        print(f"[blackw] WARNING: sanitised non-finite {kind} values (total {count})", flush=True)
        _SANITIZE_WARNED.add(kind)

from alldog_mjlab.utils import knobs
from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg
from mjlab.managers.action_manager import ActionTerm


@dataclass(kw_only=True)
class WheelLegActionCfg(BaseActionCfg):
    """One leg: 3 position residuals + 1 wheel velocity target."""

    def __post_init__(self) -> None:
        self.transmission_type = TransmissionType.JOINT

    def build(self, env) -> "WheelLegAction":
        return WheelLegAction(self, env)


class WheelLegAction(BaseAction):
    cfg: WheelLegActionCfg

    def __init__(self, cfg: WheelLegActionCfg, env) -> None:
        super().__init__(cfg=cfg, env=env)
        assert self.action_dim == 4, self.action_dim
        default_pos = self._entity.data.default_joint_pos
        assert default_pos is not None
        self._default_pos = default_pos[:, self._target_ids[:3]].clone()

    def apply_actions(self) -> None:
        # HIM/PPO exploration can produce non-finite actions on contact-heavy
        # platform terrain; sanitize before MuJoCo and report the event instead
        # of silently clipping.
        raw = self._processed_actions
        nonfinite = ~torch.isfinite(raw)
        if bool(nonfinite.any()):
            _SANITIZE_COUNTS['action'] += int(nonfinite.sum())
            _warn_sanitize('action')
            raw = torch.nan_to_num(raw, nan=0.0, posinf=100.0, neginf=-100.0)
            self._processed_actions = raw
        leg_target = raw[:, :3] + self._default_pos
        wheel_target = raw[:, 3:]
        # Deployment/eval clamps dq_des to ``command_dq_max`` (default 20 rad/s)
        # in command_limit; training had no such clamp (audit 2026-10-30).  Keep
        # it off by default (checkpoints were trained without it), but expose a
        # knob so a run can be trained under the same wheel-speed contract.
        clip = knobs.get_float("BLACKW_WHEEL_DQ_CLIP", 0.0)
        if clip > 0.0:
            wheel_target = torch.clamp(wheel_target, -clip, clip)
        self._entity.set_joint_position_target(leg_target, joint_ids=self._target_ids[:3])
        self._entity.set_joint_velocity_target(wheel_target, joint_ids=self._target_ids[3:])
