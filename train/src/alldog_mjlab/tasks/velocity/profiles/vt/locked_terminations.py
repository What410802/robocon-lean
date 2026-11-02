"""Task-level failure terminations for the locked-wheel gait.

Velocity tracking and active foot placement are task priorities equal to
"don't fall", so they must be able to terminate an episode instead of only
denting a reward:

* ``track_*``: sustained actual-vs-commanded velocity error (vx/vy/wz);
* ``static``: four feet in contact while the base makes no progress;
* ``foot``: a foot permanently airborne (tripod / dangling) or permanently in
  contact (no swing) while a motion command is active;
* ``rect``: support rectangle outside the calibrated nominal + margin.

All timers are stateful per env and reset on episode reset or command change,
so a new command gets a short grace window before it can terminate.  The
surrounding termination penalty (`-100`) makes these failures expensive.
"""

from __future__ import annotations

from typing import Any

import torch

from mjlab.managers import ManagerTermBase, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from alldog_mjlab.tasks.velocity.profiles.vt.locked import (
    LOCKED_RECT_TERM_MARGIN,
    _contact_legs,
    _feet_body_xy,
    _sensor_cols,
    command_support_box,
)


class LockedGaitFailure(ManagerTermBase):
    """Stateful velocity-tracking / support-foot failure detector."""

    def __init__(self, cfg: TerminationTermCfg, env) -> None:
        super().__init__(env)
        p = cfg.params
        self.command_name = p["command_name"]
        self.asset = env.scene[p["asset_cfg"].name]
        self.geom_cfg: SceneEntityCfg = p["wheel_cfg"]
        self.lin_abs = float(p.get("lin_abs", 0.10))
        self.lin_frac = float(p.get("lin_frac", 0.35))
        self.ang_abs = float(p.get("ang_abs", 0.08))
        self.ang_frac = float(p.get("ang_frac", 0.35))
        self.grace = float(p.get("grace", 0.30))
        self.track_timeout = float(p.get("track_timeout", 0.80))
        self.static_speed = float(p.get("static_speed", 0.03))
        self.static_timeout = float(p.get("static_timeout", 0.80))
        self.air_timeout = float(p.get("air_timeout", 0.80))
        self.stance_timeout = float(p.get("stance_timeout", 1.50))
        self.rect_timeout = float(p.get("rect_timeout", 0.50))
        self.rect_x_margin = float(p.get("rect_x_margin", 0.10))
        self.rect_y_margin = float(p.get("rect_y_margin", 0.08))
        n = env.num_envs
        self._prev_cmd = torch.zeros((n, 3), device=env.device)
        self._cmd_age = torch.zeros(n, device=env.device)
        self._track_lin = torch.zeros(n, device=env.device)
        self._track_ang = torch.zeros(n, device=env.device)
        self._static = torch.zeros(n, device=env.device)
        self._rect = torch.zeros(n, device=env.device)
        self._air = torch.zeros((n, 4), device=env.device)
        self._stance = torch.zeros((n, 4), device=env.device)
        self._cols = _sensor_cols(env)

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._prev_cmd[env_ids] = 0.0
        self._cmd_age[env_ids] = 0.0
        self._track_lin[env_ids] = 0.0
        self._track_ang[env_ids] = 0.0
        self._static[env_ids] = 0.0
        self._rect[env_ids] = 0.0
        self._air[env_ids] = 0.0
        self._stance[env_ids] = 0.0

    def __call__(self, env, **kwargs: Any) -> torch.Tensor:
        del kwargs
        dt = float(env.step_dt)
        cmd = env.command_manager.get_command(self.command_name)[:, :3]
        changed = torch.linalg.vector_norm(cmd - self._prev_cmd, dim=1) > 0.02
        self._prev_cmd = cmd.detach().clone()
        self._cmd_age = torch.where(changed, torch.zeros_like(self._cmd_age),
                                    self._cmd_age + dt)
        self._track_lin = torch.where(changed, torch.zeros_like(self._track_lin),
                                      self._track_lin)
        self._track_ang = torch.where(changed, torch.zeros_like(self._track_ang),
                                      self._track_ang)
        self._static = torch.where(changed, torch.zeros_like(self._static),
                                   self._static)
        self._rect = torch.where(changed, torch.zeros_like(self._rect),
                                 self._rect)
        self._air = torch.where(changed.unsqueeze(1), torch.zeros_like(self._air),
                                self._air)
        self._stance = torch.where(changed.unsqueeze(1), torch.zeros_like(self._stance),
                                   self._stance)

        actual_lin = self.asset.data.root_link_lin_vel_b[:, :2]
        actual_wz = self.asset.data.root_link_ang_vel_b[:, 2]
        cmd_xy = cmd[:, :2]
        cmd_norm = torch.linalg.vector_norm(cmd_xy, dim=1)
        moving = (cmd_norm > 0.03) | (torch.abs(cmd[:, 2]) > 0.03)
        active = moving & (self._cmd_age > self.grace)

        lin_err = torch.linalg.vector_norm(cmd_xy - actual_lin, dim=1)
        lin_thr = torch.maximum(
            torch.full_like(lin_err, self.lin_abs),
            self.lin_frac * cmd_norm,
        )
        ang_err = torch.abs(cmd[:, 2] - actual_wz)
        ang_thr = torch.maximum(
            torch.full_like(ang_err, self.ang_abs),
            self.ang_frac * torch.abs(cmd[:, 2]),
        )
        self._track_lin = torch.where(active & (lin_err > lin_thr),
                                      self._track_lin + dt, torch.zeros_like(self._track_lin))
        self._track_ang = torch.where(active & (ang_err > ang_thr),
                                      self._track_ang + dt, torch.zeros_like(self._track_ang))

        contact = _contact_legs(env)
        xy = _feet_body_xy(env, self.geom_cfg)
        # all four planted and no base progress
        speed = torch.linalg.vector_norm(self.asset.data.root_link_lin_vel_w[:, :2], dim=1)
        static = moving & contact.all(dim=1) & (speed < self.static_speed)
        self._static = torch.where(static, self._static + dt, torch.zeros_like(self._static))

        # per-foot air/stance persistence while a motion command is active
        airborne = (~contact).to(torch.float32)
        grounded = contact.to(torch.float32)
        self._air = torch.where(active.unsqueeze(1),
                                self._air + airborne * dt,
                                torch.zeros_like(self._air))
        self._stance = torch.where(active.unsqueeze(1),
                                   self._stance + grounded * dt,
                                   torch.zeros_like(self._stance))

        # Command-dependent trunk-centred support box (same geometry as the
        # training penalty): at cmd=0 tight around the trunk, growing with the
        # commanded stance excursion.  Termination adds a slack margin so only
        # a clearly out-of-box stance is killed.
        hx, hy = command_support_box(env)
        x_ext = xy[..., 0].abs().max(dim=1).values
        y_ext = xy[..., 1].abs().max(dim=1).values
        rect_bad = moving & ((x_ext > hx + self.rect_x_margin)
                             | (y_ext > hy + self.rect_y_margin))
        self._rect = torch.where(rect_bad, self._rect + dt, torch.zeros_like(self._rect))

        return (
            (self._track_lin > self.track_timeout)
            | (self._track_ang > self.track_timeout)
            | (self._static > self.static_timeout)
            | (self._air > self.air_timeout).any(dim=1)
            | (self._stance > self.stance_timeout).any(dim=1)
            | (self._rect > self.rect_timeout)
        )


def make_locked_gait_failure(asset_name: str = "robot",
                             wheel_geoms: tuple[str, ...] = (),
                             **params: Any) -> TerminationTermCfg:
    return TerminationTermCfg(
        func=LockedGaitFailure,
        params={
            "command_name": "twist",
            "asset_cfg": SceneEntityCfg(asset_name),
            "wheel_cfg": SceneEntityCfg(asset_name, geom_names=wheel_geoms,
                                        preserve_order=True),
            **params,
        },
    )


__all__ = ["LockedGaitFailure", "make_locked_gait_failure"]
