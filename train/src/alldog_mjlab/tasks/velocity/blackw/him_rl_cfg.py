"""HIM runner configuration for blackW.

Uses the validated N-W-wolf MjLab HIM algorithm but the blackW-specific HIM
interface: actor history group ``actor`` (6 x 57), estimator velocity group, and
the 16 joint/wheel actions.  Encoder/actor shapes are the official blackW HIM
shapes, which makes an official TorchScript policy usable for warm start.
"""
from __future__ import annotations

import os
from dataclasses import replace

from alldog_mjlab.tasks.velocity.black import him as black_him
from alldog_mjlab.tasks.velocity.common.config import BLACK_CONFIG
from alldog_mjlab.tasks.velocity.common.him_rl_cfg import (
    HimRunnerParams,
    black_him_runner_cfg,
)


def blackw_him_runner_cfg():
    cfg = black_him_runner_cfg("rough")
    cfg.him = HimRunnerParams(
        history_group=black_him.BLACK_HIM_ACTOR_GROUP,
        velocity_group=black_him.BLACK_HIM_VELOCITY_GROUP,
        command_dim=black_him.BLACK_HIM_COMMAND_DIM,
        latent_dim=16,
        terminal_ids_key=black_him.BLACK_HIM_TERMINAL_IDS_KEY,
        terminal_frame_key=black_him.BLACK_HIM_TERMINAL_FRAME_KEY,
        terminal_velocity_key=black_him.BLACK_HIM_TERMINAL_VELOCITY_KEY,
    )
    cfg.experiment_name = "blackw_velocity_him"
    cfg.max_iterations = 10_000
    cfg.save_interval = 50
    # Match the PPO/deployment clip; warm-started official actor can otherwise emit
    # extreme values on unseen official geometry and trigger a sim assert.
    cfg.clip_actions = BLACK_CONFIG.control.action_clip
    # Lower exploration noise helps the contact-heavy platform task avoid
    # Mujoco-Warp contact asserts while the official warm start supplies the mean.
    dist = dict(cfg.actor.distribution_cfg or {})
    # The official TorchScript export does not carry distribution.std_param, so
    # warm-start leaves the action std newly initialized.  Use a small std by
    # default instead of 0.3, and expose it for controlled A/B.
    dist["init_std"] = float(os.environ.get("RC_HIM_INIT_STD", "0.1"))
    cfg.actor = replace(cfg.actor, distribution_cfg=dist)
    # The custom BlackWHimOfficialRunner handles the official TorchScript warm start.
    cfg.warm_start_supported = False
    return cfg
