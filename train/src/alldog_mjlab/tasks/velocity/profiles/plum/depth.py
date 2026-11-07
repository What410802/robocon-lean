"""Depth-actor plum task: actor sees proprio history + ONE front depth frame.

This is the "student learns in the environment" branch.  Unlike
``student.py`` (depth -> predicted 1764-D fine scan -> FROZEN teacher), the
actor here is a single ``rsl_rl.CNNModel`` that maps

    depth (1x48x64) -> CNN -> 64-D latent,  cat with 324-D proprio history
    -> MLP -> 16 joint actions

Nothing privileged enters the actor.  The critic keeps the 247-D privileged
group (``height_scan`` etc.) exactly as the teacher's does, so the value
function still sees terrain while the policy does not.

Why this exists (measured, 2026-11-06): the fine scan is 4 x 441 rays over a
0.20 m patch CENTRED on each wheel -- it covers +/-0.10 m around the wheel and
nothing ahead of it, while the camera covers ground from 0.51 m to 3.0 m ahead.
Predicting the scan therefore throws away precisely the forward information the
camera uniquely has.  Zeroing the scan drops the teacher from 94.8 % to 0/384
on the official row, so perception is load-bearing; the question this task
answers is whether the ACTOR can use depth directly.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from alldog_mjlab.utils import knobs
from alldog_mjlab.tasks.velocity.profiles.plum.depth_camera import (
    CNN_LATENT_DIM, DEPTH_CNN_CFG, add_depth_camera,
)
from alldog_mjlab.tasks.velocity.profiles.plum.fine_scan import FINE_SCAN_SENSOR
from alldog_mjlab.tasks.velocity.profiles.plum.runner import PlumTeacherRunner
from alldog_mjlab.tasks.velocity.profiles.plum.teacher import (
    plum_teacher_env_cfg, plum_teacher_runner_cfg,
)

CAMERA_GROUP = "camera"


def plum_depth_env_cfg(play: bool = False, stage: int = 0):
    """Teacher env minus the fine scan, plus the depth camera on the actor."""
    cfg = plum_teacher_env_cfg(play=play, stage=stage)
    if "fine_scan" not in cfg.observations:
        raise ValueError("expected the teacher's 'fine_scan' observation group")
    cfg.observations.pop("fine_scan")
    # Drop the ray sensor too: it costs nothing but keeps the registered cfg
    # honest about what the actor consumes.
    sensors = tuple(s for s in (cfg.scene.sensors or ()) if s.name != FINE_SCAN_SENSOR)
    if len(sensors) == len(cfg.scene.sensors or ()):
        raise ValueError(f"sensor {FINE_SCAN_SENSOR!r} not found")
    cfg.scene.sensors = sensors
    add_depth_camera(cfg)
    print("[plum/depth] actor = 324-D proprio history + 1x48x64 depth "
          "(fine scan removed; critic unchanged)", flush=True)
    return cfg


def plum_depth_runner_cfg(stage: int = 0):
    """PPO with a CNN actor over (proprio history, camera)."""
    cfg = plum_teacher_runner_cfg(stage=stage)
    cfg.actor = replace(
        cfg.actor,
        class_name="CNNModel",
        cnn_cfg=dict(DEPTH_CNN_CFG),
    )
    cfg.obs_groups = {"actor": ("actor", CAMERA_GROUP), "critic": ("critic",)}
    cfg.experiment_name = "blackw_plum_depth"
    cfg.run_name = f"plum_depth_s{stage}"
    print(f"[plum/depth] runner: CNNModel actor, cnn latent {CNN_LATENT_DIM}, "
          f"obs_groups={cfg.obs_groups}", flush=True)
    return cfg


class PlumDepthRunner(PlumTeacherRunner):
    """Plum runner whose privileged warm start targets a CNN latent, not a scan.

    Initialisation routes, tried in order:

    * ``PART2_PLUM_DEPTH_BC_INIT`` -- a full CNNModel ``actor_state_dict`` from
      ``scripts/plum/bc_depth.py`` (the BC boot).  Loaded strictly.
    * ``PART2_PLUM_WARM_START`` -- the blind demo policy; its 324 proprio
      columns are copied and the 64 CNN-latent columns are zeroed, so step 0
      behaves exactly like the blind gait and PPO learns to use depth.

    ``PART2_PLUM_CRITIC_INIT`` optionally loads a ``critic_state_dict`` (e.g.
    the teacher's) so PPO starts with a calibrated value function.
    """

    def _warm_scan_dim(self) -> int:
        return CNN_LATENT_DIM

    def _warm_skip_prefixes(self) -> tuple[str, ...]:
        # The blind policy has no CNN; keep its tensors at their fresh init.
        return ("cnns.",)

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        # The base class only applies an initialisation when
        # PART2_PLUM_WARM_START is set; the depth task must also honour
        # PART2_PLUM_DEPTH_BC_INIT on its own (a BC-only boot has no blind
        # checkpoint to pass).
        super().__init__(env, train_cfg, log_dir, device)
        if (knobs.optional("PART2_PLUM_DEPTH_BC_INIT")
                and not knobs.optional("PART2_PLUM_WARM_START")):
            self._apply_warm_start(Path("bc"))
        critic_init = knobs.optional("PART2_PLUM_CRITIC_INIT")
        if critic_init:
            import torch
            ckpt = torch.load(Path(critic_init), map_location="cpu", weights_only=False)
            state = ckpt.get("critic_state_dict")
            if state is None:
                raise ValueError(f"{critic_init}: no critic_state_dict")
            missing = self.alg.critic.load_state_dict(state, strict=False).missing_keys
            print(f"[plum/depth] critic initialised from {critic_init} "
                  f"(missing: {missing or 'none'})", flush=True)

    def _apply_warm_start(self, path) -> None:
        """Load BC weights when given; otherwise use the blind warm start."""
        bc = knobs.optional("PART2_PLUM_DEPTH_BC_INIT")
        if bc:
            import torch
            ckpt = torch.load(Path(bc), map_location="cpu", weights_only=False)
            state = ckpt.get("actor_state_dict")
            if state is None:
                raise ValueError(f"{bc}: no actor_state_dict")
            self.alg.actor.load_state_dict(state, strict=True)
            # BC fit the mean only (MSE on teacher actions); rsl_rl leaves
            # `distribution.std_param` at its 1.0 default, so a strict load
            # imports std 1.0 and every rollout sample is dominated by noise.
            # Reset it to the teacher recipe's init_std (0.10) so the first
            # updates fine-tune the BC behaviour instead of abandoning it.
            std = knobs.get_float("PART2_PLUM_INIT_STD", 0.10)
            with torch.no_grad():
                self.alg.actor.distribution.std_param.fill_(std)
            print(f"[plum/depth] actor initialised from BC checkpoint {bc} "
                  f"(action std reset -> {std:g})", flush=True)
            self._warm_start_report = {"source": str(bc), "kind": "bc"}
            return
        super()._apply_warm_start(path)
