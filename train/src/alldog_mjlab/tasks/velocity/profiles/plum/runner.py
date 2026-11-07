"""Plum teacher runner: PPO + blind warm start + EP ``only_positive`` clipping.

Everything else is inherited from ``BlackVelocityOnPolicyRunner`` (reward clamp,
return clamp, curriculum checkpoint state, resume-lr override).  This class only
adds the two recipe pieces that need the actor to exist:

* ``PART2_PLUM_WARM_START`` -- path to a blind actor (a deployed ``policy.pt`` or a
  training checkpoint).  ``None`` = train from scratch.  This is an
  *initialization*, not a resume: optimizer, iteration counter and env state are
  fresh.  ``--agent.resume`` is still the way to continue a plum run.
* ``PART2_PLUM_ONLY_POSITIVE`` -- EP's ``only_positive_rewards`` clip (default on).

Both are environment variables rather than CLI flags because mjlab's
``train`` entry point builds the runner from a fixed dataclass; the repo already
uses this convention for ``RC_RESUME_LR`` and friends.
"""

from __future__ import annotations

from pathlib import Path

import torch

from alldog_mjlab.algorithms.privileged_warm_start import warm_start_teacher_from_blind
from alldog_mjlab.tasks.velocity.common.curriculum_checkpoint import (
    BlackVelocityOnPolicyRunner,
)
from alldog_mjlab.utils import knobs
from alldog_mjlab.utils.reward_safety import install_only_positive_rewards

from .teacher import ACTOR_HISTORY, COMMAND_VX
from .fine_scan import fine_scan_dim


def _load_blind_state(path: Path) -> dict[str, torch.Tensor]:
    """Actor state from a TorchScript policy.pt or a training checkpoint."""
    if path.suffix in (".pt", ".pth"):
        try:
            module = torch.jit.load(str(path), map_location="cpu")
            return dict(module.state_dict())
        except RuntimeError:
            pass
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        if "actor_state_dict" not in ckpt:
            raise ValueError(
                f"PART2_PLUM_WARM_START={path} is neither a TorchScript policy nor a "
                f"checkpoint with actor_state_dict (keys: {sorted(ckpt)[:6]})")
        return dict(ckpt["actor_state_dict"])
    raise ValueError(f"PART2_PLUM_WARM_START must be a .pt file, got {path}")


class PlumTeacherRunner(BlackVelocityOnPolicyRunner):
    """PPO runner with the plum teacher's warm start and reward clip."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        install_only_positive_rewards(
            self.env.unwrapped, knobs.get_bool("PART2_PLUM_ONLY_POSITIVE", True))
        warm = knobs.optional("PART2_PLUM_WARM_START")
        if warm:
            self._apply_warm_start(Path(warm))

    def _warm_scan_dim(self) -> int:
        """Privileged input width reserved after the proprio columns."""
        return fine_scan_dim()

    def _warm_skip_prefixes(self) -> tuple[str, ...]:
        """Parameter prefixes the blind warm start must leave untouched."""
        return ()

    def _apply_warm_start(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"PART2_PLUM_WARM_START not found: {path}")
        state = _load_blind_state(path)
        vx = 0.5 * (COMMAND_VX[0] + COMMAND_VX[1])
        report = warm_start_teacher_from_blind(
            self.alg.actor,
            state,
            history=ACTOR_HISTORY,
            scan_dim=self._warm_scan_dim(),
            fixed_command=(vx, 0.0, 0.0),
            skip_prefixes=self._warm_skip_prefixes(),
        )
        print(f"[plum/warm-start] source={path} "
              f"(step-0 = blind policy at a fixed vx={vx:.2f} m/s)", flush=True)
        self._warm_start_report = report
        warmup = knobs.get_int("PART2_PLUM_ACTOR_WARMUP_ITERS", 0)
        if warmup > 0:
            self._actor_warmup_iters = warmup
            self._original_update = self.alg.update
            self._set_actor_trainable(False)
            print(f"[plum/warm-start] actor frozen for {warmup} iterations "
                  "(critic warms up on the warm-started policy's own data; "
                  "a random critic's noisy advantages otherwise destroy the "
                  "warm start in the first PPO update)", flush=True)

    def _set_actor_trainable(self, trainable: bool) -> None:
        for p in self.alg.actor.parameters():
            p.requires_grad_(trainable)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Learn, unfreezing the actor after the critic warm-up window."""
        warmup = getattr(self, "_actor_warmup_iters", 0)
        if warmup <= 0:
            return super().learn(num_learning_iterations, init_at_random_ep_len)
        start = int(self.current_learning_iteration)
        # Unfreeze just before the first iteration whose UPDATE should be applied.
        self._set_actor_trainable(start >= warmup)
        if start < warmup:
            self.alg.update = self._critic_only_update
        print(f"[plum/warm-start] learn() from iter {start}: actor "
              f"{'trainable' if start >= warmup else 'frozen'}", flush=True)
        return super().learn(num_learning_iterations, init_at_random_ep_len)

    def _critic_only_update(self):
        """PPO update with the actor's parameters held fixed, until the window ends.

        The gate must UNINSTALL itself: the previous version re-enabled
        ``requires_grad`` in ``finally`` but left ``self.alg.update`` bound to
        this method, which starts every call by freezing the actor again.  v8
        then ran 2000 iterations with bit-identical actor weights
        (``model_0`` and ``model_1999`` differ by 0), while the critic moved
        normally -- so the run produced no policy at all.
        """
        if int(self.current_learning_iteration) >= self._actor_warmup_iters:
            self.alg.update = self._original_update
            self._set_actor_trainable(True)
            return self.alg.update()
        self._set_actor_trainable(False)
        return self._original_update()
