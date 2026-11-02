"""Training-side reward safety net for anomalous physics states.

MjLab's RewardManager already maps NaN/Inf to zero after weighting each term, but
large *finite* squared penalties (e.g. a solver blow-up producing huge joint
velocities) can still dominate PPO returns and diverge the value function.  The
clip here is deliberately far above normal reward magnitudes; it is a guard rail,
not a reward-shaping change.
"""

from __future__ import annotations

import torch


def install_reward_clamp(env, clip: float) -> None:
    """Clamp the per-step total reward seen by PPO and report clipping activity."""
    manager = env.reward_manager
    if getattr(manager, "_black_reward_clamp_installed", False):
        return
    if clip <= 0:
        print("[INFO] reward clamp disabled (clip <= 0)")
        return

    original_compute = manager.compute
    manager._black_reward_clamp_steps = 0
    manager._black_reward_clamp_count = 0
    manager._black_reward_clamp_max_abs = 0.0

    def compute(dt: float) -> torch.Tensor:
        reward = original_compute(dt)
        reward = torch.nan_to_num(reward, nan=0.0, posinf=clip, neginf=-clip)
        clipped = torch.clamp(reward, -clip, clip)
        clipped_mask = clipped != reward
        manager._black_reward_clamp_steps += 1
        manager._black_reward_clamp_count += int(clipped_mask.sum().item())
        manager._black_reward_clamp_max_abs = max(
            manager._black_reward_clamp_max_abs, float(reward.abs().max().item())
        )
        if manager._black_reward_clamp_steps % 500 == 0:
            print(
                "[INFO] reward clamp: "
                f"steps={manager._black_reward_clamp_steps}, "
                f"clipped_env_steps={manager._black_reward_clamp_count}, "
                f"max_abs_before_clip={manager._black_reward_clamp_max_abs:.3g}, "
                f"clip={clip:.3g}"
            )
        return clipped

    manager.compute = compute
    manager._black_reward_clamp_installed = True
    manager._black_reward_clip = clip
    print(f"[INFO] reward clamp installed: per-step total reward clipped to +/-{clip:.3g}")


def install_return_clamp(alg, max_abs_return: float = 1000.0, max_abs_advantage: float = 100.0) -> None:
    """Clamp GAE returns/advantages before the PPO update.

    This is the second guard after reward clipping: a single anomalous value can
    still make a return large enough to produce catastrophic value gradients.  The
    bounds are far above normal locomotion returns; clipping is logged and should
    be rare.
    """
    if getattr(alg, "_black_return_clamp_installed", False):
        return
    original_compute_returns = alg.compute_returns
    stats = {"calls": 0, "clipped": 0, "nonfinite": 0}
    alg._black_return_clamp_stats = stats

    def compute_returns(obs):
        original_compute_returns(obs)
        storage = alg.storage
        for name, limit in (("returns", max_abs_return), ("advantages", max_abs_advantage)):
            value = getattr(storage, name, None)
            if value is None:
                continue
            nonfinite = ~torch.isfinite(value)
            if nonfinite.any():
                stats["nonfinite"] += int(nonfinite.sum().item())
                value[nonfinite] = 0.0
            before = value.abs() > limit
            if before.any():
                stats["clipped"] += int(before.sum().item())
                value.clamp_(-limit, limit)
        stats["calls"] += 1
        if stats["calls"] % 50 == 0:
            print(
                "[INFO] return clamp: "
                f"calls={stats['calls']}, clipped_values={stats['clipped']}, "
                f"nonfinite_values={stats['nonfinite']}"
            )

    alg.compute_returns = compute_returns
    alg._black_return_clamp_installed = True
    print(
        "[INFO] return clamp installed: "
        f"returns +/-{max_abs_return:g}, advantages +/-{max_abs_advantage:g}"
    )
