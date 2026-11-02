"""HIM task-side integration for the blackW MjLab tasks.

The HIM algorithm itself is the validated N-W-wolf MjLab implementation; this
module only adds the blackW observation groups and terminal recorder.  The actor
history contract remains 57 x 6, identical to the official blackW deployment
policy.
"""
from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import TerminationTermCfg

from alldog_mjlab.tasks.velocity.black import him as black_him

from alldog_mjlab.tasks.velocity.profiles.part2.predicates import _nonfinite_state

from .env_cfgs import (
    blackw_competition_env_cfg,
    blackw_flat_env_cfg,
    blackw_official_mixed_env_cfg,
    blackw_platform_easy_env_cfg,
    blackw_platform_exact_env_cfg,
    blackw_plum_env_cfg,
    blackw_stairs_exact_env_cfg,
    blackw_trench_exact_env_cfg,
)


def _as_him_cfg(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    # reset NaN/Inf envs before the next mjwarp.step turns them into an assert.
    cfg.terminations["nonfinite_state"] = TerminationTermCfg(func=_nonfinite_state)
    # Dump qpos/qvel on NaN/Inf so simulator issues leave a forensic trace.
    cfg.sim.nan_guard.enabled = True
    cfg.sim.nan_guard.output_dir = "/tmp/mjlab/nan_dumps"
    # Official blackW HIM action contract uses signed wheel scale: FL/RL +10,
    # FR/RR -10. The PPO/comp policies used unsigned +10 as a separate contract.
    signs = {"FL": 1.0, "FR": -1.0, "RL": 1.0, "RR": -1.0}
    for term_name, term in list(cfg.actions.items()):
        leg = term_name.split("_")[-1].upper()
        scale = dict(term.scale)
        scale[f"{leg}_wheel_joint"] = 10.0 * signs[leg]
        cfg.actions[term_name] = replace(term, scale=scale)
    actor = cfg.observations["actor"]
    # BlackW PPO path stores flattened 6*57 history; HIM needs [B, 6, 57].
    actor.history_length = None
    actor.flatten_history_dim = False
    black_him.configure_him_observations(cfg)
    black_him.configure_him_terminal_targets(cfg)
    return cfg


def blackw_flat_him_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _as_him_cfg(blackw_flat_env_cfg(play=play))


def blackw_competition_him_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _as_him_cfg(blackw_competition_env_cfg(play=play))


def blackw_platform_exact_him_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _as_him_cfg(blackw_platform_exact_env_cfg(play=play))


def blackw_official_him_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _as_him_cfg(blackw_official_mixed_env_cfg(play=play))


def blackw_platform_easy_him_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _as_him_cfg(blackw_platform_easy_env_cfg(play=play))


def blackw_stairs_exact_him_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Exact official stairs task with the HIM observation contract."""
    return _as_him_cfg(blackw_stairs_exact_env_cfg(play=play))


def blackw_trench_exact_him_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Exact official trench task with the HIM observation contract."""
    return _as_him_cfg(blackw_trench_exact_env_cfg(play=play))


def blackw_plum_him_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Part-2 tier-1 plum field with the HIM observation contract."""
    return _as_him_cfg(blackw_plum_env_cfg(play=play))
