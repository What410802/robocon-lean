"""Part 1 competition task builder (terrain from ``profiles/part1/terrain.py``)."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from alldog_mjlab.tasks.velocity.common.config import BLACK_CONFIG
from alldog_mjlab.tasks.velocity.common.course import apply_traversal_base
from alldog_mjlab.tasks.velocity.common.velocity_env import BLACK_COMMAND_NAME
from alldog_mjlab.tasks.velocity.common.velocity_env import build_velocity_env_cfg
from alldog_mjlab.tasks.velocity.profiles.part1.terrain import (
    black_competition_terrain_generator_cfg,
)

def black_competition_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Competition-proxy terrain: flat/slope/stairs + Part1/Part2 obstacles.

    Uses the same actor/reward/termination/command/DR contract as
    ``black-basic-obstacle`` (45-D blind actor); only the terrain generator is
    replaced by the obstacle-proxy curriculum in ``competition_terrain.py``.
    """
    cfg = build_velocity_env_cfg(play=play, rough=True)
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "generator"
    generator = black_competition_terrain_generator_cfg()
    cfg.scene.terrain.terrain_generator = generator
    # Train starts at easy rows and climbs the curriculum; play/rollout samples
    # the full difficulty range so preview videos reach the obstacle rows.
    # Distance-based terrain curriculum is incompatible with goal-terminated
    # obstacle episodes (early success shows as "not enough walked distance" and
    # four course columns stayed at level 0 in the stage-3 run).  Fix the
    # difficulty distribution instead: training samples rows 0..4 (d=0.0..0.8),
    # play samples all rows.
    cfg.scene.terrain.max_init_terrain_level = (
        generator.num_rows - 1 if play else min(3, generator.num_rows - 1)
    )
    apply_traversal_base(cfg, play=play)
    return cfg


def black_competition_hist_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Competition terrain + 6-frame actor history (270-D blind actor).

    Keeps the same action/reward/termination contract; only the actor observation
    group gets MjLab native history (6 x 45 = 270, flattened).  The deployed
    contract for 270-D history policies already exists (HIM-style rl_sar cfg), so
    this is an A/B against the single-frame 45-D actor.
    """
    cfg = black_competition_env_cfg(play=play)
    cfg.observations["actor"].history_length = 6
    cfg.observations["actor"].flatten_history_dim = True
    return cfg
