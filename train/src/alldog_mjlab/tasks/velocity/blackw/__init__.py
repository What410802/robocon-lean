"""Register the blackW wheel-legged velocity tasks (official Part 1/2 obstacles).

Core (blind) PPO tasks plus the HIM variants are registered here; the research
branches (depth camera / privileged teacher / foothold / EP) were pruned in the
intra-squad repo migration.  The task/env/reward implementations live in
``env_cfgs.py`` / ``him.py``; only registration order and task ids are here.
"""

from mjlab.tasks.registry import register_mjlab_task

from alldog_mjlab.tasks.velocity.common.curriculum_checkpoint import BlackVelocityOnPolicyRunner
from alldog_mjlab.tasks.velocity.common.rl_cfg import black_ppo_runner_cfg

from .env_cfgs import (
    blackw_flat_env_cfg,
    blackw_vt_edge_env_cfg,
    blackw_vt_env_cfg,
    blackw_vt_locked_env_cfg,
    blackw_official_mixed_env_cfg,
    blackw_platform_easy_env_cfg,
    blackw_platform_exact_env_cfg,
    blackw_plum_env_cfg,
    blackw_slope_exact_env_cfg,
    blackw_stairs_exact_env_cfg,
    blackw_trench_exact_env_cfg,
    blackw_wall_exact_env_cfg,
)

register_mjlab_task(
    task_id="blackw-official",
    env_cfg=blackw_official_mixed_env_cfg(),
    play_env_cfg=blackw_official_mixed_env_cfg(play=True),
    rl_cfg=black_ppo_runner_cfg(stage="rough"),
    runner_cls=BlackVelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="blackw-plum",
    env_cfg=blackw_plum_env_cfg(),
    play_env_cfg=blackw_plum_env_cfg(play=True),
    rl_cfg=black_ppo_runner_cfg(stage="rough"),
    runner_cls=BlackVelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="blackw-flat",
    env_cfg=blackw_flat_env_cfg(),
    play_env_cfg=blackw_flat_env_cfg(play=True),
    rl_cfg=black_ppo_runner_cfg(stage="flat"),
    runner_cls=BlackVelocityOnPolicyRunner,
)

# Velocity-tracking primitive for the Part 1 position-loop handover (flat), plus
# the platform-edge variant used to make it edge-robust (bridge platforms).
register_mjlab_task(
    task_id="blackw-vt",
    env_cfg=blackw_vt_env_cfg(),
    play_env_cfg=blackw_vt_env_cfg(play=True),
    rl_cfg=black_ppo_runner_cfg(stage="flat"),
    runner_cls=BlackVelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="blackw-vt-locked",
    env_cfg=blackw_vt_locked_env_cfg(),
    play_env_cfg=blackw_vt_locked_env_cfg(play=True),
    rl_cfg=black_ppo_runner_cfg(stage="flat"),
    runner_cls=BlackVelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="blackw-vt-edge",
    env_cfg=blackw_vt_edge_env_cfg(),
    play_env_cfg=blackw_vt_edge_env_cfg(play=True),
    rl_cfg=black_ppo_runner_cfg(stage="flat"),
    runner_cls=BlackVelocityOnPolicyRunner,
)

# Single-obstacle exact-terrain tasks (official map geometry from
# ``terrain/official/part2_terrain_spec.json``; plum from ``terrain/plum/``).
for _task_id, _builder in (
    ("blackw-stairs-exact", blackw_stairs_exact_env_cfg),
    ("blackw-platform-exact", blackw_platform_exact_env_cfg),
    ("blackw-platform-easy", blackw_platform_easy_env_cfg),
    ("blackw-slope-exact", blackw_slope_exact_env_cfg),
    ("blackw-trench-exact", blackw_trench_exact_env_cfg),
    ("blackw-wall-exact", blackw_wall_exact_env_cfg),
):
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=_builder(),
        play_env_cfg=_builder(play=True),
        rl_cfg=black_ppo_runner_cfg(stage="rough"),
        runner_cls=BlackVelocityOnPolicyRunner,
    )

# HIM tasks: actor history 6x57 + estimator velocity + terminal recorder; optional
# official TorchScript warm start via RC_HIM_OFFICIAL_WARMSTART.
from .him import (
    blackw_flat_him_env_cfg,
    blackw_official_him_env_cfg,
    blackw_platform_easy_him_env_cfg,
    blackw_platform_exact_him_env_cfg,
)
from .him_rl_cfg import blackw_him_runner_cfg
from .him_runner import BlackWHimOfficialRunner

for _task_id, _builder in (
    ("blackw-flat-him", blackw_flat_him_env_cfg),
    ("blackw-platform-exact-him", blackw_platform_exact_him_env_cfg),
    ("blackw-platform-easy-him", blackw_platform_easy_him_env_cfg),
    ("blackw-official-him", blackw_official_him_env_cfg),
):
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=_builder(),
        play_env_cfg=_builder(play=True),
        rl_cfg=blackw_him_runner_cfg(),
        runner_cls=BlackWHimOfficialRunner,
    )
