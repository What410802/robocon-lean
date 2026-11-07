"""Register the blackW wheel-legged velocity tasks (official Part 1/2 obstacles).

Core (blind) PPO tasks plus the HIM variants are registered here; the research
branches (depth camera / privileged teacher / foothold / EP) were pruned in the
intra-squad repo migration.  The task/env/reward implementations live in
``env_cfgs.py`` / ``him.py``; only registration order and task ids are here.
"""

from dataclasses import replace as _dc_replace

from mjlab.tasks.registry import register_mjlab_task

from alldog_mjlab.tasks.velocity.common.curriculum_checkpoint import BlackVelocityOnPolicyRunner
from alldog_mjlab.tasks.velocity.common.rl_cfg import black_ppo_runner_cfg


def _locked_ppo_runner_cfg():
    """PPO runner config tuned for the 12-DoF locked-wheel gait.

    The default Black recipe starts with Gaussian action std 1.0.  On the
    locked-wheel robot that means +/-0.25 rad leg targets from the first step,
    which knocks the robot over within ~13 control steps (mean episode length
    0.27 s in the first S0 run), so PPO never collects a stand/gait transition.
    A smaller initial std lets stand/lock be learned; the std is learnable and
    grows again once the reset distribution is survivable.
    """
    cfg = black_ppo_runner_cfg(stage="flat")
    cfg.actor = _dc_replace(
        cfg.actor,
        distribution_cfg={**cfg.actor.distribution_cfg, "init_std": 0.2},
    )
    cfg.algorithm = _dc_replace(cfg.algorithm, entropy_coef=0.003)
    return cfg

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
    env_cfg=blackw_vt_locked_env_cfg(stage=4),
    play_env_cfg=blackw_vt_locked_env_cfg(play=True, stage=4),
    rl_cfg=_locked_ppo_runner_cfg(),
    runner_cls=BlackVelocityOnPolicyRunner,
)

# Staged locked-wheel curriculum (stand -> straight -> pure yaw -> combined ->
# full).  The final task above is S4; each stage is trainable independently and
# resumes from the previous one.
for _stage in range(5):
    register_mjlab_task(
        task_id=f"blackw-vt-locked-s{_stage}",
        env_cfg=blackw_vt_locked_env_cfg(stage=_stage),
        play_env_cfg=blackw_vt_locked_env_cfg(play=True, stage=_stage),
        rl_cfg=_locked_ppo_runner_cfg(),
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

# Plum (梅花桩): privileged teacher -> depth-CNN student.  Only the teacher is
# registered as a trainable task; the student is produced offline by
# ``train/scripts/plum/``.  See profiles/plum/ for the recipe.
from alldog_mjlab.tasks.velocity.profiles.plum.runner import PlumTeacherRunner
from alldog_mjlab.tasks.velocity.profiles.plum.teacher import (
    plum_teacher_env_cfg,
    plum_teacher_runner_cfg,
)

# Two spawn envelopes on the SAME task: stage 0 narrow (learn the gait at the
# easy end of the shrinking-cube curriculum), stage 1 full (initial-state
# robustness).  Terrain difficulty is advanced by the success-gated curriculum
# inside each stage, so "stage" here means spawn spread only.
for _stage in (0, 1, 2):
    _task = {0: "blackw-plum-teacher", 1: "blackw-plum-teacher-s1",
             2: "blackw-plum-teacher-s2"}[_stage]
    register_mjlab_task(
        task_id=_task,
        env_cfg=plum_teacher_env_cfg(stage=_stage),
        play_env_cfg=plum_teacher_env_cfg(play=True, stage=_stage),
        rl_cfg=plum_teacher_runner_cfg(stage=_stage),
        runner_cls=PlumTeacherRunner,
    )

# Depth-actor variant: the same plum task, but the actor sees ONE 48x64 depth
# frame instead of the privileged fine scan (the critic keeps the scan).  The
# fine-scan branch distils depth into a prediction of a +/-0.10 m patch centred
# under each wheel -- exactly the region a body-mounted camera cannot see --
# so this task trains the actor on depth directly instead.
from alldog_mjlab.tasks.velocity.profiles.plum.depth import (
    PlumDepthRunner,
    plum_depth_env_cfg,
    plum_depth_runner_cfg,
)

for _stage in (0, 1, 2):
    _task = {0: "blackw-plum-depth", 1: "blackw-plum-depth-s1",
             2: "blackw-plum-depth-s2"}[_stage]
    register_mjlab_task(
        task_id=_task,
        env_cfg=plum_depth_env_cfg(stage=_stage),
        play_env_cfg=plum_depth_env_cfg(play=True, stage=_stage),
        rl_cfg=plum_depth_runner_cfg(stage=_stage),
        runner_cls=PlumDepthRunner,
    )
