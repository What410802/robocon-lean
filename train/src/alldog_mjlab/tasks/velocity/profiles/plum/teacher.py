"""Privileged PPO teacher for the plum field (EP recipe + warm start + curriculum).

This is the third generation of the plum teacher.  The first two scored 0/178 and
0/4000 in-sim; the diagnosis is in ``docs`` (and in the module docstrings of
``ep_rewards.py`` / ``terrain.py``):

* the reward was support-blind -- ``progress`` paid for forward displacement of
  the lowest wheel, so vaulting the field scored the same as stepping on it, and
  v1 learned exactly that (trunk z 0.88 m, wheels at 1.17 m, then out of course);
* the run started from scratch, while the old repo's only successful plum teacher
  warm-started from the blind ``row2p_p40w40_v1`` policy;
* the command was 0.35 m/s, slow enough that the robot stalls on cube rims with a
  0.4 m gap to cross.  The old run used 0.65-0.80 m/s;
* the fixed field gave PPO no curriculum, so the 0.2 m entry step and the
  checkerboard had to be learned in one jump.

What this builder does, in order:

1. ``blackw_plum_env_cfg`` for the judge-mirrored predicates/terminations;
2. the Extreme-Parkour reward table (``ep_rewards.py``) + a sparse success bonus;
3. a fixed forward command used only by the reward and the gates -- the actor
   observation drops the command entirely (54 x 6 = 324-D);
4. ``PlumCurriculumTerrainCfg``: cubes shrink from 0.30 m half-width (a
   contiguous plateau) to the official 0.10 m checkerboard;
5. ``plum_difficulty_curriculum``: promote on repeated judge-mirrored success;
6. judge physics (euler/elliptic/100 iterations/impratio=100) and spawn z=0.55;
7. strict contact: any non-wheel body touching terrain terminates;
8. the privileged per-wheel fine scan as a second actor observation group;
9. joint-command clipping aligned with the deployment ``command_dq_max``.

The blind->teacher warm start lives in the runner (``PLUM_WARM_START``), because
it needs the actor module after rsl_rl built it.
"""

from __future__ import annotations

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from alldog_mjlab.tasks.velocity.blackw.actions import WheelLegActionCfg
from alldog_mjlab.tasks.velocity.blackw.env_cfgs import (
    _apply_deployment_sim_options, blackw_plum_env_cfg,
)
from alldog_mjlab.tasks.velocity.common.config import BLACK_CONFIG
from alldog_mjlab.tasks.velocity.common.rl_cfg import black_ppo_runner_cfg
from alldog_mjlab.tasks.velocity.common.velocity_env import BLACK_ILLEGAL_CONTACT_SENSOR
from alldog_mjlab.tasks.velocity.profiles.part2.predicates import _part2_failure

from .curriculum import plum_difficulty_curriculum
from .ep_rewards import build_reward_table
from .fine_scan import FINE_SCAN_GROUP, add_fine_scan, fine_scan_dim
from .spec import SPEC
from .terrain import PlumCurriculumTerrainCfg

TASK_ID = "blackw-plum-teacher"

# Fixed-task contract: code constants, so a run is reproducible from the task id.
# The old successful run used 0.65-0.80 m/s; a slow command cannot cross a 0.4 m
# gap with a 0.103 m wheel.
COMMAND_VX = (0.65, 0.80)
SUCCESS_BONUS = 500.0

# Curriculum rows: difficulty 0, .2, .4, .6, .8, 1.0 -> cube half 0.30 -> 0.10 m.
NUM_ROWS = 6

# Spawn: ON the entry platform (deployment hands over to this policy only after
# the ordinary locomotion policy has climbed).  Centre of the 0.8 m platform,
# plus the platform height under the usual stand height.
PLATFORM_SPAWN_X = 0.5 * (SPEC.entry_x_range[0] + SPEC.entry_x_range[1])   # 2.2
PLATFORM_SPAWN_Z = 0.55 + SPEC.platform_top                              # 0.75

# Spawn envelope on the 0.8 x 0.8 m platform (x 1.8-2.6, |y| <= 0.4).
# Widened after the v8/v9 runs: with +/-0.05 m the warm start only ever saw a
# near-centred pose, so the deployment handover (arriving at the platform from
# an arbitrary climb) was out of distribution.  Bounds are geometric, not
# arbitrary: the front wheels sit +0.214 m ahead of the root, so dx <= 0.15
# keeps them on the platform (2.2 + 0.15 + 0.214 < 2.6), and dy <= 0.18 keeps
# the outer wheel inside |y| <= 0.4.  Handover velocity is randomised separately
# by the reset event's root_velocity range (+/-0.5 m/s and rad/s).
SPAWN = {
    0: dict(x=0.10, y=0.12, yaw_deg=8.0, joint=0.06),
    1: dict(x=0.15, y=0.18, yaw_deg=20.0, joint=0.08),
    # Stage 2: the judge's spawn -- on the GROUND 2.2 m before the entry platform
    # (scene x=0, base z=0.55): the 1.8 m runway plus the 0.2 m platform face.
    # Every platform-spawned policy is out of distribution here.  Measured
    # 2026-11-07: the privileged teacher scores 0 % (743/774 episodes end in
    # illegal contact at rx 1.87, the platform face), and the 79.4 % depth actor
    # arrives at the first cube row misaligned and is judged dropped at x=2.55.
    2: dict(x=0.10, y=0.10, yaw_deg=5.0, joint=0.06, dx=-2.20, dz=-0.20),
}

# Deployment obs layout without the command term (frame-major, 54 dims/frame).
NO_COMMAND_FRAME_DIM = 54
NO_COMMAND_DEPLOY_LAYOUT = ((3, 0), (3, 3), (16, 6), (16, 22), (16, 38))
ACTOR_HISTORY = 6
PROPRIO_DIM = NO_COMMAND_FRAME_DIM * ACTOR_HISTORY   # 324

# Joint-command clip, aligned with demo/rl_sar/policy/blackW/base.yaml.
WHEEL_DQ_CLIP = 20.0
LEG_RESIDUAL_CLIP = 1.0
_JOINT_CLIP = {
    ".*_(hip|thigh|calf)_joint": (-LEG_RESIDUAL_CLIP, LEG_RESIDUAL_CLIP),
    ".*wheel_joint": (-WHEEL_DQ_CLIP, WHEEL_DQ_CLIP),
}


def _drop_command_observation(cfg: ManagerBasedRlEnvCfg) -> None:
    terms = dict(cfg.observations["actor"].terms)
    if terms.pop("command", None) is None:
        raise ValueError("expected a 'command' actor observation term")
    cfg.observations["actor"].terms = terms
    print(f"[plum/teacher] actor obs: command dropped -> {tuple(terms)}", flush=True)


def _fix_command(cfg: ManagerBasedRlEnvCfg) -> None:
    """Straight fixed command; consumed by the rewards, never by the actor."""
    cmd = cfg.commands["twist"]
    if not isinstance(cmd, UniformVelocityCommandCfg):
        raise TypeError(f"unexpected command type {type(cmd).__name__}")
    cmd.ranges.lin_vel_x = COMMAND_VX
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (0.0, 0.0)
    cmd.rel_standing_envs = 0.0
    cmd.rel_forward_envs = 1.0
    print(f"[plum/teacher] fixed command vx={COMMAND_VX} (not fed to the actor)",
          flush=True)


def _terrain_curriculum(cfg: ManagerBasedRlEnvCfg) -> None:
    """Official-footprint terrain graded by barrier height and end holes."""
    assert cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator
    gen = cfg.scene.terrain.terrain_generator
    terrain = PlumCurriculumTerrainCfg()
    gen.sub_terrains = {"plum": terrain}
    gen.num_rows = NUM_ROWS
    gen.difficulty_range = (0.0, 1.0)
    # Specialist runs pin one row (e.g. PART2_DIFF_MIN=PART2_DIFF_MAX=1.0 trains
    # the official 0.40 m field directly).  This builder ran AFTER env_cfgs
    # applied the knobs, so without the override here the pin was silently lost
    # and the depth PPO run trained row 0 for 1200 iterations.
    from alldog_mjlab.utils import knobs
    dmin = knobs.get_float("PART2_DIFF_MIN", 0.0)
    dmax = knobs.get_float("PART2_DIFF_MAX", 1.0)
    if (dmin, dmax) != (0.0, 1.0):
        gen.difficulty_range = (dmin, dmax)
        print(f"[plum/teacher] difficulty pinned to [{dmin:g}, {dmax:g}] "
              "(PART2_DIFF_MIN/MAX)", flush=True)
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.curriculum = {
        "plum_difficulty": CurriculumTermCfg(func=plum_difficulty_curriculum),
    }
    ladder = ", ".join(f"{terrain.residual_at(i / (NUM_ROWS - 1)):.2f}"
                       for i in range(NUM_ROWS))
    print(f"[plum/teacher] curriculum: {NUM_ROWS} rows, open gap "
          f"[{ladder}] m (row 0 closed, then just over the wheel diameter -> "
          f"official 0.40), OFFICIAL footprint and height (0.20 m) at every row "
          "(row-level promotion on a 40-episode window)", flush=True)


def _spawn_randomisation(cfg: ManagerBasedRlEnvCfg, stage: int) -> None:
    s = SPAWN[stage]
    events = dict(cfg.events)
    reset_base = events["reset_base"]
    params = dict(reset_base.params)
    pose = dict(params.get("pose_range") or {})
    dx = float(s.get("dx", 0.0))
    dz = float(s.get("dz", 0.0))
    pose.update({
        "x": (dx - s["x"], dx + s["x"]),
        "y": (-s["y"], s["y"]),
        "yaw": (-math.radians(s["yaw_deg"]), math.radians(s["yaw_deg"])),
    })
    if dz:
        pose["z"] = (dz, dz)
    params["pose_range"] = pose
    events["reset_base"] = replace(reset_base, params=params)
    joints = events["reset_robot_joints"]
    events["reset_robot_joints"] = replace(
        joints, params={**dict(joints.params),
                        "position_range": (-s["joint"], s["joint"])})
    cfg.events = events
    where = "on the ground runway" if (dx or dz) else "on the entry platform"
    print(f"[plum/teacher] stage {stage} spawn: {where}, centre offset "
          f"dx={dx:+.2f} dz={dz:+.2f} m, jitter +/-{s['x']:.2f}/+/-{s['y']:.2f} m, "
          f"dyaw=+/-{s['yaw_deg']} deg, joint jitter=+/-{s['joint']} rad", flush=True)


def _clip_joint_commands(cfg: ManagerBasedRlEnvCfg) -> None:
    """Per-task clip on the SCALED joint command (mjlab BaseActionCfg.clip)."""
    n = 0
    for name, term in list(cfg.actions.items()):
        if isinstance(term, WheelLegActionCfg):
            cfg.actions[name] = replace(term, clip=_JOINT_CLIP)
            n += 1
    if n != 4:
        raise ValueError(f"expected 4 WheelLegAction terms, found {n}")
    print(f"[plum/teacher] joint-command clip: wheel +/-{WHEEL_DQ_CLIP} rad/s, "
          f"leg +/-{LEG_RESIDUAL_CLIP} rad", flush=True)


def _judge_physics(cfg: ManagerBasedRlEnvCfg) -> None:
    """Official judge solver options + spawn ON the entry platform."""
    _apply_deployment_sim_options(cfg, force=True)
    robot = cfg.scene.entities.get("robot")
    if robot is not None and getattr(robot, "init_state", None) is not None:
        robot.init_state.pos = (PLATFORM_SPAWN_X, 0.0, PLATFORM_SPAWN_Z)
    print(f"[plum/teacher] judge physics: euler/elliptic/newton, iter=100, "
          f"impratio=100, spawn on the entry platform "
          f"(x={PLATFORM_SPAWN_X:.2f}, z={PLATFORM_SPAWN_Z:.2f})", flush=True)


def _strict_contact(cfg: ManagerBasedRlEnvCfg) -> None:
    """Teacher-only: ANY non-wheel body touching terrain ends the attempt.

    The judge never checks body/leg contact, so without this the teacher can
    learn to brace its trunk/hips on the cubes.  The rule is kept deliberately
    strict (user decision 2026-11-05): the old repo's successful teacher also ran
    under it.
    """
    assert cfg.scene is not None
    sensors = list(cfg.scene.sensors or ())
    found = False
    for i, sensor in enumerate(sensors):
        if sensor.name == BLACK_ILLEGAL_CONTACT_SENSOR:
            if not isinstance(sensor, ContactSensorCfg):
                raise TypeError(f"{sensor.name!r} is not a ContactSensorCfg")
            sensors[i] = replace(
                sensor,
                primary=ContactMatch(mode="body", pattern=".*", entity="robot",
                                     exclude=(r".*_foot",)),
                num_slots=2,
            )
            found = True
    if not found:
        raise ValueError(f"contact sensor {BLACK_ILLEGAL_CONTACT_SENSOR!r} not found")
    cfg.scene.sensors = tuple(sensors)
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=velocity_mdp.illegal_contact,
        params={"sensor_name": BLACK_ILLEGAL_CONTACT_SENSOR,
                "force_threshold": 1.0},
    )
    print("[plum/teacher] strict contact: all non-wheel bodies vs terrain "
          "(force > 1 N) terminate", flush=True)


def plum_teacher_env_cfg(play: bool = False, stage: int = 0) -> ManagerBasedRlEnvCfg:
    if stage not in (0, 1, 2):
        raise ValueError(f"unknown course stage {stage} (expected 0, 1 or 2)")
    cfg = blackw_plum_env_cfg(play=play)
    _drop_command_observation(cfg)
    _fix_command(cfg)
    _spawn_randomisation(cfg, stage)
    _clip_joint_commands(cfg)
    _judge_physics(cfg)
    # The masked failure term gates on the command; with a fixed forward command
    # use the plain judge-mirrored predicate instead (no hidden dependency).
    cfg.terminations["part2_failure"] = TerminationTermCfg(
        func=_part2_failure, params={"asset_cfg": SceneEntityCfg("robot")})
    _strict_contact(cfg)
    if not play:
        # Replace (not update) the Part-2 table: the masked-progress terms are
        # exactly the support-blind shaping this recipe removes.
        cfg.rewards = build_reward_table(clear_weight=SUCCESS_BONUS)
        _terrain_curriculum(cfg)
    add_fine_scan(cfg)
    return cfg


def plum_teacher_runner_cfg(stage: int = 0):
    """Standard blackW PPO; actor = (324-D proprio history, fine scan)."""
    from alldog_mjlab.utils import knobs

    cfg = black_ppo_runner_cfg(stage="rough")
    # A warm-started policy already knows how to walk; its own action std is ~0.5.
    # At the generic init_std=1.0 the sampled action is dominated by noise, the
    # behaviour being imitated never appears in the rollout, and PPO moves the
    # policy away from it (measured: tracking_goal_vel goes negative and
    # bad_orientation spikes).  The old successful teacher started at 0.101.
    init_std = knobs.get_float("PART2_PLUM_INIT_STD", 0.10)
    dist = {**cfg.actor.distribution_cfg, "init_std": init_std}
    # Optional clamp on the LEARNED std.  The entropy bonus raises it monotonically
    # at a fixed LR (0.10 -> 0.47 over 1000 iterations at 3e-5), and the
    # official-field success window closed as it crossed ~0.37 (3.5% at iter 600,
    # 0% from iter 725 on).  rsl_rl's default range is (1e-6, 1e6), i.e. no clamp.
    std_max = knobs.get_float("PART2_PLUM_STD_MAX", 0.0)
    if std_max > 0.0:
        dist["std_range"] = (1e-6, std_max)
    cfg.actor.distribution_cfg = dist
    print(f"[plum/teacher] actor init_std = {init_std:g}"
          + (f", std clamp <= {std_max:g}" if std_max > 0.0 else ""), flush=True)
    cfg.obs_groups = {"actor": ("actor", FINE_SCAN_GROUP), "critic": ("critic",)}
    cfg.experiment_name = "blackw_plum_teacher"
    cfg.run_name = f"plum_v3_s{stage}"
    cfg.clip_actions = BLACK_CONFIG.control.action_clip
    cfg.logger = "tensorboard"
    print(f"[plum/teacher] stage {stage}: actor input = {PROPRIO_DIM} + "
          f"{fine_scan_dim()} = {PROPRIO_DIM + fine_scan_dim()}-D", flush=True)
    return cfg
