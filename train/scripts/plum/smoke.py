#!/usr/bin/env python3
"""Contract smoke test for the plum teacher/student pipeline.

Run (train env, no GPU training):
    MUJOCO_GL=egl PYTHONPATH=train/src .pixi/envs/train/bin/python \
        train/scripts/plum/smoke.py

Checks the facts the recipe relies on, against the CURRENT repo's code (not the
old repo's docs): geometry, camera contract parity with the eval judge, obs
dimensions, spawn ranges, reward dt scaling, and export permutation.
"""

from __future__ import annotations

import ast
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "train" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "train" / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from alldog_mjlab.tasks.velocity.common.config import BLACK_CONFIG  # noqa: E402
from alldog_mjlab.tasks.velocity.profiles.plum.depth_camera import (  # noqa: E402
    CNN_LATENT_DIM, DEPTH_CNN_CFG, DEPTH_CONTRACT, camera_depth,
)
from alldog_mjlab.tasks.velocity.profiles.plum.fine_scan import fine_scan_dim  # noqa: E402
from alldog_mjlab.tasks.velocity.profiles.plum.spec import SPEC  # noqa: E402
from alldog_mjlab.tasks.velocity.profiles.plum.student import (  # noqa: E402
    build_permutation, build_scan_cnn, build_teacher_mlp, export_policy_dir,
)
from alldog_mjlab.tasks.velocity.profiles.plum.teacher import (  # noqa: E402
    ACTOR_HISTORY, NO_COMMAND_DEPLOY_LAYOUT, NO_COMMAND_FRAME_DIM, PROPRIO_DIM,
    SPAWN, WHEEL_DQ_CLIP, LEG_RESIDUAL_CLIP, plum_teacher_env_cfg,
    plum_teacher_runner_cfg,
)


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        raise AssertionError(f"FAIL {name}: {detail}")
    print(f"  ok  {name}" + (f"  ({detail})" if detail else ""))


def eval_depth_contract() -> dict:
    """Read DEFAULT_DEPTH_CONTRACT statically (no cross-env mujoco import)."""
    src = (ROOT / "eval" / "robocon_eval" / "policy_runner.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        target = node.target if isinstance(node, ast.AnnAssign) else (
            node.targets[0] if isinstance(node, ast.Assign) else None)
        if target is not None and getattr(target, "id", None) == "DEFAULT_DEPTH_CONTRACT":
            return ast.literal_eval(node.value)
    raise AssertionError("DEFAULT_DEPTH_CONTRACT not found in policy_runner.py")


def main() -> int:
    print("[1] geometry (single source: terrain/plum/plum_spec.json)")
    check("6 cubes, 2 rows", len(SPEC.cube_centres) == 6, str(SPEC.cube_centres))
    check("no ramps, step-down at 5.3 m", SPEC.exit_end_x == 5.3,
          f"exit_end_x={SPEC.exit_end_x}")
    check("cube x range 2.7..4.4", SPEC.cube_x_range == (2.7, 4.4))
    check("goal line 5.4", abs(SPEC.goal_x - 5.4) < 1e-9)

    print("[2] camera contract parity with eval judge")
    check("DEPTH_CONTRACT == DEFAULT_DEPTH_CONTRACT",
          DEPTH_CONTRACT == eval_depth_contract(),
          f"{DEPTH_CONTRACT['width']}x{DEPTH_CONTRACT['height']}")
    from alldog_mjlab.tasks.velocity.profiles.plum.depth_camera import (
        CAMERA_GROUP, SENSOR_NAME, add_depth_camera,
    )
    cam_cfg = plum_teacher_env_cfg(play=True)
    add_depth_camera(cam_cfg)
    cam = next(s for s in cam_cfg.scene.sensors if s.name == SENSOR_NAME)
    check("training camera pose/FOV match judge",
          tuple(cam.pos) == tuple(DEPTH_CONTRACT["pos"])
          and tuple(cam.quat) == tuple(DEPTH_CONTRACT["quat"])
          and cam.fovy == DEPTH_CONTRACT["fovy"]
          and (cam.width, cam.height) == (DEPTH_CONTRACT["width"], DEPTH_CONTRACT["height"]))
    check("training camera sees geom group 0 only", cam.enabled_geom_groups == (0,))
    check("camera creates a new sensor (camera_name=None)", cam.camera_name is None,
          f"sensor name={cam.name}")
    check("no camera DR (corruption free)",
          cam_cfg.observations[CAMERA_GROUP].enable_corruption is False)

    print("[3] dt scaling (the 'estimated per-step' convention)")
    import inspect
    from mjlab.managers.reward_manager import RewardManager
    check("scale_by_dt default True",
          inspect.signature(RewardManager.__init__).parameters["scale_by_dt"].default is True)
    check("policy step dt = 0.02 s", abs(BLACK_CONFIG.control.policy_dt - 0.02) < 1e-9,
          f"dt={BLACK_CONFIG.control.policy_dt}")

    print("[4] teacher env cfg (stage 0)")
    cfg = plum_teacher_env_cfg(stage=0)
    check("command term removed", "command" not in cfg.observations["actor"].terms)
    check("actor history 6 / flattened",
          cfg.observations["actor"].history_length == ACTOR_HISTORY
          and cfg.observations["actor"].flatten_history_dim is True)
    check("fine_scan group present", "fine_scan" in cfg.observations)
    check("no tracking rewards", not any("track" in k for k in cfg.rewards),
          str(tuple(cfg.rewards)))
    # v2: stage 0 is positive-only (no time_penalty / no penalties at all).
    check("stage 0 is positive-only",
          all(t.weight >= 0 for t in cfg.rewards.values()),
          str({k: t.weight for k, t in cfg.rewards.items()}))
    check("stage 0 has NO time penalty", "time_penalty" not in cfg.rewards)
    check("stage 0 has no lin_vel_z/upright/feet_edge",
          not ({"lin_vel_z", "upright", "feet_edge"} & set(cfg.rewards)))
    # stage 1 adds the late-course penalties.
    cfg1 = plum_teacher_env_cfg(stage=1)
    check("stage 1 adds time penalty on success step",
          "time_penalty" in cfg1.rewards and cfg1.rewards["time_penalty"].weight < 0
          and cfg1.rewards["time_penalty"].func.__name__ == "success_step_time_penalty")
    check("stage 1 adds lin_vel_z masked",
          cfg1.rewards["lin_vel_z"].func.__name__ == "lin_vel_z_masked")
    check("stage 1 keeps all stage 0 terms",
          set(cfg.rewards).issubset(set(cfg1.rewards)))
    # Spawn envelope: stage 0 narrow, stage 1 full.
    s0 = SPAWN[0]
    pose = cfg.events["reset_base"].params["pose_range"]
    check("stage 0 spawn dx/dy = 0.05 m",
          pose["x"] == (-s0["x"], s0["x"]) and pose["y"] == (-s0["y"], s0["y"]))
    check("stage 0 spawn dyaw = 5 deg",
          abs(np.degrees(pose["yaw"][1]) - s0["yaw_deg"]) < 1e-6)
    s1 = SPAWN[1]
    pose1 = cfg1.events["reset_base"].params["pose_range"]
    check("stage 1 spawn wider than stage 0", pose1["y"][1] > pose["y"][1])

    print("[4b] joint-command clipping (deployment parity)")
    from alldog_mjlab.tasks.velocity.blackw.actions import WheelLegActionCfg
    clips = [t.clip for t in cfg.actions.values() if isinstance(t, WheelLegActionCfg)]
    check("all 4 wheel-leg actions have a clip", len(clips) == 4
          and all(c is not None for c in clips))
    check("wheel clip = +/-20 rad/s = command_dq_max",
          all(c[".*wheel_joint"] == (-WHEEL_DQ_CLIP, WHEEL_DQ_CLIP) for c in clips)
          and WHEEL_DQ_CLIP == 20.0)
    check("leg residual clip = +/-1.0 rad",
          all(c[".*_(hip|thigh|calf)_joint"] == (-LEG_RESIDUAL_CLIP, LEG_RESIDUAL_CLIP)
              for c in clips))
    # Verify the clip actually resolves onto the scaled joint command at build.
    from mjlab.envs import ManagerBasedRlEnv
    probe = plum_teacher_env_cfg(stage=0)
    probe.scene.num_envs = 2
    penv = ManagerBasedRlEnv(cfg=probe, device="cuda:0")
    pclip = penv.action_manager.get_term("wheel_leg_fl")._clip[0]
    check("built action clip resolves leg+wheel",
          bool((pclip[0] == torch.tensor([-1., 1.], device=pclip.device)).all())
          and bool((pclip[3] == torch.tensor([-20., 20.], device=pclip.device)).all()),
          str(pclip.tolist()))
    penv.close()

    print("[4c] judge physics + strict contact")
    sim = cfg.sim.mujoco
    check("judge solver options", sim.integrator == "euler" and sim.cone == "elliptic"
          and sim.iterations == 100 and abs(sim.impratio - 100.0) < 1e-6,
          f"{sim.integrator}/{sim.cone}/iter={sim.iterations}/impratio={sim.impratio}")
    check("judge spawn z = 0.55",
          abs(cfg.scene.entities["robot"].init_state.pos[2] - 0.55) < 1e-9)
    check("strict illegal_contact termination present",
          "illegal_contact" in cfg.terminations)
    illegal = next(s for s in cfg.scene.sensors if s.name == "illegal_ground_contact")
    check("illegal sensor covers all non-wheel bodies",
          getattr(illegal.primary, "exclude", None) == (r".*_foot",),
          str(getattr(illegal.primary, "exclude", None)))

    rcfg = plum_teacher_runner_cfg(stage=0)
    check("runner obs_groups", rcfg.obs_groups["actor"] == ("actor", "fine_scan"))
    check("runner logger tensorboard", rcfg.logger == "tensorboard")

    print("[5] built env tensor shapes")
    cfg.scene.num_envs = 2
    from mjlab.envs import ManagerBasedRlEnv
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    obs = env.reset()
    o = obs[0] if isinstance(obs, tuple) else obs
    check("actor = 324", tuple(o["actor"].shape) == (2, PROPRIO_DIM))
    check("fine_scan = 1764", tuple(o["fine_scan"].shape) == (2, fine_scan_dim()))
    check("critic unchanged (247)", tuple(o["critic"].shape) == (2, 247))
    check("actor finite", bool(torch.isfinite(o["actor"]).all()))
    r = env.reward_manager.compute(0.02)
    check("reward finite, scale_by_dt", bool(torch.isfinite(r).all())
          and env.reward_manager._scale_by_dt is True)
    env.close()

    print("[6] export permutation")
    perm = build_permutation(fine_scan_dim())
    check("permutation length 2088", len(perm) == PROPRIO_DIM + fine_scan_dim())
    check("permutation is a bijection", set(perm.tolist()) == set(range(len(perm))))
    check("deployment frame dims sum to 54",
          sum(d for d, _ in NO_COMMAND_DEPLOY_LAYOUT) == NO_COMMAND_FRAME_DIM)

    print("[7] student modules + TorchScript export")
    scan_dim = fine_scan_dim()
    cnn, head = build_scan_cnn(scan_dim)
    z = cnn(torch.rand(2, 1, DEPTH_CONTRACT["height"], DEPTH_CONTRACT["width"]))
    check("CNN latent = 64", tuple(z.shape) == (2, CNN_LATENT_DIM),
          f"latent={tuple(z.shape)}, cfg={DEPTH_CNN_CFG['output_channels']}")
    # A synthetic teacher with the real 2088-D shape, to exercise export.
    teacher = torch.nn.Sequential(
        torch.nn.Linear(PROPRIO_DIM + scan_dim, 512), torch.nn.ELU(),
        torch.nn.Linear(512, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(),
        torch.nn.Linear(128, 16),
    )
    actor_state: dict[str, torch.Tensor] = {}
    for i, key in enumerate(("mlp.0", "mlp.2", "mlp.4", "mlp.6")):
        actor_state[f"{key}.weight"] = teacher[2 * i].weight.detach().clone()
        actor_state[f"{key}.bias"] = teacher[2 * i].bias.detach().clone()
    rebuilt = build_teacher_mlp(actor_state)
    check("teacher rebuild matches weights",
          torch.allclose(rebuilt[0].weight, teacher[0].weight))
    with tempfile.TemporaryDirectory() as tmp:
        out = export_policy_dir(Path(tmp) / "plum_student", "plum_student_smoke",
                                cnn, head, rebuilt, "smoke", {"val": 0.0})
        mod = torch.jit.load(str(out / "policy.pt"))
        d = torch.rand(1, 1, DEPTH_CONTRACT["height"], DEPTH_CONTRACT["width"])
        a = mod(torch.randn(1, PROPRIO_DIM), [d])
        check("scripted policy(obs_324,[depth]) -> 16", tuple(a.shape) == (1, 16))
        import yaml
        dep_cfg = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
        section = next(iter(dep_cfg.values()))
        check("deploy config num_observations=54", section["num_observations"] == 54)
        check("deploy config has no commands", "commands" not in section["observations"])

    print("\nALL SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
