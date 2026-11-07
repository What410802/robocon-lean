"""Single demo entry: sim_node + rl_sim + optional joy, robot/part/level aware.

Examples (from the repo root, after `source demo/install/setup.bash`):
    ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=1 policy_name:=flat
    ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=2 level:=high_platform_10cm
    ros2 launch rl_sar rl_sim_all.launch.py robot:=black part:=1 policy_name:=himloco

The MuJoCo window is ON by default (viewer:=true, matching sim_node and the legacy entry
launches). The default camera follows the robot base (`follow_camera:=true`, like the review
videos; `Ctrl+F` or `follow_camera:=false` switches to the fixed iso view). It degrades to
headless automatically when DISPLAY/WAYLAND_DISPLAY is absent. Live demos default to real time.
For headless Part2 tests pass `viewer:=false realtime:=false`. Part2 synchronizes 50 Hz
control with simulation timestamps. Part1 retains its recorded 1x wall-clock controller.

`eval_state_every` defaults to auto: 10 (50 Hz) for automatic routes, 1 (every 2 ms
physics frame) otherwise. Pass `eval_state_every:=1` for judged traces and `0` to disable
/eval_state (automatic routes retain the 50 Hz feedback they need).

`scene:=` overrides the part/level default. An automatic route selects its own part and scene.
Spawn comes from robot.yaml for Part1 and from `scenes/part2/part2_official_levels.json` for
blackW Part2; explicit spawn_* args override.
"""

from pathlib import Path
import json
import os
import runpy
import sys

import yaml

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ROOT = Path(__file__).resolve().parents[2]  # .../demo (rl_sar/launch/<this file>)
POLICY_ROOT = ROOT / "rl_sar" / "policy"
ROBOT_ROOT = ROOT / "robots"
ROUTE_SCRIPT = ROOT / "rl_sar" / "scripts" / "part1_auto_route.py"
PART2_ROUTE_SCRIPT = ROOT / "rl_sar" / "scripts" / "part2_auto_route.py"
FULL_FIELD_SCENE = ROOT / "scenes" / "part2" / "full_field" / "part2_full_field.xml"
PART2_ROUTE_PLAN = ROOT / "rl_sar" / "scripts" / "part2_route_plan.py"


def _unwrap_config(data):
    if isinstance(data, dict) and "num_observations" not in data and len(data) == 1:
        only = next(iter(data.values()))
        return only if isinstance(only, dict) else data
    return data


def _validate_contract(robot: str, policy: str) -> None:
    """P2-02 fail-loud preflight before starting nodes (launch path)."""
    robot_path = ROBOT_ROOT / robot / "robot.yaml"
    policy_path = POLICY_ROOT / robot / policy / "config.yaml"
    if not robot_path.is_file():
        raise RuntimeError(f"robot manifest missing: {robot_path}")
    if not policy_path.is_file():
        raise RuntimeError(f"policy config missing: {policy_path} (policy={policy!r})")
    man = yaml.safe_load(robot_path.read_text()) or {}
    cfg = _unwrap_config(yaml.safe_load(policy_path.read_text()) or {})
    model = policy_path.parent / cfg.get("model_name", "policy.pt")
    if not model.is_file():
        raise RuntimeError(f"policy weights missing: {model}; export/copy the local weights before launching")
    depth = cfg.get("depth") or {}
    if depth.get("enabled"):
        # Depth policies are command-free (54 x 6) and declare their own contract.
        obs = man.get("obs_contract_depth") or {}
        if not obs:
            raise RuntimeError(f"robot manifest missing obs_contract_depth: {robot_path}")
        depth_json = policy_path.parent / "depth.json"
        if not depth_json.is_file():
            raise RuntimeError(f"depth policy is missing depth.json: {depth_json}")
    else:
        obs = man.get("obs_contract", {})
    checks = [
        ("num_of_dofs", cfg.get("num_of_dofs"), man.get("num_dofs")),
        ("num_observations", cfg.get("num_observations"), obs.get("num_observations")),
        ("joint_mapping", len(cfg.get("joint_mapping") or []), man.get("num_dofs")),
        ("action_scale", len(cfg.get("action_scale") or []), man.get("num_dofs")),
    ]
    for name, got, want in checks:
        if got is not None and want is not None and got != want:
            raise RuntimeError(f"contract mismatch {robot}/{policy}: {name}={got} expected {want}")
    hist = cfg.get("observations_history")
    if hist is not None and len(hist) != obs.get("history"):
        raise RuntimeError(
            f"contract mismatch {robot}/{policy}: history={len(hist)} expected {obs.get('history')}")


def _robot_manifest(robot: str) -> dict:
    return yaml.safe_load((ROBOT_ROOT / robot / "robot.yaml").read_text()) or {}


def _part1_defaults(robot: str):
    man = _robot_manifest(robot)
    scene = ROOT / man["scene_roots"]["part1"] / "scene_part1.xml"
    return scene, tuple(man["spawn"]["part1"])


def _part2_defaults(robot: str, level: str):
    if robot != "blackW":
        raise RuntimeError(f"part 2 deployment is blackW-only in P2; got robot={robot!r}")
    man = _robot_manifest(robot)
    levels_path = ROOT / man["spawn"]["part2_source"]
    levels = {e["key"]: e for e in json.loads(levels_path.read_text())["levels"]}
    if level not in levels:
        raise RuntimeError(f"unknown part2 level {level!r}; keys: {sorted(levels)[:5]} ...")
    e = levels[level]
    spawn = e.get("spawn") or e.get("start")
    return ROOT / man["scene_roots"]["part2"] / e["scene"], tuple(spawn)


def _route_settings(context):
    route = context.perform_substitution(LaunchConfiguration("route")).strip()
    if route not in ("", "part1", "part2"):
        raise RuntimeError(f"unknown route {route!r}; supported: 'part1', 'part2'")
    part = route[-1] if route else context.perform_substitution(LaunchConfiguration("part"))
    transfer = context.perform_substitution(LaunchConfiguration("transfer_policy")).strip()
    transfer = transfer or os.environ.get("PART1_TRANSFER_POLICY", "").strip()
    transfer = transfer or "vt_locked12"
    policy = context.perform_substitution(LaunchConfiguration("policy_name")).strip()
    policy = policy or (transfer if route else "flat")
    if route:
        robot = context.perform_substitution(LaunchConfiguration("robot"))
        if robot != "blackW":
            raise RuntimeError(f"automatic routes require robot:=blackW; got {robot!r}")
    return route, part, policy, transfer


def _part2_route_defaults():
    plan = runpy.run_path(str(PART2_ROUTE_PLAN))
    scene = ROOT.parent / plan["SCENE_RELATIVE"]
    spawn = (*plan["ROUTE_START"], 0.675, 0.0)
    raw = os.environ.get("PART2_ROUTE_START", "").strip()
    if raw:
        legs = plan["LEGS"]
        selected = next((leg for leg in legs if raw.lower() == leg["leg"].lower()), None)
        if selected is None:
            try:
                index = int(raw) - 1
            except ValueError:
                index = -1
            selected = legs[index] if 0 <= index < len(legs) else None
        if selected is None:
            raise RuntimeError(f"invalid PART2_ROUTE_START={raw!r}; use a canonical leg name or 1..{len(legs)}")
        entry = selected["transit"][0] if selected["transit"] else selected["stand"]
        spawn = (*entry, 0.675, selected["lane_yaw"])
    return scene, spawn


def _sim_params(context):
    robot = context.perform_substitution(LaunchConfiguration("robot"))
    route, part, policy, transfer = _route_settings(context)
    level = context.perform_substitution(LaunchConfiguration("level"))
    explicit = context.perform_substitution(LaunchConfiguration("scene"))
    full_field = context.perform_substitution(LaunchConfiguration("full_field")).strip().lower()
    eval_raw = context.perform_substitution(LaunchConfiguration("eval_state_every")).strip()
    eval_state_every = (10 if route else 1) if eval_raw.lower() == "auto" else int(eval_raw)
    # The Part2 route runs on the separately-saved route field.
    if route == "part2":
        scene, spawn = _part2_route_defaults()
        if explicit and Path(explicit).resolve() != scene.resolve():
            raise RuntimeError(f"route:=part2 uses its canonical scene {scene}; remove the scene override")
    # The auto route consumes /eval_state as its pose feedback. Judged runs explicitly use
    # 1 (every 2 ms physics frame); an explicit `eval_state_every:=0` still enables the
    # route's 50 Hz feedback instead of leaving it without pose input.
    if route and eval_state_every == 0:
        eval_state_every = 10
    _validate_contract(robot, policy)
    if route:
        _validate_contract(robot, transfer)
    if route == "part2":
        pass  # scene/spawn already selected above
    elif explicit:
        scene, spawn = Path(explicit), (None, None, None, None)
    elif full_field in ("1", "true", "yes"):
        if robot != "blackW":
            raise RuntimeError("full_field scene is blackW-only (official ability field wrapper)")
        scene = FULL_FIELD_SCENE
        if not scene.is_file():
            raise RuntimeError(f"full field scene missing from the demo repo: {scene}")
        spawn = (None, None, None, None)
    elif part == "2":
        scene, spawn = _part2_defaults(robot, level)
    else:
        scene, spawn = _part1_defaults(robot)
    params = {
        "scene": ParameterValue(str(scene), value_type=str),
        "viewer": ParameterValue(LaunchConfiguration("viewer"), value_type=bool),
        "follow_camera": ParameterValue(LaunchConfiguration("follow_camera"), value_type=bool),
        "status_period_s": ParameterValue(LaunchConfiguration("status_period_s"), value_type=float),
        "realtime": ParameterValue(LaunchConfiguration("realtime"), value_type=bool),
        "start": ParameterValue(LaunchConfiguration("start"), value_type=str),
        "publish_every": ParameterValue(LaunchConfiguration("publish_every"), value_type=int),
        "eval_state_every": eval_state_every,
        "depth": ParameterValue(LaunchConfiguration("depth"), value_type=str),
        "depth_every": ParameterValue(LaunchConfiguration("depth_every"), value_type=int),
    }
    sx, sy, sz, syaw = spawn
    for key, arg in (("spawn_x", sx), ("spawn_y", sy), ("spawn_z", sz), ("spawn_yaw", syaw)):
        value = context.perform_substitution(LaunchConfiguration(key))
        if value.strip() in ("", "nan"):
            if arg is not None:
                params[key] = float(arg)
        else:
            params[key] = float(value)
    nodes = [Node(
        package="quadruped_ros2",
        executable="sim_node",
        name="sim_node",
        output="screen",
        parameters=[params],
    )]
    if route == "part2" and (os.environ.get("PART2_ROUTE_START") or os.environ.get("PART2_ROUTE_ONLY")):
        nodes.insert(0, LogInfo(msg="Part2 debug subset: selected steps belong to the canonical plan; this run does not verify the full route."))
    return nodes


def _route_node(context, *args, **kwargs):
    route, _, _, transfer = _route_settings(context)
    if not route:
        return []
    script = PART2_ROUTE_SCRIPT if route == "part2" else ROUTE_SCRIPT
    if not script.is_file():
        raise RuntimeError(f"route script missing: {script}")
    stop_on_fail = context.perform_substitution(LaunchConfiguration("stop_on_fail")).strip().lower() \
        not in ("0", "false", "no", "off")
    route_env = {**os.environ,
                 "PART1_TRANSFER_POLICY": transfer,
                 "PART1_ROUTE_STOP_ON_FAIL": "1" if stop_on_fail else "0"}
    if route == "part2":
        route_env["PART1_ROUTE_STEPLOCK"] = "1"
    route_proc = ExecuteProcess(
        cmd=[sys.executable, str(script)],
        output="screen",
        name="part2_auto_route" if route == "part2" else "part1_auto_route",
        env=route_env,
    )
    # When the route process exits (all PASS, or the first FAIL with stop_on_fail),
    # shut the whole launch down instead of leaving the sim running until timeout.
    return [
        route_proc,
        RegisterEventHandler(OnProcessExit(
            target_action=route_proc,
            on_exit=[EmitEvent(event=Shutdown())],
        )),
    ]


def _rl_node(context):
    _, part, policy, _ = _route_settings(context)
    return [Node(
        package="rl_sar",
        executable="rl_sim",
        name="rl_sim_node",
        output="screen",
        emulate_tty=True,
        condition=IfCondition(LaunchConfiguration("rl")),
        parameters=[{"robot_name": LaunchConfiguration("robot"),
                     "policy_name": policy, "part": int(part)}],
    )]


def generate_launch_description() -> LaunchDescription:
    joy = LaunchConfiguration("joy")
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot", default_value="blackW"),
            DeclareLaunchArgument("part", default_value="1"),
            DeclareLaunchArgument("level", default_value="high_platform_10cm",
                                  description="Part2 level key (see scenes/part2/part2_official_levels.json)"),
            DeclareLaunchArgument("policy_name", default_value="",
                                  description="initial policy (default: route transfer policy, or flat without a route)"),
            DeclareLaunchArgument("scene", default_value="", description="explicit scene xml override"),
            DeclareLaunchArgument(
                "full_field", default_value="false",
                description="load the official full Part2 ability field (scenes/part2/full_field/"
                            "part2_full_field.xml, all obstacle types and difficulties in one scene) "
                            "instead of one isolated level; manual inspection/driving only, the judge "
                            "still uses generated/part2_off_<level>.xml"),
            DeclareLaunchArgument("start", default_value="raw"),
            DeclareLaunchArgument("publish_every", default_value="1",
                                  description="publish /motor_state every N physics steps (500/N Hz)"),
            DeclareLaunchArgument("eval_state_every", default_value="auto",
                                  description="auto = 10 for routes, 1 otherwise; publish /eval_state "
                                              "every N physics steps: 1 = one sample "
                                              "per 2 ms physics frame (judged trace); 10 = 50 Hz for "
                                              "interactive runs; 0 = off"),
            DeclareLaunchArgument("depth", default_value="auto",
                                  description="sim_node offscreen depth camera on /depth_cam: "
                                              "auto = on when the scene has a depth_cam camera, "
                                              "on = require it, off = disable"),
            DeclareLaunchArgument("depth_every", default_value="10",
                                  description="render/publish /depth_cam once per N physics steps "
                                              "(10 = 50 Hz, the policy rate)"),
            DeclareLaunchArgument("route", default_value="",
                                  description="auto route: 'part1' or 'part2' starts the "
                                              "closed-loop course demo and selects its part/scene "
                                              "(Part1: 1x wall clock; Part2: 50 Hz simulation time); "
                                              "(if eval_state_every:=0 is passed it is raised to 10)"),
            DeclareLaunchArgument("transfer_policy", default_value="",
                                  description="policy used for route transfers/alignment: "
                                              "default vt_locked12 for Part1 and Part2; "
                                              "PART1_TRANSFER_POLICY supplies a default override"),
            DeclareLaunchArgument("stop_on_fail", default_value="true",
                                  description="stop the auto route (and this launch) at the first "
                                              "segment FAIL; pass false to keep going like an operator"),
            DeclareLaunchArgument(
                "viewer", default_value="true",
                description="open the MuJoCo window (auto-off when DISPLAY/WAYLAND_DISPLAY is missing; "
                            "headless runs pass viewer:=false)"),
            DeclareLaunchArgument(
                "status_period_s", default_value="5.0",
                description="how often sim_node prints its status line (sim seconds; 0 = off); "
                            "regression scripts that grep the per-second line pass 1.0"),
            DeclareLaunchArgument(
                "follow_camera", default_value="true",
                description="camera looks at the robot base every frame (like the review videos); "
                            "false = fixed iso view. Ctrl+F toggles it in the window"),
            DeclareLaunchArgument(
                "realtime", default_value="true",
                description="headless pacing only: false = full speed (ignored when the window is open, "
                            "window mode is wall-clock); keep true for Part1. Part2 always uses "
                            "50 Hz simulation-time control; eval_state_every is preserved"),
            DeclareLaunchArgument("joy", default_value="false"),
            DeclareLaunchArgument("rl", default_value="true"),
            DeclareLaunchArgument("device", default_value=""),
            DeclareLaunchArgument("spawn_x", default_value="nan"),
            DeclareLaunchArgument("spawn_y", default_value="nan"),
            DeclareLaunchArgument("spawn_z", default_value="nan"),
            DeclareLaunchArgument("spawn_yaw", default_value="nan"),
            OpaqueFunction(function=_sim_params),
            Node(
                package="quadruped_ros2",
                executable="joy_node",
                name="joy_node",
                output="screen",
                condition=IfCondition(joy),
                parameters=[{"device": LaunchConfiguration("device")}],
            ),
            OpaqueFunction(function=_route_node),
            OpaqueFunction(function=_rl_node),
        ]
    )
