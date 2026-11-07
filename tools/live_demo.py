#!/usr/bin/env python3
"""Run the existing ROS launch and preserve automatic-route failure status.

ROS launch normally returns zero when a child fails and requests shutdown.
The live-demo tasks need the route's result instead; this runner observes
process exits without adding another scene, route or controller definition.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchService
from launch.actions import EmitEvent, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource


def _shell_status(returncode: int) -> int:
    return returncode if returncode >= 0 else 128 - returncode


def run_launch(description: LaunchDescription, route: str) -> int:
    """Observe the canonical route and its required simulator/inference nodes."""
    route_result = None
    dependency_failure = None
    interrupted = False
    shutting_down = False

    def process_exit(event, context):
        nonlocal route_result, dependency_failure
        name = re.sub(r"-\d+$", "", event.process_name)
        code = int(event.returncode)
        if name == route + "_auto_route":
            route_result = _shell_status(code)
            if not context.is_shutdown:
                return [EmitEvent(event=Shutdown(reason=f"{route} route exited with {code}"))]
        elif name in ("sim_node", "rl_sim", "rl_sim_node"):
            expected_signal = shutting_down and code in (-2, -15)
            if not expected_signal and (code != 0 or (route_result is None and not shutting_down)):
                if dependency_failure is None:
                    dependency_failure = _shell_status(code) or 1
                if not context.is_shutdown:
                    return [EmitEvent(event=Shutdown(reason=f"required process {name} exited with {code}"))]
        return None

    def shutdown(event, context):
        nonlocal interrupted, shutting_down
        shutting_down = True
        interrupted = interrupted or event.due_to_sigint

    service = LaunchService(argv=[])
    service.include_launch_description(LaunchDescription([
        RegisterEventHandler(OnProcessExit(on_exit=process_exit)),
        RegisterEventHandler(OnShutdown(on_shutdown=shutdown)),
        *description.entities,
    ]))
    try:
        launch_result = service.run()
    except KeyboardInterrupt:
        return 130
    if interrupted:
        return 130
    if dependency_failure is not None:
        return dependency_failure
    if route_result is not None:
        return route_result
    # A launch error or a missing route process is never a successful demo.
    return launch_result or 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("route", choices=("part1", "part2"))
    parser.add_argument("--show-args", "--show-arguments", action="store_true")
    parser.add_argument("launch_args", nargs="*", metavar="name:=value")
    args = parser.parse_args(argv)
    values = {"robot": "blackW", "route": args.route,
              "viewer": "true", "realtime": "true",
              "eval_state_every": "10", "stop_on_fail": "true"}
    for argument in args.launch_args:
        name, separator, value = argument.partition(":=")
        if not separator or not name:
            parser.error(f"expected name:=value, got {argument!r}")
        values[name] = value
    route = values["route"]
    if route not in ("part1", "part2"):
        parser.error("demo tasks require route:=part1 or route:=part2")
    if route == "part1":
        # Reproduce the recorded physical demo. Strict live judging remains an
        # explicit opt-in; the production judge and direct ROS entry are unchanged.
        os.environ.setdefault("PART1_ROUTE_JUDGE", "0")
    if args.show_args:
        # Keep ROS's standard argument descriptions for the existing entry.
        return subprocess.call(["ros2", "launch", "rl_sar", "rl_sim_all.launch.py",
                                "--show-args", *(f"{key}:={value}" for key, value in values.items())])
    path = Path(get_package_share_directory("rl_sar")) / "launch/rl_sim_all.launch.py"
    description = LaunchDescription([IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(path)), launch_arguments=values.items())])
    result = run_launch(description, route)
    if result:
        print(f"[live_demo] {route} stopped with exit status {result}", file=sys.stderr)
    return result


if __name__ == "__main__":
    sys.exit(main())
