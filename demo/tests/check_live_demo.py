#!/usr/bin/env python3
"""Check demo-task exit statuses with tiny processes, without robot physics."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, TimerAction
from launch.events import Shutdown
from launch.utilities import perform_substitutions

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("live_demo", ROOT / "tools/live_demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)
launch_spec = importlib.util.spec_from_file_location(
    "live_launch", ROOT / "demo/rl_sar/launch/rl_sim_all.launch.py")
live_launch = importlib.util.module_from_spec(launch_spec)
launch_spec.loader.exec_module(live_launch)


def process(name, code=0, delay=0.0):
    return ExecuteProcess(name=name, cmd=[sys.executable, "-c",
        f"import time; time.sleep({delay}); raise SystemExit({code})"], output="log")


class LiveDemoPolicyTests(unittest.TestCase):
    def settings(self, route, *, transfer="", policy="", environment=""):
        context = LaunchContext()
        context.launch_configurations.update({
            "robot": "blackW", "route": route, "part": "1",
            "transfer_policy": transfer, "policy_name": policy})
        with patch.dict(os.environ, {"PART1_TRANSFER_POLICY": environment}):
            return live_launch._route_settings(context)

    def test_both_live_routes_use_physically_tested_transfer_by_default(self):
        for route, part in (("part1", "1"), ("part2", "2")):
            with self.subTest(route=route):
                self.assertEqual(self.settings(route),
                                 (route, part, "vt_locked12", "vt_locked12"))

    def test_explicit_transfer_overrides_environment_and_sets_initial_policy(self):
        self.assertEqual(self.settings("part1", transfer="vt_pose_v1", environment="flat"),
                         ("part1", "1", "vt_pose_v1", "vt_pose_v1"))

    def test_environment_override_and_explicit_initial_policy_remain_available(self):
        self.assertEqual(self.settings("part2", policy="flat", environment="vt_pose_v1"),
                         ("part2", "2", "flat", "vt_pose_v1"))


class LiveDemoTimingTests(unittest.TestCase):
    def context(self, **overrides):
        context = LaunchContext()
        for action in live_launch.generate_launch_description().entities:
            if isinstance(action, DeclareLaunchArgument):
                context.launch_configurations[action.name] = perform_substitutions(
                    context, action.default_value)
        context.launch_configurations.update(overrides)
        return context

    def environment(self, context):
        process = live_launch._route_node(context)[0]
        return {perform_substitutions(context, key):
                perform_substitutions(context, value) for key, value in process.env}

    def test_part2_live_and_full_speed_use_the_same_simulation_clock(self):
        for realtime in ("true", "false"):
            with self.subTest(realtime=realtime), patch.dict(
                    os.environ, {"PART1_ROUTE_STEPLOCK": "0"}):
                context = self.context(route="part2", realtime=realtime)
                environment = self.environment(context)
                self.assertEqual(environment["PART1_ROUTE_STEPLOCK"], "1")

    def test_part1_keeps_baseline_wall_clock_and_honors_explicit_timing_override(self):
        with patch.dict(os.environ):
            os.environ.pop("PART1_ROUTE_STEPLOCK", None)
            self.assertNotIn("PART1_ROUTE_STEPLOCK", self.environment(self.context(route="part1")))
            for value in ("0", "1"):
                os.environ["PART1_ROUTE_STEPLOCK"] = value
                self.assertEqual(self.environment(self.context(route="part1"))["PART1_ROUTE_STEPLOCK"], value)

    def test_part1_physical_preset_defaults_judge_off_and_honors_opt_in(self):
        with patch.dict(os.environ), patch.object(demo, "run_launch", return_value=0), \
                patch.object(demo, "get_package_share_directory", return_value=str(ROOT / "demo/rl_sar")):
            os.environ.pop("PART1_ROUTE_JUDGE", None)
            self.assertEqual(demo.main(["part1"]), 0)
            self.assertEqual(os.environ["PART1_ROUTE_JUDGE"], "0")
            os.environ["PART1_ROUTE_JUDGE"] = "1"
            self.assertEqual(demo.main(["part1"]), 0)
            self.assertEqual(os.environ["PART1_ROUTE_JUDGE"], "1")
            os.environ.pop("PART1_ROUTE_JUDGE")
            self.assertEqual(demo.main(["part2"]), 0)
            self.assertNotIn("PART1_ROUTE_JUDGE", os.environ)

    def test_recording_rate_is_preserved_in_both_pacing_modes(self):
        for route in ("part1", "part2"):
            for realtime in ("true", "false"):
                for every in (1, 10):
                    with self.subTest(route=route, realtime=realtime, every=every):
                        context = self.context(route=route, realtime=realtime,
                                               eval_state_every=str(every))
                        with patch.object(live_launch, "_validate_contract"), \
                                patch.object(live_launch, "Node", side_effect=lambda **kwargs: kwargs):
                            node = live_launch._sim_params(context)[-1]
                        self.assertEqual(node["parameters"][0]["eval_state_every"], every)

    def test_automatic_recording_default_is_lightweight_only_for_routes(self):
        for route, expected in (("", 1), ("part1", 10), ("part2", 10)):
            with self.subTest(route=route):
                context = self.context(route=route)
                with patch.object(live_launch, "_validate_contract"), \
                        patch.object(live_launch, "Node", side_effect=lambda **kwargs: kwargs):
                    node = live_launch._sim_params(context)[-1]
                self.assertEqual(node["parameters"][0]["eval_state_every"], expected)


class LiveDemoExitTests(unittest.TestCase):
    def test_route_success_is_zero(self):
        self.assertEqual(demo.run_launch(LaunchDescription([
            process("part2_auto_route")]), "part2"), 0)

    def test_route_failure_is_not_hidden_by_ros_shutdown(self):
        self.assertEqual(demo.run_launch(LaunchDescription([
            process("part2_auto_route", code=7)]), "part2"), 7)

    def test_simulator_early_clean_exit_aborts_the_route(self):
        self.assertNotEqual(demo.run_launch(LaunchDescription([
            process("sim_node"), process("part2_auto_route", delay=3)]), "part2"), 0)

    def test_inference_crash_is_preserved(self):
        self.assertEqual(demo.run_launch(LaunchDescription([
            process("rl_sim", code=4), process("part2_auto_route", delay=3)]), "part2"), 4)

    def test_ctrl_c_is_130(self):
        self.assertEqual(demo.run_launch(LaunchDescription([
            process("part2_auto_route", delay=3),
            TimerAction(period=0.1, actions=[EmitEvent(event=Shutdown(due_to_sigint=True))]),
        ]), "part2"), 130)

    def test_missing_route_cannot_report_success(self):
        self.assertNotEqual(demo.run_launch(LaunchDescription([]), "part2"), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
