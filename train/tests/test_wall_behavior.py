"""T1.3: high-wall success/termination behavior for the training predicate.

These are synthetic state tests of ``_part2_terrain_state`` (no rollout): they pin the
behaviour that side/end bypass is a violation and that one missing wheel cannot be success.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import unittest

from alldog_mjlab.tasks.velocity.profiles.part2.predicates import (
    _part2_terrain_state,
    PART2_WALL_X,
)


class _Data:
    def __init__(self):
        self.body_names = ["trunk", "FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self.body_link_pos_w = torch.zeros((1, 5, 3))
        # wxyz identity; the predicate maps the wheel-local cylinder pose
        # through this body orientation.
        self.body_link_quat_w = torch.zeros((1, 5, 4))
        self.body_link_quat_w[..., 0] = 1.0
        self.root_link_pos_w = torch.zeros((1, 3))
        self.root_link_lin_vel_w = torch.zeros((1, 3))


class _Robot:
    def __init__(self):
        self.body_names = ["trunk", "FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self.data = _Data()


class _Terrain:
    def __init__(self):
        self.env_origins = torch.zeros((1, 3))
        self.terrain_types = torch.zeros(1, dtype=torch.long)
        self.terrain_levels = torch.zeros(1, dtype=torch.long)
        self.cfg = SimpleNamespace(terrain_generator=SimpleNamespace(
            num_rows=2, difficulty_range=(0.0, 0.0),
            sub_terrains={"high_wall": None}))


class _Scene:
    def __init__(self):
        self.robot = _Robot()
        self.terrain = _Terrain()

    def __getitem__(self, name):
        return getattr(self, name)


class _Env:
    def __init__(self):
        self.device = torch.device("cpu")
        self.num_envs = 1
        self.common_step_counter = 0
        self.episode_length_buf = torch.tensor([2])
        self.scene = _Scene()


def _set_wheels(env, x, y, z=0.103):
    d = env.scene.robot.data
    for i in range(4):
        d.body_link_pos_w[0, 1 + i] = torch.tensor([x[i], y[i], z])
    return d


def _run(env, x, y, root_x=1.0):
    d = _set_wheels(env, x, y)
    d.root_link_pos_w[0, 0] = root_x
    env.common_step_counter += 1
    return _part2_terrain_state(env, terrain_kind="high_wall")


class TestWallBehavior(unittest.TestCase):
    def test_valid_crossing_is_success(self):
        env = _Env()
        _run(env, [1.5] * 4, [0.0, 0.0, 0.0, 0.0], root_x=1.0)
        success, violation = _run(env, [2.2] * 4, [0.0, 0.0, 0.0, 0.0], root_x=2.3)
        self.assertTrue(env._part2_wall_wheel_crossed.all())
        self.assertFalse(violation.item())
        self.assertTrue(success.item())

    def test_side_bypass_is_termination(self):
        env = _Env()
        _run(env, [1.5] * 4, [0.0, 0.0, 0.0, 0.0], root_x=1.0)
        success, violation = _run(env, [2.2] * 4, [0.0, 2.0, 0.0, 0.0], root_x=2.3)
        self.assertTrue(env._part2_wall_side_bypass.item())
        self.assertTrue(violation.item())
        self.assertFalse(success.item())

    def test_single_wheel_missing_is_not_success(self):
        env = _Env()
        _run(env, [1.5] * 4, [0.0, 0.0, 0.0, 0.0], root_x=1.0)
        success, violation = _run(env, [2.2, 2.2, 2.2, 1.5], [0.0, 0.0, 0.0, 0.0], root_x=2.3)
        self.assertFalse(env._part2_wall_wheel_crossed.all())
        self.assertFalse(violation.item())
        self.assertFalse(success.item())


if __name__ == "__main__":
    unittest.main()
