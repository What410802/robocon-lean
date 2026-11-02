"""Contract tests for the velocity-tracking primitive task (CPU only)."""

from __future__ import annotations

import unittest

from alldog_mjlab.tasks.velocity.blackw.env_cfgs import (
    blackw_vt_edge_env_cfg,
    blackw_vt_env_cfg,
    blackw_vt_locked_env_cfg,
)
from alldog_mjlab.tasks.velocity.profiles.vt import assembly
from alldog_mjlab.tasks.velocity.profiles.vt.command import (
    VTPrimitiveVelocityCommandCfg,
)

TRANSMISSION_KEYS = (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)


class TestVTCommand(unittest.TestCase):
    def test_envelope(self) -> None:
        cfg = blackw_vt_env_cfg()
        cmd = cfg.commands["twist"]
        self.assertIsInstance(cmd, VTPrimitiveVelocityCommandCfg)
        self.assertEqual(cmd.ranges.lin_vel_x, (assembly.VX_MIN, assembly.VX_MAX))
        self.assertEqual(cmd.ranges.lin_vel_y, (0.0, 0.0))
        self.assertEqual(cmd.ranges.ang_vel_z, (-assembly.WZ_MAX, assembly.WZ_MAX))
        self.assertEqual(cmd.rel_forward_envs, 0.0)
        self.assertEqual(cmd.rel_heading_envs, 0.0)
        self.assertAlmostEqual(cmd.rel_standing_envs, assembly.STANDING_FRAC)
        self.assertGreater(cmd.pure_yaw_frac, 0.0)
        self.assertGreater(cmd.pure_vx_frac, 0.0)

    def test_rewards(self) -> None:
        cfg = blackw_vt_env_cfg()
        w = {k: v.weight for k, v in cfg.rewards.items()}
        self.assertEqual(w["track_linear_velocity"], assembly.TRACK_LIN_W)
        self.assertEqual(w["track_angular_velocity"], assembly.TRACK_ANG_W)
        self.assertEqual(w["track_linear_velocity_l2"], -assembly.TRACK_ERR_L2_W)
        self.assertEqual(w["upright"], -assembly.UPRIGHT_W)
        self.assertIn("pose_dofpos", w)
        self.assertIn("pose_rotdrift", w)
        self.assertIn("wheel_plane_level", w)
        self.assertIn("wheel_rotation_follow", w)
        self.assertAlmostEqual(w["wheel_plane_level"], assembly.WHEEL_PLANE_W)
        self.assertAlmostEqual(w["wheel_rotation_follow"], assembly.WHEEL_FOLLOW_W)
        # RecoveryLoco contact/plane terms must actually be wired: an earlier
        # version passed no asset_cfg and silently indexed all 53 geoms, so both
        # terms evaluated to 0 for the whole run.
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        plane = cfg.rewards["wheel_plane_level"]
        self.assertIsInstance(plane.params.get("asset_cfg"), SceneEntityCfg)
        self.assertEqual(list(plane.params["asset_cfg"].geom_names),
                         list(assembly.WHEEL_COLLISION_GEOMS))
        self.assertNotIn("base_height", w)

    def test_events_and_terminations(self) -> None:
        cfg = blackw_vt_env_cfg()
        self.assertIn("wheel_friction", cfg.events)
        self.assertIn("push_robot", cfg.events)
        self.assertEqual(
            cfg.events["push_robot"].params["velocity_range"]["x"],
            (-assembly.PUSH_VEL_XY, assembly.PUSH_VEL_XY),
        )
        self.assertIn("illegal_contact", cfg.terminations)

    def test_locked_variant(self) -> None:
        cfg = blackw_vt_locked_env_cfg()
        self.assertEqual(cfg.actions["wheel_leg_fl"].__class__.__name__,
                         "LockedWheelLegActionCfg")
        self.assertIn("wheel_lock", cfg.rewards)
        self.assertIn("feet_air_time", cfg.rewards)
        self.assertIn("gait_phase", cfg.rewards)
        self.assertIn("foot_lift", cfg.rewards)
        self.assertIn("support_extent", cfg.rewards)
        self.assertIn("handover_ready", cfg.rewards)
        self.assertIn("foot_stance_time", cfg.rewards)
        self.assertNotIn("wheel_rotation_follow", cfg.rewards)
        self.assertNotIn("wheel_plane_level", cfg.rewards)
        self.assertEqual(cfg.rewards["track_linear_velocity"].weight, 3.0)
        # Penalty-shaped placement/support terms must be negative; a positive
        # weight here was the bug that rewarded splaying the feet.
        for name in ("foot_landing_error", "foot_swing_target", "support_extent",
                     "support_area", "support_center", "handover_ready",
                     "foot_stance_time", "foot_air_time_penalty"):
            self.assertLessEqual(cfg.rewards[name].weight, 0.0, name)

    def test_locked_support_box_grows_with_command(self) -> None:
        import torch
        from alldog_mjlab.tasks.velocity.profiles.vt.locked import (
            LOCKED_HALF_X,
            LOCKED_HALF_Y,
            support_box_half_extents,
        )

        zero = torch.zeros(1, 3)
        hx0, hy0 = support_box_half_extents(zero)
        self.assertAlmostEqual(float(hx0), LOCKED_HALF_X + 0.015, places=5)
        self.assertAlmostEqual(float(hy0), LOCKED_HALF_Y + 0.015, places=5)
        cmd = torch.tensor([[0.5, 0.2, 0.0]])
        hx, hy = support_box_half_extents(cmd)
        self.assertGreater(float(hx), float(hx0))
        self.assertGreater(float(hy), float(hy0))
        yaw = torch.tensor([[0.0, 0.0, 0.3]])
        hx_yaw, hy_yaw = support_box_half_extents(yaw)
        self.assertGreater(float(hx_yaw), float(hx0))
        self.assertGreater(float(hy_yaw), float(hy0))
        # Caps keep the box inside the 1 m platform budget.
        big = torch.tensor([[2.0, 2.0, 2.0]])
        hx_big, hy_big = support_box_half_extents(big)
        self.assertLessEqual(float(hx_big), 0.30 + 1e-6)
        self.assertLessEqual(float(hy_big), 0.30 + 1e-6)

    def test_edge_variant(self) -> None:
        cfg = blackw_vt_edge_env_cfg()
        self.assertEqual(cfg.scene.terrain.terrain_type, "generator")
        self.assertNotIn("illegal_contact", cfg.terminations)
        self.assertIn("bad_orientation", cfg.terminations)
        self.assertIn("out_of_terrain_bounds", cfg.terminations)


if __name__ == "__main__":
    unittest.main()
