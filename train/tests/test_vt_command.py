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
        self.assertNotIn("wheel_rotation_follow", cfg.rewards)
        self.assertNotIn("wheel_plane_level", cfg.rewards)
        self.assertEqual(cfg.rewards["track_linear_velocity"].weight, 3.0)

    def test_edge_variant(self) -> None:
        cfg = blackw_vt_edge_env_cfg()
        self.assertEqual(cfg.scene.terrain.terrain_type, "generator")
        self.assertNotIn("illegal_contact", cfg.terminations)
        self.assertIn("bad_orientation", cfg.terminations)
        self.assertIn("out_of_terrain_bounds", cfg.terminations)


if __name__ == "__main__":
    unittest.main()
