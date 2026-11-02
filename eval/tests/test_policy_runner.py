"""Eval-side policy runner contract (paths, joint mapping, obs contract, judge loop).

These tests are behavioural: the old runner passed a 5-step smoke test while
silently swapping blackW's RL/RR legs, so shapes/finiteness alone are not enough.
They exercise the real MuJoCo + TorchScript stack and skip when the (gitignored)
weights or generated scenes are not present locally.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from robocon_eval.part2 import official
from robocon_eval.policy_runner import (
    BASE,
    ContractError,
    PolicyRunner,
    _load_level,
    _scene_root,
)

BLACK_SCENE = BASE / "demo/robots/black/scene_part1.xml"
BLACK_POLICY = BASE / "demo/rl_sar/policy/black/himloco"
BLACKW_POLICY = BASE / "demo/rl_sar/policy/blackW/flat"
LEVEL_KEY = "high_platform_10cm"
SCENE = official.level_scene_path(official.get_level(LEVEL_KEY)) \
    if official.LEVELS_PATH.is_file() else None


class TestPathsAndManifest(unittest.TestCase):
    def test_repo_layout_paths(self):
        self.assertTrue((BASE / "eval/robocon_eval/rules.py").is_file())
        self.assertTrue((BASE / "demo/robots").is_dir())
        self.assertTrue((BASE / "demo/rl_sar/policy").is_dir())

    @unittest.skipUnless((BASE / "demo/robots/blackW/robot.yaml").is_file(),
                         "robot manifest missing")
    def test_scene_roots_are_manifest_configurable(self):
        import yaml

        manifest = yaml.safe_load(
            (BASE / "demo/robots/blackW/robot.yaml").read_text())
        part1 = _scene_root(manifest, "part1", None)
        part2 = _scene_root(manifest, "part2", None)
        self.assertEqual(part1, BASE / "demo/robots/blackW")
        self.assertEqual(part2, BASE / "demo/scenes/part2/generated")

    @unittest.skipUnless(official.LEVELS_PATH.is_file(), "manifest missing")
    def test_part1_scene_exists_for_both_robots(self):
        for robot in ("black", "blackW"):
            self.assertTrue((BASE / f"demo/robots/{robot}/scene_part1.xml").is_file())


@unittest.skipUnless(BLACK_SCENE.is_file() and BLACK_POLICY.is_dir(),
                     "black assets/weights missing")
class TestBlackJointMapping(unittest.TestCase):
    def test_black_model_mapping_is_identity(self):
        runner = PolicyRunner("black", 1, "himloco", None, steps=1)
        try:
            self.assertEqual(runner.mapping.tolist(), list(range(12)))
            self.assertEqual(runner.contract_mapping.tolist(), list(range(12)))
        finally:
            runner.close()

    def test_black_part2_is_rejected(self):
        with self.assertRaises(ContractError):
            PolicyRunner("black", 2, "himloco", LEVEL_KEY, steps=1)


@unittest.skipUnless(SCENE is not None and SCENE.is_file() and BLACKW_POLICY.is_dir(),
                     "blackW assets/weights missing")
class TestBlackWContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = PolicyRunner("blackW", 2, "flat", LEVEL_KEY, steps=1)

    @classmethod
    def tearDownClass(cls):
        cls.runner.close()

    def test_mapping_is_policy_to_actuator_order(self):
        runner = self.runner
        self.assertEqual(runner.mapping.size, 16)
        # policy order is FL FR RL RR; the MJCF actuator order is FL FR RR RL
        self.assertEqual(runner.mapping[8], 12)
        self.assertEqual(runner.mapping[12], 8)
        self.assertEqual(runner.act_ids[8], 12)
        runner.reset()
        q = np.array([runner.data.qpos[a] for a in runner.qpos_adr])
        np.testing.assert_allclose(q, runner.q_default)
        self.assertAlmostEqual(float(runner.data.qpos[runner.qpos_adr[8]]), -0.05)
        self.assertAlmostEqual(float(runner.data.qpos[runner.qpos_adr[12]]), 0.05)

    def test_observation_dim_matches_deployment_contract(self):
        self.assertEqual(self.runner.obs_dim, 57)
        self.assertEqual(self.runner.num_observations, 57)
        self.assertEqual(self.runner.history_len, 6)
        self.assertEqual(self.runner.substeps, 10)   # 20 ms policy / 2 ms physics
        self.assertEqual(self.runner.wheel_indices.tolist(), [3, 7, 11, 15])

    def test_level_scene_and_spawn(self):
        self.assertEqual(self.runner.level["key"], LEVEL_KEY)
        self.assertTrue(self.runner.scene.is_file())
        self.assertEqual(self.runner.support_frame, "wheel_lowest")
        self.assertGreaterEqual(float(self.runner.spawn[2]), 0.5)

    def test_spawn_clears_the_floor(self):
        import mujoco
        import yaml

        entry = _load_level(LEVEL_KEY)
        model = mujoco.MjModel.from_xml_path(str(self.runner.scene))
        data = mujoco.MjData(model)
        radius = 0.103
        bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_foot")
                  for leg in ("FL", "FR", "RL", "RR")]
        manifest = yaml.safe_load(
            (BASE / "demo/robots/blackW/robot.yaml").read_text())
        qadr = [model.jnt_qposadr[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in manifest["joint_names"]]
        data.qpos[:] = model.qpos0
        data.qpos[qadr] = np.asarray(manifest["default_dof_pos"])
        data.qpos[2] = float(entry["spawn_default_pose"][2])
        mujoco.mj_forward(model, data)
        bottom = min(float(data.xpos[body][2]) - radius for body in bodies)
        # default pose must not start inside the floor, and must not float either
        self.assertGreater(bottom, -0.04, bottom)
        self.assertLess(bottom, 0.06, bottom)


@unittest.skipUnless(SCENE is not None and SCENE.is_file() and BLACKW_POLICY.is_dir(),
                     "blackW assets/weights missing")
class TestRunnerJudgeLoop(unittest.TestCase):
    def test_part2_trace_meta_and_judge(self):
        from robocon_eval.policy_runner import run

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "trace.jsonl"
            result = run("blackW", 2, "flat", LEVEL_KEY, out, steps=10)
            self.assertTrue(result["finite"])
            self.assertIn(result["judge"]["status"], ("running", "success", "fail"))
            self.assertEqual(result["part"], 2)
            self.assertEqual(result["category"], "high_platform")
            meta = json.loads(out.read_text().splitlines()[0])
            self.assertEqual(meta["part"], 2)
            self.assertEqual(meta["schema"], 1)
            self.assertEqual(meta["level"]["key"], LEVEL_KEY)
            self.assertEqual(meta["support_frame"], "wheel_lowest")
            self.assertIn("policy_sha256", meta)
            self.assertIn("scene_sha256", meta)


@unittest.skipUnless(BLACK_SCENE.is_file() and (BLACK_POLICY / "best.pt").is_file(),
                     "black assets/weights missing")
class TestBlackPart1Loop(unittest.TestCase):
    def test_part1_trace_judges(self):
        from robocon_eval.policy_runner import run

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "part1.jsonl"
            result = run("black", 1, "himloco", None, out, steps=10)
            self.assertTrue(result["finite"])
            self.assertIn("states", result["judge"])
            meta = json.loads(out.read_text().splitlines()[0])
            self.assertEqual(meta["part"], 1)


if __name__ == "__main__":
    unittest.main()
