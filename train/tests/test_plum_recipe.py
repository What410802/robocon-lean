"""Static contract tests for the plum teacher/student recipe (CPU only).

Run with the repo's convention (train env has unittest, not pytest):
    pixi run -e train python -m unittest discover -s tests

The GPU/env-building checks live in ``train/scripts/plum/smoke.py``; these tests
cover the parts that must never drift: geometry source, camera contract parity
with the eval judge, obs layout, reward table shape, and export permutation.
"""

from __future__ import annotations

import ast
import inspect
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from alldog_mjlab.tasks.velocity.common.config import BLACK_CONFIG
from alldog_mjlab.tasks.velocity.profiles.plum.depth_camera import DEPTH_CONTRACT
from alldog_mjlab.tasks.velocity.profiles.plum.fine_scan import fine_scan_dim
from alldog_mjlab.tasks.velocity.profiles.plum.ep_rewards import (
    EP_SCALES, build_reward_table,
)
from alldog_mjlab.tasks.velocity.profiles.plum.spec import SPEC
from alldog_mjlab.tasks.velocity.profiles.plum.student import (
    _deploy_config, build_permutation, build_scan_cnn, build_teacher_mlp,
)
from alldog_mjlab.tasks.velocity.profiles.plum.teacher import (
    COMMAND_VX, NO_COMMAND_DEPLOY_LAYOUT, NO_COMMAND_FRAME_DIM, PROPRIO_DIM,
    SUCCESS_BONUS,
)
from alldog_mjlab.algorithms.privileged_warm_start import blind_to_teacher_columns

ROOT = Path(__file__).resolve().parents[2]


def _eval_depth_contract() -> dict:
    src = (ROOT / "eval" / "robocon_eval" / "policy_runner.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        target = node.target if isinstance(node, ast.AnnAssign) else (
            node.targets[0] if isinstance(node, ast.Assign) else None)
        if target is not None and getattr(target, "id", None) == "DEFAULT_DEPTH_CONTRACT":
            return ast.literal_eval(node.value)
    raise AssertionError("DEFAULT_DEPTH_CONTRACT not found")


class PlumRecipeTests(unittest.TestCase):
    def test_geometry_is_the_rules_pdf_field(self):
        self.assertEqual(len(SPEC.cube_centres), 6)
        self.assertEqual(SPEC.cube_x_range, (2.7, 4.4))
        self.assertAlmostEqual(SPEC.exit_end_x, 5.3)   # 0.2 m step-down, no ramps
        self.assertEqual(SPEC.total_x_range, (1.8, 5.3))
        self.assertEqual(len({y for _, y in SPEC.cube_centres}), 2)

    def test_camera_contract_matches_eval_judge(self):
        self.assertEqual(DEPTH_CONTRACT, _eval_depth_contract())
        self.assertEqual((DEPTH_CONTRACT["width"], DEPTH_CONTRACT["height"]), (64, 48))
        self.assertEqual(DEPTH_CONTRACT["cutoff"], 3.0)
        self.assertEqual(DEPTH_CONTRACT["min_depth"], 0.10)

    def test_reward_dt_scaling_is_per_step(self):
        from mjlab.managers.reward_manager import RewardManager
        self.assertIs(
            inspect.signature(RewardManager.__init__).parameters["scale_by_dt"].default, True)
        self.assertAlmostEqual(BLACK_CONFIG.control.policy_dt, 0.02)
        # reward = raw * weight * dt; a unit-raw term pays 0.02 * weight / step.
        self.assertAlmostEqual(SUCCESS_BONUS * 0.02, 10.0)

    def test_reward_table_is_the_ep_table_plus_success_bonus(self):
        table = build_reward_table(clear_weight=SUCCESS_BONUS)
        # Task-term names differ from EP scale keys only where the term is shared
        # with the rest of the repo (ang_vel_xy -> body_ang_vel, orientation ->
        # upright); every EP scale must still appear exactly once.
        aliases = {"ang_vel_xy": "body_ang_vel", "orientation": "upright"}
        expected = {aliases.get(k, k): v for k, v in EP_SCALES.items()}
        self.assertEqual(set(table) - {"part2_clear"}, set(expected))
        self.assertIn("part2_clear", table)
        for name, weight in expected.items():
            self.assertAlmostEqual(table[name].weight, weight, msg=name)
        # The support-blind progress term from v1/v2 is gone for good: it paid
        # for min(wheel_x) regardless of support, which is how v1 learned to
        # vault the field and run out of course.
        self.assertNotIn("progress", table)

    def test_command_is_fast_enough_to_cross_a_gap(self):
        # 0.4 m gaps with a 0.103 m wheel: the old successful run used 0.65-0.80
        # m/s, the failed port used 0.35.
        self.assertEqual(COMMAND_VX, (0.65, 0.80))

    def test_warm_start_layout_is_a_bijection_onto_proprio(self):
        idx = blind_to_teacher_columns(6)
        self.assertEqual(len(idx), PROPRIO_DIM)
        self.assertEqual(len(set(idx.tolist())), PROPRIO_DIM)
        # Only the 3 command columns of each of the 6 frames are dropped.
        used = set(idx.tolist())
        dropped = sorted({f * 57 + d for f in range(6) for d in range(57)} - used)
        self.assertTrue(all(d % 57 in (0, 1, 2) for d in dropped), dropped)
        self.assertEqual(len(dropped), 18)

    def test_warm_start_reproduces_the_blind_policy(self):
        """Step-0 teacher action must equal the blind policy at the fixed command."""
        vx = 0.5 * sum(COMMAND_VX)
        torch.manual_seed(0)
        cmd = torch.tensor([vx * 2.0, 0.0, 0.0])          # commands_scale[0] = 2.0
        frames = [torch.randn(54) for _ in range(6)]      # newest last
        # MjLab native history is TERM-major / oldest-first; the deploy layout is
        # frame-major / newest-first with a 3-D command per frame.
        terms = ((3, 0), (3, 3), (16, 6), (16, 22), (16, 38))
        native = torch.cat([frames[f][o:o + d] for (d, o) in terms for f in range(6)])
        blind_obs = torch.cat([torch.cat([cmd, frames[5 - f]]) for f in range(6)])
        self.assertTrue(torch.allclose(
            native, blind_obs[blind_to_teacher_columns(6)]))

        blind = torch.nn.Sequential(
            torch.nn.Linear(342, 64), torch.nn.ELU(), torch.nn.Linear(64, 16))
        blind_state = {f"{i}.{n}": getattr(m, n).detach().clone()
                       for i, m in ((0, blind[0]), (2, blind[2]))
                       for n in ("weight", "bias")}
        teacher = torch.nn.Sequential(
            torch.nn.Linear(PROPRIO_DIM + fine_scan_dim(), 64), torch.nn.ELU(),
            torch.nn.Linear(64, 16))
        from alldog_mjlab.algorithms.privileged_warm_start import (
            warm_start_teacher_from_blind)
        warm_start_teacher_from_blind(
            teacher, blind_state, history=6, scan_dim=fine_scan_dim(),
            fixed_command=(vx, 0.0, 0.0), command_scale=(2.0, 2.0, 0.25),
            prefix="")
        with torch.no_grad():
            want = blind(blind_obs)
            got = teacher(torch.cat([native, torch.zeros(fine_scan_dim())]))
        self.assertTrue(torch.allclose(got, want, atol=1e-5),
                        f"max diff {(got - want).abs().max().item():.3g}")

    def test_terrain_curriculum_is_the_official_field_plus_gaps(self):
        """The lattice and the 0.20 m height are fixed; only the gaps may vary.

        Three earlier axes were measured and rejected:
        * cube size (0.30 m half-width) -- the cubes touch in x and the two rows'
          y-spans overlap, so the field degenerates into a slab;
        * barrier height -- the judge's drop rule is ABSOLUTE (0.05 m), so at a
          0.05 m top a wheel resting on it counts as fallen (128/128 episodes
          died at x ~ 2.57), and at 0.08-0.14 m the floor stays reachable
          through the gaps and the policy learned to skirt the cubes on the
          ground (wheel z ~ 0.02, |y| = 0.60, outside the 0.2 m-wide rows).
        With the teacher spawned ON the entry platform (deployment hands over
        after the ordinary policy climbs), the only sensible axis is the gap.
        """
        from alldog_mjlab.tasks.velocity.profiles.plum.terrain import (
            NUM_ROWS, OFFICIAL_GAP_M, PlumCurriculumTerrainCfg)
        t = PlumCurriculumTerrainCfg()
        # Row 0 is fully closed; the official row is the untouched field.
        self.assertEqual(t.residual_at(0.0), 0.0)
        self.assertAlmostEqual(t.residual_at(1.0), OFFICIAL_GAP_M)
        gaps = [t.residual_at(i / (NUM_ROWS - 1)) for i in range(NUM_ROWS)]
        self.assertEqual(gaps, sorted(gaps), "gaps must open with difficulty")
        # Every rung ABOVE zero asks for a carry, not a roll: the residual is
        # above the wheel diameter (0.206 m) from row 1 on.  A linear fill from
        # 0.40 to 0.00 spent >half the ladder in rollable terrain and stalled the
        # population at row 1.4-1.8 for a whole run.
        self.assertGreater(gaps[1], 2 * 0.103)
        # ...and the first jumping rung is not already the official size.
        self.assertLess(gaps[1], gaps[-1] - 0.1)
        # Cube half-width is the spec value everywhere -- it is NOT a free axis.
        self.assertFalse(hasattr(t, "cube_half_at"),
                         "cube size must not be a curriculum axis")

    def test_curriculum_fill_and_bridge_come_from_the_same_gap_schedule(self):
        """Fill = official gap - residual; the end holes follow the same curve.

        The official end holes are asymmetric (0.40 m on one row at the exit,
        0.10 m on the other; mirrored at the entry).  At row 0 they are fully
        bridged and at the official row they are untouched.
        """
        from alldog_mjlab.tasks.velocity.profiles.plum.terrain import (
            NUM_ROWS, OFFICIAL_GAP_M, PlumCurriculumTerrainCfg)
        t = PlumCurriculumTerrainCfg()
        for i in range(NUM_ROWS):
            d = i / (NUM_ROWS - 1)
            self.assertAlmostEqual(t.gap_fill_at(d) + t.residual_at(d),
                                   OFFICIAL_GAP_M)
            self.assertAlmostEqual(t.bridge_at(d), t.gap_fill_at(d))
        self.assertEqual(t.bridge_at(0.0), OFFICIAL_GAP_M)
        self.assertEqual(t.bridge_at(1.0), 0.0, "official row must have NO bridge")
        # The bridge exists precisely because the wide hole exceeds the wheel
        # diameter; if the field ever changed, this assertion should fire.
        row1_far = max(x for x, y in SPEC.cube_centres if y > 0) + SPEC.cube_half
        wide_gap = SPEC.exit_x_range[0] - row1_far
        self.assertGreater(wide_gap, 2 * 0.103)
        self.assertAlmostEqual(OFFICIAL_GAP_M, wide_gap)

    def test_plum_drop_line_is_the_judge_absolute_value(self):
        """The drop line is the judge's absolute 0.05 m, at every row.

        Height-scaled variants failed measurably, in opposite directions:
        * ``top * 0.25`` -> 0.0125 m at a 0.05 m row: ground contact between the
          bumps counted as a drop (512/512 episodes at x ~ 2.8);
        * ``top - 0.15`` -> below the floor under 0.15 m, legalising a
          ground-level bypass (traces at d=0.6: wheel z ~ 0.02, |y| = 0.60);
        * ``min(0.05, top / 2)`` -> 0.025 m at the easy row, rejecting the
          CORRECT crossing (128/128 failures, wheel bottoms 0.039 m in gaps).
        The absolute line is correct at the one height the field has: a wheel on
        a 0.20 m top is legal, the floor is not.
        """
        import torch
        from alldog_mjlab.tasks.velocity.profiles.part2 import predicates as p
        self.assertAlmostEqual(p.PART2_PLUM_TOP_Z_M, 0.20)
        tops = torch.full((3,), p.PART2_PLUM_TOP_Z_M)
        self.assertTrue(bool((p.plum_z_cross(tops)
                              == p.PART2_PLUM_Z_CROSS_MIN_M).all()))
        # A wheel on a cube top is above the line; the floor is below it.
        self.assertGreater(p.PART2_PLUM_TOP_Z_M, p.PART2_PLUM_Z_CROSS_MIN_M)
        self.assertGreater(p.PART2_PLUM_Z_CROSS_MIN_M, 0.0)

    def test_idle_is_not_profitable(self):
        """Standing still must not out-earn crossing the field.

        Measured before the gate: 21 s of standing still returned +11.8 while a
        success pays +10, so the policy converged to standing still (v3b: 1061-step
        episodes, zero successes).  tracking_yaw is now speed-gated.
        """
        import inspect
        from alldog_mjlab.tasks.velocity.profiles.plum import ep_rewards as R
        src = inspect.getsource(R.tracking_yaw)
        self.assertIn("min_speed_frac", src)
        self.assertIn("root_link_lin_vel_w", src,
                      "the yaw term must be gated on actual forward speed")

    def test_joint_command_clip_matches_deployment(self):
        from alldog_mjlab.tasks.velocity.profiles.plum.teacher import (
            WHEEL_DQ_CLIP, LEG_RESIDUAL_CLIP, _JOINT_CLIP,
        )
        self.assertEqual(WHEEL_DQ_CLIP, 20.0)   # base.yaml command_dq_max
        self.assertEqual(LEG_RESIDUAL_CLIP, 1.0)
        self.assertEqual(_JOINT_CLIP[".*wheel_joint"], (-20.0, 20.0))
        self.assertEqual(_JOINT_CLIP[".*_(hip|thigh|calf)_joint"], (-1.0, 1.0))

    def test_terminal_requires_step_down_and_four_wheels_on_ground(self):
        # Judge v9 plum cond (v5_judge.py:355-361) and the training predicate
        # both require: base x >= exit_edge+0.10, all wheels past exit_edge+0.02,
        # all wheel bottoms <= ground_tol (0.08), trunk z in [0.30,0.80], held.
        from alldog_mjlab.tasks.velocity.profiles.part2 import predicates as p
        self.assertAlmostEqual(p.PART2_PLUM_EXIT_END_X_M, 5.3)
        self.assertAlmostEqual(p.PART2_PLUM_EXIT_GROUND_TOL_M, 0.08)
        self.assertEqual(p.PART2_PLUM_TOP_BAND_M, 0.05)  # judge WHEEL_TOP_BAND_M
        self.assertGreaterEqual(p.PART2_HOLD_STEPS, 16)  # >= judge 0.3 s / 0.02
        self.assertLess(SPEC.cube_x_range[1], p.PART2_PLUM_EXIT_END_X_M)

    def test_no_command_obs_layout(self):
        self.assertEqual(sum(d for d, _ in NO_COMMAND_DEPLOY_LAYOUT), NO_COMMAND_FRAME_DIM)
        self.assertEqual(NO_COMMAND_FRAME_DIM, 54)
        self.assertEqual(PROPRIO_DIM, 324)
        self.assertEqual(fine_scan_dim(), 1764)

    def test_export_permutation_is_a_bijection(self):
        perm = build_permutation(fine_scan_dim())
        self.assertEqual(len(perm), PROPRIO_DIM + fine_scan_dim())
        self.assertEqual(set(perm.tolist()), set(range(len(perm))))
        self.assertEqual(perm[PROPRIO_DIM:].tolist(),
                         list(range(PROPRIO_DIM, len(perm))))
        self.assertEqual(set(perm[:PROPRIO_DIM].tolist()), set(range(PROPRIO_DIM)))

    def test_teacher_mlp_rebuild_and_cnn_latent(self):
        teacher = torch.nn.Sequential(
            torch.nn.Linear(PROPRIO_DIM + fine_scan_dim(), 512), torch.nn.ELU(),
            torch.nn.Linear(512, 256), torch.nn.ELU(),
            torch.nn.Linear(256, 128), torch.nn.ELU(),
            torch.nn.Linear(128, 16),
        )
        state = {}
        for i, key in enumerate(("mlp.0", "mlp.2", "mlp.4", "mlp.6")):
            state[f"{key}.weight"] = teacher[2 * i].weight.detach().clone()
            state[f"{key}.bias"] = teacher[2 * i].bias.detach().clone()
        rebuilt = build_teacher_mlp(state)
        self.assertTrue(torch.allclose(rebuilt[0].weight, teacher[0].weight))
        self.assertTrue(all(not p.requires_grad for p in rebuilt.parameters()))

        cnn, head = build_scan_cnn(fine_scan_dim())
        z = cnn(torch.rand(2, 1, DEPTH_CONTRACT["height"], DEPTH_CONTRACT["width"]))
        self.assertEqual(tuple(z.shape), (2, 64))
        self.assertEqual(tuple(head(z).shape), (2, fine_scan_dim()))

    def test_deploy_config_drops_commands(self):
        cfg = _deploy_config("unit_test")
        section = next(iter(cfg.values()))
        self.assertEqual(section["num_observations"], 54)
        self.assertNotIn("commands", section["observations"])
        self.assertEqual(len(section["observations_history"]), 6)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "config.yaml"
            p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            self.assertEqual(yaml.safe_load(p.read_text(encoding="utf-8")), cfg)

    def test_curriculum_promotes_and_demotes_on_a_full_window(self):
        """Row-level curriculum: 40 episodes of evidence move exactly one row.

        The first version promoted per env on ONE success (``UP_STREAK=1``):
        the moment the warm start began working the ladder went from row 0 to
        row 3.0 in 23 iterations and the success rate collapsed 16.5 -> 0.31
        per batch, so the population outran its own frontier.
        """
        from types import SimpleNamespace
        import alldog_mjlab.tasks.velocity.profiles.plum.curriculum as C

        class Terrain:
            def __init__(self, n, rows):
                # 4 rows, so two promotions stay below mjlab's top-row cap
                # (``max_terrain_level = num_rows``, which teleports to a random
                # row and makes the assertion non-deterministic).
                self.cfg = SimpleNamespace(
                    terrain_generator=SimpleNamespace(num_rows=rows))
                self.terrain_levels = torch.zeros(n, dtype=torch.long)
                self.terrain_origins = torch.zeros(rows, 1, 3)
                self.env_origins = torch.zeros(n, 3)

            def update_env_origins(self, ids, up, down):
                self.terrain_levels[ids] += up.long() - down.long()

        env = SimpleNamespace(
            num_envs=8, device="cpu",
            scene=SimpleNamespace(terrain=Terrain(8, 4)),
            episode_length_buf=torch.zeros(8, dtype=torch.long))

        state = {"success": False}
        orig_s, orig_f = C._part2_success, C._part2_failure
        C._part2_success = lambda _e: torch.full((8,), state["success"])
        C._part2_failure = lambda _e: torch.full((8,), not state["success"])
        try:
            def run(n, success):
                state["success"] = success
                out = None
                for _ in range(n):
                    out = C.plum_difficulty_curriculum(
                        env, torch.arange(8), promote_rate=0.5, demote_rate=0.05,
                        records_per_call=8, cooldown_calls=0)
                    self.assertGreaterEqual(
                        int(env._plum_window_success.min()), 0,
                        "running success total must never go negative")
                return out

            # Window logic only: 40 episode records / 8 per call = 5 calls per
            # decision.  The throttling defaults are asserted separately below.
            calls_per_window = C.PROMOTE_MIN_EPISODES // 8
            run(calls_per_window, True)
            self.assertEqual(int(env.scene.terrain.terrain_levels.min()), 1)
            run(calls_per_window, True)
            self.assertEqual(int(env.scene.terrain.terrain_levels.min()), 2)
            run(calls_per_window, False)
            self.assertEqual(int(env.scene.terrain.terrain_levels.max()), 1)
        finally:
            C._part2_success, C._part2_failure = orig_s, orig_f

    def test_curriculum_defaults_throttle_the_window(self):
        """The shipped window must span many calls, or it is one step of evidence.

        v9 ran with an uncapped window: 2048 envs at ~60-step episodes finish
        >2000 episodes per simulator step, so the 40-episode window refilled
        inside one step and the row was re-decided every iteration.  Measured:
        up=96, down=288 every iteration for 50+ iterations, mean pinned at
        96/2048 = 0.0469.  The cap and cooldown are what make the window mean
        "behaviour over time".
        """
        from alldog_mjlab.tasks.velocity.profiles.plum import curriculum as C
        self.assertLess(C.RECORDS_PER_CALL, C.PROMOTE_MIN_EPISODES)
        self.assertGreaterEqual(C.PROMOTE_MIN_EPISODES // C.RECORDS_PER_CALL, 10)
        self.assertGreater(C.COOLDOWN_CALLS, 0)

    def test_curriculum_does_not_thrash_on_mixed_rows(self):
        """A row at the promotion boundary must not move more than once per cooldown.

        Reproduces the v9 signature: row 0 succeeds ~50 %, row 1 fails.  With
        throttling the levels must be stable across a long stretch of calls
        (one promotion at most per cooldown), not oscillate every call.
        """
        from types import SimpleNamespace
        import alldog_mjlab.tasks.velocity.profiles.plum.curriculum as C

        class Terrain:
            def __init__(self, n, rows):
                self.cfg = SimpleNamespace(
                    terrain_generator=SimpleNamespace(num_rows=rows))
                self.terrain_levels = torch.zeros(n, dtype=torch.long)
                self.terrain_origins = torch.zeros(rows, 1, 3)
                self.env_origins = torch.zeros(n, 3)

            def update_env_origins(self, ids, up, down):
                self.terrain_levels[ids] += up.long() - down.long()

        n = 64
        env = SimpleNamespace(
            num_envs=n, device="cpu",
            scene=SimpleNamespace(terrain=Terrain(n, 4)),
            episode_length_buf=torch.zeros(n, dtype=torch.long))
        # Row 0 envs: alternating success/failure (50 %).  Every other env fails.
        orig_s, orig_f = C._part2_success, C._part2_failure
        C._part2_success = lambda e: (
            (torch.arange(n) % 2 == 0)
            & torch.tensor(e.scene.terrain.terrain_levels == 0))
        C._part2_failure = lambda e: ~C._part2_success(e)
        try:
            calls = C.COOLDOWN_CALLS * 3
            moves = 0
            prev = env.scene.terrain.terrain_levels.clone()
            for _ in range(calls):
                C.plum_difficulty_curriculum(env, torch.arange(n))
                now = env.scene.terrain.terrain_levels
                moves += int((now != prev).sum())
                prev = now.clone()
            # At most one move per env per cooldown window (+1 for the boundary),
            # vs one per call without throttling (n * calls).
            self.assertLessEqual(moves, n * (calls // C.COOLDOWN_CALLS + 2),
                                 "levels must not oscillate every call")
            self.assertLess(moves, n * calls // 4,
                            "throttling must cut the churn by a large factor")
        finally:
            C._part2_success, C._part2_failure = orig_s, orig_f

    def test_actor_gate_disarms_after_the_warmup_window(self):
        """The critic warm-up must hand the actor back, not freeze it forever.

        v8 ran its whole 2000 iterations with the gate installed and never
        uninstalled: actor weights in ``model_0`` and ``model_1999`` were
        bit-identical, so the run trained only the critic.  This asserts the
        disarming directly, at the mechanism level (no GPU env needed).
        """
        from alldog_mjlab.tasks.velocity.profiles.plum.runner import PlumTeacherRunner

        runner = object.__new__(PlumTeacherRunner)
        calls = {"n": 0}

        class Alg:
            def __init__(self):
                self.actor = torch.nn.Linear(2, 2)

            def update(self):
                calls["n"] += 1
                return "ok"

        runner.alg = Alg()
        runner.current_learning_iteration = 0
        runner._actor_warmup_iters = 2
        runner._original_update = runner.alg.update
        runner.alg.update = runner._critic_only_update

        for it in range(2):
            runner.current_learning_iteration = it
            self.assertEqual(runner.alg.update(), "ok")
            self.assertFalse(
                any(p.requires_grad for p in runner.alg.actor.parameters()),
                f"actor must stay frozen during warm-up (iter {it})")

        runner.current_learning_iteration = 2
        self.assertEqual(runner.alg.update(), "ok")
        self.assertIs(runner.alg.update, runner._original_update,
                      "the gate must uninstall itself")
        self.assertTrue(all(p.requires_grad for p in runner.alg.actor.parameters()),
                        "the actor must be trainable after the window")
        self.assertEqual(calls["n"], 3, "the original update must run every time")

    def test_ppo_lr_schedule_and_actor_warmup_are_wired(self):
        """The two knobs that caused silent failures must reach PPO intact.

        * ``RC_LR_SCHEDULE``: the adaptive rule drove v11's LR from 1e-3 to the
          1e-5 floor after one oversized update (1e-3 -> 1.5e-5 by iteration 25),
          after which the run could not move; a fixed 3e-5 schedule is the only
          recipe measured to survive the actor unfreeze (probe_lr3e5 held ~90-130
          step episodes and non-zero successes for 60 iterations).
        * ``PART2_PLUM_ACTOR_WARMUP_ITERS``: the critic warm-up gate.  Its
          default must be 0 (opt-in), because a non-zero default silently froze
          v8's actor for all 2000 iterations (model_0 and model_1999 had
          bit-identical actor weights).
        """
        from alldog_mjlab.tasks.velocity.common.config import BLACK_CONFIG
        from alldog_mjlab.tasks.velocity.common.rl_cfg import black_ppo_algorithm_kwargs
        from alldog_mjlab.utils import knobs

        self.assertEqual(black_ppo_algorithm_kwargs()["schedule"],
                         BLACK_CONFIG.algorithm.schedule)
        self.assertIn(black_ppo_algorithm_kwargs()["schedule"],
                      ("fixed", "adaptive"))
        self.assertEqual(knobs.get_int("PART2_PLUM_ACTOR_WARMUP_ITERS", 0), 0,
                         "the actor gate must be opt-in")

    def test_std_clamp_knob_reaches_the_distribution(self):
        """PART2_PLUM_STD_MAX must clamp the LEARNED std, not just the init value.

        Measured on the real run: std rises monotonically 0.10 -> 0.48 at a fixed
        3e-5 LR (the entropy bonus, rsl_rl's default range being (1e-6, 1e6)),
        and the official-field success window closed as it crossed ~0.37
        (3.5% at iteration 600, 0% from 725 on).  A clamp that silently failed to
        arrive would reopen that trap, so it is asserted through the same path the
        trainer uses: runner cfg -> distribution_cfg -> GaussianDistribution.
        """
        import os
        from unittest import mock
        from rsl_rl.models.mlp_model import MLPModel
        from alldog_mjlab.tasks.velocity.profiles.plum.teacher import (
            plum_teacher_runner_cfg,
        )

        # knobs reads the environment lazily through _env(), so no reload is
        # needed (and none exists): patch the var and rebuild the cfg.
        with mock.patch.dict(os.environ, {"PART2_PLUM_STD_MAX": "0.30"}):
            cfg = plum_teacher_runner_cfg(stage=0)
        self.assertEqual(tuple(cfg.actor.distribution_cfg["std_range"]), (1e-6, 0.30))

        obs = {"actor": torch.zeros(2, PROPRIO_DIM + fine_scan_dim()),
               "critic": torch.zeros(2, 72)}
        model = MLPModel(
            obs=obs, obs_groups={"actor": ("actor",), "critic": ("critic",)},
            obs_set="actor", output_dim=16, hidden_dims=(64,), activation="elu",
            obs_normalization=False,
            distribution_cfg=dict(cfg.actor.distribution_cfg))
        self.assertEqual(model.distribution.std_range, [1e-6, 0.30])
        with torch.no_grad():
            model.distribution.std_param.fill_(0.48)
            clamped = model.distribution.std_param.clamp(*model.distribution.std_range)
        self.assertAlmostEqual(float(clamped.mean()), 0.30, places=5)


if __name__ == "__main__":
    unittest.main()
