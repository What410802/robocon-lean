# P1 — plum depth student: judge-physics teacher parity + DAgger with state/depth randomization

**Date**: 2026-10-31  **Author**: teammate `p1-dagger` (task-6)  **Scope**: simulation only
**Judge**: `v7.6-plum-exit-2026-10-31` (official `evaluate-policy`, plum_1, fixed command `[0.5, 0, 0]`, 18 s)
**P0 reference**: `artifacts/plum-strict/p0/` (baseline randomized robustness suite 32/59 = 54.2%)
**All P1 code/data/results/policies**: `artifacts/plum-strict/p1/` (plus `/tmp` scratch).
`eval/`, `training/`, `deployment/` source and the route-2 pipeline were **not modified** (only the new
committed `PART2_PLUM_JUDGE_PHYSICS=1` knob was used; the P0 robustness harness was copied and adapted to
take the policy path as an argument under `p1/scripts/`).

---

## 0. TL;DR / verdict

1. **Deliverable 1 — the teacher is NOT command-limited under judge physics.** With the new
   `PART2_PLUM_JUDGE_PHYSICS=1` env (judge solver options + spawn z=0.55), no training DR events, judge-like
   settle and P0-envelope spawn randomization, m1600 scores **76.5% at vx=0.5** (52/68, Wilson
   [65.1, 84.9]) — only ~5.5 pp below its own trained range (82.1% at 0.65). P0's "0.5 is OOD / command
   coverage is the dominant gap" was an artefact of the old training physics, not of the command.
   **Consequence: the teacher command fine-tune (deliverable 2) was skipped** — the task trigger was
   "<~50% at 0.5"; 76.5% is far above it, and fine-tuning a 0.76 teacher risks teaching it to slow-track
   below the 18 s judge gate for no measured ceiling gain.
2. **Deliverable 3 — 2 DAgger rounds (+1 action-loss variant) were trained and evaluated.** Both rounds use
   the judge-physics/no-DR env at fixed cmd 0.5, randomized initial state (spawn dy ±0.10 m, dyaw ±5°, joint
   ±0.03 rad / ±1.0 rad/s) and depth corruption matching every P0 failure axis (dropout 2–10%, noise
   1–5 cm-eq, delay 1–4 steps, combos); 96k then 115k on-policy frames labelled by the frozen m1600 teacher.
   The depth CNN's held-out scan error under corruption improved dramatically (dropout 15.3 cm → 2.1 cm).
   **State robustness is fixed**: joint/velocity perturbation 2/6 → **6/6**, spawn dy 7/8 → **8/8**.
   **The P0 randomized suite overall score improved from 32/59 = 54.2% to 36/59 = 61.0% (v3, Wilson
   [48.3, 72.4]); v4 = 35/59 = 59.3%** — the only target met, and within overlapping CIs, so it is a real
   but modest gain. **Depth dropout/delay robustness is NOT fixed**: dropout 0/4 and delay 1/4 for every
   P1 policy (target was ≥3/4 each). All official-judge runs stay **success / 200** (non-regression holds;
   each is ONE deterministic trial).
3. **Why depth robustness resists CNN-only DAgger** (measured, not guessed):
   * the perception side *is* fixed — held-out scan RMSE under dropout is 2.1 cm (v2: 15.3 cm), and the
     policy's clean-scan error is already 2.5 cm;
   * but the **frozen teacher MLP is extremely scan-sensitive**: a 2 % pixel dropout shifts the *pipeline
     action* by RMSE 0.084 (p95 0.19) vs the clean-pipeline-vs-teacher baseline of 0.052, and 2 cm Gaussian
     depth noise shifts it by only 0.008 (which matches the judge suite: noise 4/4 passes, dropout 0/4);
   * a 4-step depth delay is a **stale-information / control** problem (depth shows state t−4 while proprio
     is state t), not a perception problem; in-sim, 4-step delay is catastrophic (0/150) although the CNN
     predicts the *aligned* scan almost perfectly;
   * **all depth-corruption judge failures are entry-footing failures**: `dropped:wheel z≈0.03–0.05
     @ x≈3.8–4.5 m` (the first cube rows) — the clean judge pass threads a knife-edge at the field entrance,
     and a small action perturbation loses the foothold. A 1-step-delay trial even reached the exit
     (`footprint_forbidden` at x=13.90 after `exited=true`).
4. **Verdict**: P1 raised the state-robustness floor and the randomized-suite score, kept judge
   non-regression, and produced a measured reason why depth-render robustness needs a *different* next step
   (teacher-side scan-noise robustness or a temporal/representation change, not more pixels/DAgger rounds).
   Further DAgger rounds with the same recipe are **not** expected to pay off; the next round should change
   the teacher/representation or add judge-scene data (recommendations in §7).

---

## 1. Environment parity used for all P1 training/collection (committed knob, no source edits)

`PART2_PLUM_JUDGE_PHYSICS=1` (`training/src/alldog_mjlab/tasks/velocity/blackw/env_cfgs.py:816`) switches
`blackw_plum_env_cfg` to the official judge `<option>` block — verified at runtime in every run:

| item | value |
|---|---|
| integrator / cone / jacobian / solver | `euler` / `elliptic` / `sparse` / `newton` |
| iterations / ls_iterations / impratio / ccd | `100` / `50` / `100.0` / `35` |
| spawn z (`init_state.pos[2]`) | `0.55` (judge) instead of `0.52` |
| other P0 knobs kept | `PART2_FINE_SCAN=1 RES=0.01 SIZE=0.20`, `PART2_PLUM_CURRICULUM=0`, `PART2_STRICT_CONTACT=1`, `PART2_EP_ONLY_POSITIVE=1`, `PART2_SUCCESS_HOLD_STEPS=20` |
| judge-like dynamics | payload_mass / base_com / push_robot DR events removed (`--no-dr`); reset randomization kept (`reset_base`, `reset_robot_joints`) |
| P1 state envelope | `PART2_SPAWN_Y=0.10`, `PART2_SPAWN_YAW=0.0872665` (±5°), `PART2_RESET_ROOT_VEL=0` (judge) |
| settle | 50 zero-action policy steps (1.0 s) added to episode length, so every episode still gets 18 s of policy time |

Helper module: `artifacts/plum-strict/p1/scripts/p1_lib.py` (wraps the untouched
`artifacts/plum-strict/student/route2/route2_lib.py`; GPU depth augmenter; Wilson CI).

---

## 2. Deliverable 1 — teacher ceiling under judge physics (no training)

`p1/scripts/eval_parity_teacher.py` — m1600 (2106 actor = 342 proprio + 1764 fine scan), 64 envs, 50-step
settle + 900 policy steps (18 s), no DR events, fixed command, spawn randomization dy ±0.10 m / dyaw ±5°,
`part2_success` (v7.6 predicate, hold20) as success. Teacher needs no camera (fine scan only).

| vx | success / episode-ends | rate | Wilson 95% CI | termination mix (novel) |
|---|---|---|---|---|
| **0.50** | **52/68** | **0.765** | [0.651, 0.850] | 5 part2_failure, 6 illegal_contact, 5 strict_plum_wheel_ground |
| 0.55 | 53/68 | 0.779 | [0.667, 0.862] | 3 / 9 / 3 |
| 0.60 | 54/68 | 0.794 | [0.684, 0.873] | 2 / 10 / 2 |
| 0.65 | 55/67 | 0.821 | [0.713, 0.895] | 1 / 9 / 2 |
| 0.70 | 56/70 | 0.800 | [0.692, 0.877] | 1 / 12 / 1 |

(JSON per cell: `p1/results/teacher_parity_vx{050,055,060,065,070}.json`.)

**Reading**: under judge physics + judge-like determinism the teacher has an almost flat command curve
(76.5% → 82.1% from 0.5 to 0.65). The "command coverage" residual gap P0 inferred from the old-physics
control (40% at 0.5) disappears with physics parity. The remaining teacher failures are the training env's
strict-contact terminations (`illegal_contact`, `strict_plum_wheel_ground`) that the judge does not have,
so 76.5% is a conservative lower bound on the teacher's judge-side ability.

**Deliverable 2 (teacher fine-tune) — deliberately not run.** Condition in the task plan: only if the 0.5
ceiling is clearly below ~50%. It is 76.5%, so m1600 was kept as the labeler. (Fine-tuning also carries a
real hazard: training the teacher to track |v|=0.5 would slow it below the 11.55 m / 18 s ≈ 0.64 m/s
average the judge requires; the current teacher passes the judge because its 0.5-command behaviour is not
slow. This is the same reason the deterministic 0.5 judge pass is "thin" — it is one trial, not a rate.)

---

## 3. Deliverable 3 — DAgger rounds with state + depth randomization

### 3.1 Data collection

`p1/scripts/collect_dagger_p1.py`: rolls out the *student* in the §1 parity env at fixed cmd **0.5**, with
the randomized initial state (env reset events, P1 envelope), and corrupts only the depth the student sees
(per-env, per-episode category; dropout pixels → cutoff 1.0, exactly the P0 judge-suite definition):

| category | share | parameters |
|---|---|---|
| clean (forced) | 35% (r1) / 30% (r2) | — |
| pixel dropout | 1/5 of rest | p_drop ~ U(0.02, 0.10) |
| Gaussian noise | 1/5 | σ ~ U(0.01, 0.05) m-eq |
| transport delay | 1/5 | lag ~ U{1,2,3,4} steps (0.02–0.08 s) |
| dropout + noise | 1/5 | p ~ U(0.01,0.05), σ ~ U(0.01,0.02) |
| delay + dropout | 1/5 | lag ~ U{1,2}, p ~ U(0.02,0.05) |

Labels: privileged `scan` (+ `teacher_action` diagnostics) from the same env state. For transport-delay
frames the recorded depth is the stale frame and the recorded scan is the scan **of the state that produced
that depth image** (delay-aligned), so the CNN target is always semantically correct; the student action in
collection used current proprio + the stale depth, exactly like deployment.

| round | student rolled out | envs × steps | frames | episode ends | student successes in collection | wall |
|---|---|---|---|---|---|---|
| r1 | `policy_route2_v2` | 96 × 1000 | **96,000** | 199 | 10 (5.0%) | 30 s (3,239 fr/s) |
| r2 | `policies/route2_v3` | 96 × 1200 | **115,200** | 226 | 42 (18.6%) | 36 s (3,227 fr/s) |

Round-r2's on-policy success tripling (5.0% → 18.6%) under the same judge-physics/perturbed env is itself
evidence the r1 student improved off-nominal state behaviour. Raw shards:
`p1/data/dagger_r1/`, `p1/data/dagger_r2/` (with `meta.json` incl. augmentation stats).

### 3.2 Scan-CNN retraining (warm-started, includes original clean route-2 data)

Training set per round = the three original clean route-2 shard sets (259,200 frames: `data_vx07`,
`data_vx05`, `data_student_r1`) + the new corrupted DAgger shard. Validation = held-out envs
(`env_id % 5 == 0`), same split rule as route-2. Warm start from the previous round's CNN; 20 epochs,
AdamW, batch 1024.

| CNN | train / val frames | val scan RMSE (all) | obstacle-cell RMSE | notes |
|---|---|---|---|---|
| v2 (route-2) | 205,500 / 53,700 | 2.552 cm | 5.09 cm | clean-data baseline |
| r1 | 281,500 / 73,700 | 2.505 cm | 5.05 cm | +96k corrupted DAgger; warm-started from v2 |
| r2 | 296,700 / 77,700 | **2.341 cm** | 4.88 cm | +115k corrupted DAgger; warm-started from r1 |
| r3 (action-loss variant) | 296,700 / 77,700 | 2.325 cm | 4.84 cm | + auxiliary teacher-action loss λ=1; see §5 |

Old-clean-val RMSE is preserved (2.58 cm v2 → 2.53 cm r1): no catastrophic forgetting of the clean judge
domain. Offline **held-out** corruption robustness (`p1/results/cnn_robustness_heldout_r1.json`, frames from
DAgger-r1 held-out envs, `aug_cat` split):

| CNN | clean | dropout 2–10% | noise | delay | dropout+noise | delay+dropout |
|---|---|---|---|---|---|---|
| v2 | 4.63 cm | **15.31 cm** | 3.36 cm | 3.07 cm | 9.81 cm | 9.97 cm |
| r1 | 2.76 cm | **2.13 cm** | 2.47 cm | 2.46 cm | 2.79 cm | 2.34 cm |

→ the CNN front-end is genuinely robust to every P0 depth corruption after round 1 (7× better on dropout).
This is the key positive result: if perception were the bottleneck, the policy would now be robust; it is not.

### 3.3 Exported policies

`p1/scripts/export` = the untouched route-2 `export_depth_student.py` run with the new CNN and the m1600
teacher (verification: permuted-teacher vs checkpoint max|diff| 1.7e-6, pipeline vs native teacher 2.0e-6,
TorchScript 1× and 4× batch finite). All in `p1/policies/`:

| policy | CNN | official judge plum_1 fixed 0.5 | policy.pt sha256 (short) |
|---|---|---|---|
| `route2_v3` | r1 | **success / 200, exit 0** | `1a94d07c…3cdd` |
| `route2_v4` | r2 | **success / 200, exit 0** | `44c1aa58…44bf` |
| `route2_v5` | r3 (action loss) | **success / 200, exit 0** | `b5059361…a231` |
| `route2_v3_ema05` | r1 + EMA α=0.5 on predicted scan | **success / 200, exit 0** | (exploratory) |
| `route2_v3_ema03` | r1 + EMA α=0.3 on predicted scan | **fail** — `dropped:z=0.03@x=5.49` | (exploratory) |

The judge is deterministic (P0: seeds 0–7 byte-identical) → each is **one effective trial**; judge pass is
necessary but not evidence of robustness. `route2_v2` remains the fallback (also success/200).

### 3.4 Judge-side randomized robustness suite (P0 definitions, 59 trials, real judge replay)

Harness: `p1/scripts/robustness_harness_p1.py` = the P0 harness with the policy path as an argument and
per-tag trace dirs; the unmodified official judge (`replay_part2_v5`) renders every status. The P0 baseline
was **reproduced exactly** (32/59) with our copy, which validates suite parity.

| policy | trials | success | rate | Wilson 95% CI | vs v2 (paired: gained / lost) |
|---|---|---|---|---|---|
| v2 baseline (P0 reference) | 59 | 32 | **0.542** | [0.417, 0.663] | — |
| **v3 (round 1)** | 59 | **36** | **0.610** | [0.483, 0.724] | **+7 / −3 (net +4)** |
| v4 (round 2) | 59 | 35 | 0.593 | [0.466, 0.709] | +6 / −3 (net +3) |
| v5 (action loss) | 59 | 31 | 0.525 | [0.400, 0.647] | +6 / −9 (net −3) |
| v3_ema05 (temporal filter α=0.5) | 59 | 30 | 0.508 | [0.384, 0.632] | — (exploratory) |
| v3_ema03 (temporal filter α=0.3) | 59 | 2 | 0.034 | [0.009, 0.115] | — (judge fails) |

Per-axis (success/trials; v2 → v3 → v4 → v5):

| axis | v2 | v3 | v4 | v5 |
|---|---|---|---|---|
| baseline (4 command profiles, all still pass) | 4/4 | 4/4 | 4/4 | 4/4 |
| **joint/velocity perturbation** (±0.03 rad, ±0.1 rad/s) | **2/6** | **6/6** | **6/6** | 6/6 |
| **spawn dy** ±0.05/±0.10 m | 7/8 | **8/8** | **8/8** | 7/8 |
| spawn dyaw ±2/±5° | 6/8 | 6/8 | 6/8 | 3/8 |
| depth Gaussian noise (1, 2 cm-eq) | 4/4 | 4/4 | 3/4 | 4/4 |
| **depth dropout 2/5% (target ≥3/4)** | **0/4** | **0/4** | **0/4** | **0/4** |
| **depth delay 1/4 steps (target ≥3/4)** | 2/4 | 1/4 | 2/4 | 2/4 |
| depth combos (noise+dropout ± delay) | 0/4 | 0/4 | 0/4 | 0/4 |
| serial (spawn+joint+depth) | 0/4 | 0/4 | 0/4 | 0/4 |
| escalation trials | 4/9 | 4/9 | 4/9 | 3/9 |
| **all trials** | **32/59** | **36/59** | **35/59** | **31/59** |

Campaign JSON: `p1/results/robustness_{v2_baseline,v3,v4,v5}.json`; representative traces under
`p1/results/traces/` (only the last policy's set kept, by design of the copied harness).

EMA alpha=0.5 does not fix depth dropout either (0/4, delay 0/4, joint 5/6, spawn dy 7/8) and the
50-step smoothing lag breaks the deterministic judge at alpha=0.3 — more evidence the pass is a knife-edge.

**Targets:** beat P0 54.2% → **met** (61.0%, though CIs overlap 54.2% so the gain is modest);
dropout 2–5% ≥3/4 → **not met** (0/4); delay 1–2 steps ≥3/4 → **not met** (1/2 at lag 1, 0/2 at lag 4).
Per-trial paired changes v3 vs v2 (worth reading rather than the aggregate): gains are the four
joint-perturbation trials, spawn dy +0.10 adaptive, dyaw −2 adaptive, dyaw −5 fixed; losses are
`depth_delay_1step_fixed0.5` (near-success, invalidated by the lane rule at x=13.90 after `exited=true`) and
spawn dyaw +5 (fixed and adaptive). So the r1 robustness gain is dominated by *state* perturbations, plus a
re-shuffling of which lateral/spawn edge cases pass.

**Failure taxonomy of the 23 failed v3 trials** (official judge reasons): 17 ×
`dropped:wheel z≈0.00–0.05 @ x≈3.7–4.6` (all at the field entrance), 2 × `fell:tilt≈78–82°` (entry),
1 × `footprint_forbidden` (delay_1step; had completed the level physically at x=13.90), 3 × stuck/progress.
Every depth-corruption failure dies at the first cube rows, not deep in the field.

---

## 4. Diagnostics: where the residual depth fragility actually is

1. **Perception is fixed; behaviour is not.** Held-out scan RMSE under dropout is 2.1 cm (v2 15.3 cm), yet the
   judge dropout trials stay 0/4. So CNN scan regression is not the binding constraint any more.
2. **The frozen teacher amplifies scan errors into large action changes.** `p1/scripts/analyze_action_sensitivity.py`
   (8k held-out frames of the r2 collection; clean teacher action on the *true* scan is ground truth):

   | corruption | scan RMSE vs true | **pipeline action Δ vs clean pipeline** | p95 Δ |
   |---|---|---|---|
   | clean (baseline vs teacher) | 2.23 cm | 0.0518 | — |
   | dropout 2% | 4.00 cm | **0.0838** | 0.188 |
   | dropout 5% | 5.03 cm | **0.1154** | 0.262 |
   | Gaussian noise 2 cm | 2.26 cm | 0.0084 | 0.017 |
   | stale depth (4-step-shift proxy) | 10.35 cm | 0.2778 | 0.630 |

   This is exactly consistent with the judge suite: noise is harmless (4/4), dropout breaks it (0/4). Same
   result for the action-loss r3 CNN (dropout2 Δ = 0.0862), i.e. the auxiliary loss did not reduce the
   corruption-induced action shift.
3. **In-sim controls confirm it is not a judge-renderer domain gap.** `p1/scripts/eval_student_corrupt_insim.py`
   on the best round-2 policy in the judge-physics/no-DR env at nominal spawn (64 envs × 900 steps, strict
   training terminations): clean **22/64 = 34.4%**, 2% dropout **10/121 = 8.3%**, 4-step delay **0/150 = 0%**.
   Corruption breaks the policy with the *training* renderer too; state robustness is what improved.
4. **Delay is a control-information problem.** The CNN is trained/exported to predict the scan of the state
   that produced the (stale) depth image; at deployment that means the teacher acts on a 4-step-old scan with
   current proprio. No CNN-only training can recover the missing current-state information. Judge lag=4 is
   0/2 for every policy; a stale-depth proxy shifts the pipeline action by 0.278.
5. **Temporal-filter experiment (negative result, documented).** `export_depth_student_ema.py` bakes a
   one-pole EMA on the predicted scan into the exported TorchScript policy to suppress per-step dropout
   jitter (α=0.5 → ~1-step lag, stateless first step after construction verified bit-identical). Official
   judge: α=0.5 still success/200, but the randomized suite drops to 30/59 (dropout 0/4, delay 0/4); α=0.3
   **fails the judge** (`dropped:z=0.03@x=5.49`) and collapses to 2/59. Test-time smoothing is not a fix:
   the entry knife-edge is sensitive to action lag as well as jitter.
6. **V5 action-loss experiment (negative result, documented).** `train_scan_cnn_actloss.py` added
   `MSE(teacher([proprio,pred_scan]), teacher([proprio,true_scan]))` (masked off for delay frames, λ=1,
   normalised by teacher action std, warm-started from r2). Val scan RMSE slightly improved (2.325 vs
   2.341 cm) but the corruption-induced action shift did **not** (dropout2 Δ 0.0862 vs 0.0838) and the
   randomized suite regressed to 31/59 (spawn-dyaw 6/8 → 3/8). Not promoted; kept for the record.

---

## 5. Deliverables / files

| path | content |
|---|---|
| `artifacts/plum-strict/p1/REPORT.md` | this report |
| `p1/scripts/p1_lib.py` | judge-physics env wrapper, GPU depth augmenter, Wilson CI |
| `p1/scripts/eval_parity_teacher.py` | deliverable 1: teacher under judge physics |
| `p1/scripts/collect_dagger_p1.py` | DAgger collection (state randomization + depth corruption, delay-aligned labels) |
| `p1/scripts/train_scan_cnn_p1.py`, `train_scan_cnn_actloss.py` | CNN retraining (warm start / action-loss variant) |
| `p1/scripts/analyze_cnn_robustness.py`, `analyze_action_sensitivity.py` | offline corruption + behavioural-sensitivity diagnostics |
| `p1/scripts/export_depth_student_ema.py` | exploratory temporal-filter (EMA on predicted scan) exporter |
| `p1/scripts/eval_student_corrupt_insim.py` | in-sim clean/dropout/delay controls |
| `p1/scripts/robustness_harness_p1.py` | P0 judge-side randomized suite, policy path as argument |
| `p1/scripts/evaluate_policy_p1.sh`, `aggregate_p1.py` | official judge + suite runner / summary builder |
| `p1/data/{dagger_r1,dagger_r2,scan_cnn_r1,scan_cnn_r2,scan_cnn_r3}/` | on-policy shards (with `meta.json`) and CNNs |
| `p1/policies/{route2_v3,route2_v4,route2_v5}/` | exported judge/deployment policies (+ config/depth/manifest) |
| `p1/policies/{route2_v3_ema05,route2_v3_ema03}/` | exploratory temporal-filter variants (negative result; not promoted) |
| `p1/results/` | teacher parity JSONs, CNN metrics, robustness JSONs (full trial lists), judge JSONs, diagnostics, `p1_summary.json` |

Reproduce (all `cd /home/bis/RoboCon`; GPU runs with `MUJOCO_GL=egl PYTHONPATH=$PWD/training/src`):

```bash
# deliverable 1 (one vx per call; 64 envs, 50-step settle + 900 policy steps)
.pixi/envs/train/bin/python artifacts/plum-strict/p1/scripts/eval_parity_teacher.py \
  --checkpoint logs/rsl_rl/blackw_velocity_privileged/2026-10-31_11-36-32_plum_ep_success_bonus500_h15_r002/model_1600.pt \
  --vx 0.5 --num-envs 64 --steps 950 --settle-steps 50 --seed 0 \
  --out artifacts/plum-strict/p1/results/teacher_parity_vx050.json

# round 1 / 2 collection
.pixi/envs/train/bin/python artifacts/plum-strict/p1/scripts/collect_dagger_p1.py \
  --policy artifacts/plum-strict/p1/policies/route2_v3 \
  --teacher logs/rsl_rl/blackw_velocity_privileged/2026-10-31_11-36-32_plum_ep_success_bonus500_h15_r002/model_1600.pt \
  --out artifacts/plum-strict/p1/data/dagger_r2 --num-envs 96 --steps 1200 --shard-steps 600 \
  --vx 0.5 --seed 22 --clean-frac 0.30

# CNN retraining (warm start; clean route-2 shards + new corrupted round)
.pixi/envs/train/bin/python artifacts/plum-strict/p1/scripts/train_scan_cnn_p1.py \
  --data artifacts/plum-strict/student/route2/data_vx07 artifacts/plum-strict/student/route2/data_vx05 \
         artifacts/plum-strict/student/route2/data_student_r1 artifacts/plum-strict/p1/data/dagger_r2 \
  --out artifacts/plum-strict/p1/data/scan_cnn_r2 --epochs 20 --batch 1024 --seed 0 \
  --init-from artifacts/plum-strict/p1/data/scan_cnn_r1/scan_cnn.pt

# export (untouched route-2 exporter)
.pixi/envs/train/bin/python artifacts/plum-strict/student/route2/export_depth_student.py \
  --scan-cnn artifacts/plum-strict/p1/data/scan_cnn_r2/scan_cnn.pt \
  --checkpoint logs/rsl_rl/blackw_velocity_privileged/2026-10-31_11-36-32_plum_ep_success_bonus500_h15_r002/model_1600.pt \
  --out artifacts/plum-strict/p1/policies/route2_v4 --name route2_v4

# acceptance: official judge (one deterministic trial) + P0 randomized suite (59 trials, real judge replay)
bash artifacts/plum-strict/p1/scripts/evaluate_policy_p1.sh artifacts/plum-strict/p1/policies/route2_v4 v4 yes

# aggregate
.pixi/envs/eval/bin/python artifacts/plum-strict/p1/scripts/aggregate_p1.py
```

---

## 6. Constraints / caveats

- Simulation only; no deployment/eval/training source modified (git status for those paths is clean).
  The only new env behaviour used is the committed `PART2_PLUM_JUDGE_PHYSICS=1` knob; the depth override in
  the robustness harness is an instance monkeypatch of `render_depth`, as in P0.
- **The official judge is deterministic → one effective trial.** Every P1 policy passes it (non-regression),
  but that is not a robustness statement. The 59-trial randomized suite is the acceptance metric used here,
  and its Wilson CIs are wide (±12–13 pp at n=59), so 54.2% → 61.0% is a modest improvement, not a step change;
  150+ trials would be needed to separate small effects.
- Teacher-parity successes include the training env's strict-contact terminations (absent in the judge), so
  the teacher ceiling numbers are conservative; conversely the in-sim corruption controls use the strict
  training criteria, so their absolute rates understate judge-side rates (judge: clean pass; 0/4 dropout).
- `route2_v3`/`v4` were trained/evaluated in this session; the old-physics teacher is unchanged (m1600). The
  deterministic judge was re-verified 3× (v3, v4, v5 all success/200) and the P0 suite baseline was reproduced
  exactly (32/59) before using it for new policies.
- GPU peak during collection/training was ~7.5 GB (<10 GB budget); no other processes were touched.

---

## 7. P1 remaining gaps and whether a further round is warranted

**What P1 achieved**: physics parity in the training env (teacher command-coverage gap disproved), judge-like
DAgger with depth/state randomization, fixed state-perturbation robustness (joint 2/6→6/6, spawn dy 7/8→8/8),
+4 net trials on the randomized suite (54.2% → 61.0%), judge non-regression, and a measured diagnosis of the
remaining depth fragility.

**What is still open** (ranked; do these in a P1.5/P2, not as "more of the same"):
1. **Depth-corruption robustness (dropout 0/4, delay 1/4; target ≥3/4).** The binding constraint is the
   frozen teacher's action sensitivity to scan errors and the stale-information problem for delay — not the
   CNN. Highest-leverage options:
   * **teacher scan-noise robustness**: fine-tune the teacher (judge physics, cmd 0.5) with corrupted-scan
     observations during PPO (simulate student scan error with dropout/noise on the privileged fine-scan
     obs), so the teacher itself tolerates the student's 2–4 cm prediction errors; verify judge non-regression
     and the suite on the resulting student. Must keep average speed ≥0.64 m/s (18 s gate).
   * **behavioural distillation with a stronger objective**: action-loss at much larger weight (>10), or
     direct BC of the student's *estimated current scan* from stale depth, or an explicit forward-model
     (proprio + stale scan → current scan estimate) trained with current-scan labels (fixes delay).
   * **representation**: a naive EMA/temporal filter on the predicted scan was tried and *failed* (judge
     knife-edge + no dropout gain; see §4.5), so the next attempt should be a learned recurrent/forward-model
     front-end (or PUMA-style low-rank scan prior) rather than test-time smoothing. The P0 suite measures a
     policy in ~2.5 min, so iterate on the harness, not on the judge.
   * **judge-scene data**: collect depth+raycast-scan pairs in the *official judge scene* (CPU MuJoCo) so the
     CNN sees the EGL renderer/terrain exactly (P0 showed ~5–7% depth statistics gap at field states); the
     fine-scan definition (4×21×21, 1 cm, 0.25 m above wheel centre) is re-implementable with raycasts.
2. **Lateral/exit-rule robustness** is now the second-largest failure family (`spawn dyaw ±5°` 6/8; one
   near-success invalidated by `footprint_forbidden` after physically finishing). A denser trial grid on
   dyaw ±5–10° and the lane margin, plus DAgger data collected with lateral offsets, is cheap.
3. **Wider robustness grid**: 59 trials gives ±12 pp; a 150-trial grid (multi-axis serial + repeat seeds for
   the diagnostic perturbations) would let the next round claim improvements with tighter CIs.
4. Keep `route2_v2` as the certified fallback; `route2_v3` is the best P1 candidate on the randomized suite
   (61.0%) while also passing the official judge — but it is **not** certified robust; it still fails the
   depth-render perturbations that motivated the task, and one near-miss shows the lane rule can still
   invalidate a physically complete run.

**Recommendation**: a further DAgger round with the *same* recipe is **not** warranted (round 1→2 already
plateaued: 61.0% → 59.3%, CIs overlapping). The next round should change the teacher (scan-noise robustness)
or the representation/policy (temporal filtering or forward-estimation), with the P0 randomized suite +
official judge as the acceptance gate, targeting depth dropout 2–5% and delay 1–2 steps ≥3/4.
