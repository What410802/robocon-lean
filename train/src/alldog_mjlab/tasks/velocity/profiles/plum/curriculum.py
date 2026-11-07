"""Success-gated difficulty curriculum for the plum teacher.

mjlab's ``terrain_levels_vel`` promotes an env when it walks farther than
``size[0] / 2`` (8 m here) and demotes it when it walks less than half the
commanded distance.  Neither rule fits this task: the field is 3.5 m long, so
the promotion threshold is unreachable, and the commanded distance depends on a
command the actor never observes.

This term promotes on the judge-mirrored success predicate and demotes on its
failure predicate, which is the same signal the teacher is rewarded for.  It
runs once per reset (the CurriculumManager contract) and only touches terrain
levels, so the env-origin swap stays atomic within ``_reset_idx``.

Promotion is one success (``UP_STREAK = 1``).  The first version required three
consecutive successes on the same env, which looked prudent and was a deadlock:
the measured success rate is ~1.29 per 512-env batch (~0.25 % per episode), so
three in a row per env has probability ~1e-8 and the ladder never left row 0.
The v3c run therefore trained only on the easy plateau (1.2 m ramps, touching
cubes) and scored 0/269 on the official field.  Demotion stays at
``DOWN_STREAK`` consecutive failures: envs that promote past their ability fall
back, which puts the population at its own frontier.
"""

from __future__ import annotations

import torch

from alldog_mjlab.tasks.velocity.profiles.part2.predicates import (
    _part2_failure, _part2_success,
)

from .spec import SPEC

#: Episodes of evidence required before a row can be promoted.  Promotion is a
#: property of the ROW, not of a single env: one lucky success must not move a
#: whole skill level.  Measured why: with per-env ``UP_STREAK=1`` the moment the
#: warm start began working, success per batch promoted every env that happened
#: to land a run, and the ladder went from row 0 to row 3.0 in 23 iterations while
#: the success rate collapsed from 16.5 to 0.31 per batch -- the population had
#: outrun its own frontier and could no longer produce the signal it needed.
PROMOTE_MIN_EPISODES = 40
#: Records ONE call may add to a row's window.  The curriculum term runs on
#: every env reset, and 2048 envs at ~60-step episodes finish >2000 episodes per
#: simulator step, so without this cap the whole 40-episode window was replaced
#: inside a single step: every "decision" was one step of evidence.
#: Measured in v9 (no cap): up=96, down=288 on every iteration for 50+
#: iterations, mean pinned at 96/2048 = 0.0469 -- envs bounced to row 1 and
#: straight back, so no skill was ever trained at either row.
RECORDS_PER_CALL = 4
#: Calls that must pass before the same row may move again.  The term is called
#: once per simulator step (64 steps per iteration), so this allows at most one
#: decision per iteration per row -- which is what stops the promote/demote
#: thrash the uncapped window produced.
COOLDOWN_CALLS = 64
#: Row success rate that unlocks the next row (v3c measured ~0.25 % here).
PROMOTE_RATE = 0.5
DEMOTE_RATE = 0.05
#: Demotion uses the same window as promotion.  (An earlier DEMOTE_MIN_EPISODES=60
#: exceeded the 40-episode window, so the condition could never be satisfied and
#: a stuck row never fell back -- caught by the scripted test.)
DEMOTE_MIN_EPISODES = PROMOTE_MIN_EPISODES


def plum_difficulty_curriculum(
    env,
    env_ids: torch.Tensor,
    promote_rate: float = PROMOTE_RATE,
    demote_rate: float = DEMOTE_RATE,
    records_per_call: int = RECORDS_PER_CALL,
    cooldown_calls: int = COOLDOWN_CALLS,
) -> dict[str, torch.Tensor]:
    """Windowed, row-level difficulty control.

    Per row, keep the success/failure counts of the last :data:`PROMOTE_MIN_EPISODES`
    finished episodes.  The whole row moves up when that window's success rate
    reaches ``promote_rate``, and down when it falls to ``demote_rate``.  A row
    therefore holds until it has real evidence at that difficulty, which is what
    a curriculum needs; a per-env streak either deadlocks (long streak) or
    rockets to the top (streak 1).
    """
    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
        return {}
    n = env.num_envs
    device = env.device
    rows = max(1, int(terrain.cfg.terrain_generator.num_rows))
    levels = terrain.terrain_levels

    if getattr(env, "_plum_window", None) is None:
        # Per-level circular buffers of the latest episode outcomes.
        env._plum_window = torch.zeros((rows, PROMOTE_MIN_EPISODES),
                                       dtype=torch.bool, device=device)
        env._plum_window_n = torch.zeros(rows, dtype=torch.long, device=device)
        env._plum_window_head = torch.zeros(rows, dtype=torch.long, device=device)
        env._plum_window_success = torch.zeros(rows, dtype=torch.long, device=device)
        env._plum_call = 0
        env._plum_last_move = torch.zeros(rows, dtype=torch.long, device=device)
    env._plum_call += 1

    # This term is invoked once per reset (CurriculumManager contract, from
    # ``_reset_idx``), so every env in ``env_ids`` has exactly one finished
    # episode to record here -- no dedup flag is needed (an earlier one was
    # cleared only on non-reset steps and therefore never cleared at all).
    record = torch.zeros(n, dtype=torch.bool, device=device)
    record[env_ids] = True
    if bool(record.any()):
        success = _part2_success(env).detach() & record
        lv = levels.clamp(0, rows - 1)
        for row in torch.unique(lv[record]).tolist():
            sel = record & (lv == row)
            head = int(env._plum_window_head[row])
            # Sample a random subset: `success[sel]` is in env-index order, and
            # taking its head polled the SAME few envs every call, so the window
            # measured a 4-env poll rather than the row's population (v12 ran 27
            # iterations with up=0, down=0 on an identical recipe that promoted
            # at 512 envs).
            order = torch.randperm(int(success[sel].numel()), device=device)
            vals = success[sel][order].to(env._plum_window.dtype)
            # Cap the records one call may add (see RECORDS_PER_CALL): enough
            # calls must elapse for the window to represent behaviour over time
            # rather than a single simulator step.
            cnt = min(int(vals.numel()), records_per_call)
            idx = (head + torch.arange(cnt, device=device)) % PROMOTE_MIN_EPISODES
            old = int(env._plum_window[row, idx].sum())
            env._plum_window[row, idx] = vals[:cnt]
            # Slot replacement: the running total loses the overwritten entries
            # and gains the new ones.
            env._plum_window_success[row] += int(vals[:cnt].sum()) - old
            env._plum_window_head[row] = (head + cnt) % PROMOTE_MIN_EPISODES
            env._plum_window_n[row] = min(int(env._plum_window_n[row]) + cnt,
                                          PROMOTE_MIN_EPISODES)

    move_up = torch.zeros(n, dtype=torch.bool, device=device)
    move_down = torch.zeros(n, dtype=torch.bool, device=device)
    for row in range(rows):
        if int(env._plum_call) - int(env._plum_last_move[row]) < cooldown_calls:
            continue
        count = int(env._plum_window_n[row])
        if count < PROMOTE_MIN_EPISODES:
            continue
        rate = float(env._plum_window_success[row]) / count
        in_row = levels == row
        if rate >= promote_rate:
            move_up |= in_row
        elif rate <= demote_rate and count >= DEMOTE_MIN_EPISODES:
            move_down |= in_row
        if rate >= promote_rate or rate <= demote_rate:
            env._plum_last_move[row] = int(env._plum_call)
            # Clear the buffer with the counters: leaving stale entries made the
            # next cycle subtract values that were no longer counted (measured:
            # running success total went negative).
            env._plum_window[row] = False
            env._plum_window_n[row] = 0
            env._plum_window_success[row] = 0
            env._plum_window_head[row] = 0

    if bool(move_up.any() or move_down.any()):
        ids = torch.arange(n, device=device)
        terrain.update_env_origins(ids[move_up], torch.ones(int(move_up.sum()), dtype=torch.bool, device=device),
                                   torch.zeros(int(move_up.sum()), dtype=torch.bool, device=device))
        terrain.update_env_origins(ids[move_down], torch.zeros(int(move_down.sum()), dtype=torch.bool, device=device),
                                   torch.ones(int(move_down.sum()), dtype=torch.bool, device=device))

    levels = terrain.terrain_levels.float()
    return {
        "mean": torch.mean(levels),
        "max": torch.max(levels).float(),
        "up": move_up.sum().float(),
        "down": move_down.sum().float(),
    }


def cube_half_per_env(env) -> torch.Tensor:
    """Cube half-width per env for the ``feet_edge`` term.

    The cube footprint is the official one on every curriculum row (only the
    barrier height and the end holes are graded), so this is a constant.
    """
    return torch.full((env.num_envs,), SPEC.cube_half, device=env.device)
