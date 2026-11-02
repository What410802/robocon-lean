"""Measure real success rate of a policy in a Part2 exact training task.

Unlike the training logs' count-based ``Episode_Termination/*`` metric this
script counts completed episodes and their terminal success flag, so the printed
number is a true success fraction.  Used to verify that the lowest difficulty
tier of every task is solved reliably (potential-bug check).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--difficulty", type=float, default=0.0)
    ap.add_argument("--vx", type=float, default=0.3)
    ap.add_argument("--vy-range", type=float, default=0.0)
    ap.add_argument("--wz-range", type=float, default=0.0)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--terrain-size", type=float, default=0.0,
                    help="override terrain patch size in metres (0 = keep task default)")
    ap.add_argument("--clean-obs", action="store_true",
                    help="disable observation corruption/noise (judge has clean obs)")
    ap.add_argument("--exact-spawn", action="store_true",
                    help="zero the reset randomization (default pose/velocity), i.e. "
                         "the canonical judge spawn")
    ap.add_argument("--no-dr", action="store_true",
                    help="drop domain-randomization events (keep reset terms only)")
    args = ap.parse_args()

    configure_torch_backends()
    torch.manual_seed(args.seed)
    env_cfg = load_env_cfg(args.task, play=False)
    env_cfg.scene.num_envs = args.envs
    gen = env_cfg.scene.terrain.terrain_generator
    if args.terrain_size > 0:
        gen.size = (args.terrain_size, args.terrain_size)
    gen.difficulty_range = (args.difficulty, args.difficulty)
    env_cfg.scene.terrain.max_init_terrain_level = 0
    if args.exact_spawn:
        for name, ev in env_cfg.events.items():
            params = dict(ev.params)
            if name == "reset_base":
                params["pose_range"] = {}
                params["velocity_range"] = {
                    key: (0.0, 0.0) for key in params.get("velocity_range", {})}
            else:
                if "position_range" in params:
                    params["position_range"] = {
                        key: (0.0, 0.0) for key in params["position_range"]}
                if "velocity_range" in params:
                    params["velocity_range"] = (0.0, 0.0)
            env_cfg.events[name] = replace(ev, params=params)
    if args.clean_obs:
        for group in env_cfg.observations.values():
            group.enable_corruption = False
    if args.no_dr:
        env_cfg.events = {name: cfg for name, cfg in env_cfg.events.items()
                          if name.startswith("reset")}
    cmd = env_cfg.commands["twist"]
    cmd.ranges.lin_vel_x = (args.vx, args.vx)
    cmd.ranges.lin_vel_y = (-args.vy_range, args.vy_range)
    cmd.ranges.ang_vel_z = (-args.wz_range, args.wz_range)
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=None)
    runner = MjlabOnPolicyRunner(env, asdict(load_rl_cfg(args.task)), device="cuda:0")
    runner.load(args.checkpoint, load_cfg={"actor": True})
    policy = runner.alg.get_policy()
    unwrapped = env.unwrapped
    tm = unwrapped.termination_manager
    obs = env.get_observations()
    reset_terms = [t for t in tm.active_terms if "part2" in t or t in ("official_platform_success", "proxy_platform_success")]
    # Fail closed: a missing success term silently reports 0 success (the HIM line
    # hit exactly this with a play=True gate config).  Platform-exact tasks also
    # terminate on official_platform_success, so count either success predicate.
    success_terms = [t for t in tm.active_terms
                     if t in ("part2_success", "official_platform_success", "proxy_platform_success")]
    if not success_terms:
        raise SystemExit(
            f"check_tier_success: task {args.task!r} has no success termination "
            f"(active={sorted(tm.active_terms)}); refusing to report a silent 0")
    if "part2_failure" not in reset_terms:
        print(f"[warn] task {args.task!r} has no part2_failure term; "
              f"failure counts are not measurable")
    success = torch.zeros(args.envs, dtype=torch.long)
    episodes = torch.zeros(args.envs, dtype=torch.long)
    term_counts: dict[str, int] = {name: 0 for name in tm.active_terms}
    total_steps = 0
    # Derive the cap from the env instead of guessing (audit P1-3): the old
    # int(400/episodes)*episodes+2000 formula stopped before the requested
    # number of episodes.  +200 steps of slack for resets.
    max_steps = args.episodes * int(unwrapped.max_episode_length) + 200
    while int(episodes.min().item()) < args.episodes and total_steps < max_steps:
        with torch.no_grad():
            actions = policy(obs, stochastic_output=args.stochastic)
        obs, _rew, dones, _extras = env.step(actions)
        succ = torch.zeros_like(tm.get_term(success_terms[0]))
        for name in success_terms:
            term = tm.get_term(name)
            succ = succ | term.bool().to(succ.device)
        if bool(dones.any()):
            done_mask = dones.bool().cpu()
            success[done_mask] += succ[done_mask.to(succ.device)].long().cpu()
            episodes[done_mask] += 1
            for name in tm.active_terms:
                term = tm.get_term(name).cpu()
                term_counts[name] += int(term[done_mask].sum().item())
        total_steps += 1
    # Only envs that completed at least one episode contribute to per-env rates;
    # a 0/0 env must not show up as a 0% (audit P1-3).
    done_envs = episodes > 0
    rate = torch.where(
        done_envs,
        success.float() / episodes.clamp(min=1).float(),
        torch.full((success.numel(),), float("nan"), dtype=torch.float32,
                   device=success.device),
    )
    truncated = int(episodes.min().item()) < args.episodes
    out = {
        "task": args.task,
        "difficulty": args.difficulty,
        "vx": args.vx,
        "vy_range": args.vy_range,
        "wz_range": args.wz_range,
        "stochastic": args.stochastic,
        "envs": args.envs,
        "episodes": int(episodes.sum().item()),
        "successes": int(success.sum().item()),
        "success_rate": round(float(success.sum().item() / max(1, episodes.sum().item())), 4),
        "worst_env_rate": (round(float(rate[done_envs].min().item()), 4)
                            if bool(done_envs.any()) else None),
        "best_env_rate": (round(float(rate[done_envs].max().item()), 4)
                           if bool(done_envs.any()) else None),
        "episodes_per_env_min": int(episodes.min().item()),
        "truncated": truncated,
        "term_counts": term_counts,
        "part2_terms": reset_terms,
    }
    print(json.dumps(out, ensure_ascii=False))
    env.close()
    if truncated:
        print(f"[error] check_tier_success truncated: per-env min episodes "
              f"{int(episodes.min().item())} < requested {args.episodes} "
              f"(max_steps={max_steps})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
