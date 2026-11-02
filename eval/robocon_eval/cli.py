"""Command line entry point for the evaluation package.

Subcommands (kept deliberately small; see ``eval/README.md``)::

    robocon-eval run         --robot blackW --part 2 --policy flat --level high_platform_10cm
    robocon-eval replay-part1 --trace artifacts/traces/part1_run.jsonl
    robocon-eval replay-part2 --trace artifacts/traces/trench_30cm.jsonl
    robocon-eval score       --levels '{"high_platform": 30, "plum": 1}'

The judges only consume traces/JSON plus MuJoCo on the CPU; the CLI never
imports ROS and never touches the deployment stack.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import rules, scoring


def _cmd_run(args: argparse.Namespace) -> int:
    from .policy_runner import run

    result = run(
        args.robot, args.part, args.policy, args.level, Path(args.out),
        args.steps, Path(args.scene) if args.scene else None,
        duration_s=args.duration,
        command=tuple(args.command) if args.command else None,
        settle_s=args.settle, warmup_s=args.warmup,
        seed=args.seed, profile=args.profile,
        depth_contract=args.depth_contract,
    )
    if args.result:
        Path(args.result).write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    summary = {
        "robot": args.robot, "part": args.part, "policy": args.policy,
        "level": args.level, "trace": args.out, "finite": result["finite"],
        "status": result["status"], "score": result["score"],
        "reason": result["reason"], "sim_time_s": result["sim_time_s"],
        "frames": result["frames"], "stopped": result["stopped"],
        "warnings": result["warnings"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not result["finite"]:
        return 4
    if result["part"] == 2:
        return 0 if result["status"] == "success" else 3
    return 0


def _cmd_replay_part1(args: argparse.Namespace) -> int:
    from .trace import replay_part1

    report = replay_part1(args.trace)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _cmd_replay_part2(args: argparse.Namespace) -> int:
    from .trace_part2_v5 import replay_part2_v5

    report = replay_part2_v5(args.trace, levels_path=args.levels)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    """Score Part 2 from the best successful level per category (or both parts)."""
    payload = _load_levels_payload(args)
    part2 = rules.part2()
    result = scoring.part2_score(part2, payload)
    result["max_score"] = part2["max_score"]
    result["rounding"] = part2["rounding"]
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _load_levels_payload(args: argparse.Namespace) -> dict:
    if args.levels:
        return json.loads(args.levels)
    if args.levels_file:
        doc = json.loads(Path(args.levels_file).read_text())
        # Accept either a bare {category: level} map or a v5 result JSON with
        # {"categories": {cat: {"best_level": ...}}}.
        if "categories" in doc:
            return {cat: data.get("best_level")
                    for cat, data in doc["categories"].items()}
        return doc
    raise SystemExit("score: pass --levels JSON or --levels-file")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robocon-eval", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run one policy in MuJoCo and replay the judge")
    p.add_argument("--robot", choices=["black", "blackW"], required=True)
    p.add_argument("--part", type=int, choices=[1, 2], required=True)
    p.add_argument("--policy", required=True,
                   help="policy name under demo/rl_sar/policy/<robot>/ or a dir with config.yaml")
    p.add_argument("--level", default=None, help="Part2 official level key")
    p.add_argument("--scene", default=None, help="override the scene XML path")
    p.add_argument("--steps", type=int, default=None,
                   help="fixed policy steps (overrides --duration/level seconds)")
    p.add_argument("--duration", type=float, default=None,
                   help="simulated seconds (default: level seconds / 10 s for Part1)")
    p.add_argument("--command", type=float, nargs=3, default=None,
                   metavar=("VX", "VY", "WZ"))
    p.add_argument("--settle", type=float, default=1.0,
                   help="fixed-gain settle before the policy starts")
    p.add_argument("--warmup", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--profile", choices=["official", "diagnostic"], default="diagnostic")
    p.add_argument("--depth-contract", default=None,
                   help="depth-camera contract JSON (default: auto-detect <policy>/depth.json)")
    p.add_argument("--out", default="artifacts/traces/policy_trace.jsonl")
    p.add_argument("--result", default=None, help="optional result JSON path")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("replay-part1", help="replay a Part1 JSONL trace through the judge")
    p.add_argument("--trace", required=True)
    p.set_defaults(func=_cmd_replay_part1)

    p = sub.add_parser("replay-part2", help="replay a Part2 JSONL trace through the v5 judge")
    p.add_argument("--trace", required=True)
    p.add_argument("--levels", default=None,
                   help="override the level manifest path (default: generated manifest)")
    p.set_defaults(func=_cmd_replay_part2)

    p = sub.add_parser("score", help="score Part 2 from best successful levels")
    p.add_argument("--levels", default=None,
                   help='JSON map, e.g. {"high_platform":30,"trench":20,"plum":1}')
    p.add_argument("--levels-file", default=None,
                   help="JSON file with a {category: level} map or a v5 result JSON")
    p.set_defaults(func=_cmd_score)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        import sys
        print(f"robocon-eval {args.command}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
