"""Competition evaluation package for the RoboCon 2026 in-team race.

The human-readable standard is ``eval/SPEC.md``; the machine-readable standard
(every score/curve/attempt constant) is :mod:`robocon_eval.rules`.  Modules:

    robocon_eval.rules          machine-readable competition standard
    robocon_eval.scoring        Part 1 / Part 2 score aggregation (floor curves)
    robocon_eval.trace          Part 1 JSONL trace replay -> Part1JudgeV2
    robocon_eval.trace_part2_v5 Part 2 JSONL trace replay -> Part2OfficialJudge
    robocon_eval.policy_runner  independent eval-side MuJoCo policy runner
    robocon_eval.part1          Part 1 event judge + whole-course attempt protocol
    robocon_eval.part2          Part 2 terminal-state judge + level metadata
    robocon_eval.cli            ``robocon-eval run|replay-part1|replay-part2|score``

Nothing here imports ROS or the training repository: the judge is the
independent side of the score, and the runner/training wrappers must be checked
against it.
"""

__all__: list[str] = []
