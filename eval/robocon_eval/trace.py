"""Trace replay: JSONL trace -> Part 1 judge (Part 2 lives in trace_part2_v5).

A trace is JSONL: one meta line then one frame per judged sample::

    {"kind":"meta","schema":1,"part":1,"robot":"blackW","scene":"...","policy_sha256":"..."}
    {"kind":"frame","t":0.002,"base":[x,y,z],"yaw":0.0,"foot_tags":{"FL":[]},
     "ground_feet":[],"bar_z":null,"support":{"FL":[x,y,z]},"contacts":[["FL_foot","floor"]]}

Judged runs record one frame per 2 ms physics step (``eval_state_every:=1``);
the replay also accepts the legacy 20 ms cadence.  The judge works from whatever
sample stream it is given (it is event-based; Part 2 advances hold timers from
``t`` deltas), so no sampling-period constant appears here.  The replay never
imports ROS/deployment code: the judge is the independent side.
"""
from __future__ import annotations

import json
from pathlib import Path

from .part1.geom_tags import GEOM_TAGS, is_ground_geom
from .part1.judge import Part1JudgeV2

LEGS = ("FL", "FR", "RL", "RR")


def _side_of(geom: str) -> str | None:
    for side in LEGS:
        if geom.startswith(side + "_"):
            return side
    return None


def classify_contacts(contacts: list) -> tuple[dict[str, set[str]], set[str]]:
    """Map raw geom-name contact pairs to judge foot tags and ground feet."""
    tags: dict[str, set[str]] = {leg: set() for leg in LEGS}
    ground: set[str] = set()
    for pair in contacts or []:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        a, b = map(str, pair)
        sa, sb = _side_of(a), _side_of(b)
        if sa and not sb:
            side, other = sa, b
        elif sb and not sa:
            side, other = sb, a
        else:
            continue
        if is_ground_geom(other):
            ground.add(side)
        else:
            tags[side].update(GEOM_TAGS.get(other, set()))
    return tags, ground


def read_trace(path: str | Path) -> tuple[dict, list[dict]]:
    meta: dict | None = None
    frames: list[dict] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        kind = item.pop("kind", None)
        if kind == "meta":
            meta = item
        elif kind == "frame":
            frames.append(item)
    if meta is None:
        raise ValueError("trace has no meta line")
    if meta.get("schema") != 1:
        raise ValueError(f"unsupported trace schema {meta.get('schema')!r}")
    return meta, frames


def replay_part1(path: str | Path) -> dict:
    meta, frames = read_trace(path)
    if meta.get("part") != 1:
        raise ValueError(f"not a Part 1 trace: {meta.get('part')!r}")
    judge = Part1JudgeV2()
    for f in frames:
        tags = {leg: set((f.get("foot_tags") or {}).get(leg) or []) for leg in LEGS}
        ground = set(f.get("ground_feet") or [])
        if f.get("contacts"):
            ctags, cground = classify_contacts(f["contacts"])
            for leg in LEGS:
                tags[leg].update(ctags[leg])
            ground.update(cground)
        judge.update(float(f["t"]), tuple(map(float, f["base"])), float(f.get("yaw", 0.0)),
                     foot_tags=tags, ground_feet=ground,
                     bar_z=f.get("bar_z"),
                     support_xy={k: tuple(map(float, v)) for k, v in (f.get("support") or {}).items()}
                     or None)
    report = judge.finalize("trace_end")
    report["trace"] = {"path": str(path), "frames": len(frames), "meta": meta}
    return report
