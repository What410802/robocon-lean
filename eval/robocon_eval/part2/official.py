"""Canonical Part 2 level metadata (the generated runtime manifest).

The single terrain source is ``terrain/``; ``terrain/gen_part2_levels.py`` writes
the 20 official levels to::

    demo/scenes/part2/part2_official_levels.json      (manifest)
    demo/scenes/part2/generated/part2_off_<key>.xml   (one isolated scene per level)

Resolution order for the manifest path (kept configurable for CI/backfills):

1. ``ROBOCON_EVAL_LEVELS`` environment variable (absolute path);
2. ``demo/scenes/part2/part2_official_levels.json`` (the tracked generated file);
3. ``eval/fixtures/legacy/part2_official_levels.json`` (bootstrap fallback used
   only until the generator has written the manifest; SPEC documents this).

A trace is an untrusted input.  It carries a copy of the level entry so the
replay can be self-describing, but the judge thresholds must come from the
canonical manifest: otherwise a forged trace can declare a 40 cm wall with
10 cm thresholds and score 200.  Only fields that affect judging are compared
(category/level/unit and the ``judge`` block); the recorded spawn/command/scene
may legitimately differ and are not judge inputs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parents[3]          # repo root
SCENE_DIR = BASE / "demo/scenes/part2"              # manifest + generated/ scenes
LEVELS_PATH = SCENE_DIR / "part2_official_levels.json"
GENERATED_DIR = SCENE_DIR / "generated"
LEGACY_LEVELS_PATH = BASE / "eval/fixtures/legacy/part2_official_levels.json"
ENV_LEVELS = "ROBOCON_EVAL_LEVELS"
_JUDGE_COMPARE_KEYS = ("category", "level", "unit")


class LevelMismatch(ValueError):
    pass


def levels_path(path: str | Path | None = None) -> Path:
    """Resolve the manifest path (explicit > env > generated > legacy fixture)."""
    if path is not None:
        return Path(path)
    override = os.environ.get(ENV_LEVELS)
    if override:
        return Path(override)
    if LEVELS_PATH.is_file():
        return LEVELS_PATH
    return LEGACY_LEVELS_PATH


def load_levels(path: str | Path | None = None) -> list[dict]:
    doc = json.loads(levels_path(path).read_text())
    return list(doc.get("levels") or [])


def get_level(key: str, path: str | Path | None = None) -> dict:
    for entry in load_levels(path):
        if entry.get("key") == key:
            return entry
    raise LevelMismatch(f"unknown Part2 level {key!r} (manifest {levels_path(path)})")


def level_scene_path(entry: dict) -> Path:
    """Filesystem path of a level's scene XML.

    The manifest stores a bare scene filename relative to ``generated/``; an
    absolute path or a path relative to ``demo/scenes/part2`` is also accepted.
    """
    raw = Path(str(entry["scene"]))
    if raw.is_absolute():
        return raw
    candidates = [GENERATED_DIR / raw, SCENE_DIR / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def level_sha256(entry: dict) -> str:
    payload = json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _numeric_equal(a, b) -> bool:
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return a == b


def canonicalize_level(declared: dict, *, require_key: bool = True,
                       path: str | Path | None = None) -> dict:
    """Return the canonical entry, verifying the trace's declared judge subset."""
    if not isinstance(declared, dict):
        raise LevelMismatch("level metadata is not an object")
    key = declared.get("key")
    if not key:
        if require_key:
            raise LevelMismatch("level metadata has no key")
        return declared
    canonical = get_level(str(key), path)
    for field in _JUDGE_COMPARE_KEYS:
        if field in declared and not _numeric_equal(declared[field], canonical[field]):
            raise LevelMismatch(
                f"level {key}: declared {field}={declared[field]!r}, "
                f"canonical {canonical[field]!r}"
            )
    declared_judge = declared.get("judge") or {}
    if not isinstance(declared_judge, dict):
        raise LevelMismatch(f"level {key}: judge is not an object")
    for field, value in declared_judge.items():
        if field not in canonical.get("judge", {}):
            raise LevelMismatch(f"level {key}: trace adds unknown judge field {field!r}")
        if canonical["judge"][field] != value:
            raise LevelMismatch(
                f"level {key}: declared judge.{field}={value!r}, "
                f"canonical {canonical['judge'][field]!r}"
            )
    return canonical


__all__ = [
    "BASE", "SCENE_DIR", "LEVELS_PATH", "GENERATED_DIR", "LEGACY_LEVELS_PATH",
    "ENV_LEVELS", "LevelMismatch", "levels_path", "load_levels", "get_level",
    "level_scene_path", "level_sha256", "canonicalize_level",
]
