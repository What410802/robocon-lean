#!/usr/bin/env python3
"""Dump every registered task's environment config to JSON for golden comparison.

Migration tool: the pre-migration code (git branch ``legacy/ppo`` / ``legacy/him``)
and the migrated code must produce identical ``ManagerBasedRlEnvCfg`` structures
for every task except the documented intentional changes (physics timestep).
The dump converts dataclasses / tuples / numpy scalars to plain JSON so a
byte-for-byte ``diff`` is meaningful.

Usage (from the training repo root, in the environment that can import the code):

    python scripts/checks/dump_env_cfgs.py --out /tmp/cfgs.json
    python scripts/checks/dump_env_cfgs.py --tasks black-flat blackw-official --out x.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if dataclasses.is_dataclass(value):
        return {
            "__cls__": type(value).__module__ + "." + type(value).__qualname__,
            **{f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)},
        }
    if hasattr(value, "__dict__"):
        return {
            "__cls__": type(value).__module__ + "." + type(value).__qualname__,
            **{k: _jsonable(v) for k, v in vars(value).items() if not k.startswith("_")},
        }
    if isinstance(value, property):
        return {"__property__": value.fget.__qualname__ if value.fget else repr(value)}
    if callable(value):
        return {"__callable__": getattr(value, "__qualname__", repr(value))}
    return repr(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--play", action="store_true", help="dump play-mode configs instead")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    # Import the task package so the registry is populated (entry point).
    import alldog_mjlab  # noqa: F401
    from mjlab.tasks.registry import list_tasks, load_env_cfg

    tasks = args.tasks or list_tasks()
    payload = {}
    for task in sorted(tasks):
        try:
            cfg = load_env_cfg(task, play=args.play)
        except Exception as exc:  # noqa: BLE001 - dump what we can
            payload[task] = {"__error__": f"{type(exc).__name__}: {exc}"}
            continue
        payload[task] = _jsonable(cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ok = sum(1 for v in payload.values() if "__error__" not in v)
    print(f"dumped {ok}/{len(payload)} tasks -> {args.out}")
    return 0 if ok == len(payload) else 1


if __name__ == "__main__":
    raise SystemExit(main())
