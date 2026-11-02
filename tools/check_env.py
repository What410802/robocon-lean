#!/usr/bin/env python3
"""Print the versions and capabilities of the environment this runs in.

Used as `pixi run -e <env> env-check`: it detects which of the three roles the
environment can serve (train / demo / eval) and prints one JSON object. The
same output is stored under artifacts/ after environment changes (Phase-0 and
later regression checks); it is data, not documentation.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import os
import platform
import shutil
import sys
from pathlib import Path


def _version(package: str) -> str | None:
    try:
        return md.version(package)
    except md.PackageNotFoundError:
        return None


def _import(name: str):
    try:
        return __import__(name)
    except Exception as exc:  # noqa: BLE001 - probe only
        return exc


def _demo_probes() -> dict[str, bool]:
    """C++ / ROS pieces the demo environment must provide."""
    prefix = Path(sys.prefix)
    return {
        "colcon": shutil.which("colcon") is not None,
        "ros2": shutil.which("ros2") is not None,
        "rclpy": not isinstance(_import("rclpy"), Exception),
        "libtorch_cmake": any((prefix / sub / "cmake" / "Torch").is_dir()
                              for sub in ("lib", "share")),
        "mujoco_header": (prefix / "include" / "mujoco" / "mujoco.h").is_file(),
        "tbb_cmake": (prefix / "lib" / "cmake" / "TBB").is_dir(),
        "yaml_cpp": (prefix / "include" / "yaml-cpp").is_dir(),
    }


def main() -> int:
    checks: dict[str, object] = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "packages": {
            name: _version(name)
            for name in (
                "torch",
                "mujoco",
                "mujoco-warp",
                "mjlab",
                "rsl-rl-lib",
                "warp-lang",
                "numpy",
                "alldog-mjlab",
                "robocon-eval",
            )
        },
    }

    torch = _import("torch")
    if not isinstance(torch, Exception):
        checks["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
        }
    mujoco = _import("mujoco")
    if not isinstance(mujoco, Exception):
        checks["mujoco"] = {"version": mujoco.__version__}

    roles: list[str] = []
    mjlab = _import("mjlab")
    alldog = _import("alldog_mjlab")
    if not isinstance(mjlab, Exception) and not isinstance(alldog, Exception):
        roles.append("train")
        try:
            from mjlab.tasks.registry import list_tasks

            checks["tasks"] = sorted(list_tasks())
        except Exception as exc:  # noqa: BLE001
            checks["tasks_error"] = f"{type(exc).__name__}: {exc}"

    if "CONDA_PREFIX" in os.environ:
        probes = _demo_probes()
        checks["demo_probes"] = probes
        if all(probes.values()):
            roles.append("demo")
        elif probes["colcon"] and probes["rclpy"]:
            roles.append("demo-partial")

    if not isinstance(_import("robocon_eval"), Exception):
        roles.append("eval")

    expected_role = Path(sys.prefix).name
    checks["expected_role"] = expected_role
    checks["roles"] = roles
    checks["ok"] = (expected_role in roles) if expected_role in ("train", "demo", "eval") else bool(roles)
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    return 0 if checks["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
