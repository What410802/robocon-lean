#!/usr/bin/env python3
"""Verify that vendored third-party binary assets match assets/manifest.json.

Usage:  python assets/verify.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "assets/manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    files = json.loads(MANIFEST.read_text())["files"]
    bad = 0
    for rel, meta in files.items():
        path = ROOT / rel
        if not path.is_file():
            print(f"MISSING {rel}")
            bad += 1
        elif sha256(path) != meta["sha256"]:
            print(f"HASH MISMATCH {rel}")
            bad += 1
    print(f"{len(files) - bad}/{len(files)} assets verified")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
