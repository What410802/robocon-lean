#!/usr/bin/env python3
"""Compare two /motor_state CSV traces (P2-11 L1) elementwise by sim_time.

Both CSVs must have the same header layout (`sim_time`, q*, dq*, tau*).  Reports the
first divergence and per-category max abs difference; a category that never matches is
a real behavioral delta (or a different row mapping), not a pass.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path: Path) -> tuple[list[str], dict[float, dict[str, float]]]:
    with path.open() as f:
        r = csv.DictReader(f)
        fields = r.fieldnames or []
        rows = {round(float(x["sim_time"]), 6): {k: float(v) for k, v in x.items()}
                for x in r}
    return fields, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()
    fo, old = load(Path(args.old))
    fn, new = load(Path(args.new))
    if fo != fn:
        raise SystemExit(f"header mismatch: {fo} vs {fn}")
    common = sorted(set(old) & set(new))
    if not common:
        raise SystemExit("no common sim_time samples")
    cats = {"q": "q", "dq": "dq", "tau": "tau"}
    result = {"old": str(args.old), "new": str(args.new), "common_samples": len(common),
              "sim_time": [common[0], common[-1]], "categories": {}}
    for name, prefix in cats.items():
        cols = [c for c in fo if c.startswith(prefix) and c != prefix]
        worst = {"abs": 0.0, "column": None, "sim_time": None}
        first = None
        for t in common:
            for c in cols:
                d = abs(old[t][c] - new[t][c])
                if d > worst["abs"]:
                    worst = {"abs": d, "column": c, "sim_time": t}
                if first is None and d > args.tol:
                    first = {"sim_time": t, "column": c,
                             "old": old[t][c], "new": new[t][c], "abs": d}
        result["categories"][name] = {"max_abs": worst, "first_over_tol": first}
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0 if not any(v.get("first_over_tol") for v in result["categories"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
