#!/usr/bin/env python3
"""Compare every pseudo-REPL CLI plan event with the actual base motion.

Usage:
  python tools/route/compare_cli_motion.py PLAN.jsonl STATES.jsonl \
      [--out-prefix artifacts/route/cli_motion_compare]

For an event at t_i, the actual motion is base(t_i) -> base(t_{i+1}).
`move(...)` is checked against its relative fwd/left (start-yaw frame);
`turn(...)` and `turn_hold(...)` are checked against their relative yaw;
`turn_hold` additionally must keep position.  Switch/forward/mode events do
not have an endpoint, so their actual displacement is reported only.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
import statistics
from pathlib import Path

MOVE_RE = re.compile(
    r"move\(fwd=([+-]?\d+(?:\.\d+)?)m,left=([+-]?\d+(?:\.\d+)?)m"
    r"(?:,turn=([+-]?\d+(?:\.\d+)?)deg)?\)"
)
TURN_RE = re.compile(r"turn\(([+-]?\d+(?:\.\d+)?)deg\)")
TURN_HOLD_RE = re.compile(r"turn_hold\(([+-]?\d+(?:\.\d+)?)deg\)")


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_of(q) -> float:
    w, x, y, z = (float(v) for v in q[:4])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", type=Path)
    ap.add_argument("states", type=Path)
    ap.add_argument("--out-prefix", type=Path,
                    default=Path("artifacts/route/cli_motion_compare"))
    args = ap.parse_args()

    plan = [json.loads(l) for l in args.plan.read_text().splitlines() if l.strip()]
    plan.sort(key=lambda e: float(e.get("t", 0.0)))
    states = [json.loads(l) for l in args.states.read_text().splitlines() if l.strip()]
    states.sort(key=lambda e: float(e.get("t", 0.0)))
    st = [float(s.get("t", 0.0)) for s in states]

    def sample(t: float):
        i = bisect.bisect_left(st, t)
        if i >= len(states):
            i = len(states) - 1
        return states[i]

    # Collapse repeated plan events for the same primitive execution (e.g. a
    # goto emits several updates); compare first-emitted CLI at t0 with the
    # state at the next different command.
    groups = []
    for ev in plan:
        key = (ev.get("leg"), ev.get("cmd"))
        if groups and groups[-1][0] == key:
            groups[-1][1].append(ev)
        else:
            groups.append((key, [ev]))

    rows = []
    for gi, (key, evs) in enumerate(groups):
        ev = evs[0]
        t0 = float(ev.get("t", 0.0))
        if gi + 1 < len(groups):
            t1 = float(groups[gi + 1][1][0].get("t", t0))
        else:
            t1 = t0
        if t1 - t0 < 0.04:
            continue
        cli = str(ev.get("cli") or "")
        p0, p1 = sample(t0), sample(t1)
        b0, b1 = p0["base"], p1["base"]
        dx, dy = float(b1[0]) - float(b0[0]), float(b1[1]) - float(b0[1])
        y0, y1 = yaw_of(p0["quat"]), yaw_of(p1["quat"])
        dyaw = math.degrees(wrap(y1 - y0))
        dist = math.hypot(dx, dy)
        row = {
            "t": t0, "dt": t1 - t0, "leg": ev.get("leg", ""), "cli": cli,
            "cmd": ev.get("cmd", ""), "dx": dx, "dy": dy,
            "dist": dist, "dyaw_deg": dyaw, "kind": "other",
            "expected_dx": None, "expected_dy": None, "expected_dyaw": None,
            "pos_err": None, "yaw_err": None, "verdict": "",
        }
        mm = MOVE_RE.search(cli)
        tm = TURN_RE.search(cli)
        thm = TURN_HOLD_RE.search(cli)
        if mm:
            fwd, left = float(mm.group(1)), float(mm.group(2))
            turn = float(mm.group(3)) if mm.group(3) else 0.0
            ex = fwd * math.cos(y0) - left * math.sin(y0)
            ey = fwd * math.sin(y0) + left * math.cos(y0)
            pos_err = math.hypot(dx - ex, dy - ey)
            has_turn = mm.group(3) is not None
            yaw_err = abs(wrap(y1 - y0 - math.radians(turn))) if has_turn else None
            row.update(kind="move", expected_dx=ex, expected_dy=ey,
                       expected_dyaw=turn if has_turn else None, pos_err=pos_err,
                       yaw_err=math.degrees(yaw_err) if yaw_err is not None else None,
                       verdict="REVIEW" if (pos_err > 0.15 or (yaw_err is not None and yaw_err > math.radians(10))) else "ok")
        elif thm:
            exp = float(thm.group(1))
            yaw_err = abs(wrap(y1 - y0 - math.radians(exp)))
            row.update(kind="turn_hold", expected_dx=0.0, expected_dy=0.0,
                       expected_dyaw=exp, pos_err=dist,
                       yaw_err=math.degrees(yaw_err),
                       verdict="REVIEW" if (dist > 0.15 or yaw_err > math.radians(12)) else "ok")
        elif tm:
            exp = float(tm.group(1))
            yaw_err = abs(wrap(y1 - y0 - math.radians(exp)))
            row.update(kind="turn", expected_dx=0.0, expected_dy=0.0,
                       expected_dyaw=exp, pos_err=dist,
                       yaw_err=math.degrees(yaw_err),
                       verdict="REVIEW" if (dist > 0.10 or yaw_err > math.radians(10)) else "ok")
        else:
            # speed/mode/path commands have no endpoint; report actual only.
            row["verdict"] = "info"
        rows.append(row)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with args.out_prefix.with_suffix(".json").open("w") as fh:
        json.dump(rows, fh, indent=2)
    with args.out_prefix.with_suffix(".csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)

    def summarize(kind):
        rr = [r for r in rows if r["kind"] == kind]
        if not rr:
            return None
        return {
            "events": len(rr),
            "review": sum(r["verdict"] == "REVIEW" for r in rr),
            "pos_err_median": statistics.median(r["pos_err"] for r in rr),
            "pos_err_max": max(r["pos_err"] for r in rr),
            "yaw_err_median": statistics.median([r["yaw_err"] for r in rr if r["yaw_err"] is not None]) if any(r["yaw_err"] is not None for r in rr) else None,
            "yaw_err_max": max([r["yaw_err"] for r in rr if r["yaw_err"] is not None]) if any(r["yaw_err"] is not None for r in rr) else None,
        }

    summary = {k: summarize(k) for k in ("move", "turn", "turn_hold")}
    print(json.dumps(summary, indent=2))
    print("\nWorst 10 events:")
    for r in sorted(rows, key=lambda z: -((z["pos_err"] or 0.0) + 0.01 * (z["yaw_err"] or 0.0)))[:10]:
        print(f"t={r['t']:7.2f} {r['leg']:<16} {r['kind']:<9} "
              f"cli={r['cli']:<46} actual=({r['dx']:+.2f},{r['dy']:+.2f},"
              f"{r['dyaw_deg']:+.1f}deg) pos_err={r['pos_err']} "
              f"yaw_err={r['yaw_err']} {r['verdict']}")
    print(f"saved {args.out_prefix.with_suffix('.json')} / .csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
