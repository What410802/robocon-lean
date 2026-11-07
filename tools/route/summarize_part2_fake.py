#!/usr/bin/env python3
"""Summarize a Part2 fake-route run log into a PASS/FAIL table + timings.

    .pixi/envs/demo/bin/python tools/route/summarize_part2_fake.py \
        artifacts/part2_route3/v2_route.log [artifacts/part2_route3/v2_plan.jsonl]

Prints one row per leg (alignment residual at switch, cross gate, end pose) and
the total sim time from the first/last /eval_state timestamp in the plan file.
Exit code 0 only if every leg PASSed and the route finished.
"""
from __future__ import annotations

import json
import re
import sys

PAT_SWITCH = re.compile(
    r"expert switch leg=(\S+) policy=(\S+) .*actual=\(x=([+\-\d.]+),y=([+\-\d.]+),"
    r"yaw=([+\-\d.]+)deg\) e=([+\-\d.]+) along=([+\-\d.]+) "
    r"yaw_err=([+\-\d.]+)deg .*?(OK|ALIGN_FAIL)")
PAT_LEG = re.compile(
    r"(\S+): (PASS|FAIL) policy=(\S+) vx=([+\-\d.]+) end=\(([+\-\d.]+),([+\-\d.]+),"
    r"yaw=([+\-\d.]+)deg\) gate=(\w+) post=(\w+).*")
PAT_START = re.compile(r"route start at")
PAT_LEGS = re.compile(r"=== leg (\d+)/(\d+) (\S+) ")
PAT_DONE = re.compile(r"Part2 route finished: (\d+)/(\d+) legs PASS")


def main() -> int:
    log_path = sys.argv[1]
    plan_path = sys.argv[2] if len(sys.argv) > 2 else ""
    text = open(log_path, encoding="utf-8", errors="replace").read()
    switches = {}
    results = {}
    order: list[str] = []
    for line in text.splitlines():
        m = PAT_SWITCH.search(line)
        if m:
            switches[m.group(1)] = dict(policy=m.group(2), x=float(m.group(3)),
                                        y=float(m.group(4)), yaw=float(m.group(5)),
                                        e=float(m.group(6)),
                                        along=float(m.group(7)),
                                        yaw_err=float(m.group(8)),
                                        status=m.group(9))
            continue
        m = PAT_LEG.search(line)
        if m and m.group(2) in ("PASS", "FAIL"):
            results[m.group(1)] = dict(verdict=m.group(2), policy=m.group(3),
                                       vx=float(m.group(4)), x=float(m.group(5)),
                                       y=float(m.group(6)), yaw=float(m.group(7)),
                                       gate=m.group(8), post=m.group(9))
            order.append(m.group(1))
    total = None
    if plan_path:
        try:
            ts = [json.loads(l).get("t", 0.0) for l in open(plan_path, encoding="utf-8")]
            ts = [t for t in ts if t]
            if ts:
                total = max(ts) - min(ts)
        except OSError:
            pass
    print(f"{'leg':24s} {'verdict':7s} {'policy':31s} {'vx':>5s} "
          f"{'e':>7s} {'along':>7s} {'yaw_err':>8s}  end=(x,y,yaw)")
    npass = nfail = 0
    for leg in order:
        r = results[leg]
        s = switches.get(leg, {})
        v = "PASS" if r["verdict"] == "PASS" else "FAIL"
        npass += v == "PASS"
        nfail += v == "FAIL"
        if s:
            corr = (f"{s['e']:+7.3f} {s['along']:+7.3f} "
                    f"{s['yaw_err']:+8.1f}")
        else:
            corr = f"{'-':>7s} {'-':>7s} {'-':>8s}"
        print(f"{leg:24s} {v:7s} {r['policy']:31s} {r['vx']:+5.2f} {corr}  "
              f"({r['x']:+.2f},{r['y']:+.2f},{r['yaw']:+.1f}deg) "
              f"gate={r['gate']} post={r['post']}")
    completion = PAT_DONE.search(text)
    finished = bool(completion and completion.group(1) == completion.group(2)
                    and len(results) == int(completion.group(2)))
    print(f"\nlegs: {npass} PASS / {nfail} FAIL; route_finished={finished}"
          + (f"; plan sim span={total:.1f}s" if total else ""))
    if not results:
        print("no leg results found - route aborted before the first report "
              "(check the log for 'alignment failed'/'aborted')")
        return 1
    return 0 if (nfail == 0 and finished) else 1


if __name__ == "__main__":
    raise SystemExit(main())
