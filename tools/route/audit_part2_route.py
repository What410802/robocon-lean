#!/usr/bin/env python3
"""Collision / clearance audit for the Part2 full-field auto route.

Two jobs:

1. ``--dump``: load the route scene with MuJoCo and print every non-floor
   collision geom as a world-space XY AABB, grouped by terrain row (y band).
   ``terrain_label*`` / ``contype=0`` visual geoms are ignored.
2. default: import the route's ``AUDIT_SEGMENTS`` path and alignment envelopes
   from ``demo/rl_sar/scripts/part2_route_plan.py``, compute their clearance
   to every non-floor obstacle AABB and fail below the route's requirement.

Clearance semantics (2-D, robot centre path vs obstacle AABB in the XY plane):

* ``transit`` / ``align`` segments: strictly > ``--min-clear`` (0.45 m) to
  every obstacle.  These are corridor moves that must never touch anything.
* ``align_envelope`` rectangles: the entire filled alignment runway must
  clear every obstacle by ``--min-clear``, including the expert's target.
* ``approach`` / ``exit`` segments: only the explicitly named target geoms
  are ignored; everything else must clear ``--min-clear``.
* ``cross`` segments: only those named target geoms are ignored; all others must clear
  ``--min-clear``; additionally the segment is validated to start and end
  outside the crossed row and to pass over the target obstacle.
* ``slot`` stand points: before an expert leg the robot centre must sit in the
  free slot centred between the two row neighbours: clearance to the nearest
  row obstacle (the run-up distance) must be within
  ``[min_runup, max_runup]`` and both lateral neighbours must be clear of the
  robot body (>= 0.5 m).  This catches "stand too close to the step" errors
  that would make the expert climb from a standstill.

Usage (demo or eval Python environment; ROS is not required):

    .pixi/envs/demo/bin/python tools/route/audit_part2_route.py --dump
    .pixi/envs/demo/bin/python tools/route/audit_part2_route.py
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "demo/scenes/part2/full_field/part2_fixed_field.xml"
ROUTE_SCRIPT = ROOT / "demo/rl_sar/scripts/part2_route_plan.py"

ROBOT_PREFIXES = ("trunk", "FL_", "FR_", "RL_", "RR_", "imu_link", "d435_link")
# Planar rows that the route is designed to use as obstacle rows.  Anything
# with |y| <= 0.9 is the y=0 platform row; a bbox belongs to the row if its
# centre is within 0.9 m of a nominal row y.
ROW_YS = (0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 23.3, 27.0, 30.0)


# --------------------------------------------------------------------------
# MuJoCo obstacle extraction
# --------------------------------------------------------------------------
class Obstacle:
    __slots__ = ("name", "geom_id", "xmin", "xmax", "ymin", "ymax", "zmin", "zmax")

    def __init__(self, name: str, gid: int, lo, hi) -> None:
        self.name, self.geom_id = name, gid
        self.xmin, self.ymin, self.zmin = float(lo[0]), float(lo[1]), float(lo[2])
        self.xmax, self.ymax, self.zmax = float(hi[0]), float(hi[1]), float(hi[2])

    def __repr__(self) -> str:
        return (f"{self.name}[{self.xmin:.2f}..{self.xmax:.2f},"
                f"{self.ymin:.2f}..{self.ymax:.2f},zmax={self.zmax:.2f}]")

    @property
    def row_y(self) -> float:
        cy = 0.5 * (self.ymin + self.ymax)
        return min(ROW_YS, key=lambda r: abs(r - cy))


def _geom_aabb(m: mujoco.MjModel, d: mujoco.MjData, gid: int):
    """World-space AABB of one geom (computed from its placement/vertices)."""
    gtype = int(m.geom_type[gid])
    xpos = d.geom_xpos[gid]
    xmat = d.geom_xmat[gid].reshape(3, 3)
    size = m.geom_size[gid]
    if gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
        mid = int(m.geom_dataid[gid])
        vadr = int(m.mesh_vertadr[mid])
        vnum = int(m.mesh_vertnum[mid])
        verts = m.mesh_vert[vadr:vadr + vnum].reshape(-1, 3)
        world = verts @ xmat.T + xpos
        return world.min(axis=0), world.max(axis=0)
    if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        sx, sy, sz = size
        signs = np.array([[sx, sy, sz], [sx, sy, -sz], [sx, -sy, sz], [sx, -sy, -sz],
                          [-sx, sy, sz], [-sx, sy, -sz], [-sx, -sy, sz], [-sx, -sy, -sz]])
        world = signs @ xmat.T + xpos
        return world.min(axis=0), world.max(axis=0)
    if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        r = float(size[0])
        return xpos - r, xpos + r
    # capsule / cylinder / ellipsoid / hfield / plane: conservative sphere bound
    r = float(m.geom_rbound[gid])
    return xpos - r, xpos + r


def load_obstacles(scene: Path = SCENE) -> list[Obstacle]:
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    robot_bodies = {i for i, n in enumerate(names)
                    if n and any(n.startswith(p) for p in ROBOT_PREFIXES)}
    out: list[Obstacle] = []
    for gid in range(m.ngeom):
        if int(m.geom_bodyid[gid]) in robot_bodies:
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom{gid}"
        if name.startswith("terrain_label"):
            continue
        if int(m.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue  # floor
        if int(m.geom_contype[gid]) == 0 and int(m.geom_conaffinity[gid]) == 0:
            continue  # pure visual
        lo, hi = _geom_aabb(m, d, gid)
        if hi[2] < 0.005:  # flat on the floor: nothing to hit
            continue
        out.append(Obstacle(name, gid, lo, hi))
    return out


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def point_seg_distance(px: float, py: float, ax: float, ay: float,
                       bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def aabb_seg_distance(ob: Obstacle, ax: float, ay: float,
                      bx: float, by: float) -> float:
    """2-D distance from a segment to an XY AABB (0 if they intersect)."""
    # Segment intersects the box?
    inside_a = ob.xmin <= ax <= ob.xmax and ob.ymin <= ay <= ob.ymax
    inside_b = ob.xmin <= bx <= ob.xmax and ob.ymin <= by <= ob.ymax
    if inside_a or inside_b:
        return 0.0
    # Segment crosses any box edge?
    edges = [
        (ob.xmin, ob.ymin, ob.xmax, ob.ymin),
        (ob.xmax, ob.ymin, ob.xmax, ob.ymax),
        (ob.xmax, ob.ymax, ob.xmin, ob.ymax),
        (ob.xmin, ob.ymax, ob.xmin, ob.ymin),
    ]
    for ex0, ey0, ex1, ey1 in edges:
        if _segments_cross(ax, ay, bx, by, ex0, ey0, ex1, ey1):
            return 0.0
    return min(
        min(point_seg_distance(ob.xmin, ob.ymin, ax, ay, bx, by),
            point_seg_distance(ob.xmax, ob.ymin, ax, ay, bx, by),
            point_seg_distance(ob.xmax, ob.ymax, ax, ay, bx, by),
            point_seg_distance(ob.xmin, ob.ymax, ax, ay, bx, by)),
        min(point_seg_distance(px, py, ex0, ey0, ex1, ey1)
            for px, py in ((ax, ay), (bx, by))
            for ex0, ey0, ex1, ey1 in edges),
    )


def aabb_rect_distance(ob: Obstacle, bounds) -> float:
    """2-D distance to a filled XY rectangle, including contained obstacles."""
    (xmin, ymin), (xmax, ymax) = bounds
    dx = max(xmin - ob.xmax, ob.xmin - xmax, 0.0)
    dy = max(ymin - ob.ymax, ob.ymin - ymax, 0.0)
    return math.hypot(dx, dy)


def _segments_cross(ax, ay, bx, by, cx, cy, dx, dy) -> bool:
    def orient(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)

    def on_seg(px, py, qx, qy, rx, ry):
        return (min(px, qx) - 1e-9 <= rx <= max(px, qx) + 1e-9
                and min(py, qy) - 1e-9 <= ry <= max(py, qy) + 1e-9)

    o1 = orient(ax, ay, bx, by, cx, cy)
    o2 = orient(ax, ay, bx, by, dx, dy)
    o3 = orient(cx, cy, dx, dy, ax, ay)
    o4 = orient(cx, cy, dx, dy, bx, by)
    if (o1 > 1e-12 and o2 < -1e-12 or o1 < -1e-12 and o2 > 1e-12) and \
       (o3 > 1e-12 and o4 < -1e-12 or o3 < -1e-12 and o4 > 1e-12):
        return True
    if abs(o1) <= 1e-12 and on_seg(ax, ay, bx, by, cx, cy):
        return True
    if abs(o2) <= 1e-12 and on_seg(ax, ay, bx, by, dx, dy):
        return True
    if abs(o3) <= 1e-12 and on_seg(cx, cy, dx, dy, ax, ay):
        return True
    if abs(o4) <= 1e-12 and on_seg(cx, cy, dx, dy, bx, by):
        return True
    return False


# --------------------------------------------------------------------------
# Audit of the planned route polyline
# --------------------------------------------------------------------------
def audit(segments: list[dict], obstacles: list[Obstacle], min_clear: float) -> int:
    failures = 0
    print(f"obstacles: {len(obstacles)} non-floor AABBs from {SCENE.name}")
    print(f"requirement: all non-exempt segments clearance > {min_clear:.2f} m; "
          f"slot run-up in [0.45, 3.50] m, lateral neighbour > 0.50 m\n")
    for seg in segments:
        name = seg["name"]
        kind = seg.get("kind", "transit")
        pts = seg["pts"]
        crossed: list[Obstacle] = []
        ignore_names: set[str] = set()
        note = ""
        target_names = set(seg.get("target_geoms", ()))
        crossed = [ob for ob in obstacles if ob.name in target_names]
        missing = target_names - {ob.name for ob in crossed}
        if missing or (kind not in ("transit", "align_envelope") and not target_names):
            failures += 1
            print(f"[FAIL] {name:30s} kind={kind:8s} "
                  f"missing explicit target geoms: {sorted(missing) or '<none specified>'}")
            continue
        if kind in ("approach", "cross", "exit", "slot"):
            ignore_names = target_names
        if kind == "transit":
            ignore_names = set()
        ok = True
        clearance = 1e9
        if kind == "align_envelope":
            bounds = seg["bounds"]
            clearance, nearest = min(
                ((aabb_rect_distance(ob, bounds), ob.name) for ob in obstacles),
                default=(math.inf, None))
            ok = clearance > min_clear
            (xmin, ymin), (xmax, ymax) = bounds
            note = (f" bounds=({xmin:.2f},{ymin:.2f})..({xmax:.2f},{ymax:.2f})"
                    f" nearest={nearest}")
        elif kind == "slot":
            sx, sy = pts[0]
            if not crossed:
                ok = False
                note = " SLOT_NO_TARGET"
                clearance = 0.0
            else:
                runup = min(aabb_seg_distance(ob, sx, sy, sx, sy) for ob in crossed)
                neighbour_pool = [ob for ob in obstacles
                                  if ob.name not in ignore_names
                                  and aabb_seg_distance(ob, sx, sy, sx, sy) <= 1.5]
                lateral = min((aabb_seg_distance(ob, sx, sy, sx, sy)
                               for ob in neighbour_pool), default=1e9)
                clearance = runup
                ok = (0.45 <= runup <= 3.50) and lateral > 0.50
                note = (f" runup={runup:.2f} lateral_min={lateral:.2f} "
                        f"targets={sorted({ob.name.rsplit('_', 2)[0] for ob in crossed})}")
        else:
            worst = (1e9, None)
            for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
                for ob in obstacles:
                    if ob.name in ignore_names:
                        continue
                    dist = aabb_seg_distance(ob, ax, ay, bx, by)
                    if dist < worst[0]:
                        worst = (dist, (ob.name, (ax, ay), (bx, by)))
            clearance, who = worst
            ok = clearance > min_clear
            if ok and kind in ("approach", "cross", "exit"):
                note = f" exempt={len(crossed)}"
            if kind == "cross":
                (ax, ay), (bx, by) = pts[0], pts[-1]
                if not crossed:
                    ok = False
                    note += " CROSS_NO_TARGET"
                else:
                    dx, dy = bx - ax, by - ay
                    length = math.hypot(dx, dy)
                    if length < 1e-6:
                        ok = False
                        ux = uy = 0.0
                        note += " CROSS_ZERO_LENGTH"
                    else:
                        ux, uy = dx / length, dy / length
                        ps = ax * ux + ay * uy
                        pe = bx * ux + by * uy
                        bad = []
                        for ob in crossed:
                            corners = [(ob.xmin, ob.ymin), (ob.xmax, ob.ymin),
                                       (ob.xmax, ob.ymax), (ob.xmin, ob.ymax)]
                            projs = [cx * ux + cy * uy for cx, cy in corners]
                            lo, hi = min(projs), max(projs)
                            if not (min(ps, pe) < lo - 0.02 and max(ps, pe) > hi + 0.02):
                                bad.append(ob.name)
                        if bad:
                            ok = False
                            note += f" CROSS_ENDPOINTS_INSIDE_TARGET({bad[0]})"
                    note += (f" targets={sorted({ob.name.rsplit('_', 2)[0] for ob in crossed})}"
                             f" dir=({ux:+.2f},{uy:+.2f})")
            if not ok:
                detail = f"{who[0]} at {who[1]}->{who[2]}" if who and who[1] else "n/a"
                note += f"  ({detail})"
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"[{status}] {name:30s} kind={kind:8s} min_clear={clearance:5.2f} m{note}")
    print()
    if failures:
        print(f"AUDIT FAILED: {failures} segment(s)")
    else:
        print("AUDIT PASSED")
    return 0 if not failures else 1


# --------------------------------------------------------------------------
# --dump / route loading
# --------------------------------------------------------------------------
def dump(obstacles: list[Obstacle]) -> None:
    rows: dict[float, list[Obstacle]] = {}
    for ob in obstacles:
        rows.setdefault(ob.row_y, []).append(ob)
    for row_y in sorted(rows):
        obs = rows[row_y]
        print(f"\n=== row y={row_y:6.2f}  ({len(obs)} geoms) ===")
        # cluster by name prefix family and x extent
        for ob in sorted(obs, key=lambda o: (o.xmin, o.name)):
            print(f"  {ob}")


def _load_route_module():
    scripts = str(ROUTE_SCRIPT.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("part2_route_plan", ROUTE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["part2_route_plan"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=str(SCENE))
    ap.add_argument("--dump", action="store_true",
                    help="print the obstacle AABBs by row and exit")
    ap.add_argument("--min-clear", type=float, default=0.45)
    args = ap.parse_args()
    obstacles = load_obstacles(Path(args.scene))
    if args.dump:
        dump(obstacles)
        return 0
    mod = _load_route_module()
    return audit(mod.AUDIT_SEGMENTS, obstacles, args.min_clear)


if __name__ == "__main__":
    raise SystemExit(main())
