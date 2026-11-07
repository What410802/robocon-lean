"""The single Part 2 live-demo route, in field coordinates (metres).

This module has no ROS dependency.  The executor and clearance audit import the
same ordered plan. Difficulty increases within each obstacle row. The 33
crossings cover all physical tiers through platform 30 cm, stair risers 30 cm,
trench 30 cm and wall 40 cm. The 40 cm platform is explicitly omitted with
user approval after unstable descents; it is never counted as crossed.
Slopes stop at the explicitly selected 35-degree requirement; the mapped 40-
and 45-degree practice tiers remain outside that scope. The competition
plum field is the 20 cm, six-cube field at y=12; the taller practice variants
elsewhere in the map are not competition plum obstacles.

Every obstacle row is crossed west-to-east, with the next alignment point
directly ahead in the gap. Waypoints describe corridor corners, not repeated
position corrections.
"""
from __future__ import annotations

import math

ROUTE_START = (0.0, 0.0)
SCENE_RELATIVE = "demo/scenes/part2/full_field/part2_fixed_field.xml"
# User-approved temporary omission after both row and 1.8 m south-face
# approaches climbed 40 cm but failed the descent. Never count it as crossed.
SKIPPED_TIERS = {"platform": {40: "unstable descent; user-approved demo omission"}}


def _chain(x0: float, x1: float, y: float, hop: float = 4.0):
    """A straight corridor with bounded spacing, excluding its start."""
    count = max(1, math.ceil(abs(x1 - x0) / hop))
    return [(x0 + (x1 - x0) * i / count, y) for i in range(1, count + 1)]


def _leg(name, category, difficulty, geoms, stand, cross_end, *, yaw=0.0,
         transit=(), exit=(), top=0.0, bounds=None, **options):
    entry = {
        "leg": name, "category": category, "difficulty": difficulty,
        "target_geoms": tuple(geoms), "stand": tuple(stand),
        "stand_center": tuple(stand), "lane_x": float(stand[0]),
        "lane_y": float(stand[1]), "lane_yaw": yaw,
        "transit": list(transit), "cross_end": tuple(cross_end),
        "exit": list(exit), "top_m": top, "target_bounds": bounds,
        "transit_speed": 0.22, "settle_forward": 0.05,
        "along_min": -0.10, "along_max": 0.15,
        "align_xy_tol": 0.08, "align_yaw_tol_deg": 2.5,
        "align_deadline_s": 45.0, "cross_timeout": 45.0,
        "exit_overshoot_tol": 0.45,
        "align_backoff": 0.0, "no_creep": False,
        "no_recover": False, "start_dy": 0.0,
    }
    entry.update(options)
    return entry


def _build_legs():
    legs = []

    # Follow every selected platform in its physical order. Keep the
    # same expert moving through the 1.5 m gaps instead of looping around
    # each platform. The final gap also provides the northbound row exit.
    for index, cm in enumerate((5, 10, 15, 20, 25, 30)):
        cx = 3.75 + 3.0 * index
        legs.append(_leg(
            f"high_platform_{cm:02d}cm", "platform", cm,
            (f"platform_{cm:02d}cm",),
            (1.2 if index == 0 else cx - 1.60, 0.0),
            (cx + (1.10 if cm == 30 else 1.40), 0.0),
            exit=[(20.15, 1.5)] if cm == 30 else [],
            top=cm / 100,
            bounds=((cx - 0.75, -0.8, 0.0), (cx + 0.75, 0.8, cm / 100)),
            settle_forward=0.60 if index == 0 else 0.0,
            along_max=0.70 if index == 0 else 0.15,
            align_xy_tol=0.07 if index == 0 else 0.20,
            align_yaw_tol_deg=3.0 if index == 0 else 10.0,
            continuous_lane=True, no_creep=index != 0, no_recover=True,
            cross_settle_s=0.50 if cm == 30 else 0.0,
            # Brake before leaving the row; the maximum settled x=20.40
            # keeps 60 cm centre clearance from the omitted 40 cm platform.
            exit_overshoot_tol=0.55 if cm == 30 else 0.25,
            align_deadline_s=60.0, cross_timeout=30.0))

    # Each staircase has six 0.3 m treads, a 0.8 m landing, then six down.
    # The row gaps are 3.1 m wide, so successive legs need no corridor loop.
    stair_levels = ((4, 3.0), (6, 10.5), (8, 18.0), (10, 25.5),
                    (12, 33.0), (15, 40.5), (20, 48.0),
                    (25, 55.5), (30, 63.0))
    for index, (cm, x0) in enumerate(stair_levels):
        prefix = f"stairs_{cm:02d}cm"
        geoms = ([f"{prefix}_up_{i}" for i in range(1, 7)] +
                 [f"{prefix}_landing"] +
                 [f"{prefix}_down_{i}" for i in range(1, 7)])
        transit = (_chain(20.15, 0.7, 1.5) + [(0.7, 3.0)]
                   if index == 0 else [])
        legs.append(_leg(
            prefix, "stairs", cm, geoms,
            (x0 - (1.8 if index == 0 else 2.55), 3.0), (x0 + 4.95, 3.0),
            transit=transit,
            exit=[(68.6, 3.0), (68.6, 4.5)] if cm == 30 else [],
            top=6 * cm / 100,
            bounds=((x0, 2.3, 0.0), (x0 + 4.4, 3.7, 6 * cm / 100)),
            settle_forward=1.10 if cm == 30 else 0.60,
            along_max=1.60 if cm == 30 else 1.0,
            # Keep the validated forward gait through the flat gaps. Stopping
            # to turn this actor can roll it back onto the preceding staircase.
            continuous_lane=True, no_recover=True,
            # A 25 cm descent can leave band with residual lateral/yaw motion.
            # The 3.1 m ground gap permits one measured locked entry for 30 cm.
            reset_expert=cm == 30,
            cross_settle_s=0.50 if cm == 30 else 0.0,
            align_xy_tol=(0.10 if cm >= 25 else (0.12 if index == 0 else 0.30)),
            align_yaw_tol_deg=2.5 if index == 0 or cm >= 25 else 10.0,
            cross_timeout=120.0 if cm >= 25 else 45.0,
            stall_window_s=60.0 if cm >= 25 else 12.0))

    # The 1.2 m gaps between slopes are too narrow for turns.  Stay in the
    # lane throughout the row and align only with forward progress.
    for index, deg in enumerate((5, 10, 15, 20, 25, 30, 35)):
        x0 = 3.0 + 6.0 * index
        stand_x = 1.2 if index == 0 else x0 - 0.6
        prefix = f"slope_{deg:02d}deg"
        rise = 2.0 * math.tan(math.radians(deg))
        legs.append(_leg(
            prefix, "slope", deg,
            tuple(f"{prefix}_{suffix}" for suffix in ("up", "landing", "down")),
            (stand_x, 6.0), (x0 + 5.35, 6.0),
            transit=_chain(68.6, 0.7, 4.5) + [(0.7, 6.0)]
                    if index == 0 else [],
            exit=[(44.4, 7.5)] if deg == 35 else [],
            top=rise, bounds=((x0, 5.3, 0.0), (x0 + 4.8, 6.7, rise)),
            settle_forward=0.90 if index == 0 else 0.05,
            along_max=1.0 if index == 0 else 0.15,
            align_deadline_s=60.0 if index == 0 else 45.0,
            exit_overshoot_tol=0.15,
            no_creep=index != 0, no_recover=True,
            continuous_lane=True,
            align_xy_tol=0.08 if index == 0 else 0.20,
            align_yaw_tol_deg=2.5 if index == 0 else 6.0))

    for index, cm in enumerate((10, 20, 30)):
        x0, x1 = ((3.0, 7.1), (9.0, 13.2), (15.0, 19.3))[index]
        prefix = f"trench_{cm}cm"
        legs.append(_leg(
            prefix, "trench", cm,
            tuple(f"{prefix}_{suffix}" for suffix in ("entry", "near", "far", "exit")),
            (1.2 if cm == 10 else (8.15 if cm == 20 else 13.75), 9.0),
            (8.15 if cm == 10 else x1 + 0.55, 9.0),
            transit=_chain(44.4, 0.7, 7.5) + [(0.7, 9.0)]
                    if index == 0 else [],
            exit=[(20.1, 9.0), (20.1, 10.5)] if cm == 30 else [],
            top=0.30, bounds=((x0, 8.3, 0.0), (x1, 9.7, 0.30)),
            settle_forward=0.90 if index == 0 else 0.10,
            along_max=1.0 if index == 0 else (0.3 if cm == 20 else 0.6),
            align_deadline_s=60.0 if index == 0 else 45.0,
            # The first gate includes the full ground-support hold while
            # moving. Its maximum allowance still stops before the next ramp.
            exit_overshoot_tol=0.25 if index == 0 else 0.60,
            # The two low tiers share an actor. The 30 cm tier needs its
            # dedicated gap expert; runtime must measure the policy-change
            # gate instead of treating that change as same-actor continuation.
            continuous_lane=True, no_creep=index != 0,
            no_recover=True,
            cross_settle_s=0.50 if cm == 30 else 0.0,
            align_xy_tol=0.08 if index == 0 else (0.10 if cm == 30 else 0.30),
            align_yaw_tol_deg=2.5 if index == 0 else (2.0 if cm == 30 else 6.0),
            cross_timeout=35.0))

    plum_geoms = ("plum_entry", "plum_r1_01", "plum_r2_01", "plum_r1_02",
                  "plum_r2_02", "plum_r1_03", "plum_r2_03", "plum_exit")
    legs.append(_leg(
        "high_plum", "plum", 20, plum_geoms, (2.2, 12.0), (9.0, 12.0),
        # Complete the necessary right turn south of the expert lane. The
        # locked connector drifts north during that turn; arriving on y=12
        # first consumed the runway correcting the resulting lateral offset.
        transit=_chain(20.1, 1.2, 10.5) + [(1.2, 11.8)],
        exit=[(9.8, 12.0), (9.8, 13.2)], top=0.2,
        bounds=((4.0, 11.6, 0.0), (7.5, 12.4, 0.2)),
        cube_x_bounds=(4.9, 6.6),
        align_xy_tol=0.06, align_yaw_tol_deg=1.5,
        align_approach=(1.2, 11.8),
        settle_forward=1.0, along_max=0.15, align_deadline_s=75.0,
        cross_timeout=60.0,
        no_recover=True, handover_policy="vt_band_v1", handover_x=6.85,
        handover_settle_s=2.0, handover_vx=0.45,
        handover_brake_s=0.6, handover_brake_vx=-0.30))

    # Read the physical wall heights, including both 30 and 40 cm rule tiers.
    # The field has extra 4/8/12/16 cm walls and no physical 10 cm wall.
    for index, cm in enumerate((4, 8, 12, 16, 20, 30, 40)):
        centre = 3.0 + index * 1.2
        stand_x = 1.2 if index == 0 else centre - 0.6
        legs.append(_leg(
            f"high_wall_{cm:02d}cm", "wall", cm, (f"cross_bar_{index + 1}",),
            (stand_x, 21.0), (centre + (0.5 if cm == 40 else 0.6), 21.0),
            transit=(_chain(9.8, 0.7, 13.2) + [(0.7, 21.0)])
                    if index == 0 else [],
            top=cm / 100,
            bounds=((centre - 0.06, 20.3, 0.0),
                    (centre + 0.06, 21.7, cm / 100)),
            settle_forward=0.60 if index == 0 else 0.0,
            along_max=0.70 if index == 0 else 0.10,
            # Brake slightly earlier after the final selected wall. The next
            # (unselected) wall still bounds the available stopping distance.
            exit_overshoot_tol=0.18 if cm == 40 else 0.08,
            no_creep=index != 0, no_recover=True,
            align_xy_tol=0.12 if index == 0 else (0.10 if cm >= 30 else 0.20),
            continuous_lane=True, cross_settle_s=1.0 if cm == 40 else 0.0,
            # This is uninterrupted tracking by an already active expert.
            # A 0.35 x 0.30 m half-footprint at 10 degrees, plus 20 cm lane
            # error, leaves >14 cm margin inside the 1.4 m wall span.
            align_yaw_tol_deg=3.0 if index == 0 else (5.0 if cm >= 30 else 10.0),
            cross_timeout=25.0))
    return legs


LEGS = _build_legs()


def _audit_segments():
    segments = []
    previous = ROUTE_START

    def add(name, kind, points, leg=None):
        if len(points) > 1 and points[0] == points[1]:
            return
        segment = {"name": name, "kind": kind, "pts": points}
        if leg is not None:
            segment["target_geoms"] = leg["target_geoms"]
            segment["cross_pts"] = [leg["stand"], leg["cross_end"]]
        segments.append(segment)

    for leg in LEGS:
        name = leg["leg"]
        for point in leg["transit"]:
            add(name + "_transit", "transit", [previous, point])
            previous = point
        approach = tuple(leg.get("align_approach", leg["stand"]))
        add(name + "_approach", "approach", [previous, approach], leg)
        ux, uy = math.cos(leg["lane_yaw"]), math.sin(leg["lane_yaw"])
        # Alignment can begin before the expert's measured switch gate.  Plum
        # uses this runway to settle at x=2.2 rather than consume it after 2.2.
        approach_along = ((approach[0] - leg["stand"][0]) * ux
                          + (approach[1] - leg["stand"][1]) * uy)
        forward = min(leg["settle_forward"],
                      max(0.0, leg["along_max"] - approach_along))
        settle = (approach[0] + forward * ux,
                  approach[1] + forward * uy)
        if "align_approach" in leg:
            # Audit the whole shallow S-turn runway, not a straight line at
            # the incoming corner's lateral offset. The corner is accepted
            # within 15 cm and is not reacquired within 18 cm; reserve that
            # margin around both its lane offset and the final lane line.
            margin = max(0.18, leg["align_xy_tol"])
            nx, ny = -uy, ux
            approach_lateral = ((approach[0] - leg["stand"][0]) * nx
                                + (approach[1] - leg["stand"][1]) * ny)
            along_bounds = (min(approach_along - margin, leg["along_min"]),
                            max(approach_along + margin, leg["along_max"]))
            lateral_bounds = (min(approach_lateral, 0.0) - margin,
                              max(approach_lateral, 0.0) + margin)
            corners = [(leg["stand"][0] + along * ux + lateral * nx,
                        leg["stand"][1] + along * uy + lateral * ny)
                       for along in along_bounds for lateral in lateral_bounds]
            xmin, xmax = min(p[0] for p in corners), max(p[0] for p in corners)
            ymin, ymax = min(p[1] for p in corners), max(p[1] for p in corners)
            segments.append({
                "name": name + "_alignment", "kind": "align_envelope",
                "bounds": ((xmin, ymin), (xmax, ymax)),
                "pts": [(xmin, ymin), (xmax, ymin), (xmax, ymax),
                        (xmin, ymax), (xmin, ymin)],
            })
        else:
            add(name + "_settle", "approach", [approach, settle], leg)
        add(name + "_slot", "slot", [leg["stand"]], leg)
        add(name + "_cross", "cross", [leg["stand"], leg["cross_end"]], leg)
        # This checks the permitted settled gate position, not an extra drive
        # waypoint or a prediction of the policy's physical braking distance.
        overshoot = leg["exit_overshoot_tol"]
        maximum_exit = (leg["cross_end"][0] + overshoot * ux,
                        leg["cross_end"][1] + overshoot * uy)
        add(name + "_exit_allowance", "exit", [leg["cross_end"], maximum_exit], leg)
        previous = leg["cross_end"]
        for point in leg["exit"]:
            add(name + "_exit", "exit", [previous, point], leg)
            previous = point
    return segments


AUDIT_SEGMENTS = _audit_segments()
