#!/usr/bin/env python3
"""Single source of truth for the Part 2 "plum" (pile field) geometry.

Owner decision (2026-11-02)
---------------------------
The Part 2 plum field is a **no-ramp, 3-pile single-row** layout.  The older
official ``map.xml`` plum field (1.2 m entry/exit ramps + two staggered rows of
12 cubes) is *not* used for the intra-squad competition: the team owner replaced
it with this simplified field so the obstacle can be judged and trained with the
same geometry.

Confirmed geometry (metres, spawn at x=0, lane centred on y=0)::

    entry platform : 0.8 x 0.8 x 0.2  (top z = 0.2), x in [1.8, 2.6], y in [-0.4, 0.4]
    cubes          : 3 cubes 0.2 x 0.2 x 0.2, single row along +x, centres
                     x = 2.8, 3.4, 4.0 (centre spacing 0.6), y = 0
    exit platform  : 0.8 x 0.8 x 0.2  (top z = 0.2), x in [4.2, 5.0], y in [-0.4, 0.4]

Gaps: 0.1 m between the entry-platform far edge (x=2.6) and the first cube
(x=2.7); 0.1 m between the last cube (x=4.1) and the exit-platform near edge
(x=4.2).  There are no ramps: the robot steps up 0.2 m onto the entry platform at
x=1.8 and steps down 0.2 m from the exit platform at x=5.0 to normal ground.

The entry platform starts exactly at the Part 2 protocol runway (1.8 m ahead of
the spawn), so the level scenes need no x translation for plum.

What this module provides
-------------------------
* ``build_spec()`` / ``PLUM_SPEC``: constant + derived geometry as plain JSON
  data, exported to ``terrain/plum/plum_spec.json``.
* ``boxes()``: the 5 collision boxes (``plum_entry``, ``plum_r1_01..03``,
  ``plum_exit``) as dicts in the scene frame, optionally translated.
* ``mjcf_geoms()``: the same boxes as MJCF ``<geom>`` lines (used by the level
  scene generator and directly runnable in MuJoCo).
* ``write_spec()`` / CLI: write the spec JSON.

Usage (from the repository root)::

    python terrain/gen_plum.py                    # writes terrain/plum/plum_spec.json
    python terrain/gen_plum.py --out /tmp/plum.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Owner-confirmed constants (single place to edit; everything below derives).
# ---------------------------------------------------------------------------
START_X_M = 1.8          # = Part 2 protocol runway; entry platform near face
PLATFORM_SIZE_M = (0.8, 0.8, 0.2)   # entry/exit platform: size_x, size_y, height
CUBE_SIZE_M = (0.2, 0.2, 0.2)       # pile cube size
NUM_CUBES = 3
CUBE_CENTER_SPACING_M = 0.6         # distance between adjacent cube CENTRES
GAP_ENTRY_TO_FIRST_CUBE_M = 0.1     # entry platform far edge -> first cube near face
GAP_LAST_CUBE_TO_EXIT_M = 0.1       # last cube far face -> exit platform near edge
CENTER_Y_M = 0.0                    # lane centre

# Scene-frame x positions (derived; kept as floats so callers can read them).
# Derived values are rounded to 1e-9 m so the exported JSON has no float noise.
PLATFORM_HALF_X_M = round(PLATFORM_SIZE_M[0] / 2.0, 9)
PLATFORM_HALF_Y_M = round(PLATFORM_SIZE_M[1] / 2.0, 9)
PLATFORM_HEIGHT_M = PLATFORM_SIZE_M[2]
PLATFORM_TOP_Z_M = PLATFORM_HEIGHT_M

CUBE_HALF_M = round(CUBE_SIZE_M[0] / 2.0, 9)
CUBE_TOP_Z_M = CUBE_SIZE_M[2]

ENTRY_PLATFORM_X_MIN_M = START_X_M
ENTRY_PLATFORM_X_MAX_M = round(START_X_M + PLATFORM_SIZE_M[0], 9)
FIRST_CUBE_CENTER_X_M = round(
    ENTRY_PLATFORM_X_MAX_M + GAP_ENTRY_TO_FIRST_CUBE_M + CUBE_HALF_M, 9)
CUBE_CENTERS_X_M = tuple(
    round(FIRST_CUBE_CENTER_X_M + i * CUBE_CENTER_SPACING_M, 9)
    for i in range(NUM_CUBES)
)
LAST_CUBE_CENTER_X_M = CUBE_CENTERS_X_M[-1]
EXIT_PLATFORM_X_MIN_M = round(
    LAST_CUBE_CENTER_X_M + CUBE_HALF_M + GAP_LAST_CUBE_TO_EXIT_M, 9)
EXIT_PLATFORM_X_MAX_M = round(EXIT_PLATFORM_X_MIN_M + PLATFORM_SIZE_M[0], 9)
EXIT_GROUND_X_M = EXIT_PLATFORM_X_MAX_M
TOTAL_X_RANGE_M = (ENTRY_PLATFORM_X_MIN_M, EXIT_PLATFORM_X_MAX_M)

# MJCF appearance: same green / friction / condim as the organiser map geoms.
PLUM_RGBA = ".3 .75 .4 1"
PLUM_FRICTION = "1 .005 .0001"
PLUM_CONDIM = 3


def _check_geometry() -> None:
    """Fail loudly if the owner-confirmed layout is self-inconsistent."""
    tol = 1e-9

    def _close(a: float, b: float) -> bool:
        return abs(a - b) <= tol

    gaps = (
        _close(ENTRY_PLATFORM_X_MAX_M + GAP_ENTRY_TO_FIRST_CUBE_M,
               CUBE_CENTERS_X_M[0] - CUBE_HALF_M),
        _close(LAST_CUBE_CENTER_X_M + CUBE_HALF_M + GAP_LAST_CUBE_TO_EXIT_M,
               EXIT_PLATFORM_X_MIN_M),
        _close(EXIT_PLATFORM_X_MIN_M + PLATFORM_SIZE_M[0], EXIT_PLATFORM_X_MAX_M),
        _close(CUBE_CENTERS_X_M[1] - CUBE_CENTERS_X_M[0], CUBE_CENTER_SPACING_M),
    )
    if not all(gaps):
        raise AssertionError(
            "plum geometry is inconsistent (check the owner-confirmed constants): "
            f"entry[1.8,2.6] cubes={CUBE_CENTERS_X_M} exit[{EXIT_PLATFORM_X_MIN_M},"
            f"{EXIT_PLATFORM_X_MAX_M}]"
        )
    if PLATFORM_TOP_Z_M != CUBE_TOP_Z_M:
        raise AssertionError(
            "entry/exit platforms and cube tops must share one height "
            f"({PLATFORM_TOP_Z_M} != {CUBE_TOP_Z_M})"
        )


_check_geometry()


def boxes(dx: float = 0.0, dy: float = 0.0) -> list[dict]:
    """Collision boxes in the level-scene frame (optionally translated by dx/dy).

    Each entry: ``{"name", "center_m": (x, y, z), "half_size_m": (hx, hy, hz),
    "top_z_m"}``.  Boxes sit on the ground (z from 0 to their height).
    """
    out = [
        {
            "name": "plum_entry",
            "center_m": (round(ENTRY_PLATFORM_X_MIN_M + PLATFORM_HALF_X_M + dx, 9),
                         round(CENTER_Y_M + dy, 9),
                         round(PLATFORM_HEIGHT_M / 2.0, 9)),
            "half_size_m": (PLATFORM_HALF_X_M, PLATFORM_HALF_Y_M,
                            PLATFORM_HEIGHT_M / 2.0),
            "top_z_m": PLATFORM_TOP_Z_M,
        }
    ]
    for i, cx in enumerate(CUBE_CENTERS_X_M, start=1):
        out.append({
            "name": f"plum_r1_{i:02d}",
            "center_m": (round(cx + dx, 9), round(CENTER_Y_M + dy, 9),
                         round(CUBE_TOP_Z_M / 2.0, 9)),
            "half_size_m": (CUBE_HALF_M, CUBE_HALF_M, CUBE_TOP_Z_M / 2.0),
            "top_z_m": CUBE_TOP_Z_M,
        })
    out.append({
        "name": "plum_exit",
        "center_m": (round(EXIT_PLATFORM_X_MIN_M + PLATFORM_HALF_X_M + dx, 9),
                     round(CENTER_Y_M + dy, 9),
                     round(PLATFORM_HEIGHT_M / 2.0, 9)),
        "half_size_m": (PLATFORM_HALF_X_M, PLATFORM_HALF_Y_M,
                        PLATFORM_HEIGHT_M / 2.0),
        "top_z_m": PLATFORM_TOP_Z_M,
    })
    return out


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def mjcf_geoms(dx: float = 0.0, dy: float = 0.0, indent: str = "      ") -> list[str]:
    """MJCF <geom> lines for the plum field, optionally translated by dx/dy."""
    lines = []
    for box in boxes(dx=dx, dy=dy):
        cx, cy, cz = box["center_m"]
        hx, hy, hz = box["half_size_m"]
        lines.append(
            f'{indent}<geom name="{box["name"]}" type="box" '
            f'pos="{_fmt(cx)} {_fmt(cy)} {_fmt(cz)}" '
            f'size="{_fmt(hx)} {_fmt(hy)} {_fmt(hz)}" '
            f'rgba="{PLUM_RGBA}" friction="{PLUM_FRICTION}" '
            f'condim="{PLUM_CONDIM}" />'
        )
    return lines


def plum_section() -> dict:
    """The ``categories.plum`` spec section consumed by training and levels."""
    return {
        "layout": "3-pile single row, no ramps (owner decision 2026-11-02)",
        "no_ramps": True,
        "start_x_m": START_X_M,
        "runway_m": START_X_M,
        "platform_size_m": list(PLATFORM_SIZE_M),
        "platform_half_x_m": PLATFORM_HALF_X_M,
        "platform_half_y_m": PLATFORM_HALF_Y_M,
        "platform_height_m": PLATFORM_HEIGHT_M,
        "platform_top_z_m": PLATFORM_TOP_Z_M,
        "cube_size_m": list(CUBE_SIZE_M),
        "cube_half_m": CUBE_HALF_M,
        "cube_top_m": CUBE_TOP_Z_M,
        "num_cubes": NUM_CUBES,
        "cube_centers_x_m": list(CUBE_CENTERS_X_M),
        "cube_center_y_m": CENTER_Y_M,
        "cube_center_spacing_m": CUBE_CENTER_SPACING_M,
        "gap_entry_to_first_cube_m": GAP_ENTRY_TO_FIRST_CUBE_M,
        "gap_last_cube_to_exit_m": GAP_LAST_CUBE_TO_EXIT_M,
        "entry_platform_x_range_m": [ENTRY_PLATFORM_X_MIN_M, ENTRY_PLATFORM_X_MAX_M],
        "exit_platform_x_range_m": [EXIT_PLATFORM_X_MIN_M, EXIT_PLATFORM_X_MAX_M],
        "exit_ground_x_m": EXIT_GROUND_X_M,
        "step_down_m": PLATFORM_HEIGHT_M,
        "total_x_range_m": list(TOTAL_X_RANGE_M),
    }


def build_spec() -> dict:
    """Full JSON-able plum spec (constants + derived boxes + provenance)."""
    section = plum_section()
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "file": "gen_plum.py",
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rule_basis": (
                "Owner decision 2026-11-02: Part 2 plum = no ramps; entry/exit "
                "platforms 0.8x0.8x0.2 m (top z=0.2 m) centred on y=0; 3 cubes "
                "0.2x0.2x0.2 m in a single row along +x with cube-centre spacing "
                "0.6 m; 0.1 m gap between the entry platform and the first cube, "
                "0.1 m between the last cube and the exit platform; robot steps "
                "down 0.2 m from the exit platform to normal ground."
            ),
        },
        "plum": section,
        "boxes": [
            {
                "name": box["name"],
                "center_m": list(box["center_m"]),
                "half_size_m": list(box["half_size_m"]),
                "top_z_m": box["top_z_m"],
            }
            for box in boxes()
        ],
    }


def write_spec(out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(build_spec(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out


def main() -> int:
    default_out = Path(__file__).resolve().parent / "plum" / "plum_spec.json"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=default_out,
                        help=f"output spec JSON (default: {default_out})")
    args = parser.parse_args()
    out = write_spec(args.out)
    spec = build_spec()
    print(f"wrote {out}")
    print(f"  plum total x range: {spec['plum']['total_x_range_m']} m")
    print(f"  entry platform: x {spec['plum']['entry_platform_x_range_m']}, "
          f"top z={spec['plum']['platform_top_z_m']} m")
    print(f"  cubes: centres x={spec['plum']['cube_centers_x_m']} "
          f"(spacing {spec['plum']['cube_center_spacing_m']} m), "
          f"size {spec['plum']['cube_size_m']} m")
    print(f"  exit platform: x {spec['plum']['exit_platform_x_range_m']}, "
          f"step down {spec['plum']['step_down_m']} m at x="
          f"{spec['plum']['exit_ground_x_m']} m")
    for box in spec["boxes"]:
        print(f"  geom {box['name']:12s} center={box['center_m']} "
              f"half_size={box['half_size_m']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
