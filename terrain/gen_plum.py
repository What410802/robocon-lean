#!/usr/bin/env python3
"""Part 2 plum (梅花桩) geometry — rebuilt from the competition PDF figure.

The rules PDF says the plum parameters in the organiser ``map.xml`` are wrong and
asks teams to build the obstacle from the figure on page 4.  The figure specifies:

* two entry/exit platforms, ``800 x 800 x 200 mm``;
* six ``200 x 200 x 200 mm`` cubes in two staggered rows (three per row);
* same-row edge gap ``400 mm`` (cube-centre spacing ``600 mm``);
* row edge gap ``200 mm`` (row centre lines at ``y = +/- 0.2 m``);
* first cube to platform gaps ``100 mm`` (row 1) and ``400 mm`` (row 2);
* overall platform outer span ``3500 mm``, centre-symmetric.

Scene frame: the robot spawns at ``x = 0``, the entry platform starts ``1.8 m``
ahead (the Part 2 protocol runway), the lane centre is ``y = 0``, and the exit
platform far edge is the step-down to normal ground.

This module is the single source of truth for the plum geometry: the generated
judge level (``terrain/gen_part2_levels.py``), the training terrain
(``train/.../profiles/part2/terrain.py``) and the training success predicate all
read the JSON written by ``write_spec`` / ``gen_part2_spec.py``.

Usage (repository root)::

    python terrain/gen_plum.py              # writes terrain/plum/plum_spec.json
    python terrain/gen_plum.py --out /tmp/plum.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# PDF figure dimensions (metres).  Edit only here; everything below derives.
# ---------------------------------------------------------------------------
START_X_M = 1.8                     # entry-platform near face = protocol runway
PLATFORM_SIZE_M = (0.8, 0.8, 0.2)   # entry/exit platform (x, y, height)
CUBE_SIZE_M = (0.2, 0.2, 0.2)       # pile cube
SAME_ROW_EDGE_GAP_M = 0.4           # gap between cubes in one row
ROW_EDGE_GAP_M = 0.2                # gap between the two rows
ROW_CENTRE_OFFSET_M = 0.2           # row centre lines at y = +/- 0.2
FIRST_CUBE_GAP_M = (0.1, 0.4)       # entry far edge -> first cube, row 1 / row 2
CENTRE_Y_M = 0.0

PLATFORM_HALF_X_M = PLATFORM_SIZE_M[0] / 2.0
PLATFORM_HALF_Y_M = PLATFORM_SIZE_M[1] / 2.0
PLATFORM_HEIGHT_M = PLATFORM_SIZE_M[2]
PLATFORM_TOP_Z_M = PLATFORM_HEIGHT_M
CUBE_HALF_M = CUBE_SIZE_M[0] / 2.0
CUBE_TOP_Z_M = CUBE_SIZE_M[2]

ENTRY_X_MIN_M = START_X_M
ENTRY_X_MAX_M = ENTRY_X_MIN_M + PLATFORM_SIZE_M[0]

# Cube centres: row 1 starts 0.1 m after the entry platform, row 2 0.4 m after
# it; within a row the cube centres are 0.2 m (cube) + 0.4 m (gap) apart.
_ROW1_FIRST_CENTRE = ENTRY_X_MAX_M + FIRST_CUBE_GAP_M[0] + CUBE_HALF_M
_ROW2_FIRST_CENTRE = ENTRY_X_MAX_M + FIRST_CUBE_GAP_M[1] + CUBE_HALF_M
_ROW_SPACING_M = CUBE_SIZE_M[0] + SAME_ROW_EDGE_GAP_M
CUBE_CENTRES = (
    tuple((_ROW1_FIRST_CENTRE + _ROW_SPACING_M * i, +ROW_CENTRE_OFFSET_M)
          for i in range(3))
    + tuple((_ROW2_FIRST_CENTRE + _ROW_SPACING_M * i, -ROW_CENTRE_OFFSET_M)
            for i in range(3))
)
NUM_CUBES = len(CUBE_CENTRES)
NUM_ROWS = 2

# Exit platform: 0.4 m after the far row-1 cube and 0.1 m after the far row-2
# cube (mirrors the entry side), which also gives the PDF's 3.5 m outer span.
EXIT_X_MIN_M = _ROW1_FIRST_CENTRE + _ROW_SPACING_M * 2 + CUBE_HALF_M + 0.4
EXIT_X_MAX_M = EXIT_X_MIN_M + PLATFORM_SIZE_M[0]
EXIT_GROUND_X_M = EXIT_X_MAX_M

CUBE_X_MIN_M = min(c[0] for c in CUBE_CENTRES) - CUBE_HALF_M
CUBE_X_MAX_M = max(c[0] for c in CUBE_CENTRES) + CUBE_HALF_M
CUBE_Y_MIN_M = min(c[1] for c in CUBE_CENTRES) - CUBE_HALF_M
CUBE_Y_MAX_M = max(c[1] for c in CUBE_CENTRES) + CUBE_HALF_M
TOTAL_X_RANGE_M = (ENTRY_X_MIN_M, EXIT_X_MAX_M)
CENTRE_X_M = (TOTAL_X_RANGE_M[0] + TOTAL_X_RANGE_M[1]) / 2.0

# Same appearance as the organiser map's plum geoms.
PLUM_RGBA = ".3 .75 .4 1"
PLUM_FRICTION = "1 .005 .0001"
PLUM_CONDIM = 3


def _check_geometry() -> None:
    """Fail loudly if the derived layout stops matching the PDF dimensions."""
    tol = 1e-9

    def close(a: float, b: float) -> bool:
        return abs(float(a) - float(b)) <= tol

    checks = [
        ("same-row centre spacing", _ROW_SPACING_M, 0.6),
        ("row edge gap", ROW_EDGE_GAP_M, 2 * ROW_CENTRE_OFFSET_M - CUBE_SIZE_M[1]),
        ("entry->row1 first gap", CUBE_CENTRES[0][0] - CUBE_HALF_M - ENTRY_X_MAX_M,
         FIRST_CUBE_GAP_M[0]),
        ("entry->row2 first gap", CUBE_CENTRES[3][0] - CUBE_HALF_M - ENTRY_X_MAX_M,
         FIRST_CUBE_GAP_M[1]),
        ("row1 last->exit gap", EXIT_X_MIN_M - (CUBE_CENTRES[2][0] + CUBE_HALF_M), 0.4),
        ("row2 last->exit gap", EXIT_X_MIN_M - (CUBE_CENTRES[5][0] + CUBE_HALF_M), 0.1),
        ("platform outer span", EXIT_X_MAX_M - ENTRY_X_MIN_M, 3.5),
        ("platform heights", PLATFORM_TOP_Z_M, CUBE_TOP_Z_M),
    ]
    for name, got, want in checks:
        if not close(got, want):
            raise AssertionError(f"plum geometry: {name} = {got:.6f}, expected {want}")

    # 180-degree centre symmetry maps row 1 to row 2 and back.
    for (x, y) in CUBE_CENTRES:
        mirror = (2 * CENTRE_X_M - x, -y)
        if not any(close(mirror[0], cx) and close(mirror[1], cy)
                   for cx, cy in CUBE_CENTRES):
            raise AssertionError(f"plum geometry: cube {(x, y)} has no centre mirror")
    if not close(EXIT_X_MAX_M - CENTRE_X_M, CENTRE_X_M - ENTRY_X_MIN_M):
        raise AssertionError("plum geometry: platforms are not centre-symmetric")


_check_geometry()


def boxes(dx: float = 0.0, dy: float = 0.0) -> list[dict]:
    """Collision boxes in the level-scene frame (optionally translated)."""
    def box(name: str, centre, half) -> dict:
        return {
            "name": name,
            "center_m": (round(centre[0] + dx, 9), round(centre[1] + dy, 9),
                         round(centre[2], 9)),
            "half_size_m": tuple(round(v, 9) for v in half),
            "top_z_m": centre[2] + half[2],
        }

    plat = (PLATFORM_HALF_X_M, PLATFORM_HALF_Y_M, PLATFORM_HEIGHT_M / 2.0)
    cube = (CUBE_HALF_M, CUBE_HALF_M, CUBE_TOP_Z_M / 2.0)
    out = [box("plum_entry",
               (ENTRY_X_MIN_M + PLATFORM_HALF_X_M, CENTRE_Y_M, PLATFORM_HEIGHT_M / 2.0),
               plat)]
    for i, (x, y) in enumerate(CUBE_CENTRES, start=1):
        row = 1 if y > 0 else 2
        index = i if row == 1 else i - 3
        out.append(box(f"plum_r{row}_{index:02d}", (x, y, CUBE_TOP_Z_M / 2.0), cube))
    out.append(box("plum_exit",
                   (EXIT_X_MIN_M + PLATFORM_HALF_X_M, CENTRE_Y_M,
                    PLATFORM_HEIGHT_M / 2.0), plat))
    return out


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def mjcf_geoms(dx: float = 0.0, dy: float = 0.0, indent: str = "      ") -> list[str]:
    """MJCF ``<geom>`` lines for the plum field, optionally translated."""
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
    """Geometry section consumed by the level generator and training terrain."""
    return {
        "layout": "official PDF: entry/exit platforms + 6 cubes in two staggered rows",
        "no_ramps": True,
        "start_x_m": START_X_M,
        "runway_m": START_X_M,
        "platform_size_m": list(PLATFORM_SIZE_M),
        "platform_half_x_m": round(PLATFORM_HALF_X_M, 9),
        "platform_half_y_m": round(PLATFORM_HALF_Y_M, 9),
        "platform_height_m": round(PLATFORM_HEIGHT_M, 9),
        "platform_top_z_m": round(PLATFORM_TOP_Z_M, 9),
        "cube_size_m": list(CUBE_SIZE_M),
        "cube_half_m": round(CUBE_HALF_M, 9),
        "cube_top_m": round(CUBE_TOP_Z_M, 9),
        "num_cubes": NUM_CUBES,
        "num_rows": NUM_ROWS,
        "row_centre_offset_m": ROW_CENTRE_OFFSET_M,
        "cube_centre_spacing_m": round(_ROW_SPACING_M, 9),
        "same_row_edge_gap_m": SAME_ROW_EDGE_GAP_M,
        "row_edge_gap_m": ROW_EDGE_GAP_M,
        "first_cube_gap_m": list(FIRST_CUBE_GAP_M),
        "cube_centres": [[round(x, 9), round(y, 9)] for x, y in CUBE_CENTRES],
        "cube_x_range_m": [round(CUBE_X_MIN_M, 9), round(CUBE_X_MAX_M, 9)],
        "cube_y_range_m": [round(CUBE_Y_MIN_M, 9), round(CUBE_Y_MAX_M, 9)],
        "entry_platform_x_range_m": [round(ENTRY_X_MIN_M, 9), round(ENTRY_X_MAX_M, 9)],
        "exit_platform_x_range_m": [round(EXIT_X_MIN_M, 9), round(EXIT_X_MAX_M, 9)],
        "exit_ground_x_m": round(EXIT_GROUND_X_M, 9),
        "step_down_m": round(PLATFORM_HEIGHT_M, 9),
        "total_x_range_m": [round(TOTAL_X_RANGE_M[0], 9), round(TOTAL_X_RANGE_M[1], 9)],
        "centre_x_m": round(CENTRE_X_M, 9),
    }


def build_spec() -> dict:
    """Full JSON-able plum spec (section + collision boxes + provenance)."""
    return {
        "source": {
            "file": "terrain/gen_plum.py",
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rule_basis": (
                "Competition rules PDF page 4 plum figure: 800x800x200 mm "
                "entry/exit platforms, six 200x200x200 mm cubes in two staggered "
                "rows, same-row edge gap 400 mm, row edge gap 200 mm, first-cube "
                "gaps 100/400 mm, overall span 3500 mm, centre-symmetric."
            ),
        },
        "plum": plum_section(),
        "boxes": [
            {"name": b["name"], "center_m": list(b["center_m"]),
             "half_size_m": list(b["half_size_m"]), "top_z_m": b["top_z_m"]}
            for b in boxes()
        ],
    }


def write_spec(out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_spec(), indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    return out


def main() -> int:
    default_out = Path(__file__).resolve().parent / "plum" / "plum_spec.json"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=default_out)
    args = parser.parse_args()
    out = write_spec(args.out)
    sec = plum_section()
    print(f"wrote {out}")
    print(f"  plum span: {sec['total_x_range_m']} m, cubes: {sec['cube_centres']}")
    print(f"  entry {sec['entry_platform_x_range_m']} exit {sec['exit_platform_x_range_m']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
