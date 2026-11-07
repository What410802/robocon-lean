#!/usr/bin/env bash
# Run the curated demo sanity checks.
#
# These are cheap contract/geometry checks, not a MuJoCo rollout:
#   - robot manifests / policy contracts / spawn definitions
#   - Part1 and Part2 scene XMLs load in MuJoCo
#
# Usage:  pixi run -e eval demo-tests    (needs mujoco + pyyaml only)
set -euo pipefail

root="${PIXI_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$root"

python="${PYTHON:-python}"
fail=0

run() {
    echo "== $*"
    "$@" || fail=1
}

run "$python" demo/tests/check_robot_manifests.py
run "$python" demo/tests/check_policy_contracts.py
run "$python" demo/tests/check_spawns.py
run "$python" demo/tests/check_part2_route.py
run "$python" demo/tests/check_part2_replay.py

# MuJoCo load smoke for the two judge scenes.
"$python" - <<'PY' || fail=1
import mujoco
paths = [
    "demo/robots/blackW/scene_part1.xml",
    "demo/scenes/part2/generated/part2_off_high_platform_40cm.xml",
    "demo/scenes/part2/generated/part2_off_plum_1.xml",
]
for p in paths:
    m = mujoco.MjModel.from_xml_path(p)
    print(f"OK {p}: nq={m.nq} nu={m.nu} ngeom={m.ngeom}")
PY

exit "$fail"
