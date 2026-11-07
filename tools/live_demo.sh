#!/usr/bin/env bash
# Launch the canonical route in the demo pixi environment; no route data lives here.
# Usage: pixi run -e demo demo-part1 [viewer:=false ...]
#        pixi run -e demo demo-part2 [viewer:=false ...]
set -euo pipefail

root="${PIXI_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
route="${1:-}"
case "$route" in
    part1|part2) shift ;;
    *) echo "usage: tools/live_demo.sh part1|part2 [name:=value ...]" >&2; exit 2 ;;
esac

if [ ! -f "$root/demo/install/setup.bash" ]; then
    echo "demo runtime not built; run: pixi run -e demo demo-build" >&2
    exit 1
fi
cd "$root"
set +u
# shellcheck disable=SC1091
source demo/install/setup.bash
set -u

export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
exec python tools/live_demo.py "$route" "$@"
