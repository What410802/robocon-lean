#!/usr/bin/env bash
# Source the colcon workspace built by tools/build.sh.
#
#   source tools/activate.sh
#
# In the demo pixi environment this is also done by
# `pixi run -e demo ...` (the pixi task environment is not auto-sourced so that
# train/eval never load ROS's interpreter); source it manually when you built the
# workspace in your own shell.
set -u

root="${PIXI_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [ "$(basename "${CONDA_PREFIX:-}")" != "demo" ] && [ -z "${ROS_DISTRO:-}" ]; then
    echo "warning: not in the demo pixi env; source it with 'pixi shell -e demo' first" >&2
fi

if [ -f "$root/demo/install/setup.bash" ]; then
    # shellcheck disable=SC1090
    set +u
    source "$root/demo/install/setup.bash"
    set -u
else
    echo "demo/install/setup.bash not found — run 'pixi run -e demo demo-build' first" >&2
    return 1 2>/dev/null || exit 1
fi
