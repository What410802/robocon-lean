#!/usr/bin/env bash
# Build the ROS 2 packages of the demo runtime with colcon.
#
# Usage:  pixi run -e demo demo-build
#         tools/build.sh --packages-select rl_sar      # forward colcon args
set -euo pipefail

root="${PIXI_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# The demo env carries Python/headers under $CONDA_PREFIX.  ROS's
# rosidl_generator_py uses CMake FindPython which does not search the conda
# prefix unless told; pass it explicitly so the self-contained env builds
# without host Python/ROS.
python_args=()
if [ -n "${CONDA_PREFIX:-}" ]; then
    python_args=(
        "-DPython_EXECUTABLE=$CONDA_PREFIX/bin/python"
        "-DPython_ROOT_DIR=$CONDA_PREFIX"
        "-DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python"
        "-DPython3_ROOT_DIR=$CONDA_PREFIX"
    )
fi

cd "$root"
colcon build \
    --paths demo/rl_sar demo/quadruped_ros2 \
    --build-base demo/build \
    --install-base demo/install \
    --symlink-install \
    --cmake-args -DCMAKE_BUILD_TYPE=Release "${python_args[@]}" \
    "$@"

echo
echo "built. source it with:  source demo/install/setup.bash"
