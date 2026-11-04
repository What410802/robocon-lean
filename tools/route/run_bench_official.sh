#!/usr/bin/env bash
# Deployment-side expert bench on an official Part2 single-level scene.
#   tools/route/run_bench_official.sh <level> <policy> <vx> <seconds> <lane-yaw> [tag]
# Starts the real sim_node+rl_sim launch (ROS domain 87, realtime=1x) and drives
# the policy with the route's adaptive lane command via
# tools/route/bench_expert_cross.py (wall-clock pacing, NOT steplock: at
# realtime=false the sim outruns the route, which invalidates sim-time tests).
set +u
cd "$(dirname "$0")/../.."
source demo/install/setup.bash >/dev/null 2>&1
set -u
LEVEL="${1:?level e.g. high_platform_40cm}"
POLICY="${2:?policy dir}"
VX="${3:-0.5}"
SECS="${4:-12}"
LANE_YAW="${5:-0}"
TAG="${6:-bench}"
OUT="$PWD/artifacts/part2_route3"
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=87 PART1_ROUTE_STEPLOCK=1
if pgrep -af "rl_sim_node|quadruped_ros2/sim_node" | grep -v pgrep >/dev/null; then
  echo "another real sim is running; aborting" >&2
  exit 3
fi
timeout 120 ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=2 \
    level:="$LEVEL" policy_name:="$POLICY" viewer:=false realtime:=true \
    joy:=false eval_state_every:=10 > "$OUT/${TAG}_launch.log" 2>&1 &
LPID=$!
sleep 6
timeout 80 .pixi/envs/demo/bin/python tools/route/bench_expert_cross.py \
    --vx "$VX" --seconds "$SECS" --lane-yaw "$LANE_YAW" --steplock 0 \
    --out "$OUT/${TAG}_states.jsonl" > "$OUT/${TAG}_bench.log" 2>&1
BRC=$?
kill -INT "$LPID" 2>/dev/null
wait "$LPID" 2>/dev/null
echo "bench rc=$BRC (tag=$TAG)"
exit "$BRC"
