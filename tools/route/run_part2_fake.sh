#!/usr/bin/env bash
# Fake kinematic validation run for the Part2 full-field route.
#
#   tools/route/run_part2_fake.sh [tag]
#
# FAKE_RATE=<hz> sets the fake sim rate (default 50 = real feedback cadence);
# --rate 200 keeps the same sim-time route semantics and runs ~4x faster in
# wall time, which is fine for logic/PASS checks.
#
# Starts tools/route/fake_kinematic_route_sim.py (kinematic, no dynamics) on the
# real route field and runs demo/rl_sar/scripts/part2_auto_route.py against it
# with step-lock enabled; writes artifacts/part2_route3/<tag>_{states,plan}.jsonl
# plus <tag>_{sim,route}.log.  ROS domain 88 + localhost only, so it cannot
# collide with a real MuJoCo launch on the default domain.
set +u  # ROS setup.bash reads unset vars (COLCON_TRACE)
cd "$(dirname "$0")/../.."
source demo/install/setup.bash >/dev/null 2>&1
set -u
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID="${FAKE_DOMAIN_ID:-88}"
export PART1_ROUTE_STEPLOCK=1 PART1_ROUTE_JUDGE=0 PART1_TRANSFER_POLICY=vt_locked12
TAG="${1:-fake}"
OUT="$PWD/artifacts/part2_route3"
mkdir -p "$OUT"
export PART1_RECORD_PATH="$OUT/${TAG}_states.jsonl"
export PART1_RECORD_PLAN_PATH="$OUT/${TAG}_plan.jsonl"
SCENE="$PWD/demo/scenes/part2/full_field/part2_fixed_field.xml"
if pgrep -af "ros2 launch|rl_sim_node|sim_node" | grep -v pgrep >/dev/null; then
  echo "another ROS sim/launch is running; aborting (one launch at a time)" >&2
  exit 3
fi
.pixi/envs/demo/bin/python tools/route/fake_kinematic_route_sim.py \
    --scene "$SCENE" --rate "${FAKE_RATE:-50}" --spawn-x 0 --spawn-y 0 --spawn-yaw 0 \
    > "$OUT/${TAG}_sim.log" 2>&1 &
SIM=$!
sleep 1
timeout "${ROUTE_TIMEOUT:-1150}" .pixi/envs/demo/bin/python demo/rl_sar/scripts/part2_auto_route.py \
    2>&1 | tee "$OUT/${TAG}_route.log"
RC=${PIPESTATUS[0]}
kill "$SIM" 2>/dev/null
wait "$SIM" 2>/dev/null
echo "route exit=$RC (tag=$TAG)"
exit "$RC"
