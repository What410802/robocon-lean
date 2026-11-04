#!/usr/bin/env bash
# Deployment-stack one-trial harness: enter RL via /joy, hold a forward command,
# record /eval_state at the judged rate, and leave a JSONL trace for
# `pixi run -e eval robocon-eval replay-part2`.
#
# Adapted from the legacy workspace `deployment/tests/run_joy_trial.sh` for this
# repo's `demo/` layout.  Use it to check whether an expert *actually* passes a
# Part2 level through the real ROS deployment stack (the eval-side PolicyRunner
# is a separate implementation; the two can disagree).
#
# Run from the repo root after `pixi run -e demo demo-build`:
#
#   bash demo/tests/run_joy_trial.sh
#   PART=2 LEVEL=high_platform_40cm POLICY=row2p_p40w40_v1 AXIS_VX=0.333 \
#     OUT=$PWD/artifacts/traces/trial_platform40.jsonl SECONDS_REC=25 \
#     bash demo/tests/run_joy_trial.sh
#   pixi run -e eval robocon-eval replay-part2 --trace $PWD/artifacts/traces/trial_platform40.jsonl
#
# Two command modes:
#   ADAPTIVE=0 (default): hold raw /joy axes; AXIS_VX is the raw axes[1] value;
#     the runtime multiplies it by JOY_COMMAND_SCALE (1.5), so AXIS_VX=0.333
#     => body vx 0.5 m/s.
#   ADAPTIVE=1: run `tools/trial/adaptive_joy.py`, which reproduces the
#     training-side adaptive lane command (vy/wz from /eval_state) around
#     VX=<body vx>, CENTER_Y, LANE_YAW_DEG; use this to test an expert the way
#     the Part2 route drives it (the eval-side AdaptiveCommand).
# Cleanup kills only this trial's own process group (setsid); it never pkills
# other launches, so it is safe next to a concurrent route run on another domain.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBOT="${ROBOT:-blackW}"
PART="${PART:-2}"
LEVEL="${LEVEL:-high_platform_40cm}"
POLICY="${POLICY:-row2p_p40w40_v1}"
OUT="${OUT:-$ROOT/artifacts/traces/joy_trial_${LEVEL}.jsonl}"
SECONDS_REC="${SECONDS_REC:-25}"
AXIS_VX="${AXIS_VX:-0.333}"
TIMEOUT="${TIMEOUT:-60}"
LAUNCH_LOG="${LAUNCH_LOG:-/tmp/joy_trial_launch.log}"
REC_LOG="${REC_LOG:-/tmp/joy_trial_rec.log}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-$((150 + RANDOM % 50))}"
mkdir -p "$(dirname "$OUT")"

set +u
source "$ROOT/demo/install/setup.bash"
set -u

SCENE="$ROOT/demo/scenes/part2/generated/part2_off_${LEVEL}.xml"
LEVEL_ARGS=()
if [ "$PART" = "2" ]; then LEVEL_ARGS=(level:="$LEVEL"); fi
: >"$LAUNCH_LOG"
: >"$REC_LOG"

python3 "$ROOT/demo/tests/record_eval_trace.py" --part "$PART" --robot "$ROBOT" \
  --level "$LEVEL" --scene "$SCENE" --seconds "$SECONDS_REC" --out "$OUT" \
  >"$REC_LOG" 2>&1 &
rec=$!

setsid timeout -s INT "$TIMEOUT" ros2 launch rl_sar rl_sim_all.launch.py robot:="$ROBOT" part:="$PART" \
  "${LEVEL_ARGS[@]}" policy_name:="$POLICY" viewer:=false realtime:=true joy:=false \
  publish_every:=${PUBLISH_EVERY:-1} eval_state_every:=${EVAL_EVERY:-1} >"$LAUNCH_LOG" 2>&1 &
simlaunch=$!

pub_joy() {
  ros2 topic pub -r 20 --times 5 /joy sensor_msgs/msg/Joy "$1" >/dev/null 2>&1
}
wait_log() {
  local pat=$1 i
  for i in $(seq 1 40); do
    grep -qa "$pat" "$LAUNCH_LOG" && return 0
    sleep 0.5
  done
  return 1
}

cleanup() {
  # Kill only our own process group (setsid); never pkill by name: a concurrent
  # route launch on another ROS domain must not be affected.
  kill -INT -"$simlaunch" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$simlaunch" 2>/dev/null || break; sleep 0.2; done
  kill -9 -"$simlaunch" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT

# Stand up: retry A until GetUp is logged.
for _ in $(seq 1 6); do
  pub_joy "{buttons: [1,0,0,0,0,0,0,0,0,0,0], axes: [0,0,0,0,0,0,0,0]}"
  wait_log "RLFSMStateGetUp" && break
done
# Enter RL: retry RB+D-pad up until RL_Locomotion is logged.
for _ in $(seq 1 8); do
  pub_joy "{buttons: [0,0,0,0,0,1,0,0,0,0,0], axes: [0,0,0,0,0,0,0,1.0]}"
  wait_log "RLFSMStateRL_Locomotion" && break
done
sleep 1
if [ "${ADAPTIVE:-0}" = "1" ]; then
  # training-side adaptive lane command (vy/wz feedback from /eval_state)
  python3 "$ROOT/tools/trial/adaptive_joy.py" --vx "${VX:-0.5}" \
    --center-y "${CENTER_Y:-0}" --lane-yaw-deg "${LANE_YAW_DEG:-0}" \
    --seconds "$((SECONDS_REC - 8))" >/dev/null 2>&1
elif awk "BEGIN{exit !(${AXIS_VX} != 0)}"; then
  timeout $((SECONDS_REC - 8)) ros2 topic pub -r 20 /joy sensor_msgs/msg/Joy \
    "{buttons: [0,0,0,0,0,0,0,0,0,0,0], axes: [0,${AXIS_VX},0,0,0,0,0,0]}" >/dev/null 2>&1
fi
wait "$rec" 2>/dev/null || true
cleanup
trap - EXIT
echo "launch log: $LAUNCH_LOG"
echo "trace: $OUT"
grep -a "RLFSMStateRL_Locomotion" "$LAUNCH_LOG" | head -2
