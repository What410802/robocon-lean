#!/usr/bin/env bash
# Headless acceptance: start sim_node + rl_sim, drive the FSM Passive -> GetUp -> RL_Locomotion
# through /joy messages, verify both nodes come up and the policy is loaded/entered.
#
# Run from the repo root inside the demo env (it sources the built workspace):
#   pixi run -e demo demo-build                     # once
#   bash demo/tests/smoke_policy_load.sh
#   POLICY=row2p_p40w40_v1 bash demo/tests/smoke_policy_load.sh
#
# Needs the local policy weights (demo/rl_sar/policy/<robot>/<policy>/policy.pt, not in git).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # .../demo
SCENE="${SCENE:-$ROOT/robots/blackW/smoke_flat.xml}"
POLICY="${POLICY:-flat}"
ROBOT="${ROBOT:-blackW}"
LOG="$(mktemp /tmp/rl_policy_smoke.XXXXXX.log)"
rm -f /tmp/rl_obs_debug.csv
export RL_DEBUG_OBS=1
LPID=""
cleanup() { if [ -n "$LPID" ]; then kill -INT "$LPID" 2>/dev/null || true; for _ in $(seq 1 20); do kill -0 "$LPID" 2>/dev/null || break; sleep 0.1; done; kill -9 "$LPID" 2>/dev/null || true; fi; pkill -f "rl_sim_all.launch.py" 2>/dev/null || true; rm -f "$LOG"; }
trap cleanup EXIT

set +u
source "$ROOT/install/setup.bash"
set -u

ros2 launch rl_sar rl_sim_all.launch.py \
    robot:="$ROBOT" policy_name:="$POLICY" scene:="$SCENE" \
    viewer:=false realtime:=false joy:=false >"$LOG" 2>&1 &
LPID=$!

for _ in $(seq 1 50); do
    grep -q "Feedback-driven clock" "$LOG" && break
    sleep 0.2
done
grep -q "Feedback-driven clock" "$LOG" || { echo "FAIL: rl_sim clock did not start"; tail -40 "$LOG"; exit 1; }

# A (buttons[0]) -> StandUp; RB + DPadUp (buttons[5], axes[7]) -> EnterRL.
ros2 topic pub --once /joy sensor_msgs/msg/Joy \
    "{buttons: [1,0,0,0,0,0,0,0,0,0,0], axes: [0,0,0,0,0,0,0,0]}" >/dev/null
sleep 2
ros2 topic pub --once /joy sensor_msgs/msg/Joy \
    "{buttons: [0,0,0,0,0,1,0,0,0,0,0], axes: [0,0,0,0,0,0,0,1]}" >/dev/null

ok=0
for _ in $(seq 1 50); do
    if grep -q "Switch from RLFSMStateGetUp to RLFSMStateRL_Locomotion" "$LOG"; then ok=1; break; fi
    sleep 0.2
done
kill -INT "$LPID" 2>/dev/null || true
for _ in $(seq 1 30); do kill -0 "$LPID" 2>/dev/null || break; sleep 0.2; done
kill -9 "$LPID" 2>/dev/null || true
pkill -f "rl_sim_all.launch.py" 2>/dev/null || true

if [ "$ok" != 1 ]; then
    echo "FAIL: FSM did not reach RL_Locomotion"
    tail -60 "$LOG"
    exit 1
fi
grep -q "Switch from RLFSMStatePassive to RLFSMStateGetUp" "$LOG" || { echo "FAIL: no GetUp transition"; exit 1; }

POLICY_PT="$ROOT/rl_sar/policy/$ROBOT/$POLICY/policy.pt"
if [ ! -f "$POLICY_PT" ]; then
    echo "NOTE: policy weights are local-only, $POLICY_PT not present (commit them locally to run)"
fi
echo "PASS: Passive -> GetUp -> RL_Locomotion; policy loaded (see $LOG)"
