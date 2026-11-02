#!/usr/bin/env bash
# Smoke: headless sim, enter RL via /joy, then switch the selected policies with the gamepad
# combos that docs/joystick.md §3 documents. Asserts the rl_sim log lines, so it proves the
# binding -> intent -> SwitchPolicy -> InitRL chain end to end without a physical gamepad.
#
#   bash demo/tests/smoke_policy_switch.sh            # Part2 robot (blackW, default level)
#   PART=1 bash demo/tests/smoke_policy_switch.sh     # Part1 + fixed drives
#   POLICY=row2p_p40w40_v1 ...                        # direct launch by policy name
#
# Run from the repo root in the demo env (sources demo/install/setup.bash).
# Needs the local policy weights (not in git). Exit 0 = every expected log line appeared and
# no [ERROR]/dead process. Logs stay in /tmp.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # .../demo
PART="${PART:-2}"
LEVEL="${LEVEL:-high_platform_10cm}"
ROBOT="${ROBOT:-blackW}"
POLICY="${POLICY:-flat}"   # initial policy loaded on RL entry (also proves direct launch by policy name)
OUT="${OUT:-/tmp/policy_switch_smoke}"
mkdir -p "$OUT"
LAUNCH_LOG="$OUT/launch.log"
fail=0
export ROS_DOMAIN_ID=$(( (RANDOM % 90) + 10 ))

LEVEL_ARGS=()
if [ "$PART" = "2" ]; then LEVEL_ARGS=(level:="$LEVEL"); fi
timeout -s INT 40 bash -lc "source $ROOT/install/setup.bash && \
    ros2 launch rl_sar rl_sim_all.launch.py robot:=$ROBOT part:=$PART \
    ${LEVEL_ARGS[*]} policy_name:=\"$POLICY\" viewer:=false realtime:=false joy:=false \
    eval_state_every:=10 publish_every:=10" >"$LAUNCH_LOG" 2>&1 &
launch_pid=$!

pub_joy() { ros2 topic pub -r 20 --times 4 /joy sensor_msgs/msg/Joy "$1" >/dev/null 2>&1; }
wait_log() {  # $1 = fixed string; retries for up to 6 s
    local pat=$1 i
    for i in $(seq 1 24); do
        grep -qaF "$pat" "$LAUNCH_LOG" && return 0
        sleep 0.25
    done
    return 1
}

expect() {  # $1 = description, $2 = fixed string
    if wait_log "$2"; then
        echo "PASS  $1"
    else
        echo "FAIL  $1  (missing: $2)"
        fail=1
    fi
}

# --- enter RL (same gesture as the old joy trial): A stand up, then RB + D-pad up.
for _ in $(seq 1 8); do
    pub_joy "{buttons: [1,0,0,0,0,0,0,0,0,0,0], axes: [0,0,0,0,0,0,0,0]}"
    wait_log "RLFSMStateGetUp" && break
done
for _ in $(seq 1 8); do
    pub_joy "{buttons: [0,0,0,0,0,1,0,0,0,0,0], axes: [0,0,0,0,0,0,0,1.0]}"
    wait_log "RLFSMStateRL_Locomotion" && break
done
sleep 0.5
expect "entered RL with initial policy $POLICY" "Entered RL_Locomotion"
if [ "$POLICY" != "flat" ]; then
    # same-policy switch is a documented no-op, so seed a different one first.
    pub_joy "{buttons: [0,0,0,0,0,0,0,1,0,0,0], axes: [0,0,0,0,0,0,-1.0,0]}"   # Start + left
    expect "seed switch to flat (so the next switch is not a no-op)" "[policy] switched to flat"
fi

# --- selected policy switches: Start+D-pad down = PolicyMain, RB+D-pad down = PolicyTStairs.
pub_joy "{buttons: [0,0,0,0,0,0,0,1,0,0,0], axes: [0,0,0,0,0,0,0,-1.0]}"   # Start + down
expect "Start+D-pad down -> row2p_p40w40_v1" "[policy] switched to row2p_p40w40_v1"

pub_joy "{buttons: [0,0,0,0,0,1,0,0,0,0,0], axes: [0,0,0,0,0,0,0,-1.0]}"   # RB + down
expect "RB+D-pad down -> st30_ppo_st30_s100_600" "[policy] switched to st30_ppo_st30_s100_600"

pub_joy "{buttons: [0,0,0,0,0,1,0,0,0,0,0], axes: [0,0,0,0,0,0,1.0,0]}"    # RB + right
expect "RB+D-pad right -> s1d_trench_reward_v75_m100" "[policy] switched to s1d_trench_reward_v75_m100"

if [ "$PART" = "1" ]; then
    # fixed-action states are Part1-only: Back+D-pad down = low_bar_drive.
    pub_joy "{buttons: [0,0,0,0,0,0,1,0,0,0,0], axes: [0,0,0,0,0,0,0,-1.0]}"  # Back + down
    expect "Back+D-pad down -> RLFSMStateLowBarDrive" "RLFSMStateLowBarDrive"
    # return to RL and re-check that a policy switch works from there again.
    for _ in $(seq 1 8); do
        pub_joy "{buttons: [0,0,0,0,0,1,0,0,0,0,0], axes: [0,0,0,0,0,0,0,1.0]}"  # RB + up
        wait_log "Entered RL_Locomotion" && break
    done
    pub_joy "{buttons: [0,0,0,0,0,0,0,1,0,0,0], axes: [0,0,0,0,0,0,0,-1.0]}"
    expect "back in RL -> switch to row2p again" "[policy] switched to row2p_p40w40_v1"
fi

kill -INT "$launch_pid" 2>/dev/null || true
pkill -f "rl_sim_all.launch.py.*robot:=$ROBOT.*part:=$PART" 2>/dev/null || true
wait "$launch_pid" 2>/dev/null || true
wait 2>/dev/null || true

if grep -qaE "\[ERROR\]|process has died" "$LAUNCH_LOG"; then
    echo "FAIL  launch log contains errors:"; grep -aE "\[ERROR\]|process has died" "$LAUNCH_LOG" | tail -5; fail=1
else
    echo "PASS  no [ERROR]/dead process in launch log"
fi
echo "log: $LAUNCH_LOG"
echo "fail=$fail"
exit "$fail"
