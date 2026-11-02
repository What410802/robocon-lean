#!/usr/bin/env bash
# Evaluate one training checkpoint as a deployment policy + acceptance probes.
#
#   tools/tracking/eval_checkpoint.sh <run_dir> <checkpoint.pt> <policy_name>
#
# Steps: export (CPU TorchScript) -> copy into demo/rl_sar/policy/blackW/<name>/
# (weights local, config template from vt_pose_v1) -> F1-F7 gate -> support
# rectangle -> turn footprint -> position loop harness. JSONs under
# artifacts/velocity-tracking/<name>/.
#
# Environment: run inside the repo pixi environments, e.g.
#   PY_EVAL="pixi run -e eval python" PY_TRAIN="pixi run -e train python" ...
# or pass absolute interpreters (the values must be executable prefixes).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="${1:?usage: eval_checkpoint.sh <run_dir> <checkpoint.pt> <policy_name>}"
CKPT="${2:?}"
NAME="${3:?}"
PY_EVAL="${PY_EVAL:-python}"
PY_TRAIN="${PY_TRAIN:-python}"
OUT="$ROOT/artifacts/velocity-tracking/$NAME"
POLICY_DIR="$ROOT/demo/rl_sar/policy/blackW/$NAME"

mkdir -p "$OUT/export" "$POLICY_DIR"
echo "=== export $RUN/$CKPT -> $NAME ==="
PYTHONPATH="$ROOT/train/src" MUJOCO_GL=egl $PY_TRAIN \
    "$ROOT/train/scripts/export/export_history_policy.py" \
    --task blackw-vt --robot blackW \
    --checkpoint "$RUN/$CKPT" --out "$OUT/export/policy.pt" --device cpu || exit 1
cp "$OUT/export/policy.pt" "$POLICY_DIR/policy.pt"
cp "$OUT/export/manifest.json" "$POLICY_DIR/manifest.json"
cp "$ROOT/demo/rl_sar/policy/blackW/vt_pose_v1/config.yaml" "$POLICY_DIR/config.yaml"

export PYTHONPATH="$ROOT/eval" MUJOCO_GL=egl
echo "=== F1-F7 gate ==="
$PY_EVAL "$ROOT/tools/tracking/command_probe.py" --policy "$NAME" --part 1 --gate \
    --out "$OUT/command_probe.json" || true

echo "=== support rectangle ==="
$PY_EVAL "$ROOT/tools/tracking/probe_support_rectangle.py" --policies "$NAME" \
    --out "$OUT/support_rect.json" || true

echo "=== turn footprint ==="
$PY_EVAL "$ROOT/tools/tracking/probe_turn_footprint.py" --policies "$NAME" \
    --out "$OUT/turn_footprint.json" || true

echo "=== position harness ==="
$PY_EVAL "$ROOT/tools/tracking/position_harness.py" --policies "$NAME" \
    --out "$OUT/position_harness.json" || true

echo "=== robustness (RecoveryLoco-style) ==="
$PY_EVAL "$ROOT/tools/tracking/robustness_probe.py" --policies "$NAME" \
    --out "$OUT/robustness.json" || true

echo "artifacts: $OUT"
