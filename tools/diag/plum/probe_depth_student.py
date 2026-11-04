#!/usr/bin/env python3
"""Depth-CNN student feasibility probe on the diagnostic competition-plum level.

The only old-repo policy that ever passed the regulation plum field is the
depth-CNN student `route2_v3` (``~/RoboCon/artifacts/plum-strict/p1/policies/route2_v3``,
copied to ``artifacts/part2_experts/plum/policies/route2_v3``).  This script
answers two questions with the lean eval runner:

1. Does the *optional depth path* of ``eval/robocon_eval/policy_runner.py``
   actually run that checkpoint (TorchScript CNNModel, ``policy(obs, [depth])``)?
2. Does it transfer to the *competition orientation* of the corrected plum
   geometry, and does it reproduce the old-repo ``plum_reg_1`` success on the
   y-mirrored geometry (same relative field as the old regulation level)?

Deployment blocker (recorded in the REPORT): the lean ROS runtime has no depth
camera path -- ``demo/rl_sar/src/rl_sim.cpp:Forward()`` calls
``model.forward({obs})`` only; there is no camera/renderer/d435 code in
``demo/rl_sar`` (grep depth|camera|d435 -> 0 hits), so a CNNModel checkpoint
cannot run in the deployment route at all.

Usage (from the repo root; eval env -- MUJOCO_GL=egl is set by pixi):
    pixi run -e eval python tools/diag/plum/probe_depth_student.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DIAG = Path(__file__).resolve().parent
LOCAL_POLICY = str(ROOT / "artifacts/part2_experts/plum/policies/route2_v3")
RESULTS = ROOT / "artifacts/part2_experts/plum/results"

os.environ["ROBOCON_EVAL_LEVELS"] = str(DIAG / "plum_diag_levels.json")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robocon_eval.policy_runner import AdaptiveCommand, run  # noqa: E402

SCENE_COMP = "tools/diag/plum/plum_comp_diag.xml"
SCENE_MIRROR = "tools/diag/plum/plum_comp_diag_mirror.xml"
MODES = {
    "fixed05": (0.5, False),
    "fixed08": (0.8, False),
    "adaptive05": (0.5, True),
    "adaptive08": (0.8, True),
}


def _run(tag: str, scene: str, level: str, mode: str, dy: float,
         duration: float, apply_offset: bool = True) -> dict:
    vx, adaptive = MODES[mode]
    outdir = RESULTS / "depth_student"
    outdir.mkdir(parents=True, exist_ok=True)
    trace = outdir / f"{tag}.jsonl"
    command_fn = (AdaptiveCommand(vx=vx, k_y=0.6, k_yaw=1.2) if adaptive else None)
    started = time.time()
    result = run(
        "blackW", 2, LOCAL_POLICY, level, trace,
        scene=scene,
        duration_s=duration,
        command=(vx, 0.0, 0.0),
        settle_s=1.0, warmup_s=0.0, seed=0, stop_on_fall=True,
        profile="diagnostic", command_fn=command_fn,
        spawn_offset=(dy, 0.0) if apply_offset else None,
    )
    result["probe"] = {"tag": tag, "scene": scene, "level": level, "mode": mode,
                       "vx": vx, "adaptive": adaptive, "dy": dy,
                       "apply_spawn_offset": apply_offset,
                       "duration_s": duration,
                       "depth_enabled": result.get("depth") is not None}
    result["wall_time_s"] = round(time.time() - started, 3)
    (outdir / f"{tag}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    detail = (result.get("judge") or {}).get("detail") or {}
    return {
        "tag": tag, "orientation": ("mirror_oldreg" if "mirror" in scene
                                    else "competition"),
        "level": level, "mode": mode, "dy": dy,
        "spawn_offset": [dy, 0.0] if apply_offset else None,
        "status": result.get("status"), "score": result.get("score"),
        "reason": result.get("reason"),
        "sim_time_s": result.get("sim_time_s"),
        "stopped": result.get("stopped"),
        "depth_enabled": result.get("depth") is not None,
        "judge_detail": detail,
        "wall_time_s": result["wall_time_s"],
        "result_path": str(outdir / f"{tag}.json"),
    }


def main() -> int:
    rows: list[dict] = []
    started = time.time()
    # 1. competition orientation, judge-style spawn (all 4 command modes)
    for mode in MODES:
        rows.append(_run(f"comp__judge__{mode}", SCENE_COMP, "plum_comp_judge_1",
                         mode, 0.0, duration=8.0))
    # 2. y-mirrored orientation == old plum_reg_1 relative geometry
    for mode in MODES:
        rows.append(_run(f"mirror__judge__{mode}", SCENE_MIRROR,
                         "plum_comp_judge_mirror_1", mode, 0.0, duration=8.0))
    # 3. reproduction of the old-repo *baseline* (no post-settle mj_forward).
    #    The old report says the offset API's extra mj_forward flips the run.
    rows.append(_run("comp__judge__fixed05__nooffset", SCENE_COMP,
                     "plum_comp_judge_1", "fixed05", 0.0, 8.0,
                     apply_offset=False))
    rows.append(_run("mirror__judge__fixed05__nooffset", SCENE_MIRROR,
                     "plum_comp_judge_mirror_1", "fixed05", 0.0, 8.0,
                     apply_offset=False))
    # 4. knife-edge robustness on the orientation it was trained for
    for dy in (0.05, -0.05, 0.10, -0.10):
        rows.append(_run(f"mirror__judge__fixed05__dy{dy:+.2f}", SCENE_MIRROR,
                         "plum_comp_judge_mirror_1", "fixed05", dy, 8.0))
    # 5. full-field route-style spawn (on the ground, must climb the platform)
    for mode in MODES:
        rows.append(_run(f"comp__field__{mode}", SCENE_COMP, "plum_comp_field_1",
                         mode, 0.0, duration=12.0))
    summary = {
        "policy": LOCAL_POLICY,
        "policy_source": ("~/RoboCon/artifacts/plum-strict/p1/policies/route2_v3"),
        "depth_contract": "auto-detected depth.json in the policy dir",
        "scene_competition": SCENE_COMP,
        "scene_mirror_oldreg": SCENE_MIRROR,
        "wall_time_s": round(time.time() - started, 3),
        "successes": sum(1 for r in rows if r["status"] == "success"),
        "attempts": len(rows),
        "rows": rows,
    }
    out = RESULTS / "depth_student__summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    for r in rows:
        print(f"{r['tag']:42s} {r['status']:7s} {r['reason'] or '':28s} "
              f"max_x={(r['judge_detail'] or {}).get('max_x')} "
              f"depth={r['depth_enabled']}")
    print(f"\n{summary['successes']}/{summary['attempts']} success; {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
