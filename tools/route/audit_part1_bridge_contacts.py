#!/usr/bin/env python3
"""Replay Part1 bridge B collision points without integrating or rendering.

Example (the train environment includes matplotlib):
  .pixi/envs/train/bin/python tools/route/audit_part1_bridge_contacts.py \
    artifacts/route_fix/part1_final_v67_states.jsonl \
    --out artifacts/route_fix/bridge_b_contact_audit/v67

sim_node publishes qpos after its Euler integration, while feet and contacts
describe the preceding collision configuration. Undo the scalar-joint Euler
increment and recover the earlier trunk transform from its four recorded foot
body positions. This avoids mixing post-step wheel positions with pre-step
contacts. Contact normals identify top/edge/side geometric contacts; this trace
does not contain contact forces, so these are not force-weighted load samples.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "demo/robots/blackW/scene_part1.xml"
BANDS = {"bridge_bands_0": (5.25, 5.45), "bridge_bands_2": (5.85, 6.05)}
WHEELS = ("FL", "FR", "RL", "RR")


def read_frames(path):
    """Read complete JSONL records, including a prefix of a live trace."""
    with path.open() as source:
        for line in source:
            if not line.endswith("\n"):
                break
            if line.strip():
                yield json.loads(line)


def quantiles(values):
    if not values:
        return None
    names = ("min", "p05", "p25", "p50", "p75", "p95", "max")
    return dict(zip(names, map(float, np.quantile(values, (0, .05, .25, .5, .75, .95, 1)))))


def yaw_deg(q):
    w, x, y, z = q
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def outer_pairs(pairs):
    return collections.Counter(
        tuple(pair) for pair in pairs
        if any(name in BANDS for name in pair)
        and any(name and name.endswith("_wheel_collision") for name in pair)
    )


def contact_summary(rows):
    if not rows:
        return None
    return {
        "contacts": len(rows),
        "frames": len({row["t"] for row in rows}),
        "time_range_s": [min(row["t"] for row in rows), max(row["t"] for row in rows)],
        "x_m": quantiles([row["x"] for row in rows]),
        "inner_clearance_m": quantiles([row["inner_clearance"] for row in rows]),
        "outer_clearance_m": quantiles([row["outer_clearance"] for row in rows]),
        "normal_types": dict(collections.Counter(row["contact_type"] for row in rows)),
        "frames_by_normal_type": {
            kind: len({row["t"] for row in rows if row["contact_type"] == kind})
            for kind in ("top", "edge", "side")
        },
    }


def plot_contacts(rows, output, trace_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    fig, (ax, margins) = plt.subplots(1, 2, figsize=(11, 7), gridspec_kw={"width_ratios": [1.25, 1]})
    colors = dict(zip(WHEELS, ("#0072b2", "#e69f00", "#009e73", "#cc79a7")))
    for band, (left, right) in BANDS.items():
        ax.add_patch(Rectangle((left, -3), right - left, 1.5, facecolor="0.9", edgecolor="0.5"))
        inner = right if band.endswith("_0") else left
        ax.plot([inner, inner], [-3, -1.5], color="black", linestyle="--", linewidth=1)
        ax.text((left + right) / 2, -1.47, band.replace("bridge_bands_", "band "), ha="center", fontsize=9)
    for wheel in WHEELS:
        rr = [row for row in rows if row["wheel"] == wheel]
        if not rr:
            continue
        for kind, marker, size in (("top", ".", 9), ("edge", "o", 20), ("side", "x", 13)):
            selected = [row for row in rr if row["contact_type"] == kind]
            if selected:
                ax.scatter([row["x"] for row in selected], [row["y"] for row in selected],
                           c=colors[wheel], marker=marker, s=size, alpha=.55, linewidths=.6)
        by_time = collections.defaultdict(list)
        for row in rr:
            by_time[row["t"]].append(row["inner_clearance"] * 100)
        times = sorted(by_time)
        margins.plot(times, [np.median(by_time[t]) for t in times], label=wheel,
                     color=colors[wheel], linewidth=1.4)
    ax.set(xlim=(5.21, 6.09), ylim=(-3.05, -1.40), xlabel="World x (m)", ylabel="World y (m)",
           title="Actual outer-band contact points\nDashed lines: inner edges")
    ax.set_aspect("equal")
    handles = [Line2D([], [], color=color, marker=".", linestyle="", label=wheel)
               for wheel, color in colors.items()]
    handles += [Line2D([], [], color="0.25", marker=marker, linestyle="", label=label)
                for marker, label in ((".", "top normal"), ("o", "edge normal"), ("x", "side normal"))]
    ax.legend(handles=handles, loc="lower center", ncol=2, fontsize=8)
    margins.axhline(0, color="black", linestyle="--", linewidth=1)
    margins.set(xlabel="Recorded simulation time (s)", ylabel="Inner-edge clearance (cm)",
                title="Median contact-point clearance per frame")
    margins.grid(alpha=.2)
    margins.legend()
    fig.suptitle(f"{trace_name}\nPart1 bridge B: collision replay, no physics integration", fontsize=11)
    fig.text(.5, .015, "Geometric contacts; no contact-force data. Top: |normal z| ≥ 0.9; side: |normal z| ≤ 0.1.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .96))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("states", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--start", type=float, help="Override automatic first-contact time minus 0.1 s")
    parser.add_argument("--end", type=float, help="Override automatic last-contact time plus 0.1 s")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib dependency; still write CSV and JSON")
    args = parser.parse_args()
    first, last = None, None
    for frame in read_frames(args.states):
        if outer_pairs(frame.get("contacts", [])):
            first = frame["t"] if first is None else first
            last = frame["t"]
    if first is None:
        parser.error("trace has no recorded wheel contacts with outer bridge bands 0/2")
    start = args.start if args.start is not None else first - .1
    end = args.end if args.end is not None else last + .1
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    if model.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER:
        parser.error("exact pre-step reconstruction requires the scene's Euler integrator")
    data = mujoco.MjData(model)
    qadr = [int(model.jnt_qposadr[int(model.actuator_trnid[i, 0])]) for i in range(model.nu)]
    foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, wheel + "_foot") for wheel in WHEELS]
    bar_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "height_bar_slide")
    rows, frames, errors, pair_matches = [], [], [], []
    for frame in read_frames(args.states):
        if not start <= frame["t"] <= end:
            continue
        data.qpos[:3], data.qpos[3:7] = 0, [1, 0, 0, 0]
        data.qpos[qadr] = np.asarray(frame["q"]) - model.opt.timestep * np.asarray(frame["dq"])
        mujoco.mj_kinematics(model, data)
        local = data.xpos[foot_ids].copy()
        recorded = np.asarray([frame["feet"][wheel + "_foot"] for wheel in WHEELS])
        local_center, world_center = local.mean(0), recorded.mean(0)
        u, _, vt = np.linalg.svd((local - local_center).T @ (recorded - world_center))
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        base = world_center - rotation @ local_center
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rotation.ravel())
        data.qpos[:3], data.qpos[3:7] = base, quat
        if bar_id >= 0 and frame.get("bar_z") is not None:
            data.qpos[model.jnt_qposadr[bar_id]] = frame["bar_z"] - .325
        mujoco.mj_forward(model, data)
        error = float(np.max(np.abs(data.xpos[foot_ids] - recorded)))
        errors.append(error)
        if error > 1e-5:
            raise ValueError(f"{frame['t']}: foot fit error {error:g} m; trace/scene or Euler reconstruction differs")
        current, pairs = [], []
        for contact in data.contact[:data.ncon]:
            names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) for g in contact.geom]
            if not outer_pairs([names]):
                continue
            pairs.append(names)
            band = next(name for name in names if name in BANDS)
            wheel = next(name[:2] for name in names if name and name.endswith("_wheel_collision"))
            x, y, z = map(float, contact.pos)
            nx, ny, nz = map(float, contact.frame[:3])
            left, right = BANDS[band]
            inner = right - x if band.endswith("_0") else x - left
            outer = x - left if band.endswith("_0") else right - x
            kind = "top" if abs(nz) >= .9 else ("side" if abs(nz) <= .1 else "edge")
            current.append(dict(t=frame["t"], wheel=wheel, band=band, x=x, y=y, z=z,
                                nx=nx, ny=ny, nz=nz, dist=float(contact.dist), contact_type=kind,
                                inner_clearance=inner, outer_clearance=outer, base_x=float(base[0]),
                                base_y=float(base[1]), base_yaw_deg=yaw_deg(quat),
                                published_base_x=frame["base"][0], published_base_y=frame["base"][1],
                                published_base_yaw_deg=yaw_deg(frame["quat"])))
        pair_matches.append(outer_pairs(pairs) == outer_pairs(frame.get("contacts", [])))
        if current:
            wheels = {row["wheel"] for row in current}
            phase = ("four_wheels" if len(wheels) == 4 else "entry_front_only" if wheels <= {"FL", "FR"}
                     else "exit_rear_only" if wheels <= {"RL", "RR"} else "mixed_transition")
            for row in current:
                row["phase"] = phase
            rows.extend(current)
            frames.append(dict(t=frame["t"], base_x=float(base[0]), base_y=float(base[1]),
                               yaw_deg=yaw_deg(quat), phase=phase, wheels=sorted(wheels)))
    if not rows:
        parser.error("selected interval has no reconstructed outer-band contacts")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "contacts.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_phase = {}
    for phase in sorted({frame["phase"] for frame in frames}):
        ff = [frame for frame in frames if frame["phase"] == phase]
        rr = [row for row in rows if row["phase"] == phase]
        by_phase[phase] = {
            "frames": len(ff), "time_range_s": [ff[0]["t"], ff[-1]["t"]],
            "base_x_m": quantiles([frame["base_x"] for frame in ff]),
            "yaw_deg": quantiles([frame["yaw_deg"] for frame in ff]),
            "wheels": {wheel: contact_summary([row for row in rr if row["wheel"] == wheel]) for wheel in WHEELS},
        }
    summary = {
        "trace": str(args.states), "scene": str(args.scene), "mujoco_version": mujoco.__version__,
        "method": "CPU mj_forward only; undo Euler scalar-joint increment and rigid-fit recorded feet to recover collision-time trunk pose. No integration/rendering. Geometric contacts, not force-weighted load measurements.",
        "normal_types": "top: abs(nz)>=0.9; side: abs(nz)<=0.1; edge: otherwise",
        "scene_sha256": hashlib.sha256(args.scene.read_bytes()).hexdigest(),
        "robot_sha256": hashlib.sha256((args.scene.parent / "blackW_description.xml").read_bytes()).hexdigest(),
        "window_s": [start, end], "replay_frames": len(pair_matches),
        "exact_outer_contact_pair_count_frames": sum(pair_matches), "foot_body_fit_max_error_m": max(errors),
        "band_bounds_m": BANDS,
        "all_contact_frames_base_x_m": quantiles([frame["base_x"] for frame in frames]),
        "all_contact_frames_yaw_deg": quantiles([frame["yaw_deg"] for frame in frames]),
        "by_wheel": {wheel: contact_summary([row for row in rows if row["wheel"] == wheel]) for wheel in WHEELS},
        "by_phase": by_phase,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not args.no_plot:
        plot_contacts(rows, args.out / "contacts.png", args.states.stem)
    print(json.dumps({key: summary[key] for key in ("replay_frames", "exact_outer_contact_pair_count_frames",
          "foot_body_fit_max_error_m", "all_contact_frames_base_x_m", "all_contact_frames_yaw_deg")}, indent=2))
    print(f"Artifacts: {args.out}")


if __name__ == "__main__":
    main()
