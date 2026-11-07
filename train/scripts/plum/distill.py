#!/usr/bin/env python3
"""Train the depth-CNN student and export a judgeable policy directory.

Only the CNN + scan head are optimised; the teacher MLP is rebuilt from the
checkpoint and kept frozen.  Loss = action MSE + 0.1 * scan MSE.

    MUJOCO_GL=egl PYTHONPATH=train/src .pixi/envs/train/bin/python \
      train/scripts/plum/distill.py --data artifacts/plum/data/round0 \
      --teacher <model_xxx.pt> --out artifacts/plum/student_v1 \
      --policy-name plum_student_v1
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from alldog_mjlab.tasks.velocity.profiles.plum.fine_scan import fine_scan_dim
from alldog_mjlab.tasks.velocity.profiles.plum.student import (
    build_scan_cnn, build_teacher_mlp, distillation_loss, export_policy_dir,
    student_action,
)


def load_shards(data_dir: Path, device: str):
    files = sorted(glob.glob(str(data_dir / "shard_*.npz")))
    if not files:
        raise SystemExit(f"no shards in {data_dir}")
    cols = {k: [] for k in ("depth", "proprio", "gt_scan", "teacher_action")}
    for f in files:
        z = np.load(f)
        for k in cols:
            cols[k].append(z[k])
    depth = np.concatenate(cols["depth"]).reshape(-1, *cols["depth"][0].shape[2:])
    tensors = {
        "depth": torch.from_numpy(depth).float(),
        "proprio": torch.from_numpy(np.concatenate(cols["proprio"]).reshape(-1, cols["proprio"][0].shape[-1])).float(),
        "gt_scan": torch.from_numpy(np.concatenate(cols["gt_scan"]).reshape(-1, cols["gt_scan"][0].shape[-1])).float(),
        "teacher_action": torch.from_numpy(np.concatenate(cols["teacher_action"]).reshape(-1, cols["teacher_action"][0].shape[-1])).float(),
    }
    finite = torch.ones(tensors["depth"].shape[0], dtype=torch.bool)
    for t in tensors.values():
        finite &= torch.isfinite(t).flatten(1).all(1)
    n_bad = int((~finite).sum())
    print(f"[plum/distill] dropping {n_bad} non-finite frames "
          f"({n_bad / len(finite):.1%})", flush=True)
    return {k: v[finite].to(device) for k, v in tensors.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--policy-name", default="plum_student_v1")
    ap.add_argument("--init-student", default=None,
                    help="warm-start CNN/head from a previous student.pt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--action-weight", type=float, default=1.0)
    ap.add_argument("--scan-weight", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device

    ckpt = torch.load(args.teacher, map_location="cpu", weights_only=False)
    teacher = build_teacher_mlp(ckpt["actor_state_dict"], device)
    data = load_shards(args.data, device)
    n = data["depth"].shape[0]
    print(f"[plum/distill] frames={n} scan_dim={fine_scan_dim()}", flush=True)

    cnn, head = build_scan_cnn(fine_scan_dim(), device)
    if args.init_student:
        s = torch.load(args.init_student, map_location="cpu", weights_only=False)
        cnn.load_state_dict(s["cnn_state"])
        head.load_state_dict(s["scan_head"])
        print(f"[plum/distill] warm-start CNN from {args.init_student}", flush=True)

    idx = torch.randperm(n, device=device)
    n_val = int(n * args.val_frac)
    val, tr = idx[:n_val], idx[n_val:]
    opt = torch.optim.Adam(list(cnn.parameters()) + list(head.parameters()), lr=args.lr)

    best = float("inf")
    best_metrics: dict = {}
    for ep in range(args.epochs):
        cnn.train()
        head.train()
        perm = tr[torch.randperm(len(tr), device=device)]
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            loss, _, _ = distillation_loss(
                teacher, cnn, head, data["proprio"][b], data["depth"][b],
                data["gt_scan"][b], data["teacher_action"][b],
                args.action_weight, args.scan_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        cnn.eval()
        head.eval()
        with torch.no_grad():
            b = val
            a_student, pred = student_action(
                teacher, cnn, head, data["proprio"][b], data["depth"][b])
            a_teacher = teacher(torch.cat([data["proprio"][b], data["gt_scan"][b]], dim=-1))
            act_mse = float(F.mse_loss(a_student, a_teacher))
            scan_rmse = float(torch.sqrt(F.mse_loss(pred, data["gt_scan"][b])))
            act_std = float(a_teacher.std())
        print(f"[plum/distill] ep{ep:02d} train_loss={tot / len(tr):.5f} "
              f"val_action_rmse={np.sqrt(act_mse):.4f} "
              f"(teacher std {act_std:.4f}) val_scan_rmse={scan_rmse:.4f}", flush=True)
        if act_mse < best:
            best = act_mse
            best_metrics = {
                "val_action_rmse": float(np.sqrt(act_mse)),
                "val_scan_rmse": scan_rmse,
                "teacher_action_std": act_std,
                "frames": n,
                "epoch": ep,
            }
            torch.save({
                "cnn_state": cnn.state_dict(),
                "scan_head": head.state_dict(),
                "latent_dim": 64,
                "scan_dim": fine_scan_dim(),
                "teacher_checkpoint": str(args.teacher),
                "metrics": best_metrics,
            }, args.out / "student.pt")

    # Export uses a FRESH teacher: export_policy_dir permutes its first Linear
    # columns in place and must not touch the frozen training teacher.
    export_teacher = build_teacher_mlp(ckpt["actor_state_dict"], "cpu")
    cnn_cpu, head_cpu = build_scan_cnn(fine_scan_dim(), "cpu")
    saved = torch.load(args.out / "student.pt", map_location="cpu", weights_only=False)
    cnn_cpu.load_state_dict(saved["cnn_state"])
    head_cpu.load_state_dict(saved["scan_head"])
    export_policy_dir(
        args.out / "policy", args.policy_name, cnn_cpu, head_cpu, export_teacher,
        str(args.teacher), best_metrics)
    (args.out / "metrics.json").write_text(json.dumps(best_metrics, indent=1) + "\n",
                                           encoding="utf-8")
    print(f"[plum/distill] best val_action_rmse={best_metrics['val_action_rmse']:.4f}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
