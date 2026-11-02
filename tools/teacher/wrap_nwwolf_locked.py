#!/usr/bin/env python3
"""Wrap the N-W-wolf `black_locked_wheels` JIT policy into the 16-DoF/57x6 contract.

The teacher is a 12-action, 45x6 observation policy.  Our demo/rl_sdk and
PolicyRunner expect 16 actions and 57x6 observations.  This wrapper:
  * reshapes [B, 6*57] -> [B, 6, 57] (history newest-first, term-major);
  * extracts commands / body ang vel / gravity / leg q / leg dq / leg actions;
  * converts our dof_pos relative to the blackW default (hip +-0.05) to the
    teacher's point-foot default (hip 0);
  * calls the teacher and maps its 12 leg actions back to 16 actions (wheels 0).
The resulting TorchScript has input 342/output 16 and can be dropped into the
existing lock_wheels deployment path.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch

LEG_IDX = torch.tensor([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14], dtype=torch.long)
OUR_DEFAULT = torch.tensor([0.05,0.8014,-1.527,0.0, -0.05,-0.8014,1.527,0.0,
                            -0.05,0.8014,-1.527,0.0, 0.05,-0.8014,1.527,0.0])
TEACHER_DEFAULT = torch.tensor([0.0,0.8014,-1.527,0.0, 0.0,-0.8014,1.527,0.0,
                                0.0,0.8014,-1.527,0.0, 0.0,-0.8014,1.527,0.0])

class NwwolfLocked16(torch.nn.Module):
    def __init__(self, teacher: torch.jit.ScriptModule):
        super().__init__()
        self.teacher = teacher
        self.register_buffer("leg_idx", LEG_IDX)
        self.register_buffer("our_default", OUR_DEFAULT)
        self.register_buffer("teacher_default", TEACHER_DEFAULT)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [B, 342] = 6 frames x 57; newest frame first.
        b = obs.shape[0]
        x = obs.reshape(b, 6, 57)
        frames = []
        for i in range(6):
            f = x[:, i, :]
            cmd = f[:, 0:3]       # already commands_scale
            ang = f[:, 3:6]       # already ang_vel_scale
            grav = f[:, 6:9]
            qrel16 = f[:, 9:25]
            dq16 = f[:, 25:41]
            act16 = f[:, 41:57]
            q_abs = qrel16 + self.our_default
            q_teacher = q_abs[:, self.leg_idx] - self.teacher_default[self.leg_idx]
            dq_teacher = dq16[:, self.leg_idx]
            act_teacher = act16[:, self.leg_idx]
            frames.append(torch.cat((cmd, ang, grav, q_teacher, dq_teacher, act_teacher), dim=1))
        teacher_obs = torch.cat(frames, dim=1)  # newest-first, 270
        a12 = self.teacher(teacher_obs)
        out = torch.zeros((b, 16), dtype=a12.dtype, device=a12.device)
        out[:, self.leg_idx] = a12
        return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    teacher = torch.jit.load(args.teacher, map_location="cpu")
    teacher.eval()
    mod = NwwolfLocked16(teacher)
    mod.eval()
    scripted = torch.jit.script(mod)
    scripted.save(args.out)
    print("wrapped ->", args.out)
    # smoke test: input 342 -> output 16
    y = scripted(torch.zeros(2, 342))
    print("smoke shape", tuple(y.shape), "wheel cols", y[0, [3, 7, 11, 15]].tolist())

if __name__ == "__main__":
    main()
