"""BlackW HIM runner with an optional official TorchScript warm start.

The official blackW deployment policies are HIM exports (actor + source encoder)
and match the MjLab HIMPolicy shapes exactly.  Set
``RC_HIM_OFFICIAL_WARMSTART=/path/policy.pt`` before launching training to copy
the actor/source-encoder tensors into the randomly initialized HIM policy.  The
target encoder/prototypes and critic remain freshly initialized, as they are
training-only components absent from the deployment export.
"""
from __future__ import annotations

import os

import torch

from alldog_mjlab.tasks.velocity.common.him_runner import BlackHimOnPolicyRunner
from alldog_mjlab.utils import knobs
from alldog_mjlab.utils.reward_safety import install_reward_clamp, install_return_clamp


class _ActionGuardWrapper:
    """Sanitize policy actions before they enter the simulator.

    Action-term guards are necessary but not sufficient: PPO/HIM can emit a bad
    vector that is used by the wrapper/observation pipeline before the term runs.
    This wrapper is the last line before ``mjwarp.step``.
    """

    def __init__(self, env, clip_actions: float = 100.0) -> None:
        self.env = env
        self.clip_actions = float(clip_actions)
        self.nonfinite_actions = 0
        self.max_abs_action = 0.0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def __len__(self) -> int:
        return len(self.env)

    def step(self, action):
        if not torch.isfinite(action).all():
            self.nonfinite_actions += int((~torch.isfinite(action)).sum().item())
            action = torch.nan_to_num(action, nan=0.0, posinf=self.clip_actions, neginf=-self.clip_actions)
        action = torch.clamp(action, -self.clip_actions, self.clip_actions)
        self.max_abs_action = max(self.max_abs_action, float(action.abs().max().item()))
        return self.env.step(action)


def _map_official(policy, path: str) -> None:
    official = torch.jit.load(path, map_location="cpu").eval()
    mapped = {}
    for name, value in official.named_parameters():
        if name.startswith("actor."):
            target = "mlp." + name[len("actor."):]
        elif name.startswith("estimator."):
            target = "estimator.encoder." + name[len("estimator."):]
        elif name.startswith("mlp."):
            # Native HIM TorchScript export (this project's export_him_policy.py).
            target = name
        elif name.startswith("encoder."):
            target = "estimator.encoder." + name[len("encoder."):]
        else:
            raise ValueError(f"unexpected official parameter: {name}")
        mapped[target] = value
    missing, unexpected = policy.load_state_dict(mapped, strict=False)
    missing_mlp = [name for name in missing if name.startswith("mlp.")]
    if missing_mlp:
        raise RuntimeError(
            f"warm start incomplete: missing actor MLP keys {missing_mlp}; "
            "refusing to silently train from a partial/zero actor."
        )
    print(
        f"[INFO] official warm start: {path} mapped={len(mapped)} "
        f"missing={missing} unexpected={unexpected}"
    )


class BlackWHimOfficialRunner(BlackHimOnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu", **kwargs):
        clip = float(
            (train_cfg.get("clip_actions") if isinstance(train_cfg, dict) else getattr(train_cfg, "clip_actions", 100.0))
            or 100.0
        )
        if not isinstance(env, _ActionGuardWrapper):
            env = _ActionGuardWrapper(env, clip_actions=clip)
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        reward_clip = float(os.environ.get("BLACKW_REWARD_CLIP", "50.0"))
        install_reward_clamp(self.env.unwrapped, reward_clip)
        install_return_clamp(self.alg)
        # This MUST be inside __init__.  An earlier patch accidentally left it
        # after ``return super().save(...)``, so warm-start silently never ran.
        path = os.environ.get("RC_HIM_OFFICIAL_WARMSTART")
        if path:
            _map_official(self.alg.actor, path)
            # The deployment export has no distribution.std_param; keep this
            # visible instead of silently treating the warm start as complete.
            dist = self.alg.actor.distribution
            log_std = getattr(dist, "log_std_param", None) if dist is not None else None
            scalar_std = getattr(dist, "std_param", None) if dist is not None else None
            if log_std is not None:
                std_value = float(torch.exp(log_std.detach()).mean())
            elif scalar_std is not None:
                std_value = float(scalar_std.detach().mean())
            else:
                std_value = float("nan")
            print(
                "[WARN] official warm start does not restore action std; "
                f"current std≈{std_value:.4f} "
                f"(RC_HIM_INIT_STD={os.environ.get('RC_HIM_INIT_STD', '0.1')})"
            )
            self.alg.set_policy_anchor()
        freeze_std = os.environ.get("RC_HIM_FREEZE_STD", "")
        if freeze_std:
            dist = self.alg.actor.distribution
            log_param = getattr(dist, "log_std_param", None) if dist is not None else None
            scalar_param = getattr(dist, "std_param", None) if dist is not None else None
            if log_param is not None:
                if freeze_std == "precheck":
                    with torch.no_grad():
                        log_param.fill_(-10.0)
                    print("[INFO] HIM log_std frozen to precheck value (std≈0)")
                else:
                    log_param.requires_grad_(False)
                    print("[INFO] HIM log_std frozen at warm-start/init value")
            elif scalar_param is not None:
                if freeze_std == "precheck":
                    with torch.no_grad():
                        scalar_param.fill_(1e-5)
                    print("[INFO] HIM scalar std frozen to precheck value (std≈1e-5)")
                else:
                    scalar_param.requires_grad_(False)
                    print("[INFO] HIM scalar std frozen at warm-start/init value")

    def save(self, path, infos=None):
        try:
            raw = self.env.unwrapped
            sim = raw.sim
            data = sim.data
            nefc = getattr(data, "nefc", None)
            overflow = getattr(data, "overflow", None)
            nefc_max = int(nefc.max().item()) if nefc is not None else -1
            overflow_n = int((overflow != 0).sum().item()) if overflow is not None else -1
            print(
                f"[HIM guard] nefc_max={nefc_max} overflow_nonzero={overflow_n} "
                f"nonfinite_actions={self.env.nonfinite_actions} "
                f"max_abs_action={self.env.max_abs_action:.3f}"
            )
            # Reset both counters: an un-reset max_abs_action hides whether the
            # policy is still growing between saves (job24 reported 45 for 90
            # iterations because the running max was never cleared).
            self.env.nonfinite_actions = 0
            self.env.max_abs_action = 0.0
        except Exception as exc:  # noqa: BLE001 - monitoring must never stop training
            print(f"[HIM guard] monitor unavailable: {type(exc).__name__}: {exc}")
        return super().save(path, infos)
