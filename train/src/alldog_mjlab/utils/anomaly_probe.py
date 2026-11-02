"""Temporary investigation probe for the ``Warp CUDA error 710`` blow-ups.

Enabled with ``RC_ANOMALY_PROBE=1``; otherwise installing it is a no-op.  It
wraps ``RslRlVecEnvWrapper.step`` and keeps a short rolling history of per-step
extremes (action, actor obs, joint velocity).  Soft warnings are printed when a
new per-env action maximum is reached; when an env crosses a hard threshold
(``RC_PROBE_ACTION``/``RC_PROBE_OBS``/``RC_PROBE_JOINT_VEL``) or the
state goes non-finite, it prints the triggering envs with the recent history and
raises ``SystemExit(0)`` so training stops *before* the CUDA assert.

This file is a diagnostic instrument, not part of the training contract; it can
be deleted once the instability is understood.
"""

from __future__ import annotations

from collections import deque

import torch

from alldog_mjlab.utils import knobs


def _threshold(name: str, default: float) -> float:
    return knobs.optional(name) and knobs.get_float(name, default) or default


def install_anomaly_probe(vec_env) -> None:
    """Wrap ``vec_env.step`` and abort on the first anomalous env state."""
    if not knobs.get_bool("RC_ANOMALY_PROBE"):
        return
    if getattr(vec_env, "_black_anomaly_probe_installed", False):
        return

    unwrapped = vec_env.unwrapped
    thr_action = _threshold("RC_PROBE_ACTION", 100.0)
    thr_obs = _threshold("RC_PROBE_OBS", 200.0)
    thr_jv = _threshold("RC_PROBE_JOINT_VEL", 100.0)
    history: deque = deque(maxlen=12)
    step_counter = {"n": 0}
    best = {"action": 0.0}
    original_step = vec_env.step

    robot = unwrapped.scene["robot"]
    data = robot.data

    def describe(env_ids: torch.Tensor, actions: torch.Tensor, obs) -> str:
        ids = env_ids[:8].tolist()
        lines = []
        episode_len = unwrapped.episode_length_buf
        for i in ids:
            a = actions[i].detach().cpu()
            jv = data.joint_vel[i].detach().cpu()
            root_lin = data.root_link_lin_vel_w[i].detach().cpu()
            root_ang = data.root_link_ang_vel_w[i].detach().cpu()
            root_pos = data.root_link_pos_w[i].detach().cpu()
            obs_actor = obs["actor"][i].detach().cpu()
            lines.append(
                "  env {} ep_len={} |a|max={:.4g} act={} |jv|max={:.4g} "
                "root_lin={} root_ang={} root_pos={} obs_absmax={:.4g}".format(
                    i,
                    int(episode_len[i]),
                    float(a.abs().max()),
                    [round(float(x), 2) for x in a.tolist()],
                    float(jv.abs().max()),
                    [round(float(x), 2) for x in root_lin.tolist()],
                    [round(float(x), 2) for x in root_ang.tolist()],
                    [round(float(x), 3) for x in root_pos.tolist()],
                    float(obs_actor.abs().max()),
                )
            )
        return "\n".join(lines)

    def wrapper(actions: torch.Tensor):
        step_counter["n"] += 1
        obs, rew, dones, extras = original_step(actions)
        action_abs = actions.abs().amax(dim=1)
        jv_abs = data.joint_vel.abs().amax(dim=1)
        actor_obs = obs["actor"] if "actor" in obs else None
        obs_abs = actor_obs.abs().amax(dim=1) if actor_obs is not None else torch.zeros_like(jv_abs)
        nonfinite = ~torch.isfinite(actions).reshape(actions.shape[0], -1).all(dim=1)
        nonfinite |= ~torch.isfinite(actor_obs).reshape(actor_obs.shape[0], -1).all(dim=1)
        nonfinite |= ~torch.isfinite(data.joint_vel).all(dim=1)

        step_max_action = float(action_abs.max())
        step_max_jv = float(jv_abs.max())
        step_max_obs = float(obs_abs.max())
        counts = {
            "gt5": int((action_abs > 5).sum()),
            "gt10": int((action_abs > 10).sum()),
            "gt50": int((action_abs > 50).sum()),
        }
        history.append(
            {
                "step": step_counter["n"],
                "action": step_max_action,
                "obs": step_max_obs,
                "joint_vel": step_max_jv,
                **counts,
            }
        )
        if step_max_action > best["action"]:
            best["action"] = step_max_action
            idx = int(action_abs.argmax())
            print(
                "[PROBE] new |action| max {:.3f} at step {} env {} "
                "(|jv|={:.2f} ep_len={} counts={})".format(
                    step_max_action,
                    step_counter["n"],
                    idx,
                    float(jv_abs[idx]),
                    int(unwrapped.episode_length_buf[idx]),
                    counts,
                )
            )
        bad = (
            (action_abs > thr_action)
            | (jv_abs > thr_jv)
            | (obs_abs > thr_obs)
            | nonfinite
        )
        if not bool(bad.any()):
            return obs, rew, dones, extras

        ids = torch.nonzero(bad, as_tuple=False).flatten()
        print("\n[PROBE] anomaly detected")
        print(
            f"[PROBE] step={step_counter['n']} envs={ids[:8].tolist()} "
            f"thresholds action>{thr_action:g} joint_vel>{thr_jv:g} obs>{thr_obs:g} "
            f"nonfinite={int(nonfinite.sum())}"
        )
        for rec in history:
            print(
                "  history step={step} |action|max={action:.4g} |obs|max={obs:.4g} "
                "|joint_vel|max={joint_vel:.4g} gt5={gt5} gt10={gt10} gt50={gt50}".format(**rec)
            )
        print(describe(ids, actions, obs))
        raise SystemExit(0)

    vec_env.step = wrapper
    vec_env._black_anomaly_probe_installed = True
    print(
        "[PROBE] installed: hard action>{:g} joint_vel>{:g} obs>{:g}".format(
            thr_action, thr_jv, thr_obs
        )
    )
