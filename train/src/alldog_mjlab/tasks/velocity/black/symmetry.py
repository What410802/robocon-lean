"""Left-right mirror symmetry for the Black actor observation/action contract.

Used by RSL-RL's ``Symmetry`` extension for an auxiliary equivariance loss:

    pi(mirror(o)) == mirror(pi(o))

The contract is fixed here instead of being inferred from term names, because the
policy observation layout is a frozen deployment contract (45 dims, policy joint
order FL -> FR -> RL -> RR).
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

# Policy order is FL -> FR -> RL -> RR, each leg hip -> thigh -> calf.
# Mirroring swaps FL<->FR and RL<->RR and negates every joint-space value
# (default pose is left/right antisymmetric).
_LEG_SWAP = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)

# Actor layout: command(3) | base_ang_vel(3) | projected_gravity(3) |
# joint_pos(12) | joint_vel(12) | actions(12).
_ACTOR_COMMAND = slice(0, 3)
_ACTOR_ANG_VEL = slice(3, 6)
_ACTOR_GRAVITY = slice(6, 9)
_ACTOR_JOINT_POS = slice(9, 21)
_ACTOR_JOINT_VEL = slice(21, 33)
_ACTOR_ACTIONS = slice(33, 45)


def _mirror_vec3(values: torch.Tensor, signs: tuple[float, float, float]) -> torch.Tensor:
    scale = torch.tensor(signs, dtype=values.dtype, device=values.device)
    return values * scale


def mirror_actions(actions: torch.Tensor) -> torch.Tensor:
    """Mirror a 12-dim policy action without allocating the full observation."""
    assert actions.shape[-1] == 12, actions.shape
    mirrored = actions[..., list(_LEG_SWAP)] * -1.0
    return mirrored


# MjLab native history is *term-major* (each term's frames are contiguous):
#   [command_t0..tH-1, ang_vel_t0.., gravity_t0.., joint_pos_t0.., joint_vel_t0.., actions_t0..]
# so a 45*k observation must be mirrored per term, not as k contiguous 45-D frames.
_TERM_LAYOUT: tuple[tuple[int, tuple[float, ...] | None, bool], ...] = (
    (3, (1.0, -1.0, -1.0), False),   # command
    (3, (-1.0, 1.0, -1.0), False),   # base_ang_vel
    (3, (1.0, -1.0, 1.0), False),    # projected_gravity
    (12, None, True),                # joint_pos
    (12, None, True),                # joint_vel
    (12, None, True),                # actions
)


def mirror_actor_obs(actor_obs: torch.Tensor) -> torch.Tensor:
    """Mirror the PPO actor observation (45-D single frame or 45*k native history).

    The observation is already scaled, so the reflection is applied on the scaled
    values.  Native (term-major) history blocks are mirrored per term with their
    frame order unchanged.  Layout is asserted by shape and slices;
    ``mirror(mirror(x)) == x``.
    """
    dim = actor_obs.shape[-1]
    assert dim % 45 == 0, actor_obs.shape
    frames = dim // 45
    out = actor_obs.clone()
    offset = 0
    for term_dim, signs, swap_legs in _TERM_LAYOUT:
        block = actor_obs[..., offset : offset + term_dim * frames].reshape(
            *actor_obs.shape[:-1], frames, term_dim
        )
        if swap_legs:
            mirrored = block[..., list(_LEG_SWAP)] * -1.0
        else:
            scale = torch.tensor(signs, dtype=block.dtype, device=block.device)
            mirrored = block * scale
        out[..., offset : offset + term_dim * frames] = mirrored.reshape(
            *actor_obs.shape[:-1], term_dim * frames
        )
        offset += term_dim * frames
    return out


def mirror_obs_actions(
    env,
    obs: TensorDict | None,
    actions: torch.Tensor | None,
):
    """RSL-RL symmetry callback.

    The callback always returns the original batch concatenated with the mirrored
    batch (``2N`` rows).  RSL-RL's mirror-loss path calls it twice: once with
    observations and once with action means, and compares the mirrored halves.
    """
    del env  # contract is purely a tensor transformation; no env state is needed.
    obs_out: TensorDict | None = None
    if obs is not None:
        n = obs.batch_size[0]
        tensors: dict[str, torch.Tensor] = {}
        for key in obs.keys():
            value = obs[key]
            if key == "actor":
                mirrored = mirror_actor_obs(value)
            else:
                # This unit only enables mirror loss (not data augmentation), so
                # non-actor groups are repeated.  Keep this explicit so future
                # critic augmentation must add a real mirror transform here.
                mirrored = value
            tensors[key] = torch.cat([value, mirrored], dim=0)
        obs_out = TensorDict(tensors, batch_size=[2 * n], device=obs.device)

    actions_out: torch.Tensor | None = None
    if actions is not None:
        actions_out = torch.cat([actions, mirror_actions(actions)], dim=0)
    return obs_out, actions_out


def install_mirror_loss(alg, env, coefficient: float) -> None:
    """Attach RSL-RL's mirror loss to a PPO algorithm instance.

    Data augmentation is intentionally off for this unit: it would also require a
    correct critic/height-scan mirror transform.  The mirror loss only needs the
    actor observation/action contract, so it is the safer first A/B knob.
    """
    if coefficient <= 0.0:
        return
    from rsl_rl.extensions import Symmetry

    alg.symmetry = Symmetry(
        env=env,
        data_augmentation_func=f"{__name__}:mirror_obs_actions",
        use_data_augmentation=False,
        use_mirror_loss=True,
        mirror_loss_coeff=coefficient,
    )
    print(f"[INFO] mirror-loss symmetry enabled: coefficient={coefficient:.3g}")
