"""PPO (blind) -> privileged teacher warm start (initialization, not resume).

The plum teacher observes ``proprio history (324) + privileged fine scan (1764)``.
A blind PPO/deploy policy observes ``57 x 6 = 342`` (frame-major, newest first,
with a 3-D command per frame).  Starting the teacher from the blind policy means
the teacher begins as a competent wheel-leg locomotor and only has to learn the
foot placement, instead of relearning a gait on a 0.2 m step.

The two layouts differ in THREE ways, so a plain prefix copy is wrong:

1. **command columns dropped** -- the fixed plum task removes the velocity
   command from the actor, so the blind's per-frame [0:3] block has no target.
2. **history order** -- the deploy layout is frame-major / newest-first; MjLab's
   native history is term-major / oldest-first.
3. **term order within a frame** -- the deploy contract is
   ``commands, ang_vel_body, gravity_vec, dof_pos, dof_vel, actions``.

``blind_to_teacher_columns`` therefore returns an explicit scatter index.  It is
built from named term layouts rather than magic numbers and is asserted to be a
bijection by the unit tests and by ``scripts/plum/smoke.py``.

Only the actor is copied.  The critic observes the 247-D privileged critic group
in both policies, but the blind checkpoint's critic was trained against a
different reward scale, so it is left at its fresh init (same reasoning as
``RC_RESUME_LOAD_CFG=actor`` for cross-reward fine-tuning).
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: Blind deployment frame: (term, dim, offset within the 57-D frame).
BLIND_FRAME_LAYOUT = (
    ("commands", 3, 0),
    ("ang_vel_body", 3, 3),
    ("gravity_vec", 3, 6),
    ("dof_pos", 16, 9),
    ("dof_vel", 16, 25),
    ("actions", 16, 41),
)
#: Teacher native frame (MjLab term-major / oldest-first): (term, dim).
TEACHER_TERM_LAYOUT = (
    ("ang_vel_body", 3),
    ("gravity_vec", 3),
    ("dof_pos", 16),
    ("dof_vel", 16),
    ("actions", 16),
)
DROPPED_TERMS = ("commands",)


def _blind_frame_dim() -> int:
    return sum(dim for _, dim, _ in BLIND_FRAME_LAYOUT)


def _teacher_frame_dim() -> int:
    return sum(dim for _, dim in TEACHER_TERM_LAYOUT)


def blind_to_teacher_columns(history: int = 6) -> torch.Tensor:
    """Scatter index mapping blind deployment columns onto teacher native columns.

    Returns ``idx`` of length ``teacher_frame_dim * history`` (324).  For each
    column ``j`` in the teacher's native layout, ``idx[j]`` is the column in the
    blind's 342-D deployment vector that carries the same quantity.  Use as::

        teacher_w[:, :324] = blind_w[:, idx]

    Raises ``ValueError`` if a term is missing on either side, which is the
    fail-loud guard against a silently misaligned warm start.
    """
    if history <= 0:
        raise ValueError(f"history must be positive, got {history}")
    blind_dim = _blind_frame_dim()
    teacher_dim = _teacher_frame_dim()
    blind_terms = {term: (dim, off) for term, dim, off in BLIND_FRAME_LAYOUT}
    teacher_terms = dict(TEACHER_TERM_LAYOUT)

    dropped = [t for t in DROPPED_TERMS if t in teacher_terms]
    if dropped:
        raise ValueError(f"teacher layout must not contain dropped terms: {dropped}")
    missing = [t for t, _ in TEACHER_TERM_LAYOUT if t not in blind_terms]
    if missing:
        raise ValueError(f"blind frame layout is missing teacher terms: {missing}")

    idx = torch.zeros(teacher_dim * history, dtype=torch.long)
    offset = 0
    for term, dim in TEACHER_TERM_LAYOUT:
        b_dim, b_off = blind_terms[term]
        if b_dim != dim:
            raise ValueError(
                f"term {term!r} dim mismatch: blind {b_dim} vs teacher {dim}")
        for frame in range(history):
            native_frame = frame                    # teacher: oldest first
            blind_frame = history - 1 - frame       # blind: newest first
            for d in range(dim):
                idx[offset + frame * dim + d] = blind_frame * blind_dim + b_off + d
        offset += dim * history
    if offset != teacher_dim * history:
        raise AssertionError(f"column count {offset} != {teacher_dim * history}")
    return idx


def first_linear_names(mlp: nn.Module, prefix: str = "") -> tuple[str, str]:
    """Return the (weight, bias) state-dict names of the first ``nn.Linear``.

    ``prefix`` is the name the MLP is registered under in the actor's
    ``state_dict`` (rsl_rl's MLPModel uses ``"mlp."``); pass ``""`` for a bare
    ``nn.Sequential``, as the unit tests do.
    """
    for name, module in mlp.named_modules():
        if isinstance(module, nn.Linear):
            full = f"{prefix}{name}." if name else prefix
            return f"{full}weight", f"{full}bias"
    raise ValueError("warm start: actor MLP has no nn.Linear layer")


def warm_start_teacher_from_blind(
    actor: nn.Module,
    blind_state: dict[str, torch.Tensor],
    history: int = 6,
    scan_dim: int | None = None,
    fixed_command: tuple[float, float, float] | None = None,
    command_scale: tuple[float, float, float] = (2.0, 2.0, 0.25),
    prefix: str = "",
    skip_prefixes: tuple[str, ...] = (),
) -> dict[str, object]:
    """Copy a blind actor into the teacher actor's proprio columns in place.

    ``blind_state`` is either a training ``actor_state_dict`` or the state dict of
    an exported TorchScript policy (both use ``mlp.<i>.(weight|bias)`` keys).
    Columns beyond ``teacher_frame_dim * history`` (the fine scan) are zeroed, so
    step-0 behaviour equals the blind policy plus a scan that contributes
    nothing until PPO moves those weights.

    ``fixed_command`` makes step-0 behaviour *exactly* the blind policy rather
    than the blind policy with a zero command: the first layer is linear and the
    task command is constant, so the contribution of the dropped command columns
    is ``W[:, cmd_cols] @ (scale * command)`` per history frame, which is folded
    into the bias.  Without it the teacher would start as "blind policy at
    standstill", which is the worst possible prior for a forward task.
    """
    weight_name, bias_name = first_linear_names(actor, prefix)
    target = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    if weight_name not in blind_state:
        raise ValueError(f"warm start: blind state has no {weight_name!r}")
    if weight_name not in target:
        raise ValueError(f"warm start: teacher actor has no {weight_name!r}")

    src_w = blind_state[weight_name].to(target[weight_name].device)
    dst_w = target[weight_name]
    if src_w.ndim != 2 or dst_w.ndim != 2:
        raise ValueError("warm start: first Linear weight is not 2-D")
    blind_expected = _blind_frame_dim() * history
    if src_w.shape[1] != blind_expected:
        raise ValueError(
            f"warm start: blind actor expects {src_w.shape[1]} inputs, "
            f"expected {blind_expected} (57 x {history})")
    idx = blind_to_teacher_columns(history)
    committed = len(idx)
    if scan_dim is not None and dst_w.shape[1] != committed + scan_dim:
        raise ValueError(
            f"warm start: teacher input {dst_w.shape[1]} != "
            f"{committed} proprio + {scan_dim} scan")
    if dst_w.shape[0] != src_w.shape[0]:
        raise ValueError("warm start: first Linear hidden dims differ")

    new_w = torch.zeros_like(dst_w)
    new_w[:, :committed] = src_w[:, idx.to(src_w.device)].to(new_w.dtype)
    new_b = blind_state[bias_name].to(device=dst_w.device, dtype=dst_w.dtype).clone()

    folded = 0
    if fixed_command is not None:
        if len(command_scale) != len(fixed_command):
            raise ValueError("warm start: command_scale / fixed_command length differ")
        blind_dim = _blind_frame_dim()
        cmd = torch.tensor(
            [c * s for c, s in zip(fixed_command, command_scale)],
            dtype=new_w.dtype, device=new_w.device)
        # Sum over history frames: every frame sees the same task command.
        for frame in range(history):
            base = frame * blind_dim
            # BLIND_FRAME_LAYOUT puts "commands" at offset 0; assert rather than assume.
            cmd_off = next(off for term, _, off in BLIND_FRAME_LAYOUT
                           if term == "commands")
            new_b += src_w[:, base + cmd_off:base + cmd_off + len(cmd)].to(
                new_w.dtype) @ cmd
            folded += len(cmd)
    target[weight_name] = new_w
    target[bias_name] = new_b.to(device=dst_w.device)

    # TorchScript exports carry only the MLP; the Gaussian's std is not in them.
    # Require every MLP key, allow the distribution keys to stay at their init.
    optimiser_keys: list[str] = []
    copied = 0
    for key in sorted(target):
        if key in (weight_name, bias_name):
            continue
        if skip_prefixes and key.startswith(tuple(skip_prefixes)):
            # Modules the blind policy does not have (e.g. the depth CNN).
            # Leave them at their fresh initialisation.
            continue
        if key not in blind_state:
            if key.startswith("distribution."):
                optimiser_keys.append(key)
                continue
            raise ValueError(f"warm start: blind state missing layer {key!r}")
        if blind_state[key].shape != target[key].shape:
            raise ValueError(
                f"warm start: shape mismatch at {key!r}: "
                f"{tuple(blind_state[key].shape)} vs {tuple(target[key].shape)}")
        target[key] = blind_state[key].to(device=target[key].device, dtype=target[key].dtype)
        copied += 1
    actor.load_state_dict(target, strict=True)
    print(
        f"[plum/warm-start] blind {src_w.shape[1]} -> teacher {dst_w.shape[1]} "
        f"(copied {committed} proprio columns via the layout map, "
        f"zeroed {dst_w.shape[1] - committed} scan columns, "
        f"{copied} later-layer tensors, folded constant command="
        f"{fixed_command} into the bias)", flush=True)
    return {
        "source_actor_input_dim": int(src_w.shape[1]),
        "target_actor_input_dim": int(dst_w.shape[1]),
        "copied_columns": committed,
        "zeroed_columns": int(dst_w.shape[1] - committed),
        "later_tensors_copied": copied,
        "folded_command_values": folded,
    }
