"""blackW wheel-action contract helpers.

The 16-D action layout is ``[hip, thigh, calf, wheel] x [FL, FR, RL, RR]``.
The wheel joints are velocity targets with scale +/-10.  ``signed`` is the
official deployment/HIM contract (FL/RL +10, FR/RR -10); ``unsigned`` is the
nwwolf PPO contract (+10 all).  Weights are meaningless without this tag.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

SIGNED = "signed"
UNSIGNED = "unsigned"
WHEEL_INDICES = (3, 7, 11, 15)
SIGNED_WHEEL_SCALE = (10.0, -10.0, 10.0, -10.0)
UNSIGNED_WHEEL_SCALE = (10.0, 10.0, 10.0, 10.0)


def contract_from_wheel_scales(scales) -> str | None:
    vals = tuple(float(v) for v in scales)
    if vals == SIGNED_WHEEL_SCALE:
        return SIGNED
    if vals == UNSIGNED_WHEEL_SCALE:
        return UNSIGNED
    return None


def contract_from_action_scale(action_scale) -> str | None:
    scale = [float(v) for v in action_scale]
    if len(scale) != 16:
        return None
    return contract_from_wheel_scales([scale[i] for i in WHEEL_INDICES])


def expected_wheel_scale(contract: str) -> tuple[float, ...]:
    if contract == SIGNED:
        return SIGNED_WHEEL_SCALE
    if contract == UNSIGNED:
        return UNSIGNED_WHEEL_SCALE
    raise ValueError(f"unknown action contract {contract!r}")


def current_training_contract() -> str:
    """PPO task contract from the BLACKW_ACTION_UNSIGNED knob."""
    from alldog_mjlab.utils import knobs

    return UNSIGNED if knobs.get_bool("BLACKW_ACTION_UNSIGNED", False) else SIGNED


def checkpoint_contract(ckpt: dict[str, Any]) -> str | None:
    value = ckpt.get("action_contract")
    if value:
        return str(value)
    infos = ckpt.get("infos") or {}
    value = infos.get("action_contract")
    return str(value) if value else None


def tag_checkpoint(path: str | Path, contract: str) -> None:
    import torch

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    ckpt["action_contract"] = contract
    torch.save(ckpt, str(path))


def check_checkpoint_contract(path: str | Path, expected: str) -> str | None:
    import torch

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    actual = checkpoint_contract(ckpt)
    if actual is not None and actual != expected:
        raise ValueError(
            f"{path}: action_contract={actual!r} but expected {expected!r}"
        )
    return actual


def npz_contract(path: str | Path) -> str | None:
    with np.load(path, allow_pickle=True) as d:
        for key in ("action_contract", "action_convention"):
            if key in d:
                arr = d[key]
                if arr.size == 1:
                    return str(arr.reshape(-1)[0])
        meta = d["meta"] if "meta" in d else None
        if meta is not None and len(meta) and isinstance(meta[0], dict):
            return meta[0].get("action_contract")
    return None


def tag_npz(path: str | Path, contract: str) -> None:
    with np.load(path, allow_pickle=True) as d:
        data = {k: d[k] for k in d.files}
    data["action_contract"] = np.array(contract)
    np.savez_compressed(path, **data)


def assert_same_dataset_contract(paths) -> str | None:
    seen = {}
    for p in paths:
        c = npz_contract(p)
        if c is not None:
            seen[str(p)] = c
    if len(set(seen.values())) > 1:
        raise ValueError(f"mixed action contracts in datasets: {seen}")
    return next(iter(seen.values())) if seen else None
