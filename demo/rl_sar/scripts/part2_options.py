"""Pure Part2 CLI validation shared by launch and the route controller."""
import math


def timeout_scale(raw):
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError("timeout_scale must be a finite number >= 0 (0 disables route timeouts)") from None
    if not math.isfinite(value) or value < 0:
        raise ValueError("timeout_scale must be a finite number >= 0 (0 disables route timeouts)")
    return value


def resolve_start(legs, raw):
    raw = str(raw).strip()
    if not raw:
        return 0
    for index, leg in enumerate(legs):
        if raw.lower() == leg["leg"].lower():
            return index
    try:
        index = int(raw) - 1
    except ValueError:
        index = -1
    if not 0 <= index < len(legs):
        names = ", ".join(leg["leg"] for leg in legs)
        raise ValueError(f"invalid resume_before={raw!r}; use 1..{len(legs)} or: {names}")
    return index


def approach_point(leg):
    """Spawn directly at the flat alignment approach before the selected obstacle."""
    return tuple(leg.get("align_approach", leg["stand"]))
