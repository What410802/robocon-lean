"""blackW wheel-legged robot (16 joints) for the mjlab training repo."""

from .blackw_constants import (
    BLACKW_FOOT_NAMES,
    BLACKW_JOINT_NAMES,
    BLACKW_WHEEL_JOINTS,
    BLACKW_LOCKED_WHEEL_DAMPING,
    BLACKW_LOCKED_WHEEL_STIFFNESS,
    get_blackw_locked_robot_cfg,
    get_blackw_robot_cfg,
)

__all__ = [
    "BLACKW_FOOT_NAMES",
    "BLACKW_JOINT_NAMES",
    "BLACKW_WHEEL_JOINTS",
    "BLACKW_LOCKED_WHEEL_DAMPING",
    "BLACKW_LOCKED_WHEEL_STIFFNESS",
    "get_blackw_locked_robot_cfg",
    "get_blackw_robot_cfg",
]
