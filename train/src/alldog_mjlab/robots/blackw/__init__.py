"""blackW wheel-legged robot (16 joints) for the mjlab training repo."""

from .blackw_constants import (
    BLACKW_FOOT_NAMES,
    BLACKW_JOINT_NAMES,
    BLACKW_WHEEL_JOINTS,
    get_blackw_robot_cfg,
)

__all__ = [
    "BLACKW_FOOT_NAMES",
    "BLACKW_JOINT_NAMES",
    "BLACKW_WHEEL_JOINTS",
    "get_blackw_robot_cfg",
]
