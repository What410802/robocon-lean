"""Velocity-tracking primitive (VT) profile for blackW.

Purpose: a Part 1 *handover / position-loop* model.  The route controller
(``part1_auto_route``) closes a position loop by sending body-frame
``(vx, 0, wz)`` commands; the policy must follow those commands accurately,
stop when told, reverse, and turn without translating away from the spot.

The recipe is a wheel-leg adaptation of the point-foot robust locomotion
recipes (RecoveryLoco / legged_gym family):

* full command support (forward / reverse / stop / low speed / in-place yaw),
  including explicit low-speed sampling for the dead zones;
* tracking-dominant reward + explicit L2 tracking penalty + upright and
  joint-space stance anchors (the pose recipe's fixes for lean/crouch);
* ``rotation_drift_rate`` penalty on pure-yaw commands = "turn without
  translating" (small swept footprint);
* RecoveryLoco-style random states at reset (random root pose/velocity, wider
  joint offsets), periodic push disturbances, and wheel-friction domain
  randomization (only possible after the wheel collision geoms got names);
* optional platform-edge terrain variant (``blackw-vt-edge``) for the edge
  robustness the Part 1 handover needs (bridge platforms / stair lips).

Training-only code: the deployment/eval contract (57x6 obs, 16 actions,
50 Hz policy / 2 ms physics) is unchanged.
"""

from alldog_mjlab.tasks.velocity.profiles.vt.assembly import apply_vt_contract

__all__ = ["apply_vt_contract"]
