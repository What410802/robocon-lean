"""Pose-primitive profile: flat-task extensions for the Part1 move primitive.

The plan-A route bridges obstacles with a flat velocity-tracking policy used as
a closed-loop move primitive.  This package adds the two training-side
ingredients the 2026-11-01 v2 round needs (user direction: no vy investment;
fix low-speed dead zones; add platform-edge robustness):

* ``command``: low-speed mixture sampling on top of the RC_POSE_CMD envelope
  (dead-zone fix — uniform envelope sampling puts too little mass where the
  dead zones live);
* ``terrain``: platform/block edge sub-terrain for the flat task (the
  northeast-platform scenario: turn on a 0.2 m platform, wheel drops off the
  edge, the policy must recover onto the platform under command);
* ``assembly``: cfg-level wiring used by ``blackw_flat_env_cfg``.
"""

from alldog_mjlab.tasks.velocity.profiles.pose.assembly import (
	apply_pose_edge_terrain,
	apply_pose_low_speed_mixture,
)
from alldog_mjlab.tasks.velocity.profiles.pose.command import (
	PoseEnvelopeVelocityCommand,
	PoseEnvelopeVelocityCommandCfg,
	make_pose_envelope_command_cfg,
)
from alldog_mjlab.tasks.velocity.profiles.pose.terrain import (
	PoseEdgePlatformTerrainCfg,
	pose_edge_terrain_generator_cfg,
)

__all__ = [
	"apply_pose_edge_terrain",
	"apply_pose_low_speed_mixture",
	"PoseEnvelopeVelocityCommand",
	"PoseEnvelopeVelocityCommandCfg",
	"make_pose_envelope_command_cfg",
	"PoseEdgePlatformTerrainCfg",
	"pose_edge_terrain_generator_cfg",
]
