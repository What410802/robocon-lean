"""Part 1 scene geom-name -> judge surface-tag mapping (stdlib only).

Kept separate from the MuJoCo runner because it is stdlib-only: a scene/tag
coverage check can compare ``demo/robots/<robot>/scene_part1.xml`` geom names
against this map without importing MuJoCo (the 2026-10-22 BYOK audit caught
missing bridge_blocks/bands tags this way).
"""

GEOM_TAGS: dict[str, set[str]] = {
    "bridge_blocks_0": {"bridge"}, "bridge_blocks_1": {"bridge"}, "bridge_blocks_2": {"bridge"},
    "bridge_blocks_3": {"bridge"}, "bridge_blocks_4": {"bridge"},
    "bridge_bands_0": {"bridge"}, "bridge_bands_1": {"bridge"}, "bridge_bands_2": {"bridge"},
    "bridge_blocks_platform_l": {"platform_left"},
    "bridge_platform_r": {"platform_right", "platform_high"},
    "bridge_bands_platform_low": {"platform_low"},
    "slope_a": {"slope"},
    "slope_b": {"slope"},
    "gravel_pit_a": {"gravel"},
    "gravel_pit_b": {"gravel"},
    "high_wall": {"wall"},
    "ramp_blocks_l": {"ramp"},
    "ramp_blocks_r": {"ramp"},
    # T stairs: three arms (south / north / east) with explicit level tags.
    "t_stairs_y_low1": {"step:s:1"},
    "t_stairs_y_low2": {"step:s:2"},
    "t_stairs_y_low3": {"step:s:3"},
    "t_stairs_y_high1": {"step:n:1"},
    "t_stairs_y_high2": {"step:n:2"},
    "t_stairs_y_high3": {"step:n:3"},
    "t_stairs_x_low1": {"step:e:1"},
    "t_stairs_x_mid2": {"step:e:2"},
    "t_stairs_x_high3": {"step:e:3"},
    "t_stairs_top": {"step:top"},
}


# Ground classification shared by the Part 1 runners. The 2026-10-23 flash acceptance
# found that only plane geoms were treated as ground, missing named non-plane approach
# surfaces (bridge ramps). Ground = plane / floor / ground* names / known approach ramps.
GROUND_GEOM_NAMES = {"floor", "trench_bottom", "level_floor",
                     "ramp_blocks_l", "ramp_blocks_r", "ground", "groundplane",
                     "ground_left", "ground_right", "ground_plane"}


def is_ground_geom(name: str, is_plane: bool = False) -> bool:
    return bool(is_plane) or name in GROUND_GEOM_NAMES
