"""blackW wheel-legged robot configuration (16 joints: 4 x hip/thigh/calf/wheel).

Model provenance: ``xmls/blackW_description.xml`` and ``xmls/assets/*.STL`` are
copied verbatim from the organiser's ``blackW.zip`` (2026-10-13); the per-file
sha256 is recorded in the workspace ``assets.manifest.json``.


The MJCF/STL assets are the organiser's blackW model used by the deployed rl_sar stack.
Policy joint order is FL, FR, RL, RR (hip, thigh, calf, wheel); the MJCF actuator/body
order is FL, FR, RR, RL, so callers must use the explicit names below and let MjLab
resolve by name.
"""

from pathlib import Path

import mujoco

from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

_HERE = Path(__file__).parent
BLACKW_XML: Path = _HERE / "xmls" / "blackW_description.xml"

assert BLACKW_XML.exists(), f"blackW MJCF not found: {BLACKW_XML}"

BLACKW_FOOT_NAMES = ("FL", "FR", "RL", "RR")
BLACKW_WHEEL_JOINTS = tuple(f"{leg}_wheel_joint" for leg in BLACKW_FOOT_NAMES)
BLACKW_JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in BLACKW_FOOT_NAMES
    for joint in ("hip", "thigh", "calf", "wheel")
)

# Deployed blackW contract: legs kp=50/kd=1.2, wheels kp=0/kd=1.0;
# torque limits hip/thigh=23.7, calf=59.25, wheel=20 N*m.
BLACKW_LEG_STIFFNESS = 50.0
BLACKW_LEG_DAMPING = 1.2
BLACKW_WHEEL_DAMPING = 1.0
BLACKW_EFFORT = {
    "hip": 23.7,
    "thigh": 23.7,
    "calf": 59.25,
    "wheel": 20.0,
}

# Default stand pose used by the deployment: hip +/-0.05, thigh +/-0.8014,
# calf -/+1.527, wheel 0.
BLACKW_DEFAULT_JOINT_POS = {
    "FL_hip_joint": 0.05,
    "FL_thigh_joint": 0.8014,
    "FL_calf_joint": -1.527,
    "FL_wheel_joint": 0.0,
    "FR_hip_joint": -0.05,
    "FR_thigh_joint": -0.8014,
    "FR_calf_joint": 1.527,
    "FR_wheel_joint": 0.0,
    "RL_hip_joint": -0.05,
    "RL_thigh_joint": 0.8014,
    "RL_calf_joint": -1.527,
    "RL_wheel_joint": 0.0,
    "RR_hip_joint": 0.05,
    "RR_thigh_joint": -0.8014,
    "RR_calf_joint": 1.527,
    "RR_wheel_joint": 0.0,
}


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(BLACKW_XML))
    # The organiser MJCF ships its own 16 <motor> actuators and a default keyframe.
    # MjLab's IdealPdActuator adds the actuators it manages, so the pre-existing motors
    # and the keyframe (whose ctrl length refers to them) must be removed.
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for key in list(spec.keys):
        spec.delete(key)
    return spec


def _leg_actuator(joint_name: str) -> IdealPdActuatorCfg:
    kind = joint_name.split("_")[1]
    return IdealPdActuatorCfg(
        target_names_expr=(joint_name,),
        stiffness=BLACKW_LEG_STIFFNESS,
        damping=BLACKW_LEG_DAMPING,
        effort_limit=BLACKW_EFFORT[kind],
    )


def _wheel_actuator(joint_name: str) -> IdealPdActuatorCfg:
    return IdealPdActuatorCfg(
        target_names_expr=(joint_name,),
        stiffness=0.0,
        damping=BLACKW_WHEEL_DAMPING,
        effort_limit=BLACKW_EFFORT["wheel"],
    )


BLACKW_ACTUATOR_CFGS = tuple(
    _wheel_actuator(name) if name in BLACKW_WHEEL_JOINTS else _leg_actuator(name)
    for name in BLACKW_JOINT_NAMES
)

BLACKW_ARTICULATION = EntityArticulationInfoCfg(
    actuators=BLACKW_ACTUATOR_CFGS,
    soft_joint_pos_limit_factor=0.9,
)

INIT_STATE = EntityCfg.InitialStateCfg(
    # The deployed blackW spawn uses support height + wheel offset (~0.676 m on flat);
    # training starts a little above contact and lets the policy settle.
    pos=(0.0, 0.0, 0.69),
    joint_pos=dict(BLACKW_DEFAULT_JOINT_POS),
    joint_vel={".*": 0.0},
)


def get_blackw_robot_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=get_spec,
        init_state=INIT_STATE,
        articulation=BLACKW_ARTICULATION,
        sort_actuators=True,
    )


if __name__ == "__main__":
    from mjlab.entity import Entity

    robot = Entity(get_blackw_robot_cfg())
    model = robot.spec.compile()
    print("blackW Entity compiled successfully")
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody}")
