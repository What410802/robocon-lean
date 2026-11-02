"""Black flat/rough 的单一人工训练参数入口。

修改后由 env_cfgs.py / terrain.py / rl_cfg.py 组装为 MjLab v1.6.0 原生配置；
这里不替代 ManagerBasedRlEnvCfg，也不定义 term 顺序或 policy joint order。
机器人固有参数、default pose 与 actuator 见 robots/black/。

POLICY / DEPLOYMENT CONTRACT-SENSITIVE：control timestep/decimation、action
scale 与 actor observation scales。修改它们须建立新训练契约并重验导出和部署兼容性。
"""

from dataclasses import dataclass, field
from typing import Literal
import math

from alldog_mjlab.utils import knobs


@dataclass(frozen=True)
class EnvParams:
    # Black 训练默认 4096 env，play 默认 1；CLI 显式指定时仍以 CLI 为准。
    train_num_envs: int = 4096
    play_num_envs: int = 1
    episode_length_s: float = 20.0


@dataclass(frozen=True)
class ControlParams:
    """Frequency and action contract shared by training, the judge and deployment.

    Three independent periods are stored; the two integer decimations are derived
    so they can never go out of sync:

    * ``physics_dt``    MuJoCo substep (official MJCF option timestep = 0.002 s);
    * ``control_period`` deployment motor-command period (organiser controller 5 ms);
    * ``policy_period`` policy inference period (50 Hz).

    ``sim_decimation = policy_period / physics_dt`` (10) and
    ``deployment_decimation = policy_period / control_period`` (4).
    """

    physics_dt: float = knobs.get_float("RC_PHYSICS_DT", 0.002)
    control_period: float = knobs.get_float("RC_CONTROL_PERIOD", 0.005)
    policy_period: float = knobs.get_float("RC_POLICY_PERIOD", 0.020)
    action_scale: float = 0.25
    # Deployment clips the raw network output to +/-100 (rl_sar config
    # clip_actions_*).  Applying the same bound in training removes a
    # train/deploy mismatch and caps the last-action feedback term; <=0 disables.
    action_clip: float = 100.0

    def __post_init__(self) -> None:
        for name in ("physics_dt", "control_period", "policy_period"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        sim = self.policy_period / self.physics_dt
        dep = self.policy_period / self.control_period
        for label, value in (("policy_period/physics_dt", sim),
                             ("policy_period/control_period", dep)):
            if abs(value - round(value)) > 1e-9:
                raise ValueError(
                    f"frequency contract violated: {label} = {value:.6f} is not an "
                    "integer; frequencies must be integer multiples"
                )
        override = knobs.optional("RC_SIM_DECIMATION")
        if override is not None and int(override) != round(sim):
            print(
                f"[frequency] WARNING: RC_SIM_DECIMATION={override} overrides the "
                f"derived sim_decimation={round(sim)}; the env cfg will not match the "
                "declared policy period",
                flush=True,
            )

    @property
    def sim_decimation(self) -> int:
        return round(self.policy_period / self.physics_dt)

    @property
    def deployment_decimation(self) -> int:
        return round(self.policy_period / self.control_period)

    @property
    def policy_dt(self) -> float:
        return self.policy_period


@dataclass(frozen=True)
class ObservationParams:
    # POLICY / DEPLOYMENT CONTRACT-SENSITIVE：仅数值 scale；term 顺序在 env_cfgs.py。
    command_scale: tuple[float, float, float] = (2.0, 2.0, 0.25)
    base_ang_vel_scale: float = 0.25
    projected_gravity_scale: float = 1.0
    joint_pos_scale: float = 1.0
    joint_vel_scale: float = 0.05
    last_action_scale: float = 1.0
    actor_corruption_enabled: bool = True
    # Deployment clips the concatenated observation to +/-clip_obs (100).
    # Per-term clipping here is equivalent after scaling and bounds the
    # last-action feedback loop in rare falling states; <=0 disables.
    actor_obs_clip: float = 100.0


# =============================================================================
# Depth camera (C2 vision mainline)
# =============================================================================


@dataclass(frozen=True)
class DepthCameraParams:
    """Simulated front depth camera (mount: robot/d435_link).

    This is a *simulated real sensor*: the rules allow a camera with realistic
    noise/latency/extrinsics, and the policy must not see the privileged
    heightmap/foot sensors.  The camera renders through mujoco_warp's
    RenderContext (GPU), so it needs no EGL context.

    64x48 / 70 deg / 0.1-3.0 m is the first cost-smoke point; DR (noise, frame
    drops, extrinsic jitter) is a follow-up item, not implemented yet.
    """

    # Render resolution. Lowered via env vars only for cost A/B on one machine.
    width: int = 64
    height: int = 48
    fovy: float = 70.0
    # Mount pose: RealSense D435 body at trunk front, lens looking forward and
    # 15 deg down.  Quaternion is body<-camera in (w, x, y, z).
    parent_body: str = "d435_link"
    position: tuple[float, float, float] = (0.01, 0.0, 0.0)
    quaternion: tuple[float, float, float, float] = (0.5610, 0.4304, -0.4304, -0.5610)
    # Depth range and observation normalization.
    min_depth: float = 0.10
    cutoff_distance: float = 3.0
    # Uniform observation noise on the normalized depth image (training only).
    obs_noise: float = 0.02


# =============================================================================
# Command
# =============================================================================


@dataclass(frozen=True)
class CommandCurriculumParams:
    """Black forward-speed command curriculum 冻结参数。

    语义来源：super-dog Black 后期 ``update_command_curriculum()``（buffer / EMA /
    pass streak 状态机），不是 official HIMLoco 的直接映射。评价样本是每个完成
    episode 的线性速度 tracking ratio；全部阈值与统计参数在这里集中冻结，
    不散落在 curriculum term 实现内。
    """

    # play 模式下 curriculum 完全不运行（不采样、不评估、不扩 range）。
    enabled: bool = True
    # 初始范围沿用 CommandParams.lin_vel_x；这里显式列出便于切换默认值。
    initial_lin_vel_x: tuple[float, float] = (-1.0, 1.0)
    # 目标范围上界：vx_min >= -max_abs_vx、vx_max <= +max_abs_vx，永不越界。
    # RC_CMD_CURR_MAX_VX 默认 2.0（冻结 baseline）；velocity-tracking Phase 1 用
    # 1.2 收紧上界，避免课程把 vx 扩到没有评估/部署需求的高速区。
    max_abs_vx: float = knobs.get_float("RC_CMD_CURR_MAX_VX", 2.0)
    # 每次推进双边同时扩展 step：vx_min -= step、vx_max += step。
    step: float = 0.1
    # low-speed 组（0.2 < |vx| <= 0.6V）的 tracking ratio EMA 需超过该阈值。
    threshold_low: float = 0.70
    # high-speed 组阈值 = threshold_low - threshold_offset（super-dog：0.70 - 0.10）。
    threshold_offset: float = 0.10
    # EMA 平滑系数：ema = (1-alpha)*old + alpha*new_mean。
    ema_alpha: float = 0.20
    # 连续这么多次成功 evaluation（EMA 双双过线）才扩一次 range。
    required_passes: int = 2
    # buffer 累计这么多个有效 episode sample 才触发一次 low/high 评估。
    buffer_min: int = 256
    # 组计数下限：评估要求 low 与 high 样本各自至少这么多个；不足时保留 buffer
    # 继续（不清空），EMA / pass streak 也不更新，等待更多 episode 样本。
    min_low_count: int = 8
    min_high_count: int = 4
    # 组划分下限：|vx| <= low_speed_min 的样本（含 standing / 极小指令）不属于任何
    # 组，只保留在 buffer 里占位；low/high 上限分界 = low_high_split_ratio * V。
    low_speed_min: float = 0.2
    low_high_split_ratio: float = 0.6

    def threshold_high(self) -> float:
        """high-speed 组阈值 = low 阈值 - 阈值偏移（super-dog 语义：0.70 - 0.10）。"""
        return self.threshold_low - self.threshold_offset

    def validate(self) -> None:
        if not (0.0 < self.ema_alpha < 1.0):
            raise ValueError("command curriculum ema_alpha must be in (0, 1)")
        if self.required_passes < 1:
            raise ValueError("command curriculum required_passes must be >= 1")
        if self.buffer_min < 1:
            raise ValueError("command curriculum buffer_min must be >= 1")
        if self.step <= 0.0 or self.max_abs_vx <= 0.0:
            raise ValueError("command curriculum step / max_abs_vx must be positive")
        initial_abs = max(abs(self.initial_lin_vel_x[0]), abs(self.initial_lin_vel_x[1]))
        if self.max_abs_vx < initial_abs:
            raise ValueError(
                f"RC_CMD_CURR_MAX_VX={self.max_abs_vx} is below the command-curriculum "
                f"initial range {self.initial_lin_vel_x}; the curriculum ceiling would "
                "already exclude the sampling range. Raise RC_CMD_CURR_MAX_VX "
                f"to >= {initial_abs}."
            )
        if self.threshold_high() <= 0.0:
            raise ValueError(
                "command curriculum high threshold must be positive: "
                f"{self.threshold_low} - {self.threshold_offset}"
            )
        if self.min_low_count < 1 or self.min_high_count < 1:
            raise ValueError("command curriculum group counts must be >= 1")


@dataclass(frozen=True)
class CommandParams:
    """Black command contract：初始 vx 范围 + native sampler + forward-speed 课程。

    生成器为 MjLab v1.6 `UniformVelocityCommand`：body-frame 速度指令，按
    `resampling_time` 重采样。行/角速度单位为 m/s 与 rad/s。

    ``lin_vel_x`` 是 ​​**课程起点**：训练中由 command curriculum 按策略 tracking
    表现逐步扩展到 ±max_abs_vx（见 CommandCurriculumParams）；``lin_vel_y`` 与
    ``ang_vel_z`` 训练全程固定。play 模式不运行 curriculum（使用 initial 范围）。
    """

    resampling_time: tuple[float, float] = (10.0, 10.0)

    lin_vel_x: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y: tuple[float, float] = (-1.0, 1.0)
    ang_vel_z: tuple[float, float] = (-math.pi, math.pi)

    # native sampler 比例：standing env 指令置 0；forward-only env 取 vx >= 0.3
    # 且 vy = ang_vel_z = 0；world-frame env 与 reset 初速度随机均未启用。
    # 课程学习只改 ranges.lin_vel_x，sampler 比例训练全程不变。
    standing_fraction: float = 0.1
    forward_fraction: float = 0.2
    world_fraction: float = 0.0
    init_velocity_prob: float = 0.0

    # 性能驱动的 forward-speed command curriculum（super-dog Black 后期语义）。
    # lin_vel_y / ang_vel_z 不参与课程学习；唯一被改写的量是 lin_vel_x。
    command_curriculum: CommandCurriculumParams = field(
        default_factory=CommandCurriculumParams
    )


# =============================================================================
# Observation noise
# =============================================================================


@dataclass(frozen=True)
class NoiseParams:
    """Actor observation 各分量的 raw 值均匀噪声幅值。

    MjLab pipeline 为 compute → noise → clip → scale，因此这里写加在 raw 值上的
    噪声：进入 policy 的最终幅值 = raw noise × observation scale。
    """

    enabled: bool = True
    base_ang_vel: tuple[float, float] = (-0.3, 0.3)
    projected_gravity: tuple[float, float] = (-0.05, 0.05)
    joint_pos: tuple[float, float] = (-0.08, 0.08)
    joint_vel: tuple[float, float] = (-2.0, 2.0)


# =============================================================================
# Reset
# =============================================================================


@dataclass(frozen=True)
class ResetParams:
    """Episode reset 的 pose / velocity 采样范围。

    root_pose 为空 dict 表示不随机（default pose + zero offset，root 高度取 default
    0.45 m）；root_velocity 的 key 为 MjLab v1.6.0 的 SE(3) 轴名。
    joint_position 是以 default joint pose 为均值的对称 offset，三组 support 均完全
    落在 soft joint limits 内（不依赖 clamp）；joint velocity 不随机。
    """

    root_pose: dict[str, tuple[float, float]] = field(default_factory=dict)

    root_velocity: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.5, 0.5),
            "roll": (-0.5, 0.5),
            "pitch": (-0.5, 0.5),
            "yaw": (-0.5, 0.5),
        }
    )

    joint_position: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "hip": (0.0, 0.0),
            "thigh": (-0.4007, 0.4007),
            "calf": (-0.5945, 0.5945),
        }
    )

    joint_velocity: tuple[float, float] = (0.0, 0.0)


# =============================================================================
# Termination
# =============================================================================


@dataclass(frozen=True)
class TerminationParams:
    """终止阈值。

    illegal_contact_force [N]：illegal_contact_bodies 内 body 与 terrain 接触力
    超过它即终止；illegal_contact_history 取一个 control step 内的 physics
    substep 数。illegal_contact_bodies 是 legged_gym 式
    ``termination_contact_names`` 的等价配置项：触发碰撞终止的 body 集合，
    默认为冻结的 trunk + 四 thigh（不含 hip / calf / foot）。名字必须与 robot
    body 名一致，错误名字在 env 构建时 fail-loud。
    stuck_* [s] / [m/s] / [m/s] / [s]：有效 planar command 下沿指令方向无 progress
    的连续时长超过 stuck_timeout 即终止，stuck_grace 内不计时。
    """

    illegal_contact_force: float = 1.0
    illegal_contact_history: int = 4
    illegal_contact_bodies: tuple[str, ...] = (
        "trunk",
        "FL_thigh",
        "FR_thigh",
        "RL_thigh",
        "RR_thigh",
    )

    stuck_timeout: float = 4.0
    stuck_velocity_threshold: float = 0.05
    stuck_command_threshold: float = 0.2
    stuck_grace: float = 1.0

    # 数值安全终止（不是能力判据）：物理状态进入非有限值或极端速度时终止该 env，
    # 避免 "怪动作 -> 观测反馈 -> 更大动作 -> NaN -> CUDA 710" 链条传下去。
    # 阈值远高于正常步态（正常 |qvel| < ~40、|root ang vel| < ~5）。
    max_joint_vel: float = 100.0
    max_root_ang_vel: float = 40.0
    max_root_lin_vel: float = 20.0


# =============================================================================
# Reward
# =============================================================================


@dataclass(frozen=True)
class RewardScales:
    # Command tracking weights。RC_TRACK_LIN_W / RC_TRACK_ANG_W 的默认值就是冻结的
    # flat baseline（1.0 / 0.5），不设 env var 时旧 run 的 reward 表逐值不变。
    tracking_linear: float = knobs.get_float("RC_TRACK_LIN_W", 1.0)
    tracking_angular: float = knobs.get_float("RC_TRACK_ANG_W", 0.5)
    lin_vel_z: float = -2.0
    ang_vel_xy: float = -0.05
    orientation: float = -0.2
    base_height: float = -1.0
    foot_clearance: float = -0.01
    dof_acc: float = -2.5e-7
    joint_power: float = -2e-5
    action_rate: float = -0.01
    smoothness: float = -0.01


@dataclass(frozen=True)
class RewardParams:
    """Black flat v1 reward baseline 的系数与非系数参数。

    tracking_sigma 是指数分母，不是 sigma²；base_height_target 是期望 root
    高度，与 reset 的 0.45 m 职责不同。权重集中在 scales。

    ``RC_TRACK_SIGMA`` / ``RC_TRACK_ERR_L2_W`` 只改 tracking 项本身：默认值 0.25 /
    0.0 与冻结 baseline 逐值一致，不设 env var 时旧 run 的 reward 表不变。
    ``tracking_err_l2`` 是显式 L2 平面 tracking 误差惩罚的幅值（>0 时注册
    ``track_linear_velocity_l2``，weight = -tracking_err_l2；0 = 关闭/legacy）。
    """

    tracking_sigma: float = knobs.get_float("RC_TRACK_SIGMA", 0.25)
    tracking_err_l2: float = knobs.get_float("RC_TRACK_ERR_L2_W", 0.0)
    base_height_target: float = 0.43
    clearance_height_target: float = 0.09
    scales: RewardScales = field(default_factory=RewardScales)

    def __post_init__(self) -> None:
        if not math.isfinite(self.tracking_sigma) or self.tracking_sigma <= 0.0:
            raise ValueError(
                f"RC_TRACK_SIGMA must be a positive finite number, got {self.tracking_sigma}"
            )
        if not math.isfinite(self.tracking_err_l2) or self.tracking_err_l2 < 0.0:
            raise ValueError(
                "RC_TRACK_ERR_L2_W must be a non-negative finite number "
                f"(it is a penalty magnitude; 0 = off), got {self.tracking_err_l2}"
            )


# =============================================================================
# Domain randomization
# =============================================================================


@dataclass(frozen=True)
class DomainRandomizationParams:
    """Black flat v1 的 domain randomization（train 生效，play 全部移除）。

    startup（每 env 采样一次并在 episode 内保持）：friction、payload_mass、
    com_offset、encoder_bias；
    reset（每次 episode reset 重新采样）：pd_gains；
    interval（训练中周期性施加 xy 速度增量）：push_interval / push_velocity。
    """

    friction_enabled: bool = True
    payload_enabled: bool = True
    com_enabled: bool = True
    pd_gain_enabled: bool = True
    encoder_bias_enabled: bool = True
    push_enabled: bool = True

    # Contact
    friction: tuple[float, float] = (0.2, 1.25)

    # Rigid body（trunk payload 与 nominal COM 的 offset）
    payload_mass: tuple[float, float] = (-1.0, 2.0)
    com_offset: dict[int, tuple[float, float]] = field(
        default_factory=lambda: {
            0: (-0.05, 0.05),
            1: (-0.05, 0.05),
            2: (-0.05, 0.05),
        }
    )

    # Actuator（相对 nominal Kp=40 / Kd=1.2 的缩放）
    kp_scale: tuple[float, float] = (0.9, 1.1)
    kd_scale: tuple[float, float] = (0.9, 1.1)

    # Sensor（固定 encoder calibration bias，不是 observation noise）
    encoder_bias: tuple[float, float] = (-0.015, 0.015)

    # Disturbance（间隔 [s] 与 root xy 速度增量 [m/s]）
    # 2026-10-24: was a fixed 16 s, i.e. exactly one push at the same phase of every
    # 20 s episode -> the policy could learn the schedule and pushes were missing from
    # most of the experience.  Sample the interval per env from this range instead
    # (event manager samples lower..upper at reset and after every push; with 20 s
    # episodes some envs now get zero pushes and others one or two).
    push_interval: tuple[float, float] = (6.0, 20.0)
    push_velocity: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-1.0, 1.0),
            "y": (-1.0, 1.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
    )


# =============================================================================
# Terrain
# =============================================================================


@dataclass(frozen=True)
class TerrainParams:
    """Black rough v1 的 terrain generator 数值。

    patch 尺寸与高度场分辨率与 legacy Black 一致（4 m 半径 / 0.1 m / 0.005 m）。
    MjLab v1.6 curriculum 的 row difficulty 按 ``row / (num_rows - 1)`` 在
    ``difficulty_range`` 内插值，因此取 ``(0.0, 0.9)`` + ``num_rows = 10`` 以恢复
    legacy 的 ``row / num_rows`` 语义（0.0, 0.1, ..., 0.9，不含 1.0）。

    proportions 是 curriculum 模式下每类 terrain 的 env 分配权重（不是列数）。
    单项可以为 0；总和必须大于 0。
    """

    # Patch 与网格
    size: tuple[float, float] = (8.0, 8.0)
    num_rows: int = 10
    difficulty_range: tuple[float, float] = (0.0, 0.9)
    # MjLab border 为 z = 0 的 flat apron（与 legacy heightfield border_size=25 m
    # 语义不同），取 MjLab native rough preset 的 20.0 m。
    border_width: float = 20.0

    # 高度场分辨率（legacy 同值）
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    platform_width: float = 3.0

    # 各 terrain 的 env 分配权重
    proportions: dict[str, float] = field(
        default_factory=lambda: {
            "flat": 0.10,
            "smooth_slope_up": 0.05,
            "smooth_slope_down": 0.05,
            "rough_slope": 0.10,
            "discrete_obstacles": 0.20,
            "stairs_up": 0.25,
            "stairs_down": 0.25,
        }
    )

    # smooth slope：max slope = 0.7 x 0.9 = 0.63
    slope_range: tuple[float, float] = (0.0, 0.7)

    # rough slope 噪声：amplitude = rough_noise_base + rough_noise_gain x difficulty
    rough_noise_base: float = 0.015
    rough_noise_gain: float = 0.1
    rough_noise_step: float = 0.005
    rough_noise_downsample: float = 0.2
    rough_base_thickness_ratio: float = 1.0

    # discrete obstacles：height = 0.06 + difficulty x 0.2
    obstacle_height_range: tuple[float, float] = (0.06, 0.26)
    obstacle_width_range: tuple[float, float] = (1.0, 2.0)
    obstacle_count: int = 20

    # stairs（native box 金字塔台阶，up / down）：
    # step_height = stair_step_height_base + difficulty x stair_step_height_gain
    #             = 0.05 + 0.18 x d（super-dog HEAD lineage）；
    # step_width 恒 0.3 m（legacy 中的宽度课程学习插值是 dead code，实际恒 0.3）。
    # 几何实现为 MjLab native BoxPyramidStairsTerrainCfg /
    # BoxInvertedPyramidStairsTerrainCfg（真实竖直台阶沿，legacy 为 heightfield
    # 0.1 m 网格量化；台缘形状为有意 framework difference）。
    stair_step_height_base: float = 0.05
    stair_step_height_gain: float = 0.18
    stair_step_width: float = 0.3

    # 初始 terrain level 上限（inclusive，与 legacy max_init_terrain_level 同义）
    max_init_terrain_level: int = 5

    def validate(self) -> None:
        expected = {
            "flat",
            "smooth_slope_up",
            "smooth_slope_down",
            "rough_slope",
            "discrete_obstacles",
            "stairs_up",
            "stairs_down",
        }
        if set(self.proportions) != expected:
            raise ValueError(f"Black terrain proportions must have keys {sorted(expected)}")
        if any(
            not math.isfinite(value) or value < 0
            for value in self.proportions.values()
        ):
            raise ValueError("Black terrain proportions must be finite and nonnegative")
        if sum(self.proportions.values()) <= 0:
            raise ValueError("Black terrain proportions must have positive total weight")


@dataclass(frozen=True)
class SimulationParams:
    # MJWarp per-world constraint capacity（njmax）。
    #
    # 旧 MjLab velocity baseline = 1500。下限由 MJWarp put_data 对模板 spawn state 的
    # 硬性要求决定（njmax >= mjd.nefc）：flat seed = 144 / rough seed = 220（约成：
    # 12 friction_dof + 4 joint limit + spawn contact 的 pyramidal rows；Black nv=18<=32
    # → dense，tile 16 对齐）。
    #
    # Black rough 实测（check_black_sim_capacity，3000 env / 2.4M world-substep samples）：
    # max nefc = 80、p99.9 = 52，overflow = NO。
    #
    # 当前候选 = 256：seed floor 的 ~1.16x、实测 runtime max 的 3.2x；
    # 这是 capacity tuning（减少 constraint arrays / solver workspace 预分配），
    # 不改变正常 constraint solver 数学行为；前提是运行过程中 nefc 不超过容量
    # （overflow bit = YES 即失效，需回退）。
    njmax: int = 256
    # MJWarp GPU contact-capacity workaround；不是 legacy Black 行为。flat 保持 native 35。
    rough_nconmax: int = 128


@dataclass(frozen=True)
class PolicyParams:
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    # POLICY / DEPLOYMENT CONTRACT-SENSITIVE：actor 保持不做 running normalization。
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = True
    # Guarded fine-tuning from a distilled checkpoint freezes the exploration
    # noise: BC policies are sensitive to action-space noise, and PPO otherwise
    # pushes std up (observed 0.30 -> 0.52-0.68 on the specialist runs).
    initial_std: float = knobs.get_float("RC_INIT_STD", 1.0)
    std_type: str = "scalar"
    learn_std: bool = knobs.get_bool("RC_LEARN_STD", True)


@dataclass(frozen=True)
class AlgorithmParams:
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = knobs.get_float("RC_ENTROPY_COEF", 0.01)
    num_learning_epochs: int = knobs.get_int("RC_EPOCHS", 5)
    num_mini_batches: int = knobs.get_int("RC_MINIBATCHES", 4)
    normalize_advantage_per_mini_batch: bool = knobs.get_bool("RC_NORM_ADV_PER_MB")
    # Fine-tuning from a distilled checkpoint needs a much smaller step than
    # from-scratch PPO; override with RC_LR for warm-start runs.
    learning_rate: float = knobs.get_float("RC_LR", 1.0e-3)
    schedule: str = knobs.get_str("RC_LR_SCHEDULE", "adaptive")
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0


@dataclass(frozen=True)
class HimParams:
    """Black HIM estimator / latent 参数（HIM-only）。

    PPO 侧超参数（actor/critic dims、learning rate、gamma、lam、KL、max_grad_norm 等）
    继续复用 PolicyParams / AlgorithmParams，不在 HIM section 重复。
    official HIM released code 默认：encoder 128→64→latent、target 128→64→latent、
    32 prototypes、temperature 3.0、Adam lr 1e-3、max_grad_norm 10.0。

    ``estimator_learning_rate`` 只是 estimator optimizer 的**构造初始值**；official 语义下
    每个 update 都会用 PPO 当前/adaptive ``learning_rate`` 覆盖它（见
    ``HIMPPO._update_estimator``），不是独立的 estimator LR schedule。
    """

    latent_dim: int = 16
    encoder_hidden_dims: tuple[int, ...] = (128, 64)
    target_encoder_hidden_dims: tuple[int, ...] = (128, 64)
    num_prototypes: int = 32
    temperature: float = 3.0
    # 构造初始值；训练时每个 update 被 PPO self.learning_rate 覆盖。
    estimator_learning_rate: float = 1.0e-3
    estimator_max_grad_norm: float = 10.0


@dataclass(frozen=True)
class StageRunnerParams:
    run_name: str
    load_run: str


@dataclass(frozen=True)
class RunnerParams:
    # MjLab v1.6 runner 默认 seed=42；由官方 CLI 覆盖时以 CLI 为准。
    seed: int = 42
    experiment_name: str = "black_velocity"
    # 同一 checkpoint-compatible policy family；stage 仅决定 run 名称与默认来源。
    flat: StageRunnerParams = field(
        default_factory=lambda: StageRunnerParams(run_name="flat", load_run=r".*_flat$")
    )
    rough: StageRunnerParams = field(
        default_factory=lambda: StageRunnerParams(run_name="rough", load_run=r".*_rough$")
    )
    gravel_flat: StageRunnerParams = field(
        default_factory=lambda: StageRunnerParams(
            run_name="gravel_flat", load_run=r".*_gravel_flat$"
        )
    )
    gravel_free: StageRunnerParams = field(
        default_factory=lambda: StageRunnerParams(
            run_name="gravel_free", load_run=r".*_gravel_free$"
        )
    )
    # C2 vision mainline: same basic-obstacle terrain, camera in the actor.
    depth: StageRunnerParams = field(
        default_factory=lambda: StageRunnerParams(
            run_name="basic_obstacle_depth", load_run=r".*_basic_obstacle_depth$"
        )
    )
    # HIM stage 使用独立 run 名，避免与普通 PPO run 混用（Unit 3 未注册 task，
    # 也不再做 PPO→HIM warm start；命名只用于将来的正式 HIM run）。
    flat_him: StageRunnerParams = field(
        default_factory=lambda: StageRunnerParams(
            run_name="flat_him", load_run=r".*_flat_him$"
        )
    )
    rough_him: StageRunnerParams = field(
        default_factory=lambda: StageRunnerParams(
            run_name="rough_him", load_run=r".*_rough_him$"
        )
    )
    save_interval: int = 50
    num_steps_per_env: int = 64
    max_iterations: int = 10_000
    # resume 时 command curriculum state 的默认恢复策略（PPO / HIM 共用；语义见
    # curriculum_checkpoint.resolve_restore_mode）。
    #
    # auto:
    #   same-stage resume   -> full
    #   cross-stage resume  -> range
    #   old checkpoint 无 curriculum state -> none + warning
    # full:
    #   逐值恢复 range + EMA + streak + pending buffer
    # range:
    #   只恢复 vx range；EMA / streak / buffer fresh
    # none:
    #   不恢复 curriculum state
    #
    # CLI ``--agent.command-curriculum-restore`` 可临时覆盖人工默认值；
    # PPO→HIM warm start 不经过本字段（固定 range，见 him_runner.py）。
    command_curriculum_restore: Literal["auto", "none", "range", "full"] = "range"


@dataclass(frozen=True)
class BlackConfig:
    env: EnvParams = field(default_factory=EnvParams)
    control: ControlParams = field(default_factory=ControlParams)
    command: CommandParams = field(default_factory=CommandParams)
    observation: ObservationParams = field(default_factory=ObservationParams)
    noise: NoiseParams = field(default_factory=NoiseParams)
    reset: ResetParams = field(default_factory=ResetParams)
    termination: TerminationParams = field(default_factory=TerminationParams)
    reward: RewardParams = field(default_factory=RewardParams)
    domain_rand: DomainRandomizationParams = field(default_factory=DomainRandomizationParams)
    terrain: TerrainParams = field(default_factory=TerrainParams)
    depth: DepthCameraParams = field(default_factory=DepthCameraParams)
    simulation: SimulationParams = field(default_factory=SimulationParams)
    policy: PolicyParams = field(default_factory=PolicyParams)
    him: HimParams = field(default_factory=HimParams)
    algorithm: AlgorithmParams = field(default_factory=AlgorithmParams)
    runner: RunnerParams = field(default_factory=RunnerParams)


BLACK_CONFIG = BlackConfig()
