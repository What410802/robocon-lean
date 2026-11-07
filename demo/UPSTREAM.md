# 上游来源与许可（demo/）

`demo/` 是 RoboCon 2026 比赛的 ROS 2 + MuJoCo 仿真运行时（`rl_sim` + `sim_node`），
由下列上游派生并裁剪。浏览器/评审需要确认“哪些是上游、哪些是本仓新增”时看本文件。

## 1. 上游与许可

| 上游 | 版本 / commit | 许可 | 用途 |
|---|---|---|---|
| [`fan-ziqi/rl_sar`](https://github.com/fan-ziqi/rl_sar) | `e5c2f41`（2025-07-25） | **Apache-2.0** | 运行时框架层：`library/core/{rl_sdk,observation_buffer,fsm,loop}`、policy 配置结构（`base.yaml` / `config.yaml` / `manifest.json`）、`fsm.hpp` 上游骨架 |
| 上游来源仓（`@20261005_ros2` / `@20261007_assignment` / `@20261024_blackw`） | 迁移时逐文件标注 | 同上游 / 上游之外 | MuJoCo `sim_node` + viewer、`joy_node`、`MitCommand`/`MotorState`/`ControlStatus` 消息、`rl_sim` 的 ROS 2 适配、命令安全限幅、FSM intent 改造 |
| 主办方材料（RoboCon 2026 赛事包） | `assets/{robots,policies,terrain}` / `terrain.zip` | **主办方赛事材料**（非开源许可；仅用于本赛事仿真复现） | blackW 机器人模型 `blackW_description.xml` + 18 个 mesh（入库 `robots/blackW/assets/`）、官方 `flat`/`stair`/`obstacle` 策略配置、Part1/Part2 官方地形（`scenes/part1/official/`、`scenes/part2/official/map.xml`、`scenes/part2/full_field/labels/`） |

修改过的第三方文件保留原版权头，并按 Apache-2.0 §4 以
`Modified for RoboCon 2026`（或迁移时的具体任务标签，如 `Modified for @20261007_assignment`）
标注修改。框架层文件与上游 `e5c2f41` 的差异清单/等价性证明来自迁移记录（`MIGRATION_NOTES.md`）。

## 2. 本仓新增（相对上游 rl_sar）

- **MuJoCo `sim_node`**（`quadruped_ros2/src/sim_node.cpp`）：上游没有仿真器；本仓用 MuJoCo 3.11
  跑 2 ms 物理、发布 `MotorState` / `/eval_state`（接触/轮地信息供判分）、可选 GLFW 窗口与跟随相机。
- **`joy_node` + `input_map.hpp`**：把主机手柄/键盘归一化为 `/joy`，组合键边沿解析为 Intent；
  支持运行时热切换策略与 Part1 固定动作。
- **消息定义**（`quadruped_ros2/msg/`）：`MitCommand`（控制律 → 仿真）、`MotorState`（仿真 → 控制律）、
  `ControlStatus`。
- **`rl_sim` 的 ROS 2 适配**：上游 ROS 1/真机 SDK 分支重写为 ROS 2 节点；新增 policy 热切换、
  命令安全限幅（`command_limit.hpp`）、`/eval_state` 遥测、robot.yaml 契约预检。
- **策略工程化**：`policy/<robot>/{base.yaml,knobs.hpp,fsm.hpp,behaviors/}` 的团队策略/specialist
  配置与固定动作姿态。
- **深度相机（本轮新增，上游没有）**：`quadruped_ros2/include/quadruped_ros2/depth_camera.hpp`
  + `msg/DepthFrame.msg`。`sim_node` 用 EGL 设备显示离屏渲染相机贴图 64x48 公制平面深度
  （只画 geom group 0；`mjRND_SEGMENT|mjRND_IDCOLOR`、`readDepthMap=mjDEPTH_ZEROFAR`、
  reverse-Z→米、上下翻转，与 mujoco-python 的 depth 渲染逐位一致），有窗口时复用 GLFW 上下文，
  每 `depth_every` 步先于 `/motor_state` 发布 `/depth_cam`；失败一次性告警后停发，仿真与盲走不受影响。
  `rl_sim` 新增深度策略输入路径（`policy(obs_1d, [1x1xHxW])`）与 `RL_DEPTH_DEBUG` 对拍 dump。
- **`/policy_switch`（本轮新增）**：`std_msgs/String`（payload = policy 目录名）直接加载 TorchScript，
  供 Part2 自动路线在 leg 内切换深度专家/下降专家；不复用手柄组合键（~0.7 s 太慢）。
- **构建**：`rl_sar` 的 C++ 标准 17→20（PyPI torch 2.14 的 ATen 头文件硬性要求），
  `find_package(Torch)` 改为从 demo 环境的 Python 解释器定位 PyPI torch 的 CMake 包。

## 3. BL-6：blackW 轮碰撞 geom 命名

官方 `robots/blackW/blackW_description.xml` 的四个轮子碰撞几何原本**没有 name**，判分器无法从
接触信息里区分四个轮子（`ground_feet` / 2 足触地规则无法判）。本仓为四个轮碰撞 geom 增加 name：

```xml
FL_wheel_collision   FR_wheel_collision   RL_wheel_collision   RR_wheel_collision
```

**official blackW model unchanged physically; only names added for judge contact classification**

- 改动仅四个 `<geom>` 元素的 `name` 属性；`size` / `pos` / `quat` / `contype` / `conaffinity` /
  `friction` / `mass` / `priority` / `group` / `rgba` 等其余属性一字未改（文本 diff 只有这 4 行）。
- 机器验证：MuJoCo 3.11 编译前后对比 481 个 `mjModel` 数值数组，物理相关数组全部逐位相同
  （唯一差异是 `name_geomadr` / `names_map` 等名称索引表，因新增 name 而必然变化）；
  四个新 name 均可通过 `mj_id2name` 取到，且分别挂在 `FL_foot` / `FR_foot` / `RL_foot` / `RR_foot`。
- 详见 `MIGRATION_NOTES.md` 的 BL-6 条目。

## 4. 权重与资产入库策略

- **策略权重不入库**：`policy.pt` / `best.pt`（以及 onnx/trt/engine）被根 `.gitignore` 排除，
  需按 `config.yaml` + `manifest.json` 在本地补齐。选中保留的配置：
  - blackW：`flat`、`stair`、`obstacle`、`row2p_p40w40_v1`、`st30_ppo_st30_s100_600`、
    `s1d_trench_reward_v75_m100`、`vt_band_v1`、`vt_pose_v1`、`vt_edge_v1`
    （+ `behaviors/*.yaml` 固定动作）；`vt_band_v1` 为 2026-11-04 新增的 Part2 楼梯 30cm 专家
    （来源 `~/RoboCon/deployment/ws/src/rl_sar/policy/blackW/vt_band_v1`）；
  - black：`himloco`。
- **机器人 mesh 入库**：`robots/blackW/assets/`（18 个 STL，约 40 MB）是唯一一份；场景通过
  `scenes/…/assets -> ../../../robots/blackW/assets` 软链接引用（MuJoCo 的 `compiler meshdir`
  是相对**主场景文件**目录解析的），`scenes/` 下不再复制 mesh。
- **主办方标签纹理入库**：`scenes/part2/full_field/labels/`（85 张 PNG，约 2.2 MB），只服务
  整场人工查看场景 `full_field:=true`；单关判分场景不需要它们。
