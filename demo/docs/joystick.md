# 手柄 / 键盘操作与策略切换（blackW）

本文件是**操作手入口**：手柄按键表、可运行时切换的策略名册、Part1 各关卡按键 playbook。
键映射本身是数据表 `rl_sar/include/input_map.hpp` 的 `kBindings`，本文件 §2 与它同源；
改键 = 改 `kBindings` 一行 + 同步本文件 + 跑 `input_map_selftest`（见 §8）。

## 0. 启动前

```bash
# 一次性构建（仓库根目录）
pixi run -e demo demo-build
source demo/install/setup.bash
```

## 1. 设备与消息约定

`/joy` 由 `quadruped_ros2/scripts/joy_node.py` 从 `/dev/input/event*`（python-evdev，非 SDL）读取，
输出**规范化**布局（与主办方 `gamepads.yaml` 的 xbox profile 一致，正负号也已归一化）：

```text
axes[0..7]     = 左摇杆 X、左摇杆 Y、LT、右摇杆 X、右摇杆 Y、RT、十字键 X、十字键 Y
buttons[0..11] = A、B、X、Y、LB、RB、Back、Start、Guide、LS、RS、恒 0
正负号：摇杆 上/左 = +1；十字键 上/右 = +1
```

`rl_sim` 侧（`rl_sar/src/rl_sim.cpp::JoyCallback`）：`x = axes[1]*joy_command_scale`（前推为正）、
`y = axes[0]*joy_command_scale`、`yaw = axes[3]*joy_command_scale`；`joy_command_scale` 是
`rl_sim` 节点参数，默认 **1.5**。所有切换/动作键都是**边沿**：组合键全部按下的那一帧触发一次，
按住不重复（`rl_sar/include/input_map.hpp` 的 `KeyBits::pressed`）。

启动时 `joy:=true` 才会起 `joy_node`（默认 `false`）；真机手柄：

```bash
ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=1 joy:=true device:=/dev/input/event7
# device:= 留空则自动找（名字含 Xbox / gamepad 的设备）
```

纯键盘操作请用 `joy:=false`：`JoyCallback` 每帧会用摇杆值重写 `x/y/yaw`，键盘 jog 会被清零。

## 2. 完整键表

### 2.1 动作 / 模式（单键）

| 手柄 | 键盘 | Intent | 作用 |
|---|---|---|---|
| A | `0` | StandUp | 站立（Passive → GetUp） |
| B | `9` | LieDown | 趴下（GetDown） |
| RB + 十字上 | `1` | EnterRL | 进入 RL 行走（GetUp → RL_Locomotion） |
| LB + X | `P` | Damp | 阻尼（Passive） |
| RB + Y | `R` | ResetSim | 重置仿真（回到初始位） |
| X | `N` | ToggleNavigation | 导航模式开/关 |

### 2.2 策略切换（RL 行走中热切换）

| 手柄 | 键盘 | Intent | 策略目录 | 用途 |
|---|---|---|---|---|
| Start + 十字左 | `2` | PolicyFlat | `flat` | 平地 / 绕杆（主办方策略） |
| Start + 十字上 | `3` | PolicyStair | `stair` | 楼梯通用（主办方策略） |
| Start + 十字右 | `4` | PolicyObstacle | `obstacle` | 砂砾坑 / 障碍（主办方策略） |
| **Start + 十字下** | **`8`** | **PolicyMain** | **`row2p_p40w40_v1`** | **Part2 主策略；Part1 高墙 / 积木桥 / 斜坡** |
| **RB + 十字下** | **`F`** | **PolicyTStairs** | **`st30_ppo_st30_s100_600`** | **Part1 T 型楼梯专用** |
| **RB + 十字右** | **`T`** | **PolicyTrench30** | **`s1d_trench_reward_v75_m100`** | **Part2 沟壑 10/20/30cm 专用（LY≈0.8–1.0 m/s）** |
| LB + 十字左 | — | PolicyPose | `vt_pose_v1` | 低速闭环对线 / 落位（自动路线用；手动排障） |
| Back + 十字上 | — | PolicyEdge | `vt_edge_v1` | 轮子掉边恢复（自动路线用；手动排障） |

### 2.3 Part1 固定动作（仅 `part:=1`，进入独立 FSM 状态）

| 手柄 | 键盘 | Intent | 行为 yaml | 用途 |
|---|---|---|---|---|
| Back + 十字下 | `5` | LowBarDrive | `policy/blackW/behaviors/low_bar_drive.yaml` | Part1 限高杆（低姿通过） |
| Back + 十字右 | `6` | BridgeDrive | `policy/blackW/behaviors/bridge_drive.yaml` | Part1 窄带木桥（宽轮距姿态） |
| Back + 十字左 | `7` | CarDrive | `policy/blackW/behaviors/car_drive.yaml` | 备用低姿（未在正式任务验证） |

进入固定动作后：先 150 个控制周期（0.75 s）把腿插值到 yaml 目标姿态，之后左摇杆 `LY` 控制前后、
`RX` 控制转向（轮速 = `wheel_velocity_scale*x ∓ yaw_to_wheel_velocity*yaw`）。`RB + 十字上` 回 RL；
`A`/`B`/`LB+X` 分别回 GetUp / GetDown / Passive。**Part2 规则只允许 RL，`part:=2` 时这三个键无效**
（`RLFSMStateRL_Locomotion::CheckChange` 里 `rl.part == 1` 才切）。

### 2.4 键盘 jog（仅键盘）

`W/S/A/D` = 前后左右 jog，`Q/E` = 左右转 jog，`Space` = 指令清零。每按一次走一步，用于无摇杆时挪位。

## 3. 策略名册（本地权重按需补齐）

所有策略同契约：16 DoF / 57 obs / history `[0..5]` / 200 Hz 控制 + 50 Hz 推理 / 轮子 action
signed（FR、RR 为 `-10`），与 `robots/blackW/robot.yaml` 一致；运行时 `InitRL` 会做契约校验。
权重（`policy.pt`）不入库，按 `demo/rl_sar/policy/<name>/` 的 `config.yaml` + `manifest.json` 本地补齐。

| 策略目录 | 用途 / 覆盖任务 |
|---|---|
| `row2p_p40w40_v1` | Part2 主策略（16/20）；Part1 high_wall / bridge_blocks / slope |
| `st30_ppo_st30_s100_600` | Part1 T 型楼梯专用（成功；判据见 `eval/SPEC.md`） |
| `s1d_trench_reward_v75_m100` | Part2 trench_10/20/30cm（需 `LY≈0.8–1.0 m/s`；0.5 m/s 会失败） |
| `obstacle` | Part1 gravel_pit 满分；通用障碍（主办方策略） |
| `flat` | Part1 slalom（绕杆，配合速度指令）+ 平地（主办方策略） |
| `stair` | 主办方楼梯通用策略（历史保留） |
| `vt_pose_v1` | 低速三轴命令跟随/对线（自动路线衔接，无键盘键，手柄 LB+←） |
| `vt_track_v1` | 新速度跟随衔接模型（Part1 位置环专用；无热切换键，launch/自动路线用；训练/验收见 `docs/VT-RECIPE.md`） |
| `vt_edge_v1` | 轮子掉边恢复专家（自动路线排障，手柄 Back+↑） |

Part2 覆盖（2026-10-31 评审）：`row2p_p40w40_v1` 单独 16/20，加上 `s1d_trench_reward_v75_m100`
后为 **17/20**；仍无前沿模型覆盖的关卡：`stairs_30cm`、`slope_45deg`、`plum_1`。
`vt_pose_v1` / `vt_edge_v1` / `vt_track_v1` 是操作辅助模型（**不是障碍策略**），只在 launch 或自动路线里使用。
`vt_track_v1` 用 `PART1_TRANSFER_POLICY=vt_track_v1` 让自动路线使用；首次使用先用
`tools/tracking/position_harness.py` 复核落位增益。

## 4. 切换语义与限制

1. **热切换**：在 `RL_Locomotion` 状态按组合键 → 立即 `SwitchPolicy`：重读
   `policy/blackW/<name>/config.yaml`、`torch::jit::load` 加载 `policy.pt`、清空 `x/y/yaw`、
   输出队列与 `episode_length`；下一个推理周期（≤20 ms 仿真时间）生效。
   终端打印 `[policy] switched to <name>`。
2. **非 RL 状态按切换键**（Passive / GetUp / GetDown / 固定动作）：只记 pending，
   下次进入 RL 时加载，打印 `[policy] next RL entry will load <name>`。
   注意：进入 RL 的按键是 `RB + 十字上`；固定动作里先回 RL 再切策略。
3. **同策略重复按无操作**（`SwitchPolicy` 先比较当前名）。
4. **不换场景**：Part2 的 `level:=`（地形）只能在 launch 时选；手柄只切模型，不切关卡/出生点。
5. **失败处理**：`[policy] switch to <name> failed: ...` = 目录/config/权重契约问题，
   用 `pixi run -e demo python demo/tests/check_policy_contracts.py` 检查；失败后
   `rl_init_done=false`，机器人保持最后指令，建议 `R` 重置或重新 launch。
6. 策略名必须是 `policy/blackW/` 下的目录名（含 `config.yaml` + `policy.pt` + `manifest.json`）。

## 5. Part1 任务 → 按键 playbook

前置：`ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=1 policy_name:=row2p_p40w40_v1`
（或先 `flat`，进入 RL 后按 Start+下）。站起 `A` → 进入 RL `RB+十字上` → 按任务选键：

| Part1 任务 | 按键 | 说明与操作要点 |
|---|---|---|
| height_bar 限高杆 | Back+下（low_bar_drive） | 等 0.75 s 插值成低姿后再前进；**越杆时 base 中心 z 需低于杆底 0.325 m**（实测低姿 base z≈0.28 m），杆不能碰落；直行慢速通过。 |
| high_wall 高墙 | Start+下（row2p） | 正对墙体直行翻越；速度约 0.5 m/s。 |
| bridge_blocks 积木桥 | Start+下（row2p） | 沿积木方向直行；速度稳定，别在块上转向。 |
| slope 斜坡 | Start+下（row2p） | 正对坡面直行，保持速度。 |
| gravel_pit 砂砾坑 | Start+右（obstacle） | 上坑坡放慢，过坡顶后**加油门冲下**（判据用 base 中心记录北侧离开边沿，需 ≤1 轮触地）；落地后收油，别原地掉头。 |
| t_stairs T 型楼梯 | RB+下（st30） | 直上直下；判据见 `eval/SPEC.md`（本仓已修正下行接触次序问题）。 |
| slalom 绕杆 | Start+左（flat） | 低速（0.2–0.3 m/s）沿官方 S 弯；过杆方向左右交替，过 3 根杆后过 corner 检查点，最后进 B 区。 |
| bridge_bands 窄带木桥（比赛非必选） | Back+右（bridge_drive） | 宽轮距姿态；先把车横向对到带子上（仿真对线窗口约 ±0.08 m，中心 x≈5.60–5.76 m），再以约 0.5 m/s 直行；对线偏了会掉缝。 |

## 5.1 自动整场路线（Part1，演示/回归用）

```bash
source demo/install/setup.bash
# 可选：PART1_TRANSFER_POLICY=vt_track_v1 使用新的速度跟随衔接模型（默认 vt_pose_v1）
ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=1 \
    policy_name:=vt_pose_v1 route:=part1 viewer:=false realtime:=true
```

`route:=part1` 会同时启动 `rl_sar/scripts/part1_auto_route.py`（虚拟手柄，发布 `/joy`）并自动完成
A 站立 → 进入 RL → 逐段走完备赛路线。**`realtime:=true` 必须显式传**（默认 `false` 是全速回归，
会把墙钟节拍的路线控制器甩开）。`eval_state_every` 默认 1（2 ms 一帧，判分采样率）；route 在
`eval_state_every:=0` 时会自动抬到 10。详细流程、按键与几何限制见
[`part1-auto-route-operator.md`](part1-auto-route-operator.md)。

## 6. Part2 使用示例

Part2 每个关卡是独立场景，**关卡在 launch 时选定**；模型推荐 = `row2p_p40w40_v1`：

```bash
source demo/install/setup.bash
ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=2 \
    level:=high_platform_10cm policy_name:=row2p_p40w40_v1 joy:=true
# 也可以先 policy_name:=flat，进入 RL 后按 Start+十字下 切到 row2p
```

- 关卡清单：`scenes/part2/part2_official_levels.json`（20 关，场景在 `scenes/part2/generated/`）。
- 推荐巡航速度：摇杆 `LY` 对应 x≈0.5 m/s；platform30 / slope35 的成功域较窄，别冲太快。
- **trench_30cm 例外**：切到 `s1d_trench_reward_v75_m100`（`RB+十字右` / `T`）后，`LY` 要给
  **0.8–1.0 m/s**（0.5 m/s 会失败，0.75–1.0 稳定、仿真 gate 100%）；trench_10/20cm 也可以用它。
- Part2 为 RL-only：固定动作（Back+…）在 `part:=2` 下不生效。
- 官方整场地图人工查看：`full_field:=true`（场景 `scenes/part2/full_field/`，只用于人工检查/驾驶，
  **判分仍走单关场景**）。
- 仍无前沿模型的关卡：`stairs_30cm`、`slope_45deg`、`plum_1`。

## 7. 排错

| 现象 | 原因 / 处理 |
|---|---|
| 按组合键没反应 | ①确认 /joy 有数据（`ros2 topic hz /joy`）；②组合键要**同时**按住 modifier + 十字键（边沿需要一帧同时为真）；③策略切换只在 RL 状态立即生效，不在 RL 时看是否打印 `next RL entry will load`。 |
| 切换后机器人不动 | 正常：切换会清空 `x/y/yaw`，重新推杆即可。 |
| `[policy] switch to ... failed` | 策略目录/config/权重契约错误；跑 `demo/tests/check_policy_contracts.py`。 |
| 手柄按了没反应 / 设备找不到 | `joy:=true`；`device:=` 指定 `/dev/input/eventX`；确认当前用户在 `input` 组。 |
| 键盘 jog 无效 | 用了 `joy:=true`（每帧摇杆值会覆盖 jog）；键盘操作请 `joy:=false`。 |
| Part1 按 Back+… 无反应 | `part:=1` 才有效；Part2 规则 RL-only。 |
| Part2 换 level 没变化 | level 只能 launch 参数切换，重新 launch。 |
| MuJoCo 场景加载失败 | 场景 include 的机器人/贴图路径不对；跑 `pixi run -e eval demo-tests` 定位。 |

## 8. 自检

```bash
# 纯按键解析（无 ROS 图、无仿真、无手柄）
ros2 run rl_sar input_map_selftest

# 无头仿真：进入 RL 后注入 /joy，断言两条切换日志（Part2）
pixi run -e demo bash demo/tests/smoke_policy_switch.sh
PART=1 pixi run -e demo bash demo/tests/smoke_policy_switch.sh   # 含固定动作状态切换

# 全部策略与 robot.yaml 的契约
pixi run -e eval python demo/tests/check_policy_contracts.py
```

## 9. 变更记录

- **2026-10-31**：接入 `PolicyMain` = `row2p_p40w40_v1`（Start+十字下 / 键盘 `8`）与
  `PolicyTStairs` = `st30_ppo_st30_s100_600`（RB+十字下 / 键盘 `F`）。
- **2026-10-31（当天第二批）**：接入 Part2 沟壑专项 `s1d_trench_reward_v75_m100`
  （RB+十字右 / 键盘 `T`），补上 trench_30cm 缺口（x 速度需 0.8–1.0 m/s）。
- **2026-11-02 迁移**：只保留 8 个 blackW 策略目录 + `black/himloco`；权重不入库；
  `vt_flat_*` 等研究轮次已删除，操作辅助模型为 `vt_pose_v1` / `vt_edge_v1`。
