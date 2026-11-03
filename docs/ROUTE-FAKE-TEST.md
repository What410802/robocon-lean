# Part1 路线：理想速度假模型与锁轮衔接模型实测（2026-11-03）

## 1. 目的与结论

本轮把「路线逻辑」和「衔接模型能力」分开测：

1. **理想速度假模型**：一个按指令精确积分体速度的 kinematic 假 sim，跑现有
   `part1_auto_route.py` 全场路线，用来暴露路线本身的排序/超时/几何问题。
2. **当前锁轮衔接模型**：导出 S2 checkpoint（`vt_locked_s2`）到部署目录，跑同一条
   路线，观察实际 handover 表现。
3. 两个过程都生成视频。

结论：

- **路线本身（理想模型）可以全程 PASS**，包括用户提到的东北角 bridge A 东台 →
  spin → bridge B → 长坡；东北角在理想模型下 `target_err=0.118 m`、spin drift 0。
- 假模型暴露并修复了一个**路线问题**：slalom 的 7 个 keyframe `goto` 超时 9 s 不够
  （先大约 90–130° 转向、再平移 >1 m），理想模型也只到 5/7；改为 15 s 后 7/7。
- **当前 S2 衔接模型在位置环里能精密移动**：position harness
  `forward_ok=True, turns_ok=True, turn_drift_max=0.059 m, stop_and_go_ok=True,
  accepted=True`；但 `l_shape` 仍失败（横向路径）。
- **整场路线仍不及预期**：S2 从 terrain expert 切换过来后，在 high wall 终点附近
  5 s 内从 z≈0.52、tilt≈8° 演化为 z≈0.18、tilt≈90°（倒下）。这是
  **expert→connector handover 的初态鲁棒性**问题，不是矩形框或位置环精度问题。
  已试过 stop-before-switch、post-switch settle、FSM get-up 三种 handover 修复；
  前两者仍倒，get-up 会卡在 GetUp 状态不接受速度命令，已回退。

## 2. 理想速度假模型

文件：`tools/route/fake_kinematic_route_sim.py`（demo 环境运行）。

- 订阅 route 的 `/joy`，按 `axes[1]*1.5 / axes[0]*1.5 / axes[3]*1.5` 解码 vx/vy/wz，
  直接用体速度积分 base；没有动力学、没有打滑。
- 用真实 `scene_part1.xml` 的 MuJoCo ray cast 取四个轮心下方地面高度（排除机器人
  geom），生成 base z、roll/pitch、四轮 `feet` 和 `contacts`；`/eval_state` JSON 与
  sim_node 同字段。
- route 用 `PART1_ROUTE_STEPLOCK=1`（一帧一 tick），`PART1_ROUTE_JUDGE=0`（只测路线
  预检，不用 live judge；假模型没有真实的支撑/接触证据）。

运行：

```bash
source demo/install/setup.bash
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=88
PART1_ROUTE_STEPLOCK=1 PART1_ROUTE_JUDGE=0 PART1_TRANSFER_POLICY=vt_pose_v1 \
PART1_RECORD_PATH=$PWD/artifacts/route/fake_route_states2.jsonl \
PART1_RECORD_PLAN_PATH=$PWD/artifacts/route/fake_route_plan2.jsonl \
bash -c 'timeout 400 .pixi/envs/demo/bin/python tools/route/fake_kinematic_route_sim.py --rate 50 &
         sleep 1
         timeout 360 .pixi/envs/demo/bin/python demo/rl_sar/scripts/part1_auto_route.py'
```

### 2.1 路线结果（修复后）

| 段 | 结果 |
|---|---|
| high_wall / t_stairs / bridge_blocks | PASS |
| bridge A east handover | `moved=0.610 m target_err=0.100 m`；PASS |
| bridge A east centre spin | PASS，drift 0 |
| bridge B entry | **x=5.66, y=-0.90, yaw=-89.9°**（修正前 x=5.84/5.79，偏东） |
| bridge_bands / long slope / height_bar / gravel | PASS |
| slalom | 修复前 5/7（两个 keyframe 超时），**修复后 7/7 PASS** |
| return_start | PASS；`wheel recoveries: 0` |

修复：

1. slalom `goto` 超时 9 s→15 s（转向 + 平移在理想模型下也需要 >9 s）。
2. slalom 关键帧从距杆 0.35 m 拉到 **0.60–0.65 m**（杆心半径 0.35 m），
   拐点保留在 corner checkpoint 0.30 m 内；7/7 PASS。
3. bridge B 前横向对带：原 `adaptive_lane_cmd(axis="y", lane_yaw=-pi/2)` 的横向
   符号是反的，理想模型被推到 x=8.71；改为专用 `vy=-0.8*(x-5.65)` +
   yaw 保持，实测进入固定动作前 **x=5.66 / yaw=-89.9°**。
4. handover 日志由 `drift=pre→post` 改为 `moved=... target_err=...`，避免把
   “从 x=5.25 继续前进到 5.77”误读成 drift。

视频：`artifacts/route/fake_route_v2.mp4`（220.7 s，全场；含 bridge B 对带和
加宽后的 slalom）。

### 2.2 高墙后切换模型时的关节状态（S2 路线失败的直接证据）

数据：`artifacts/route/highwall_switch_joints.csv`（t=28–34 s，16 关节 q/dq/tau），
图：`artifacts/route/highwall_wheel_q.png`。

四轮关节（q 是关节角 [rad]，dq 是角速度 [rad/s]）：

| t [s] | base z | q wheels [FL,FR,RR,RL] | dq wheels | tau wheels |
|---|---|---|---|---|
| 29.0 | 0.557 | 15.97, -10.36, -11.99, 15.89 | -5.47, 5.23, 5.47, -5.54 | -0.9, 2.4, 0.9, -0.5 |
| 30.1 | 0.493 | 10.25, -4.58, -6.41, 10.15 | -5.78, 5.39, 5.40, -5.43 | 1.0, 1.7, 1.4, -1.2 |
| 30.5 | 0.328 | 8.28, -2.50, -4.35, 7.98 | -5.11, 4.98, 5.67, -5.39 | -3.1, 1.5, 1.2, -1.4 |
| 31.5 | 0.298 | 3.15, -0.01, -0.04, 2.63 | -5.21, -0.05, 0.87, -5.42 | -2.4, 0.7, 0.8, -1.2 |
| 32.0 | 0.300 | 0.75, -0.01, 0.01, 0.05 | -4.46, 0.07, -0.01, -1.56 | -7.3, 0.6, -0.9, 0.4 |
| 33.0 | 0.297 | 0.04, 0.00, -0.01, 0.00 | -0.0, 0.0, 0.0, -0.0 | -3.1, -0.2, 0.4, -0.0 |

解释：`q` 是**关节位置角**，`dq` 是角速度。滚动专家（row2p）期间四轮累计转到了
约 ±10–25 rad；切到锁轮策略后，C++ `lock_wheels` 把轮目标设成默认角 **0**，位置 PD
（rl_kp=80 / rl_kd=3，轮力矩上限 20 N·m）以约 **5.4 rad/s 的固定反向速度**把 q 拉回
0，同时 base z 从 0.557 掉到 0.30、机器人侧倒。修复方向是**切换时把轮目标设为当前
q（或先低速回零）**，而不是固定 0；本轮试过 hold-current 的 C++ 补丁，但该版本在
route 启动后策略不前进，已回退，需要单独调通后再接路线。

## 3. 当前锁轮模型整场路线

导出：

```bash
PYTHONPATH=train/src .pixi/envs/train/bin/python \
  train/scripts/export/export_history_policy.py \
  --task blackw-vt-locked-s2 \
  --checkpoint logs/rsl_rl/black_velocity/2026-11-03_00-12-37_locked3_s2b/model_898.pt \
  --out demo/rl_sar/policy/blackW/vt_locked_s2/policy.pt --robot blackW
```

部署配置从 `nwwolf_locked16/config.yaml` 复制，只改 top key 和模型名；保留
`lock_wheels: true` + wheel rl_kp=80/rl_kd=3，leg rl_kp=50/rl_kd=1.2。

PolicyRunner sim2sim（导出后）：

| cmd | 实测 |
|---|---|
| vx 0 / 0.1 / 0.2 / 0.3 | 0.022 / 0.100 / 0.191 / 0.289 |
| vy 0.2 | 0.119 |
| wz ±0.3 | +0.160 / −0.145 |

位置环 `position_harness.py --policies vt_locked_s1,vt_locked_s2`：

```
vt_locked_s2: forward_ok=True turns_ok=True turn_drift_max=0.0594
              l_shape_ok=False stop_and_go_ok=True accepted=True
vt_locked_s1: forward_ok=True turns_ok=False turn_drift_max=0.2791 accepted=False
```

整场路线（`stop_on_fail:=false` 以便看 handover 之后的行为）：高墙段本身通过，
切回锁定策略后 5 s 内倒下；`bridge A east` 之后无法继续。视频
`artifacts/route/locked_s2_route.mp4`（48.2 s，记录到高墙/stairs 附近失败）。

### 3.1 已试 handover 修复

| 修复 | 结果 |
|---|---|
| stop before switching + 0.6 s | 仍在 5 s 内倒 |
| 再增加 post-switch 1.2 s zero settle | 仍在 5 s 内倒 |
| FSM get-up（A 键）再 switch | GetUp→RL 未完成，卡在 GetUp，速度命令无效；已回退 |
| S2 宽 reset 鲁棒性微调（joint ±0.25, vel ±2） | position harness forward_ok=False，负结果；未用于交付 |

## 4. 矩形框是否罚住 vy/wz？

不是。S2 实测：

- `Episode_Reward/support_extent ≈ -0.014 /s`（训练早期曾 -1.5 /s，已收敛）；
- 实际触地轮相对躯干框的最大越界 ≤ **0.007 m**；
- vy/wz 指令下框的增长：`vy=0.2` 约 +0.016 m，`wz=0.3` 约 +0.011 m，仍足够容纳
  正常步态接触脚。

当前瓶颈是策略授权/训练分布（教师模型本身没有纯 yaw/vy 能力）和 handover 初态
鲁棒性，而不是支撑矩形过紧。下一步若按用户意见去掉 `foot_slip` 惩罚，需要用
小权重 `foot_lift` + `feet_air_time`/`support_extent` 防止纯滑行，并在 S2/S3 重训后
复测 F4/F5、position harness 与整场路线。

## 5. 视频

| 视频 | 说明 |
|---|---|
| [fake_route_v2.mp4](artifacts/route/fake_route_v2.mp4) | 理想速度假模型，全场路线 PASS（bridge B x=5.66、slalom 7/7） |
| [highwall_wheel_q.png](artifacts/route/highwall_wheel_q.png) | 高墙 handover 四轮 q/dq 曲线 |
| [locked_s2_route.mp4](artifacts/route/locked_s2_route.mp4) | 当前 S2 锁轮模型；显示 expert→connector handover 后倒下，路线未完成 |
