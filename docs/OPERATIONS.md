# 操作手册（Part1 / Part2 仿真）

所有命令都在仓库根目录执行。判分跑要求 **独占 CPU**、**一次只跑一个 launch 实例**
（`/joy`、`/eval_state` 会串台）；并行 MuJoCo 探针会拖垮 50 Hz 控制环。

## 0. 一次性准备

```bash
pixi install -e demo          # ROS 2 Humble + MuJoCo 3.11 + CPU libtorch
pixi install -e eval          # 判分器
pixi run -e demo demo-build   # colcon 构建 demo/rl_sar + demo/quadruped_ros2
source demo/install/setup.bash
```

权重不入库。把训练导出的 `policy.pt` + `config.yaml` + `manifest.json` 放到
`demo/rl_sar/policy/<robot>/<name>/`；`demo/rl_sar/policy/blackW/` 已有 8 个目录的配置模板
（`row2p_p40w40_v1`、`vt_pose_v1`、`vt_edge_v1`、`st30_ppo_st30_s100_600`、
`s1d_trench_reward_v75_m100`、`flat`、`obstacle`、`stair`），本地权重按需补。

## 1. Part 1：手柄 / 键盘

```bash
ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=1 \
    policy_name:=vt_pose_v1 joy:=true viewer:=true realtime:=true
```

按键（边沿触发，完整表见 `demo/README.md`）：

| 手柄 | 键盘 | 作用 |
|---|---|---|
| A | `0` | 站立（Passive→GetUp） |
| B | `9` | 趴下 |
| RB + 十字上 | `1` | 进入 RL 行走 |
| LB + X | `P` | 阻尼 |
| RB + Y | `R` | 重置仿真 |
| Start + 十字下 | `8` | 切 `row2p_p40w40_v1`（Part2 主策略；Part1 高墙/积木桥/斜坡） |
| RB + 十字下 | `F` | 切 `st30_ppo_st30_s100_600`（T 台阶） |
| RB + 十字右 | `T` | 切 `s1d_trench_reward_v75_m100`（沟壑 10/20/30 cm） |
| Start + 十字左/上/右 | `2`/`3`/`4` | `flat` / `stair` / `obstacle`（主办方策略） |
| Back + 十字下/右/左 | `5`/`6`/`7` | Part1 固定动作：限高杆 / 窄带木桥 / 备用低姿 |

> Part2 规则只允许 RL：`part:=2` 时固定动作键无效（代码里 `rl.part == 1` 才切）。

### Part1 一键自动整场（演示/回归）

```bash
ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=1 \
    policy_name:=vt_pose_v1 route:=part1 viewer:=false realtime:=true
```

- `route:=part1` 起 `part1_auto_route` 虚拟手柄节点，自动完成站立→进 RL→逐段走完全场；
  **`realtime:=true` 必须显式传**（否则仿真全速跑，路线控制器会被甩开）。
- 路线依赖策略：`vt_pose_v1`、`vt_edge_v1`、`row2p_p40w40_v1`、`st30_ppo_st30_s100_600`、
  `obstacle` + `behaviors/{low_bar_drive,bridge_drive}.yaml`。
- 速度衔接模型推荐 `vt_track_v2`（四轮共面/跟随奖励；训练/验收见 `docs/training/VT-RECIPE.md` §4.2）：
  `ros2 launch ... transfer_policy:=vt_track_v2`（直接跑脚本时用环境变量
  `PART1_TRANSFER_POLICY=vt_track_v2`）；`vt_track_v1` 保留作对照。v2 平移偏保守（支撑矩形
  比 0.63），首次使用先跑 `tools/tracking/position_harness.py` 并在 controller 里把速度指令
  上调约 20%，或调整 `--max-vx`。
- 已知弱项：桥 A 西台 1.0×1.0 m 原地 90° 转向需要约 1.1×1.1 m 净空（几何限制，不是策略问题），
  S 弯/返程精度不足。MPC 接入方案见 `tools/mpc/README.md`。

## 2. Part 2：单关独立评测

Part2 场景由 `terrain/` 生成（单关、通道中线 y=0、出生点 x=0），关卡清单：
`demo/scenes/part2/part2_official_levels.json`（20 关：平台 4、楼梯 4、斜坡 4、高墙 4、沟壑 3、梅花桩 1）。

```bash
ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=2 \
    level:=high_platform_40cm policy_name:=row2p_p40w40_v1 viewer:=false realtime:=true
```

要点：

- 一关一个场景，`level`/`scene` 在 launch 时选定，手柄只切模型不切关卡。
- 官方地图整场人工检查：`full_field:=true`（人工驾驶用，**不用于判分**）。
- 关卡 `seconds` 只是 120 s 安全超时（规则不计时）。
- 连续走完六类的"一次启动"全场 demo 是非计分演示（规划中），计分仍按单关独立。

## 3. 判分：录制 2 ms trace → replay

判分跑必须让 `sim_node` 每 2 ms（== 每个物理步）发一次 `/eval_state`：

```bash
# 终端 A：无窗口、实时、判分采样
ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=2 \
    level:=trench_30cm policy_name:=s1d_trench_reward_v75_m100 \
    viewer:=false realtime:=true eval_state_every:=1 publish_every:=1

# 终端 B：录制（demo env）
python demo/tests/record_eval_trace.py --part 2 --robot blackW \
    --level trench_30cm --seconds 30 \
    --out artifacts/traces/trench_30cm.jsonl

# 终端 C：判分（eval env，无需 ROS）
pixi run -e eval robocon-eval replay-part2 --trace artifacts/traces/trench_30cm.jsonl
pixi run -e eval robocon-eval replay-part1 --trace artifacts/traces/part1_run.jsonl
```

- 正常演示/调试可把 `eval_state_every` 调大（10 → 50 Hz）省 IO；**判分 trace 用 1**。
- Part1 自动路线自身消费 `/eval_state` 做位姿反馈（50 Hz）；判分跑同时开 `eval_state_every:=1`
  时它仍只读最新帧，不影响路线控制。
- 判分器不 import ROS：`eval/` 只吃 trace/JSON+MuJoCo CPU。

## 4. 复盘视频

```bash
pixi run -e eval python tools/record_video.py \
    --trace artifacts/traces/plum_1.jsonl --out artifacts/videos/plum_1.mp4
```

`record_video.py` 用同一策略/同一命令重放并叠加判分状态；默认 640×360 / 50 fps（评审质量），
需要更高分辨率时按脚本参数调整。

### 4.1 固定路线录像（Part 1 自动路线）

自动路线节点支持把 `/eval_state` 原始帧与计划（leg/target）同时落盘：

```bash
source demo/install/setup.bash
PART1_RECORD_PATH=$PWD/artifacts/route/states.jsonl \
PART1_RECORD_PLAN_PATH=$PWD/artifacts/route/plan.jsonl \
ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=1 \
    route:=part1 policy_name:=vt_pose_v1 transfer_policy:=vt_track_v2 \
    viewer:=false realtime:=true eval_state_every:=10 stop_on_fail:=true
# 结束后（或 timeout 后）用 eval env 离线渲染，叠加路线/目标/四轮矩形：
pixi run -e eval python tools/video/replay_route_video.py \
    artifacts/route/states.jsonl artifacts/route/route.mp4 artifacts/route/plan.jsonl
```

`stop_on_fail:=true`（默认）在第一个路线预检 FAIL 处停止并关掉 launch；`false` 则继续（旧行为）。
四轮跟随/接触平面可单独量：`pixi run -e eval python tools/tracking/probe_wheel_follow.py
--policies vt_track_v2`。
路线 PASS/FAIL 是排序用 precheck，不是判分；判分用独立 `robocon-eval replay-part1`（差异与
对齐做法见 `demo/docs/part1-auto-route-operator.md` §2.1）。

`replay_route_video.py` 是运动学回放（`mj_forward`，不重新步进物理），忠实显示录制到的机器人
状态；视频上部标注当前 leg / target / CLI / 轮地接触，右上有路线俯视 inset。要实时看窗口就去掉
`viewer:=false`。

## 5. 自检与排错

| 现象 | 原因/处理 |
|---|---|
| 开局乱跑/漂出场地 | 忘了 `realtime:=true`（全速模式） |
| `no /eval_state` | launch 没带 `route:=part1` 或 `eval_state_every:=0` |
| `policy switch failed` | 策略目录缺 `config.yaml`/`policy.pt`，或 obs/action 契约不符 |
| MuJoCo 场景加载失败 | 先跑 `pixi run -e eval demo-tests`；检查 `demo/install/setup.bash` 是否 source |
| replay 分数为 0 | 检查 trace 是否为 2 ms 帧、`--part`/`--level` 是否与场景匹配 |
| 固定动作键无效 | `part:=2` 按规则禁用了固定动作；Part1 才可用 |

快速自检：

```bash
pixi run -e eval eval-pytest
pixi run -e eval demo-tests
pixi run -e eval python tools/check_env.py
```
