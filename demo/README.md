# demo — ROS 2 + MuJoCo 仿真演示运行时

`rl_sar` 派生的机器人运行时：`sim_node`（quadruped_ros2）用 **MuJoCo** 跑物理并发布状态，
`rl_sim`（rl_sar）加载 **TorchScript 策略**并输出 MIT 指令。同一套代码支持 Part1 障碍赛和
Part2 越障能力测试，可无窗口全速回归，也可开窗口 + 手柄人工驾驶。

- 机器人：**blackW**（16 DoF，轮腿式，Part1/Part2 主用）；black（12 DoF，历史对照）。
- 物理：MuJoCo 3.11，2 ms 步长；控制 200 Hz、推理 50 Hz（`policy/<robot>/base.yaml`）。
- 策略权重**不入库**：`policy.pt` / `best.pt` 本地按需补齐（见下）。

## 目录

```text
demo/
├── rl_sar/              策略推理节点、FSM、输入映射、安全限幅、策略配置（policy/<robot>/<name>/）
├── quadruped_ros2/      MuJoCo sim_node + viewer + 手柄 joy_node + MitCommand/MotorState 消息
├── robots/              机器人唯一资产库：描述 MJCF + robot.yaml + mesh（blackW/assets）、Part1 运行时场景
├── scenes/              官方 Part1/Part2 场景、Part2 单关生成场景、整场查看场景（不放机器人资产）
├── tests/               契约/出生点检查、trace 录制、无头策略冒烟（见 tests/README.md）
├── docs/                joystick.md（操作手键表/策略名册）、part1-auto-route-operator.md（自动路线）
├── README.md  UPSTREAM.md
```

## 构建

环境来自仓库根 `pixi.toml` 的 `demo` 环境（ROS 2 Humble + MuJoCo + torch 2.14.1+cpu，PyPI CPU 轮子；与 eval 同版本，libtorch 由策略侧 C++ 直接链接）：

```bash
# 仓库根目录
pixi install -e demo
pixi run -e demo demo-build        # tools/build.sh: colcon build demo/rl_sar demo/quadruped_ros2
source demo/install/setup.bash
```

即 `colcon build --paths demo/rl_sar demo/quadruped_ros2 --build-base demo/build --install-base demo/install`；
产物在 `demo/build`、`demo/install`、`demo/log`（均不入库）。

## 跑一个演示

```bash
# Part1：官方 Part1 场景，开窗口，手柄可用
ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=1 \
    policy_name:=row2p_p40w40_v1 joy:=true viewer:=true realtime:=true

# Part2：单关场景（关卡 launch 时选），默认 level=high_platform_10cm
ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=2 \
    level:=high_platform_40cm policy_name:=row2p_p40w40_v1 viewer:=true realtime:=true

# Part2 官方整场人工查看/驾驶（不用于判分）
ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=2 \
    full_field:=true policy_name:=row2p_p40w40_v1 joy:=true

# Part1 一键自动整场（演示/回归）
ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=1 \
    policy_name:=vt_pose_v1 route:=part1 viewer:=false realtime:=true

# Part2 一键自动整场（六类全开：平台/楼梯/斜坡/沟壑/梅花桩/高墙；约 20 min 1×）
PART1_ROUTE_JUDGE=0 \
PART1_RECORD_PATH=$PWD/artifacts/part2_route3/real_v23_states.jsonl \
PART1_RECORD_PLAN_PATH=$PWD/artifacts/part2_route3/real_v23_plan.jsonl \
ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=1 route:=part2 \
    policy_name:=vt_locked12 transfer_policy:=vt_locked12 \
    viewer:=false realtime:=true eval_state_every:=10 stop_on_fail:=true

# black（12 DoF，历史对照）
ros2 launch rl_sar rl_sim_all.launch.py robot:=black part:=1 policy_name:=himloco
```

关键 launch 参数：`robot:=blackW|black`、`part:=1|2`、`level:=<Part2 key>`、`policy_name:=...`、
`scene:=`（显式覆盖场景）、`viewer`（无 `DISPLAY` 时自动无窗口）、`realtime`（无窗口时是否按墙钟节流）、
`joy`、`follow_camera`、`status_period_s`、`publish_every` / `eval_state_every`、
`depth`（`auto|on|off`，默认 auto：场景有 `depth_cam` 就发 `/depth_cam`）、
`depth_every`（多少物理步渲一帧，默认 10 = 50 Hz）。

话题（本仓库新增）：

- `/depth_cam`（`quadruped_ros2/msg/DepthFrame`）：64x48 公制平面深度，50 Hz；先于同一步
  `/motor_state` 发布，同一 `d->time`。深度策略（plum）消费；盲目策略不受影响。
- `/policy_switch`（`std_msgs/msg/String`）：payload 为 policy 目录名，`rl_sim` 直接加载
  TorchScript（实测 ~3 ms），供自动路线切换深度专家/下降专家；手柄键位不变。

无头/自动化一律 `viewer:=false realtime:=false`（全速）；**1x 实时演示**用
`viewer:=false realtime:=true`。

## 判分 trace（2 ms 采样）

判分只看 trace：`eval_state_every` 默认 **1**（每个 2 ms 物理帧一帧 `/eval_state`）：

```bash
# 终端 A：无窗口实时跑一关，按物理帧采样
ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=2 \
    level:=trench_30cm policy_name:=s1d_trench_reward_v75_m100 \
    viewer:=false realtime:=true eval_state_every:=1 publish_every:=1

# 终端 B：录 trace（demo env）
python demo/tests/record_eval_trace.py --part 2 --robot blackW --level trench_30cm \
    --scene demo/scenes/part2/generated/part2_off_trench_30cm.xml --seconds 30 \
    --out artifacts/traces/trench_30cm.jsonl

# 终端 C：判分（eval env，无 ROS）
pixi run -e eval robocon-eval replay-part2 --trace artifacts/traces/trench_30cm.jsonl
```

- 交互/调试想省 IO 时传 `eval_state_every:=10`（50 Hz）；判分 trace 必须用 **1**。
- 关卡 manifest `scenes/part2/part2_official_levels.json` 里的 `seconds` 只是 **120 s 安全超时**
  （规则不计时，没有任何计分时间上限）。
- 一次只跑一个 launch 实例（`/joy`、`/eval_state` 会串台）；判分跑独占 CPU。

## 策略与权重（本地）

策略目录：`rl_sar/policy/<robot>/<policy_name>/`，每个目录含 `config.yaml` + `manifest.json` +
本地权重（`policy.pt`，不提交）。选中保留的 blackW 策略：

`flat`、`stair`、`obstacle`、`row2p_p40w40_v1`、`st30_ppo_st30_s100_600`、
`s1d_trench_reward_v75_m100`、`vt_band_v1`、`vt_pose_v1`、`vt_edge_v1`、
`plum_depth_final_v1`（深度相机梅花桩专家，含 `depth.json` 契约）；
black：`himloco`（`best.pt`）。固定动作姿态在 `policy/blackW/behaviors/`。
训练导出一个新策略 = 把 `config.yaml`/`manifest.json`/`policy.pt` 拷进对应目录（见 `train/README.md`）。

## 手柄 / 键盘

- 键表、可热切换策略名册、Part1 各任务 playbook：**[docs/joystick.md](docs/joystick.md)**。
- Part1 自动整场路线的流程/排错：**[docs/part1-auto-route-operator.md](docs/part1-auto-route-operator.md)**。
- 无手柄时键盘可用（`joy:=false`）：`0` 站立、`9` 趴下、`1` 进 RL、`W/A/S/D/Q/E` jog、`R` 复位。

## 自检

```bash
pixi run -e eval demo-tests     # 契约/出生点 + Part1/Part2 场景 MuJoCo 加载（tools/demo_tests.sh）
pixi run -e eval python demo/tests/check_robot_manifests.py
pixi run -e eval python demo/tests/check_policy_contracts.py
pixi run -e eval python demo/tests/check_spawns.py
pixi run -e demo bash demo/tests/smoke_policy_load.sh    # 需已构建 + 本地权重
pixi run -e demo bash demo/tests/smoke_policy_switch.sh
pixi run -e eval python demo/tests/check_depth_inference.py \
    --policy demo/rl_sar/policy/blackW/plum_depth_final_v1  # 需先 RL_DEPTH_DEBUG=1 跑过一小段
```

深度相机离线自检（不需要 ROS）：`demo/install/quadruped_ros2/lib/quadruped_ros2/sim_node
--depth-self-test <scene.xml>` 会打印统计并把原始 float32 写到 `/tmp/depth_selftest.f32`。

## 场景

场景布局、生成来源与 `assets` 软链接约定见 **[scenes/README.md](scenes/README.md)**。
Part2 单关场景由 `terrain/` 生成，不要手改 `scenes/part2/generated/`。

## 来源与许可

派生自 [fan-ziqi/rl_sar](https://github.com/fan-ziqi/rl_sar)（Apache-2.0）；主办方提供的
blackW 模型/官方策略/地形归赛事材料。上游 commit、本仓新增内容、BL-6 命名与“官方模型物理零改动”
说明见 **[UPSTREAM.md](UPSTREAM.md)**。
