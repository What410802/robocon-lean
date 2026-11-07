# tests — demo 自检脚本

纯检查类脚本只依赖 Python + MuJoCo + pyyaml（`pixi run -e eval python …` 即可）；
ROS 相关脚本需要在 demo 环境里先构建并 `source demo/install/setup.bash`。

| 脚本 | 作用 | 运行 |
|---|---|---|
| `check_robot_manifests.py` | `robots/*/robot.yaml` 契约自检（长度/置换/场景路径） | `python demo/tests/check_robot_manifests.py` |
| `check_policy_contracts.py` | 每个策略 `config.yaml`/`manifest.json` 对照 `robot.yaml`；缺本地权重时 `[SKIP]` | `python demo/tests/check_policy_contracts.py` |
| `check_spawns.py` | Part1/Part2 出生点与场景文件存在性（含 20 关清单） | `python demo/tests/check_spawns.py` |
| `check_part2_route.py` | 当前 33 段路线的实际尺寸、低档位覆盖、实体选择与转场净空 | `pixi run -e eval python demo/tests/check_part2_route.py` |
| `check_part2_replay.py` | 精确障碍映射、向东跨墙支撑证据、低档位零分、缺段拒绝与每类最高分汇总 | `pixi run -e eval python demo/tests/check_part2_replay.py` |
| `check_route_controller.py` | Part2 转场控制器的横漂、拐角、越过路点和失败终止回归 | `pixi run -e demo python demo/tests/check_route_controller.py` |
| `check_part1_handover.py` | 实际 Part1 碎石交接运动：超越目标倒退、提前交接前进、保持朝向、倒地停止，并以旧 goto 大角度回头作负控制 | `pixi run -e demo python demo/tests/check_part1_handover.py` |
| `check_part2_controller.py` | 同步假模型执行当前 33 段控制循环，检查对线、方向、错误障碍和深度策略失败卸载 | `pixi run -e demo python demo/tests/check_part2_controller.py` |
| `check_live_demo.py` | 两条演示路线默认策略、显式覆盖与 route/sim/RL 退出状态（不启动机器人仿真） | `pixi run -e demo python demo/tests/check_live_demo.py` |
| `check_route_timing.py` | Part2 在 2 ms / 20 ms 反馈下相同的 50 Hz 控制与保持时长、事件唤醒竞争、失去反馈超时 | `pixi run -e demo python demo/tests/check_route_timing.py` |
| `wheel_lock_selftest`（C++） | 真实 SDK/FSM：12DoF 切入只捕获一次轮角、120 周期 q 目标固定/dq=0、限力矩只降增益、16DoF 轮速及 Part1 固定动作保留、同策略重入重新捕获；本地权重存在时实际加载模型 | 构建后 `pixi run -e demo ctest --test-dir demo/build/rl_sar -R wheel_lock_selftest --output-on-failure` |
| `record_eval_trace.py` | 录 `/eval_state` 为判分 trace JSONL（判分跑 `eval_state_every:=1` = 2 ms 帧） | 见下 |
| `record_motor_state.py` | 录 `/motor_state` 为 CSV（回归对比） | ROS 运行时 |
| `record_state_command.py` | 录 `/motor_state` + 最近 `/mit_command`（状态/指令归因） | ROS 运行时 |
| `compare_motor_traces.py` | 逐 sim_time 对比两份 motor_state CSV | `python compare_motor_traces.py a.csv b.csv` |
| `smoke_policy_load.sh` | 无头仿真：A 站立 → RB+↑ 进 RL，断言策略加载成功 | 需构建 + 本地权重 |
| `smoke_policy_switch.sh` | 无头仿真：注入 `/joy` 热切换 row2p / st30 / s1d / vt_band（Part1 含固定动作） | 需构建 + 本地权重 |
| `run_joy_trial.sh` | **部署栈单关试验**：真实 launch 里 FSM A→EnterRL，按固定置杆或训练侧自适应命令跑一关，录判分 trace（配合 `robocon-eval replay-part2`）；`ADAPTIVE=0/1`、`VX=`/`AXIS_VX=`、`LEVEL=`、`POLICY=`、`OUT=` | 需构建 + 本地权重；判分前先 build 后 source（见文件头示例） |

判分 trace 录制示例（终端 A 起 launch，终端 B 录）：

```bash
ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=2 level:=trench_30cm \
    policy_name:=s1d_trench_reward_v75_m100 viewer:=false realtime:=true \
    eval_state_every:=1 publish_every:=1
python demo/tests/record_eval_trace.py --part 2 --robot blackW --level trench_30cm \
    --scene demo/scenes/part2/generated/part2_off_trench_30cm.xml --seconds 30 \
    --out artifacts/traces/trench_30cm.jsonl
```

`pixi run -e eval demo-tests`（`tools/demo_tests.sh`）会依次跑契约、出生点、Part2 路线检查 + Part1/Part2 场景的
MuJoCo 加载冒烟。ROS 冒烟脚本在未构建或没有本地 `policy.pt` 时不要运行（会起不来/加载失败）。
