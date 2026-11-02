# tests — demo 自检脚本

纯检查类脚本只依赖 Python + MuJoCo + pyyaml（`pixi run -e eval python …` 即可）；
ROS 相关脚本需要在 demo 环境里先构建并 `source demo/install/setup.bash`。

| 脚本 | 作用 | 运行 |
|---|---|---|
| `check_robot_manifests.py` | `robots/*/robot.yaml` 契约自检（长度/置换/场景路径） | `python demo/tests/check_robot_manifests.py` |
| `check_policy_contracts.py` | 每个策略 `config.yaml`/`manifest.json` 对照 `robot.yaml`；缺本地权重时 `[SKIP]` | `python demo/tests/check_policy_contracts.py` |
| `check_spawns.py` | Part1/Part2 出生点与场景文件存在性（含 20 关清单） | `python demo/tests/check_spawns.py` |
| `record_eval_trace.py` | 录 `/eval_state` 为判分 trace JSONL（判分跑 `eval_state_every:=1` = 2 ms 帧） | 见下 |
| `record_motor_state.py` | 录 `/motor_state` 为 CSV（回归对比） | ROS 运行时 |
| `record_state_command.py` | 录 `/motor_state` + 最近 `/mit_command`（状态/指令归因） | ROS 运行时 |
| `compare_motor_traces.py` | 逐 sim_time 对比两份 motor_state CSV | `python compare_motor_traces.py a.csv b.csv` |
| `smoke_policy_load.sh` | 无头仿真：A 站立 → RB+↑ 进 RL，断言策略加载成功 | 需构建 + 本地权重 |
| `smoke_policy_switch.sh` | 无头仿真：注入 `/joy` 热切换 row2p / st30 / s1d（Part1 含固定动作） | 需构建 + 本地权重 |

判分 trace 录制示例（终端 A 起 launch，终端 B 录）：

```bash
ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=2 level:=trench_30cm \
    policy_name:=s1d_trench_reward_v75_m100 viewer:=false realtime:=true \
    eval_state_every:=1 publish_every:=1
python demo/tests/record_eval_trace.py --part 2 --robot blackW --level trench_30cm \
    --scene demo/scenes/part2/generated/part2_off_trench_30cm.xml --seconds 30 \
    --out artifacts/traces/trench_30cm.jsonl
```

`pixi run -e eval demo-tests`（`tools/demo_tests.sh`）会依次跑前三个检查 + Part1/Part2 场景的
MuJoCo 加载冒烟。ROS 冒烟脚本在未构建或没有本地 `policy.pt` 时不要运行（会起不来/加载失败）。
