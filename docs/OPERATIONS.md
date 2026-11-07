# 操作手册（Part1 / Part2 仿真）

所有命令都在仓库根目录执行。判分跑要求 **独占 CPU**、**一次只跑一个 launch 实例**
（`/joy`、`/eval_state` 会串台）；并行 MuJoCo 探针会拖垮 50 Hz 控制环。

## 0. 一次性准备

```bash
pixi install -e demo          # ROS 2 Humble + MuJoCo 3.11 + torch 2.14.1+cpu（PyPI CPU 轮子，与 eval 同版本）
pixi install -e eval          # 判分器（torch 2.14.1+cpu，与 demo/train 同一 torch 版本线）
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
pixi run -e demo demo-part1
```

- `demo-part1` 使用统一 launch 的 `route:=part1`，自动完成站立→进 RL→逐段走路线。
  默认开窗口、按 1× 实时运行并使用 `vt_locked12` 衔接；无窗口加 `viewer:=false`。
- 路线依赖策略：`vt_locked12`、`vt_edge_v1`、`row2p_p40w40_v1`、`st30_ppo_st30_s100_600`、
  `obstacle` + `behaviors/{low_bar_drive,bridge_drive}.yaml`。
- 调试其它衔接模型时显式传 `transfer_policy:=...`；也接受 `PART1_TRANSFER_POLICY`。
  `vt_track_v2` 的历史训练/验收见 `docs/training/VT-RECIPE.md` §4.2。
- `vt_locked12` 的策略仅输出 12 个腿关节动作；部署层在每次进入该策略时保存四个实测轮角，
  持续以该角度和零目标角速度做 PD 保持。力矩限幅通过降低增益实现，不移动锁轮目标；
  切到 16DoF 轮式专家或 Part1 固定动作后使用各自原有轮控制。
- Part1 默认源码已恢复为冻结基线 `part2_full_live_v72_sources/demo/rl_sar/scripts/part1_auto_route.py`，
  SHA256 为 `c556739c374e9cb40393cfaddedefa6aaf7a6de33cc0a3cefc1717c1e97a75a5`。
  该源码与 v56/v72 快照哈希一致，保留 v55/v67 阶段的路线行为：桥 A 原交接门限、桥 B 原对线、低杆原路点，
  砂砾坑入口 `x=5.70`、循迹 `x_hold=5.95`、专家门限 `y<-11.95`，再定向到角点转西。
- [v67 开窗口 1× 整场](../artifacts/route_fix/part1_final_v67_evidence.json) 全部门限 PASS、0 次恢复、
  退出状态 0，13109 帧、262.16 s 返回起点 `(3.561,-8.939,0.557)`；恢复源码后未重跑物理整场。
  v55 与更早 full40 / 基线记录仍保留各自历史范围。
- 本次请求的桥 A 延后交接、桥 B 精对线、低杆纠偏和四轮上坑即换模仍未解决。
  v85 / v86 过两桥后低杆前对线失败，v87 / v88 在桥 B 对线失败。
  [v88 实验源码及失败证据](../artifacts/route_fix/part1_final_calibration_v88_evidence.json) 保留，
  所有这些新校准均未作为默认启用，不将失败校准当作已完成改进。
- Part1 保留 1× 墙钟控制，保持 `realtime:=true`。快捷入口默认 `PART1_ROUTE_JUDGE=0`，
  显式 `PART1_ROUTE_JUDGE=1` 启用原版 live judge；直接 ROS launch 的 judge 默认与正式 judge 未改。
  历史 full40 视频的原版 judge 回放为 500 分；v67 未据此宣称正式得分，
  实体运动完成不等于正式全通过，详见操作者手册 §2.1 / §4。

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
- 全场自动路线依次跨越所选最高规格及每类全部较低实体（见下），计分仍按单关独立。

### 2.1 全场 1× 演示（唯一 33 段路线，含深度相机梅花桩）

新版最佳连续前缀仍为 v82 / v83：平台六档与楼梯至 25 cm 共 14 段通过，30 cm 楼梯前对齐失败。
最新 v84 的前四段通过严格回放，随后在 25 cm 平台卡住，回放为 `not_on_platform`。
完整 33 段尚无全部成功记录，40 cm 平台仍是明确跳过的能力缺口。

```bash
pixi run -e demo demo-part2
# 无窗口实时调试：
pixi run -e demo demo-part2 viewer:=false
```

- 当前执行平台至 30 cm、楼梯单级至 30 cm、坡至 35°、沟宽至 30 cm、墙至 40 cm，包含全部较低实体。
  平台规则上限仍为 40 cm；长助跑重试仍失败后，经用户确认暂跳过此档，计划通过 `SKIPPED_TIERS`
  明确记录能力缺口，不能算作通过。
  按此前约定本轮不做 45° 坡，也不额外跨越 40° 练习坡；比赛梅花桩为 20 cm 六桩。
  路线、实际规格及验证状态见 [`part2/ROUTE.md`](part2/ROUTE.md)。
  唯一路线数据在 `demo/rl_sar/scripts/part2_route_plan.py`；启动器、净空审计与运行时读取同一份数据。
- `route:=part2` 自动选择 Part2 场景、RL `part=2`；长直线使用 `flat`，转向与精对线使用
  `vt_locked12`。A 站立在两个 part 都可用，
  Part2 仍禁止固定动作过障。
- 自动路线默认按 1× 实时运行，实时和全速模式均按仿真时间同步 50 Hz 控制。
  `eval_state_every` 独立控制记录频率：快捷入口默认 10，判分 trace 显式传 1。
  第一段失败就停，跑完自动退出。
- `PART2_ROUTE_START=<leg名或序号>` 从同一路线的指定段开始，launch 自动把出生点放在该段入口；
  `PART2_ROUTE_ONLY=<leg名或序号>` 在指定段后停止。显式 `spawn_*` 仍可覆盖；子段调试不代表全场完成。

### 2.2 深度相机（梅花桩专家）

- `sim_node` 自动检测场景里的 `depth_cam`（blackW 的 `d435_link`），EGL 离屏渲染 64x48 公制
  平面深度并发布 `/depth_cam`（50 Hz，`depth_every:=10`；`depth:=auto|on|off`）。
- 离线自检：`demo/install/quadruped_ros2/lib/quadruped_ros2/sim_node --depth-self-test <scene.xml>`；
  与 eval 侧 Python 渲染逐位对拍用 `tools/` 的 depth_parity 脚本逻辑。
- 深度策略的 C++ 推理与 Python 同输入对拍：
  `RL_DEPTH_DEBUG=1 RL_DEPTH_DEBUG_MAX=6` 跑一小段后，
  `pixi run -e eval python demo/tests/check_depth_inference.py --policy demo/rl_sar/policy/blackW/plum_depth_final_v1`。
- 无窗口/EGL 不可用时深度相机会打印一次告警并停发，仿真与盲目策略不受影响。

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
pixi run -e demo demo-part1 viewer:=false
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

### 4.2 全场 Part2 路线录像

```bash
MUJOCO_GL=egl pixi run -e eval python tools/video/replay_route_video.py \
    artifacts/part2_route/states.jsonl artifacts/part2_route/route.mp4 \
    artifacts/part2_route/plan.jsonl \
    --scene demo/scenes/part2/full_field/part2_fixed_field.xml --cli-only
```

录制时设置 `PART1_RECORD_PATH=$PWD/artifacts/part2_route/states.jsonl` 和
`PART1_RECORD_PLAN_PATH=$PWD/artifacts/part2_route/plan.jsonl` 再运行 §2.1 的启动命令。
回放不会重新计算物理；旧版 `real_v27_*` 只记录每类一段，不能证明当前路线的全部障碍覆盖。
`--cli-only` 在画面外只保留 `> CLI:` 标注；完整命令和局部样片见 [视频工具说明](../tools/video/README.md)。

现有 [v56 全程视频](../artifacts/route_fix/part2_full_live_v56_cli.mp4) 是旧 25 段计划的单次
开窗口 1× 记录，时长 20 min 20.4 s：集成门限 25/25、退出状态 0，
[支撑点回放 24/25](../artifacts/route_fix/part2_full_live_v56_eval.json)，最后 20 cm 墙未满足稳定保持。
该录像用于复盘旧版完整实体运动，新版 33 段验证状态以路线文档为准。

## 5. 自检与排错

| 现象 | 原因/处理 |
|---|---|
| 想按 1× 现场演示 | 保持默认 `realtime:=true`；Part1 使用原墙钟控制，Part2 支持仿真时间全速回归 |
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
