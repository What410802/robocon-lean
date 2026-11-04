# robocon-lean

RoboCon 2026 足式机器人仿真赛题的最小可读仓库：**Part 1 障碍赛（900 分）** 与
**Part 2 越障能力测试（1300 分）** 的仿真、训练、评测与演示。一个人从零读完本文件即可跑起
demo；规则依据、判分细节和训练配方分别在 `docs/` 与各目录 README 中。

> 状态：2026-11-02 已从旧比赛工作区迁移重建完成（新仓独立 git 历史，不引用旧仓）。
> 仓库不带历史权重/旧 artifacts：权重按 `demo/rl_sar/policy/*/` 的配置重新训练/导出。
> 同日已按本 README 在干净环境验证 `eval` / `demo` / `train` 三个 pixi 环境、colcon 构建、
> 固定路线（含 `stop_on_fail`、live Part1 judge 门控）与路线录像。

## 三条流水线

```
terrain/  ──►  demo/scenes/  ──►  demo/ (MuJoCo + ROS 2)  ──►  traces  ──►  eval/ (judge)  ──►  score
 官方地图         单关场景           策略推理 / 手柄 / 自动路线        2 ms 状态+接触         Part1/Part2
   │
   └──────────►  train/ (mjlab)  ──►  TorchScript policy ──►  demo/rl_sar/policy/<name>/
```

## 目录

| 目录 | 内容 | 入口 |
|---|---|---|
| `demo/` | ROS 2 + MuJoCo 运行时（rl_sar 派生）、机器人模型、场景、策略配置、操作手册 | `demo/README.md` |
| `eval/` | 独立判分器（Part1 judge、Part2 v5 judge）、trace 回放、评分曲线、SPEC | `eval/SPEC.md` |
| `train/` | mjlab 训练任务（black / blackW）、域随机化、导出 | `train/README.md` |
| `terrain/` | **唯一地形来源**：官方 `map.xml`、Part1 `scene_terrain.xml`、场景/关卡生成器 | `terrain/README.md` |
| `assets/` | 第三方资产清单/恢复脚本（不入库大文件） | `assets/manifest.json` |
| `tools/` | 构建、录视频、状态检查、Part1 MPC（规划中） | `tools/README.md` |
| `docs/` | 规则依据、架构、操作手册；`part2/` 专家与全场路线；`training/` 训练配方 | `docs/RULES-BASIS.md` |

## 快速开始

```bash
# 0) 只装需要的环境（三选一/按需；单份 pixi.lock，不用 --frozen）
pixi install -e eval
pixi install -e demo
pixi install -e train

# 1) 判分器自测（CPU，最快验证）
pixi run -e eval eval-pytest

# 2) 构建 ROS 2 运行时（首次几分钟）
pixi run -e demo demo-build

# 3) 跑一关 Part1（无窗口、实时）
source demo/install/setup.bash
ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=1 \
    policy_name:=row2p_p40w40_v1 viewer:=false realtime:=true

# 4) 跑一关 Part2（单关场景由 terrain/ 生成）
ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=2 \
    level:=high_platform_40cm policy_name:=row2p_p40w40_v1 viewer:=false realtime:=true
```

# 5) Part 1 固定路线自检 + 录像（可选；约 4–6 min）
source demo/install/setup.bash
PART1_RECORD_PATH=$PWD/artifacts/route/states.jsonl \
PART1_RECORD_PLAN_PATH=$PWD/artifacts/route/plan.jsonl \
ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=1 route:=part1 \
    policy_name:=vt_pose_v1 transfer_policy:=vt_track_v2 \
    viewer:=false realtime:=true eval_state_every:=10 stop_on_fail:=true
pixi run -e eval python tools/video/replay_route_video.py \
    artifacts/route/states.jsonl artifacts/route/route.mp4 artifacts/route/plan.jsonl
```

`stop_on_fail:=true`（默认）在第一个路线预检 FAIL 处停止并关掉 launch，避免卡在掉落点
空等；要像操作员一样继续就传 `stop_on_fail:=false`。路线 PASS 只是 precheck，官方分用
`demo/tests/record_eval_trace.py` 同时录 trace 后 `pixi run -e eval robocon-eval replay-part1`
判定（两者差异见 `demo/docs/part1-auto-route-operator.md` §2.1）。

判分跑必须记录 2 ms trace（`eval_state_every:=1`），然后：

```bash
pixi run -e eval robocon-eval replay-part2 --trace <trace.jsonl>
```

## 机器人 / 契约

- 唯一机型：**blackW**（16 DoF：每条腿 hip/thigh/calf + 驱动轮）；black（12 DoF）仅作历史对照保留。
- 策略契约：50 Hz 推理、200 Hz 控制、2 ms 物理；obs 57×6 history，action 16；
  `kp=50, kd=1.2`（hip/thigh/calf），轮子 kp=0；力矩上限 hip/thigh 25 N·m、calf 60 N·m（规则要求）。
- 允许传感器：IMU、关节编码器、相机/雷达（仿真相机）；**禁止** body_lin_vel、heightmap、
  foot_force、foot_height、body_height 以及特权地形扫描进网络。世界位姿真值只可用于生成指令。

## 文档索引

- **通用**：`docs/RULES-BASIS.md`（规则原文/实现约定/资产哈希）、`docs/ARCHITECTURE.md`
  （数据流/坐标系/判分链路）、`docs/OPERATIONS.md`（手柄/自动路线/录制/判分操作手册）
- **Part 2**：`docs/part2/EXPERTS.md`（各地形专家选型、证据、入场对齐容差、梅花桩缺口）、
  `docs/part2/ROUTE.md`（修正版全场自动路线：专家 + `vt_locked12` 衔接、转场/对齐/审计）
- **训练**：`docs/training/VT-RECIPE.md`（速度跟随衔接模型配方/训练记录/验收）、
  `train/README.md`（训练任务、导出、改造入口、权重管理）
- **判分**：`eval/SPEC.md`（人读版单一标准）
- **迁移**：`MIGRATION_NOTES.md`（迁移取舍、已废弃内容、未决事项）

## 许可与来源

`demo/` 派生自 [fan-ziqi/rl_sar](https://github.com/fan-ziqi/rl_sar)（Apache-2.0），
训练基座来自 nwwolf mjlab 发布，官方场景/地图来自赛事主办方。具体 commit、许可与“官方模型物理
未改动”的说明见各目录 `UPSTREAM.md` 与 `docs/RULES-BASIS.md`。
