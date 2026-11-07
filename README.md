# robocon-lean

RoboCon 2026 足式机器人仿真赛题的最小可读仓库：**Part 1 障碍赛（900 分）** 与
**Part 2 越障能力测试（1300 分）** 的仿真、训练、评测与演示。一个人从零读完本文件即可跑起
demo；规则依据、判分细节和训练配方分别在 `docs/` 与各目录 README 中。

> 状态：2026-11-02 已从旧比赛工作区迁移重建完成（新仓独立 git 历史，不引用旧仓）。
> 仓库包含演示所需的 14 份策略权重；路径、大小与 SHA-256 见 [WEIGHTS.json](demo/rl_sar/policy/WEIGHTS.json)。旧 artifacts 不入库。
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

# 3) Part1 现场自动演示（开窗口、实时、自动站立并运行路线）
pixi run -e demo demo-part1

# 4) Part2 现场自动演示（33 段，逐档越障，包含梅花桩）
pixi run -e demo demo-part2

# 无窗口仍按 1× 实时运行
pixi run -e demo demo-part2 viewer:=false
# Part2 从指定障碍前对线后继续；0 关闭路线超时/停滞退出，Ctrl+C 手动停止
pixi run -e demo demo-part2 resume_before:=stairs_30cm timeout_scale:=0
```

以上演示依次运行，一次只开一个 launch。两个快捷入口均自动 source 工作区，默认开窗口、
1× 实时和 50 Hz 位姿记录，无需操作手柄；Part1 使用 `vt_locked12` 衔接，Part2 长直线使用
`flat`、转向与精对线使用 `vt_locked12`。上一场自动退出后再启动下一场。

Part1 保留一条默认整场路线：桥 A 等后轮越过末槽、落上东台后再交接，桥 B 沿 `x=5.65` 跨越。
砂砾坑保留原 `require + rotate_hold` 对准器，入口改为 `x=5.84`；四轮上坑即切 `vt_locked12`，
再沿东臂到角点转西，角点 lookahead 为 0.30。
限高杆在降至低姿态后重新精对正，停稳复核偏角小于 1°才前进。
[当前源码 v104 整场证据](artifacts/route_fix/part1_lowbar_full_v104_evidence.json)：开窗口、1× 实时，
全部路线预检通过并返回起点，退出 0、0 次恢复、263.80 s；实际过杆偏角约 0.69°，无机器人杆柱接触记录。
这是关闭 live judge 的实体运动演示，未宣称正式满分或广泛扰动下的鲁棒性。
Part2 当前唯一计划执行 **33 段**：平台至 30 cm、楼梯单级至 30 cm、坡至 35°、沟宽至
30 cm、墙至 40 cm，逐档跨越较低实体及比赛梅花桩。40 cm 平台在加长助跑后仍摔倒，
经用户确认暂时跳过，并在计划中明确记录为能力缺口；规则最高平台仍是 40 cm。
按此前约定本轮不做 45° 坡。
历史 [v56 开窗口 1× 全程视频](artifacts/route_fix/part2_full_live_v56_cli.mp4) 完成旧 25 段集成门限、
退出状态 0，但 [支撑点回放为 24/25](artifacts/route_fix/part2_full_live_v56_eval.json)，末段 20 cm 墙
未满足越墙后稳定保持；其中梅花桩成功投影 200 分。新版最佳连续前缀仍为 v82 / v83 的前 14 段，
均在 30 cm 楼梯前对齐失败；最近从原点试跑 v84 的前四段通过严格回放，随后在 25 cm 平台卡住，
回放为 `not_on_platform`。新版 33 段整场尚未完成，
最新状态见 [路线说明](docs/part2/ROUTE.md)。
v91 / v92 已实测命令行恢复的出生点与正常对线；两次均在原路线门限退出 1，未视为成功演示。

Part1 路线录制（可选）：

```bash
PART1_RECORD_PATH=$PWD/artifacts/route/states.jsonl \
PART1_RECORD_PLAN_PATH=$PWD/artifacts/route/plan.jsonl \
pixi run -e demo demo-part1 viewer:=false
pixi run -e eval python tools/video/replay_route_video.py \
    artifacts/route/states.jsonl artifacts/route/route.mp4 artifacts/route/plan.jsonl
```

`stop_on_fail:=true`（默认）在第一个路线预检 FAIL 处停止并关掉 launch，避免卡在掉落点
空等；要像操作员一样继续就传 `stop_on_fail:=false`。路线 PASS 只是 precheck，官方分用
`demo/tests/record_eval_trace.py` 同时录 trace 后 `pixi run -e eval robocon-eval replay-part1`
判定（两者差异见 `demo/docs/part1-auto-route-operator.md` §2.1）。
Part2 同次运行录制与仅含 `> CLI:` 标注的整场视频命令见 [视频工具说明](tools/video/README.md)。

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
- 仿真相机：`sim_node` 渲染场景里的 `depth_cam`（64x48、fovy 70、公制平面深度、cutoff 3.0 m），
  以 `/depth_cam`（`DepthFrame`）50 Hz 发布；梅花桩深度专家 `plum_depth_final_v1` 用它作为
  网络第二输入（`policy(obs_1d_324, [1x1x48x64])`）。相机契约（`depth.json`）与训练/eval 同源。

## 文档索引

- **通用**：`docs/RULES-BASIS.md`（规则原文/实现约定/资产哈希）、`docs/ARCHITECTURE.md`
  （数据流/坐标系/判分链路）、`docs/OPERATIONS.md`（手柄/自动路线/录制/判分操作手册）
- **Part 2**：`docs/part2/EXPERTS.md`（各地形专家选型、证据、入场对齐容差；六类最高档判分侧
  **1200/1300**，含本轮新增的深度相机梅花桩专家）、
  `docs/part2/ROUTE.md`（唯一 33 段自动路线：所选最高规格及全部低档、转场/对齐/净空审计与运行命令）
- **训练**：`docs/training/VT-RECIPE.md`（速度跟随衔接模型配方/训练记录/验收）、
  `train/README.md`（训练任务、导出、改造入口、权重管理）
- **判分**：`eval/SPEC.md`（人读版单一标准）
- **迁移**：`MIGRATION_NOTES.md`（迁移取舍、已废弃内容、未决事项）

## 许可与来源

`demo/` 派生自 [fan-ziqi/rl_sar](https://github.com/fan-ziqi/rl_sar)（Apache-2.0），
训练基座来自 nwwolf mjlab 发布，官方场景/地图来自赛事主办方。具体 commit、许可与“官方模型物理
未改动”的说明见各目录 `UPSTREAM.md` 与 `docs/RULES-BASIS.md`。
