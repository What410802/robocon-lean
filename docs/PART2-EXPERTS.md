# Part2 越障专家选型（blackW，2026-11-04）

本文回答“Part2 队内赛要求走过的地形，各用哪个专家权重、证据是什么、对入场对齐有什么要求”。
路线编排/衔接模型见 [`PART2-ROUTE.md`](PART2-ROUTE.md)；判分规范见 [`../eval/SPEC.md`](../eval/SPEC.md)。

- 场地：`demo/scenes/part2/full_field/part2_fixed_field.xml`（`map_fixed.xml`，梅花桩已按竞赛 PDF 修正）。
- 专家来源分两层：**旧仓 `~/RoboCon` 的证据**（`success-videos.md` + `artifacts/vtaudit-20261031/`
  + 全部 summary 扫描）与**本仓复测**（1640 次单关 eval-side 仿真，`artifacts/part2_experts/`）。
- 预算：单关判分用 `eval/robocon_eval/policy_runner.py`（与部署同一观测/动作/PD 契约，
  Part2 v5 judge），deterministic；鲁棒性用 spawn offset（dy/dyaw）而不是 seed。

## 1. 结论表（每个地形一个部署专家）

判分侧（eval-side `policy_runner`，与旧仓官方门禁同一条链路）与**真实部署栈**（`rl_sim`+`sim_node` ROS 运行）
在本轮出现系统性差异，因此分两列给结论；全场路线采用**部署栈实测通过**的难度（§2.3）。

| 地形（要求项） | 判分侧最高 | 部署栈实测（真实 ROS，重复试验） | 专家（policy 目录） | 命令模式 | 单关得分（判分侧 / 部署侧） |
|---|---|---|---|---|---|
| 高台 | `high_platform_40cm`（200） | **场地 20cm 2/2 PASS（路线用）；30cm 场地 1/4、官方单关 2/3；40cm 0/4** | `row2p_p40w40_v1` | adaptive vx=0.5 | 200 / 50 |
| 楼梯 | `stairs_30cm`（200） | **`stairs_30cm` vx0.5 1/1、vx0.6 1/1（vx0.55/0.65/0.7 失败）** | **`vt_band_v1`（本次新接入）** | adaptive vx=0.5 | 200 / 200 |
| 斜坡 | `slope_35deg`（100；45° 按用户要求暂不做） | **`slope_35deg` 3/3** | `row2p_p40w40_v1` | adaptive vx=0.5 | 100 / 100 |
| 沟壑 | `trench_30cm`（300） | **`trench_30cm` 3/3** | `s1d_trench_reward_v75_m100` | **adaptive vx=0.8**（fixed 0.8 也可，但 +2° 偏航会失败） | 300 / 300 |
| 梅花桩（竞赛 PDF 几何） | `plum` 0（盲走 0/200） | 不可判分（见 §4） | **无可部署专家（诚实缺口）** | 路线用 `obstacle` 仅做非计分贯通 | 0 / 0 |
| 高墙（用户清单外、Part2 也计分） | `high_wall_40cm`（200） | **`high_wall_40cm` 2/3（官方单关 1.8 m 助跑）**；场地实例只有 0.54 m 助跑 → 路线用 20 cm `cross_bar_5`（旧 real8 已验证） | `vt_band_v1`（40cm）/ `row2p`（20cm） | adaptive vx=0.5 | 200 / 50（场地实例） |

- 判分侧（eval-side）可部署专家合计：**1000 / 1300**（高台 200 + 楼梯 200 + 斜坡 100 + 沟壑 300 + 高墙 200；缺 slope_45 的 200 与梅花桩的 200）。
- **全场 1× 真跑最终档（`real_v21` 6/6 legs PASS）**：高台 20（50）+ 楼梯 20（100，st30@0.45）+
  斜坡 35（100）+ 沟壑 30（300）+ 梅花桩绕行（0）+ 高墙 20 场地实例（50）= **600 / 1300**；
  evaluate 用的 200 分楼梯 30 档（vt_band@0.60）在 v14/v17 整场走过、v15/v18 会在顶层平台 stall
  （约 50%），可按 §2.4 的命令切换，不设为默认。
- 若只按“除 45° 外的官方最高难度”计：判分侧 1000/1100；部署侧可复现 600/1100
  （楼梯 30 档运气好时 700/1100）。

## 2. 旧仓证据 → 本仓复测

### 2.1 旧仓（`~/BiS.d/tmp/success-videos.md` + `~/RoboCon/artifacts/`）结论

- 19/19 个非 slope45 官方便利关有成功记录；**slope45 全仓 0 条成功**（失败视频见
  `vtaudit-20261031/videos/part2/vt_mode_v2__adaptive_vx0.5__slope_45deg__failure.mp4`）。
- `row2p_p40w40_v1`（= 本仓同名策略，`vt_mode_v2` 在障碍段与它完全等价）覆盖平台 40/高墙 40/
  斜坡 35/楼梯 20/沟壑 20；`vtaudit-20261031/matrix_summary.json` 的
  `vt_mode_v2__adaptive05__seed17` / `row2p_p40w40_v1__adaptive05__seed17` 合计 750 分。
- 楼梯 30cm 唯一通过者：`vt_band_v1` adaptive vx=0.5（旧仓 `campaign3-adaptive-vt_band_v1`
  trials 22–24 = 3/3，v7.6 judge；`success-videos.md` 中 `stairs_30cm` 只有 trace 无视频）。
- 沟壑 30cm：`s1d_trench_reward_v75_m100` fixed vx=0.8/1.0 10/10；vx=0.5 必然失败
  （`gap:wheel_in_slot`）。
- 梅花桩：旧仓所有成功记录（17 条）都属于 depth-CNN 学生族 `route2_*`，盲走策略 0/384。

### 2.2 本仓复测（`tools/campaign/run_expert_matrix.py`，1640 runs，0 error）

```
pixi run -e eval python tools/campaign/run_expert_matrix.py           # 缓存矩阵 + 报告
pixi run -e eval python tools/campaign/run_expert_matrix.py --aggregate-only
```

| 复测项 | 结果 |
|---|---|
| 官方 20 关单关矩阵（18 policy × 多模式 × 多偏移） | `artifacts/part2_experts/summary.md` |
| 与旧仓权威结论对照 | 16/16 复现（含 `vt_band_v1 stairs_30cm`、`s1d trench_30cm`、`row2p platform40`） |
| 楼梯 30cm 细盆扫描 | `artifacts/part2_experts/stairs30_basin.json`（vx=0.5/0.55/0.70 三张 13×11 图 + vx=0.7 的 9×9 细图） |
| 斜坡 45° | 25/25 失败（本仓同样证实无解，按用户要求不做） |

### 2.3 部署栈复测（真实 ROS 部署：`rl_sim` + `sim_node`）

判分侧（PolicyRunner 进程内 MuJoCo）复现旧仓门禁 16/16，但**真实部署栈**在相同 policy/scene/config
下会系统性偏弱：例如 `high_platform_40cm` 判分侧 row2p/st30 可以过，部署栈 0/4（4 个专家/命令组合，
全部在 0.40 m 台阶面 stalls，base z≈0.68、pitch≈20°）；`platform_30cm` 部署栈 2/3。
根因已排除：/joy 双倍缩放、obs/action 契约（`RL_DEBUG_OBS` 的 action 与 Python TorchScript 逐位一致）、
policy 加载、命令限幅窗口——差异在策略输出之后的执行链路（S4 发布/看门狗、S5 限幅/力矩预算或 sim_node
伺服），本轮未继续追（耗时不划算），改用**部署栈实测档位**编排路线。

部署栈单关试验工具（本轮新增）：`demo/tests/run_joy_trial.sh` + `tools/trial/adaptive_joy.py`。
它跑真实 launch（FSM A→EnterRL），按 50 Hz 训练侧自适应命令或固定置杆推 28–34 s，录判分 trace，再
`robocon-eval replay-part2` 判定。命令示例：

```bash
ADAPTIVE=1 VX=0.5 SECONDS_REC=34 LEVEL=high_platform_30cm POLICY=row2p_p40w40_v1 \
  OUT=$PWD/artifacts/traces/trial.jsonl bash demo/tests/run_joy_trial.sh
pixi run -e eval robocon-eval replay-part2 --trace artifacts/traces/trial.jsonl
```

部署栈逐关结果（重复试验，`artifacts/traces/deploy*/`）：

| 关卡 | 试验 | 结果 |
|---|---|---|
| high_platform_20cm | row2p adaptive05 | 官方单关 1/1、**场地 S→N 2/2**；v19/v20/v21 整场 1× 均通过 |
| high_platform_30cm | row2p adaptive05 | 官方单关 2/3；**场地 S→N 1/4**（对齐良好时也可能台阶面 stall，接触混沌） |
| high_platform_40cm | row2p/st30/vt_freezehid/vtg09 adaptive/fixed | 0/4 fail（台阶面 stall） |
| stairs_30cm | vt_band adaptive vx=0.5 / 0.6 | 各 1/1 success（vx 0.55/0.65/0.7 fail） |
| slope_35deg | row2p adaptive05 | **3/3 success** |
| trench_30cm | s1d adaptive08 | **3/3 success** |
| high_wall_40cm | vt_band adaptive05 | 2/3 success（官方单关 1.8 m 助跑） |

结论：全场路线取“部署栈实测档位”；判分侧的最高档（平台 40/高墙 40）仍作为**单关能力**证据保留，
但不在带 0.54–0.75 m 助跑槽的整场地形上强行使用。

### 2.4 真场整场试跑（1× `realtime:=true`，逐步收敛）

部署栈单关能过 ≠ 整场能过：整场还叠加了转场、对齐、恢复。本轮真场试跑记录（详细表格见
`PART2-ROUTE.md` §7.3）：

| 试跑 | 档位 | 结果 |
|---|---|---|
| `real_v8` | platform20/stairs30 | leg1 PASS；leg2 转场北漂蹭 `stairs_08` 西端 → 转场改走南侧 y=−2.5 大走廊 |
| `real_v9` | platform20/stairs30 | leg1 PASS；leg2 北爬 x=60.50 被 turn-in 横漂带到 `stairs_25` 东缘 → 北爬移到缺口中心 61.45 |
| `real_v10` | platform20/stairs30 | leg1 PASS；leg2 stairs30 对齐 OK、爬完上坡，顶层平台 stall（x≈64.7,z≈2.15）；恢复倒车摔倒 → 加“高地不倒车”守卫 |
| `real_v12` | platform20/stairs20 | leg1 **PASS**、leg2 stairs20 **PASS**；leg3 slope35 对齐失败（旋转横漂 0.2 m，settle 前向只有 5 cm）→ 新增横向 `_sidestep_lane` 基元 |
| `real_stairs30_v60` | stairs30 单测 vt_band@0.60 | **PASS**（(67.88,3.20,−1.4°)）→ 楼梯 30 用 **vx=0.60** 的 200 分档在 field 可用 |
| `real_v13` | platform20/stairs20/slope30（无 sidestep） | slope 对齐失败 → 被 sidestep 修复覆盖 |
| `real_v14` | platform20/stairs30@0.6/slope35（sidestep） | leg1–4 PASS；plum `obstacle` 穿越 stall+摔倒 → plum 改绕行 |
| `real_v15`/`real_v18` | platform20/stairs30@0.6 | leg1 PASS，stairs30 在顶层平台 stall（stairs30 变异性） |
| `real_v16`/`real_v17` | 安全/最佳档 + 斜坡东侧出口 | leg1–3 PASS，斜坡→沟壑东侧 1.2 m 缺口转场失败 → 改西端绕行 |
| `real_v19` | 安全档 + 西端绕行 + plum bypass | **leg1–5 PASS**；wall20 对齐差 0.183 m 失败 → bar5 专用宽容差 + row2p@0.40 |
| `real_v20` | 安全档 + wall off | **PASS：`route finished: plum bypassed`**（leg1–5） |
| `real_v21` | 安全档 + wall bar5 修复 | **6/6 PASS：`route finished: 6/6 legs PASS (plum bypassed)`** = 最终交付档 |

结论：部署栈整场可复现档 = **平台 20 + 楼梯 20（st30@0.45）或楼梯 30（vt_band@0.60）+ 斜坡 35 +
沟壑 30 + 梅花贯通 + 高墙 20**；所有失败点都定位到几何/对齐/恢复实现，不是专家能力缺失
（单关 3/3、2/3 的证据在 §2.3）。

**楼梯 30cm 通过盆（关键）**：`vt_band_v1` 在 vx=0.5 时“原点格单点”通过
（相邻 1 cm / 1° 就可能翻转，属接触混沌，不是容差盒）；**vx=0.70** 时通过率 65%（93/143），
其中 `dy=−0.030 m` 一行对 `dyaw∈[−5°,+5°]` **11/11 全过**；细扫另给出 `dy∈[−0.050,−0.040] m`、
`dyaw∈[−1°,0°]` 的 3×3 全过区。路线因此采用 **vx=0.70 + 预对齐 y=3.000−0.045
（保守取 −0.045，容差 ±0.02 m / ±1.5°）**，见 `PART2-ROUTE.md` §5。

## 3. 专家权重与接入

| 专家 | 部署目录（`demo/rl_sar/policy/blackW/`） | 权重来源 | 备注 |
|---|---|---|---|
| `row2p_p40w40_v1` | 已有 | `logs/rsl_rl/black_velocity/2026-10-29_22-59-42_numrows31_mixed/model_750.pt` | Part2 主策略 |
| `vt_band_v1` | **本次新增** | `~/RoboCon/deployment/ws/src/rl_sar/policy/blackW/vt_band_v1`（policy sha256 `5f5a07f5…`） | 楼梯 30cm / 高墙 40 专家 |
| `s1d_trench_reward_v75_m100` | 已有 | `~/RoboCon/logs/rsl_rl/blackw_velocity_foothold/2026-10-31_01-49-46_s1d_trench_reward_v75/model_100.pt` | 沟壑 30 专家 |
| `st30_ppo_st30_s100_600` | 已有 | 旧 `@20261024_blackw` policy tree | 楼梯 20 / Part1 T 台阶 |
| `obstacle` | 已有 | **主办方官方策略**（`assets/policies/official`） | 梅花桩按非计分贯通段使用；也是高墙备选 |
| `vt_locked12` | 已有 | Part1 的 12-DoF 锁轮衔接模型 | 只做转场/对齐，不做障碍 |

- 权重 `policy.pt` 不入库（`.gitignore`），clone 后需按 `config.yaml`+`manifest.json` 本地补齐；
  本机已就位。新增 `vt_band_v1` 是唯一新部署策略。
- 手柄/键盘热切换绑定（本次新增）：`RB+D-pad 左` / 键盘 `b` = `PolicyBand` → `vt_band_v1`
  （`RB+D-pad 上` 仍是 EnterRL，不能占用）。改 `demo/rl_sar/include/input_map.hpp` +
  `demo/rl_sar/src/rl_sim.cpp`，`pixi run -e demo demo-build` 重建，`input_map_selftest` 通过。
  键表见 `demo/docs/joystick.md`。

## 4. 梅花桩：为什么没有可部署专家

竞赛 PDF 修正后的几何（`map_fixed.xml` row y=12）：入口平台 0.8×0.8×0.2 m，6 个 0.2 m 方块交错两排
（排内中心距 0.6 m、排间 y 距 0.4 m、首桩距平台 0.1/0.4 m），出口平台，无斜坡，全长 3.5 m。

判分要求轮子**全程在 0.2 m 顶面上**（任一轮底 <0.05 m 即 `dropped`）。结论：

- **盲走（只有 IMU+编码器，57×6 obs）无解**：本仓探针 5 个候选 × 2 种出生点 × 4 种命令 × 5 个
  y 偏移 = **0/200**；旧仓 13 个盲走策略 **0/384**。所有盲走第一次判规违反都是桩间轮子落地
  （judge 出生点下 100/100 是右前轮掉进 row-B 的 0.4 m 间隙）。
- **先上平台再切策略也没用**：0.2 m 入口平台上得去（盲走 72/100，`obstacle` 20/20），
  卡点是交错桩阵。
- **深度学生 `route2_v3`**（depth-CNN → 预测 1764 维 fine-scan → 冻结 RL teacher）：
  - 只在 eval 侧能跑（`MUJOCO_GL=egl`；`eval/robocon_eval/policy_runner.py` 的 depth 契约）；
  - 在**修正地图朝向**下 0/9，只在 **y 镜像**（= 旧仓 `plum_reg_1` 朝向）下 2/9（knife-edge）；
  - **部署不可用**：`demo/rl_sar` 没有 depth/camera 链路（`rl_sim.cpp` 只 `model.forward({obs})`），
    `rl_sim` 无法跑 `policy(obs,[depth])`。
- **建议**：梅花桩按“缺口”上报（0 分）；本仓全场路线只用 `obstacle` 做**非计分贯通段**
  （物理上能跨过桩阵，判分会 `dropped`），路线日志已标注 `TRANSIT_ONLY`。若后续要拿这 200 分：
  给 `demo/rl_sar` 补深度相机渲染/推理链路（工作量大），或重训一个盲/深度新专家并做判分门禁。

> **一致性风险（需队内确认）**：`terrain/gen_plum.py` 与生成的单关 `part2_off_plum_1.xml`
> 仍是 2026-11-02 的“3 桩单排”旧几何，与修正地图（6 桩两排）不一致；本轮只修了地图/路线，
> 没有改判分关卡生成器。另：修正地图的桩阵列相对旧仓 `plum_reg_1` 是 **y 镜像**，
> 复用任何旧梅花桩 checkpoint 前建议按 PDF 图核对左右手性。

## 5. 入场对齐要求（衔接模型能否达到）

专家对入场位姿的实测容差（eval-side，spawn offset 扫描）：

| 地形 | 横向 | 偏航 | 说明 |
|---|---|---|---|
| 平台 40 | ±0.05 m | −2°..+5° | ±0.10 / −5° 失败（非单调，接触混沌） |
| 楼梯 30 | 需 dy 目标 −0.03..−0.045 附近，±0.02 m | ±1.5° | vx 0.7 的 −0.03 行全 yaw 通过；其它 dy 很窄 |
| 斜坡 35 | ±0.10 m | ±5° | 最鲁棒 |
| 沟壑 30 | ±0.10 m | ±2° | adaptive08 比 fixed08 对称（fixed 在 +2° 失败） |
| 高墙 40 | −0.10..+0.10 m | −5°..+5°（8/9） | `vt_band_v1`/`obstacle`；`row2p` 3/9 不要用 |

`vt_locked12`（Part1 12-DoF 锁轮模型）的实测能力（旧 real8 全场 + 本轮真跑）：

- **车道保持基本无效**：纯 vy 自适应车道在 11–34 m 转场里系统横漂 ≈0.008 m/m
  （0.16–0.27 m），所以转场改为 **`goto()` 折线（≤10 m/段）**，靠 turn-drive-turn 纠偏。
- `goto`/位置闭环可达 ~2 cm 级，但**低速不能爬进 2 cm 圆**：真跑实测在 4–5 cm 残差处会
  振荡（v0.08 m/s + 锁轮死区/蠕动）；因此门限放宽到 0.05 m（楼梯 0.02 m 另配 vx=0.7 的
  容差行），并用“纯偏航伺服 + 沿车道短直行 settle”消除旋转带来的 5–6 cm 漂移。
- **真实部署栈（ROS）实测**：turn-drive-turn 对齐 48 s 内可到 `|e|≈0.02 m / |yaw|≈1°`（全场日志
  每条 `expert switch` 都记录 actual/e/along/yaw_err）；专家切换后的容差按上表（部署栈实测：
  平台 30 / 楼梯 30 / 斜坡 35 / 沟壑 30 均在该精度下走过）。注意部署栈需要 **1× 实时**：
  `PART1_ROUTE_STEPLOCK=1 + realtime:=false` 会让仿真跑到路线前面，命令/超时失去物理意义
  （实测平台 30 在该模式下失败、1× 下通过），所以整场真跑用 `realtime:=true`，只有假运动学
  模型用来做路线逻辑回归。
- 结论：**衔接模型可以达到各专家需要的入场精度**；其中楼梯 30 在部署栈的最佳命令是 vx=0.5
  （0.5/0.6 通过、0.55/0.65/0.7 失败，非单调），入场用普通车道中心（`start_dy=0`）。

## 6. 复现

```bash
# 1) 专家矩阵 + 楼梯通过盆（CPU，约 6 分钟）
pixi run -e eval python tools/campaign/run_expert_matrix.py
# 2) 单关判分（示例：楼梯 30 / 沟壑 30 / 平台 40）
pixi run -e eval robocon-eval run --robot blackW --part 2 --level stairs_30cm \
    --policy vt_band_v1 --out artifacts/traces/stairs30.jsonl --result artifacts/traces/stairs30.json
pixi run -e eval robocon-eval run --robot blackW --part 2 --level trench_30cm \
    --policy s1d_trench_reward_v75_m100 --out artifacts/traces/trench30.jsonl
pixi run -e eval robocon-eval run --robot blackW --part 2 --level high_platform_40cm \
    --policy row2p_p40w40_v1 --out artifacts/traces/platform40.jsonl
# 3) Part2 记分（判分侧最高档；部署侧实际可复现档见 §2.4）
pixi run -e eval robocon-eval score \
  --levels '{"high_platform":40,"stairs":30,"slope":35,"high_wall":40,"trench":30,"plum":0}'
# 4) 部署栈单关复测（真实 launch；例：平台 20 自适应 0.5）
ADAPTIVE=1 VX=0.5 SECONDS_REC=34 LEVEL=high_platform_20cm POLICY=row2p_p40w40_v1 \
  OUT=$PWD/artifacts/traces/trial.jsonl bash demo/tests/run_joy_trial.sh
pixi run -e eval robocon-eval replay-part2 --trace artifacts/traces/trial.jsonl
# 5) 整场 1×（默认档 = 最终验证档，见 PART2-ROUTE.md §8）
PART1_ROUTE_JUDGE=0 ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=1 \
  route:=part2 policy_name:=vt_locked12 transfer_policy:=vt_locked12 viewer:=false \
  realtime:=true eval_state_every:=10 stop_on_fail:=true
```

证据目录：`artifacts/part2_experts/`（`summary.md`、`summary.json`、`runs/` 1640 条、
`stairs30_basin.json`、`plum/REPORT.md`、`plum/results/`）；部署栈 trace 在
`artifacts/traces/`，整场 `artifacts/part2_route3/real_v21_*`（含 `real_v21_route.mp4`）。
