# Part2 越障专家选型（blackW，2026-11-07）

本文回答“Part2 比赛要求走过的地形，各用哪个专家权重、证据是什么、对入场对齐有什么要求”。
路线编排/衔接模型见 [`ROUTE.md`](ROUTE.md)；判分规范见 [`../../eval/SPEC.md`](../../eval/SPEC.md)。

- 场地：`demo/scenes/part2/full_field/part2_fixed_field.xml`（`map_fixed.xml`，梅花桩已按竞赛 PDF 修正）。
- 专家来源分两层：**旧仓 `~/RoboCon` 的证据**（`success-videos.md` + `artifacts/vtaudit-20261031/`
  + 全部 summary 扫描）与**本仓复测**（1640 次单关 eval-side 仿真，`artifacts/part2_experts/`）。
- 预算：单关判分用 `eval/robocon_eval/policy_runner.py`（与部署同一观测/动作/PD 契约，
  Part2 判据），deterministic；鲁棒性用 spawn offset（dy/dyaw）而不是 seed。
- 判据口径（v9）：高台按官方“上到上平面即得分”，不要求从远侧下去；楼梯/斜坡/沟壑/高墙/梅花桩
  仍按“到达目标面 → 下到另一侧正常地面”的操作化终端态。
- **环境**：三个 pixi 环境统一 torch **2.14.1**（train CUDA 轮子，demo/eval 同版本 PyPI CPU 轮子）；
  MuJoCo 3.11。torch 版本会改变策略数值（plum 深度策略在 2.12 下 fail、2.14 下 success），
  所以本表所有本轮证据都在统一环境后重跑，路径见 §2.5。

## 1. 结论表（每个地形一个部署专家）

判分侧（eval-side `policy_runner`，与旧仓官方门禁同一条链路）与**真实部署栈**（`rl_sim`+`sim_node` ROS 运行）
在本轮出现系统性差异，因此分两列给结论；全场路线采用**部署栈实测通过**的难度（§2.3）。

| 地形（要求项） | 判分侧最高 | 部署栈实测（真实 ROS，重复试验） | 专家（policy 目录） | 命令模式 | 单关得分（判分侧 / 部署侧） |
|---|---|---|---|---|---|
| 高台 | `high_platform_40cm`（200） | **场地 20cm 2/2 PASS（路线用）；30cm 场地 1/4、官方单关 2/3；40cm 0/4** | `row2p_p40w40_v1` | adaptive vx=0.5 | 200 / 50 |
| 楼梯 | `stairs_30cm`（200） | **`stairs_30cm` vx0.5 1/1、vx0.6 1/1（vx0.55/0.65/0.7 失败）** | **`vt_band_v1`** | adaptive vx=0.5 | 200 / 200 |
| 斜坡 | `slope_35deg`（100；45° 按用户要求暂不做） | **`slope_35deg` 3/3** | `row2p_p40w40_v1` | adaptive vx=0.5 | 100 / 100 |
| 沟壑 | `trench_30cm`（300） | **`trench_30cm` 3/3** | `s1d_trench_reward_v75_m100` | **adaptive vx=0.8**（fixed 0.8 也可，但 +2° 偏航会失败） | 300 / 300 |
| 梅花桩（竞赛 PDF 几何） | **`plum_1`（200，2026-11-07 新增）** | **组合穿越：深度专家 + 出口台阶轮式专家（`vt_band_v1`）；判分 trace success 200（单段）；整场路线已接入（见 §4）** | **`plum_depth_final_v1`（深度相机）** | 无 command 项（自走；`/joy` 无效）+ 出口段轮式 0.45 m/s | 200 / 200 |
| 高墙（**默认开**） | `high_wall_40cm`（200） | **`high_wall_40cm` 2/3（官方单关 1.8 m 助跑）**；场地实例只有 0.54 m 助跑 → 路线用 20 cm `cross_bar_5`（旧 real8 已验证，本轮整场已接入） | `vt_band_v1`（40cm）/ `row2p`（20cm） | adaptive vx=0.5 | 200 / 50（场地实例） |

- 判分侧（eval-side）可部署专家合计：**1200 / 1300**（高台 200 + 楼梯 200 + 斜坡 100 + 沟壑 300 +
  高墙 200 + 梅花桩 200；只缺按用户要求本轮不做的 slope_45 的 100）。证据在
  `artifacts/part2_evals/torch214/`（六类每类一份 official-profile trace+result，`score.json` 合计 1200）。
- **全场 1× 六类全开档（`real_v27`，本轮交付证据）**：平台 20（S→N）→ 楼梯 20（st30@0.45）→
  斜坡 30 → 沟壑 20 → 梅花桩组合穿越（判分 200）→ 高墙 20 bar5（判分 50）；**路线 6/6 段 PASS，
  eval 判据逐段投影 6/6 段 success、合计 600/1300**（73712 帧，约 1475 s 仿真；
  `artifacts/part2_route3/real_v27_*`，含 mp4/overview）。
  全场连续跑不是官方计分单元，路线档位刻意低于判分侧最高档（可靠优先），更高档位仍可 env 切换。
- 旧的 `real_v22`（保守档、plum 绕行、wall 关）5/5 段集成门限 PASS、判分投影 350/1300；
  `real_v21`（plum 绕行、wall 20 开）6/6 段 PASS、投影 600/1300，保留作回退证据。
- 部署侧可复现档位（不含判分侧最高档）：平台 20 + 楼梯 20 + 斜坡 30 + 沟壑 20 + 梅花桩组合 +
  高墙 20（场地实例）。

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
| 官方 20 关单关矩阵（18 policy × 多模式 × 多偏移） | `artifacts/part2_experts/summary.md`（2026-11-07 在统一 torch 2.14.1 后整表重跑） |
| 与旧仓权威结论对照 | 旧矩阵（torch 2.12，2026-11-04）16/16 复现；**统一 torch 2.14.1 后 10/16**——差异项是 `vt_mode_v2`/`fixed-stairs30-final`（旧仓策略未迁入，config 缺失）与 `vt_band_v1 stairs_30cm`、`s1d fixed08 trench30`（30 s 测试时长下的接触混沌；按官方 120 s 时限的单关运行仍 success，见 §2.5）。矩阵的“推荐 mode”因此有轻微漂移（stairs→`st30 fixed08`、wall→`obstacle fixed05`），但**单关判分证据以实际 official run 为准**，不用矩阵推荐代替。 |
| 楼梯 30cm 细盆扫描 | `artifacts/part2_experts/stairs30_basin.json`（vx=0.5/0.55/0.70 三张 13×11 图 + vx=0.7 的 9×9 细图；2.12 环境图表，档位结论仍由部署 trace 定） |
| 斜坡 45° | 25/25 失败（本仓同样证实无解，按用户要求不做） |

### 2.3 部署栈复测（真实 ROS 部署：`rl_sim` + `sim_node`）

判分侧（PolicyRunner 进程内 MuJoCo）复现旧仓门禁 16/16（torch 2.12 环境；统一 2.14.1 后的
差异见 §2.2/§2.5），但**真实部署栈**在相同 policy/scene/config
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
`ROUTE.md` §7.3）：

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
| `real_v21` | 安全档 + wall bar5 修复 | **6/6 PASS**（含 wall20）；高难回退证据 |
| `real_v22` | **保守档**：platform20/stairs20/slope30/trench20/plum bypass，wall 默认关 | **5/5 段 PASS，1224.8 s**；旧保守档（回退用；本轮默认已改六类全开） |
| `real_v23` | 六类全开首跑（plum 组合未刹车） | leg1–4 PASS；plum 深度段后在出口边缘跌倒（判分 `fell`） |
| `real_v24` | 六类全开（plum 组合 + 刹车） | **leg1–5 全 PASS；判分投影 550/1300，含 plum success 200**；leg5 后连接器 goto 失败（锁轮前未停稳）→ 加 gate 后站定 |
| `real_v25` | 六类全开（加 gate 站定） | leg1–4 PASS；plum 下降段跌倒（组合的混沌边界） |
| `real_v26` | 六类全开（默认档） | leg1–2 PASS；leg3 slope 对齐失败（yaw 23.8°）→ 重跑 |
| `real_v27` | 六类全开（默认档） | **路线 6/6 PASS、判分投影 6/6 success、合计 600/1300**（含 plum 200、wall 50；73712 帧）；**本轮交付证据** |

结论：部署栈整场 1× 默认档 = **平台 20 + 楼梯 20（st30@0.45）+ 斜坡 30 + 沟壑 20 +
梅花桩组合穿越 + 高墙 20（bar5，默认开）**，`real_v27` 6/6 PASS（逐段判分投影 600/1300）；
更高档位仍可切换（平台 30/40、楼梯 30 vt_band@0.60、斜坡 35、沟壑 30、高墙 bar6/bar7）。
整场是**非计分集成演示**，可靠性仍受两个混沌边界限制：plum 组合下降（约 2/3 单段成功率）与
个别 leg 的对齐/接触（slope 对齐偶发失败）；失败时最便宜的处理是重跑整场。

**楼梯 30cm 通过盆（关键）**：`vt_band_v1` 在 vx=0.5 时“原点格单点”通过
（相邻 1 cm / 1° 就可能翻转，属接触混沌，不是容差盒）；**vx=0.70** 时通过率 65%（93/143），
其中 `dy=−0.030 m` 一行对 `dyaw∈[−5°,+5°]` **11/11 全过**；细扫另给出 `dy∈[−0.050,−0.040] m`、
`dyaw∈[−1°,0°]` 的 3×3 全过区。路线因此采用 **vx=0.70 + 预对齐 y=3.000−0.045
（保守取 −0.045，容差 ±0.02 m / ±1.5°）**，见 `ROUTE.md` §5。

### 2.5 统一 torch 2.14.1 后的六类单关复核（2026-11-07）

三个环境统一到 torch 2.14.1 后，用**官方单关口径**（`--profile official`、自适应专家命令、
manifest 最高档）重跑六类，全部 success，合计 **1200/1300**：

| 地形 | level | policy | 命令 | 结果 | 证据 |
|---|---|---|---|---|---|
| 高台 | `high_platform_40cm` | `row2p_p40w40_v1` | adaptive 0.5 | success 200 | `artifacts/part2_evals/torch214/high_platform_40cm.*` |
| 楼梯 | `stairs_30cm` | `vt_band_v1` | adaptive 0.5 | success 200 | `…/stairs_30cm.*` |
| 斜坡 | `slope_35deg` | `row2p_p40w40_v1` | adaptive 0.5 | success 100 | `…/slope_35deg.*` |
| 沟壑 | `trench_30cm` | `s1d_trench_reward_v75_m100` | adaptive 0.8 | success 300 | `…/trench_30cm.*` |
| 高墙 | `high_wall_40cm` | `vt_band_v1` | adaptive 0.5 | success 200 | `…/high_wall_40cm.*` |
| 梅花桩 | `plum_1` | `plum_depth_final_v1` | 无 command（深度） | success 200 | `…/plum_1.*` |

记分：`robocon-eval score --levels '{"high_platform":40,"stairs":30,"slope":35,"high_wall":40,"trench":30,"plum":1}'`
→ `{"total": 1200, "max_score": 1300}`（`…/score.json`）。漏掉的 100 分 = 用户要求本轮不做的 slope_45。

复现命令（六条同样式；plum 不加 `--adaptive`）：

```bash
pixi run -e eval robocon-eval run --robot blackW --part 2 --level high_platform_40cm \
    --policy row2p_p40w40_v1 --profile official --adaptive --adaptive-vx 0.5 \
    --out artifacts/part2_evals/torch214/high_platform_40cm.jsonl \
    --result artifacts/part2_evals/torch214/high_platform_40cm.json
```

> 注意：矩阵复跑的“推荐 mode”在 30 s 测试时长下对若干混沌边界项会漂移（§2.2）；单关判分证据
> 一律用上面这种 official-profile 整时限 run，矩阵只用于选型参考与鲁棒性扫描。

## 3. 专家权重与接入

| 专家 | 部署目录（`demo/rl_sar/policy/blackW/`） | 权重来源 | 备注 |
|---|---|---|---|
| `row2p_p40w40_v1` | 已有 | `logs/rsl_rl/black_velocity/2026-10-29_22-59-42_numrows31_mixed/model_750.pt` | Part2 主策略 |
| `vt_band_v1` | **本次新增** | `~/RoboCon/deployment/ws/src/rl_sar/policy/blackW/vt_band_v1`（policy sha256 `5f5a07f5…`） | 楼梯 30cm / 高墙 40 专家 |
| `s1d_trench_reward_v75_m100` | 已有 | `~/RoboCon/logs/rsl_rl/blackw_velocity_foothold/2026-10-31_01-49-46_s1d_trench_reward_v75/model_100.pt` | 沟壑 30 专家 |
| `st30_ppo_st30_s100_600` | 已有 | 旧 `@20261024_blackw` policy tree | 楼梯 20 / Part1 T 台阶 |
| `obstacle` | 已有 | **主办方官方策略**（`assets/policies/official`） | 梅花桩按非计分贯通段使用；也是高墙备选 |
| `vt_locked12` | 已有 | Part1 的 12-DoF 锁轮衔接模型 | 只做转场/对齐，不做障碍 |
| `plum_depth_final_v1` | **本轮新增** | `logs/rsl_rl/blackw_plum_depth/2026-11-07_01-34-48_depth_ppo_stage2_r0/model_1150.pt`（深度学生；导出脚本 `train/scripts/plum/export_depth_policy.py`） | 梅花桩深度专家；无 command 项，忽略 `/joy`；manifest 含 official run / spawn 网格 / 组合部署证据 |

- 权重 `policy.pt` 不入库（`.gitignore`），clone 后需按 `config.yaml`+`manifest.json` 本地补齐；
  本机已就位。新增 `vt_band_v1` 是唯一新部署策略。
- 手柄/键盘热切换绑定（本次新增）：`RB+D-pad 左` / 键盘 `b` = `PolicyBand` → `vt_band_v1`
  （`RB+D-pad 上` 仍是 EnterRL，不能占用）。改 `demo/rl_sar/include/input_map.hpp` +
  `demo/rl_sar/src/rl_sim.cpp`，`pixi run -e demo demo-build` 重建，`input_map_selftest` 通过。
  键表见 `demo/docs/joystick.md`。

## 4. 梅花桩：深度相机学生（本轮已可用）

竞赛 PDF 修正后的几何（`map_fixed.xml` row y=12）：入口平台 0.8×0.8×0.2 m，6 个 0.2 m 方块交错两排
（排内中心距 0.6 m、排间 y 距 0.4 m、首桩距平台 0.1/0.4 m），出口平台，无斜坡，全长 3.5 m。
判分要求轮子**全程在 0.2 m 顶面上**（任一轮底 <0.05 m 即 `dropped`），到出口平台后还要下到正常地面。

结论与现状：

- **盲走无解**（历史结论，仍然成立）：本仓 5 个候选 × 2 出生点 × 4 命令 × 5 y 偏移 = 0/200；
  旧仓 13 个盲走策略 0/384。桩阵交错间隙必须靠视觉/深度。
- **本轮训练了深度学生 `plum_depth_final_v1`**：actor = 324 维 proprio history（54×6）+ 1 帧
  48×64 深度（CNN latent 64 → MLP），critic 保留特权 fine-scan；教师 = 1764 维 fine-scan 特权
  RL（`blackw-plum-teacher-s2`），学生用 privileged warm-start + 教师动作蒸馏
  （`blackw-plum-depth-s2`，脚本 `train/scripts/plum/`）。训练环境成功 98.4%（316 eps / 311）。
- **判分侧**：`plum_1` official-profile 单关 **success 200**（统一 torch 2.14.1 后；torch 2.12 会
  fail，这也是三环境统一版本的原因）。证据 `artifacts/part2_evals/torch214/plum_1.*`。
- **部署侧**：深度相机链路已接入（`sim_node` EGL 渲染 `/depth_cam` → `rl_sim` 策略第二输入，
  C++ 渲染/推理与 eval 侧逐位一致）。但**该 actor 自己的出口 0.2 m 台阶下降在部署栈是刀锋状态**：
  身体前倾翻下，判分投影 `not_descended_exit_platform`/`fell`。所以全场路线把梅花桩拆成
  “深度专家穿越桩阵 → 在出口平台（field x=6.80）切轮式专家（`vt_band_v1`）→ 倒车 0.6 s 刹车 →
  站定 2 s → 0.45 m/s 走下出口台阶”，组合 trace 判分 **success 200**
  （`artifacts/part2_route3/plum_comp_v4/v5*`；切在平台是必须的——切在桩阵上轮式专家会踩空，
  judge `dropped`）。两半都是 RL，判分只看支撑点，仍是合法一次尝试；细节见 `ROUTE.md` §6.4。
- **鲁棒性仍是刀锋**：同环境 spawn-offset 网格 **2/15**（±0.05 m/±2° 就翻车；策略确定性，
  网格测的是边界不是噪声）。官方每类允许 4 次尝试且用默认 spawn，所以判分证据成立；但
  “反复稳定越过”的备注要求未满足。**建议下一步重训**（更宽初始位姿/速度随机化 + 出口下降奖励），
  或给玩家侧加多 attempt 策略。
- 旧证据（保留作溯源）：旧仓 depth 学生 `route2_*` 只在 y 镜像几何下 knife-edge 通过；本轮新
  学生训练在修正朝向上；旧 3 桩 checkpoint/证据不再适用。

## 5. 入场对齐要求（衔接模型能否达到）

专家对入场位姿的实测容差（eval-side，spawn offset 扫描）：

| 地形 | 横向 | 偏航 | 说明 |
|---|---|---|---|
| 平台 40 | ±0.05 m | −2°..+5° | ±0.10 / −5° 失败（非单调，接触混沌） |
| 楼梯 30 | 需 dy 目标 −0.03..−0.045 附近，±0.02 m | ±1.5° | vx 0.7 的 −0.03 行全 yaw 通过；其它 dy 很窄 |
| 斜坡 35 | ±0.10 m | ±5° | 最鲁棒 |
| 沟壑 30 | ±0.10 m | ±2° | adaptive08 比 fixed08 对称（fixed 在 +2° 失败） |
| 高墙 40 | −0.10..+0.10 m | −5°..+5°（8/9） | `vt_band_v1`/`obstacle`；`row2p` 3/9 不要用 |
| 梅花桩（深度） | dy=0 / −0.05 两行中仅 dy=0、−0.05 的 yaw=0 通过 | 0°（±2° 全灭） | 同环境网格 2/15；判分只用默认 spawn；部署路线靠对齐收紧到 0.05 m/1.5° + 组合下降兜底 |

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
  --levels '{"high_platform":40,"stairs":30,"slope":35,"high_wall":40,"trench":30,"plum":1}'
#    -> total 1200 / max 1300（只缺 slope_45）
# 4) 部署栈单关复测（真实 launch；例：平台 20 自适应 0.5）
ADAPTIVE=1 VX=0.5 SECONDS_REC=34 LEVEL=high_platform_20cm POLICY=row2p_p40w40_v1 \
  OUT=$PWD/artifacts/traces/trial.jsonl bash demo/tests/run_joy_trial.sh
pixi run -e eval robocon-eval replay-part2 --trace artifacts/traces/trial.jsonl
# 5) 整场 1×（默认档 = 最终验证档，见 ROUTE.md §8）
PART1_ROUTE_JUDGE=0 ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=1 \
  route:=part2 policy_name:=vt_locked12 transfer_policy:=vt_locked12 viewer:=false \
  realtime:=true eval_state_every:=10 stop_on_fail:=true
```

证据目录：`artifacts/part2_experts/`（`summary.md`、`summary.json`、`runs/`（矩阵会重跑覆盖）、
`stairs30_basin.json`、`plum/REPORT.md`、`plum/results/`）；**六类最高档 official-profile 单关证据**
在 `artifacts/part2_evals/torch214/`（含 `score.json`）；部署栈 trace 在 `artifacts/traces/`；
整场 `artifacts/part2_route3/real_v27_*`（六类全开档，含录像/概览图）与
`plum_comp_v4/v5*`（梅花桩组合单段判分证据）。
