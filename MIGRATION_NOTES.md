# 迁移说明与规则核对记录（2026-11-02）

本仓是 **全新起点**：独立 `git init`，不引用旧工作区路径、不依赖旧仓 git 历史/tag；旧仓保持原样。
唯一保留的外部引用是上游开源项目的 provenance（`rl_sar`、nwwolf mjlab 基座、主办方资产），
见各目录 `UPSTREAM.md`。

## 1. 迁移取舍

### 进来

| 内容 | 位置 | 说明 |
|---|---|---|
| 判分核心（Part1 judge、Part2 v5 judge、scoring、trace、policy_runner） | `eval/` | 语义不动，仅做规定的修复（见 §3） |
| 判分规范 | `eval/SPEC.md` | 人读单一标准；替代旧 `contract.json` + 多份 eval 文档 |
| 训练任务（black/blackW velocity、profiles part1/part2、导出） | `train/` | 研究分支不迁 |
| 仿真运行时（rl_sar + quadrup_ros2）、机器人、选中策略配置、场景 | `demo/` | 权重不入库 |
| 官方地形/场景 + 生成器 | `terrain/` | 唯一地形来源；Part2 场景由它生成 |
| 操作/架构/规则文档 | `docs/` | 压缩到 5 篇 |
| 状态录制/策略契约/机器人清单检查脚本 | `demo/tests/` | 精简后保留 |
| 构建、录视频、状态检查工具 | `tools/` | 精简后保留 |

### 不进来（旧仓保留，无需修改）

- `eval/` 治理层：`qa/hashes/freeze/calibration/consistency/audit/matrix`、`campaign` 自适应工具、
  `part2/v2_judge`（black Part2 已不参加）、20 篇旧 eval 文档、13 个 fixtures 中的历史杂物；
- 24 篇 agent-docs、3 篇研究/审查报告、研究探针脚本（depth/HIM/foothold/PUMA/EP 等）；
- 14 GB `artifacts/`、16 GB `logs/`、67 个 campaign 目录、全部旧视频；
- 275 个旧策略权重（只保留 8 个目录的配置作为模板，本地权重点到为止，见 `demo/UPSTREAM.md`）；
- 重复的 geometry 副本（`training/.../official_terrain_spec.json` 改为读 `terrain/` 单一来源）；
- 旧 `pixi.lock`/`--frozen` 流程：新仓每环境独立解析，只钉 MuJoCo 3.11 与训练验证集。

### 可读性预算（验收标准）

- tracked 文件 ≤ ~450；`docs/*.md` ≤ 8；CLI 子命令 ≤ 5；无旧 artifacts/weights/media 入库；
- 每个目录有 README；README 的命令可直接复制执行；
- 判分逻辑只有一处来源：`eval/robocon_eval/` + `eval/SPEC.md`；
- 地形数据只有一处来源：`terrain/`（demo 场景与 train 任务都从它生成/读取）。

## 2. 规则依据（本队确认）

权威文件（`~/BiS.d/tmp/`）：`队内赛.pdf`、`terrain.zip`、`scene_terrain (1).xml`。
已核对：

- Part1 地图：`terrain/official/scene_terrain.xml` 与主办方文件 **sha256 完全一致**
  (`730ccebd…`)，斜坡为 10° → 按官方 25th **V1.0** 内容执行（不是 V2.0 的 11.3°/每侧 1 m）。
- Part2 地图：`terrain/official/map.xml` 与 `terrain.zip` 内 `terrain/map.xml` **sha256 一致**
  (`57531fe2…`)；唯 **梅花桩** 例外（规则明说提供地图的梅花桩参数有误，见 §3）。
- Part1 判分：每障碍 100、T 台阶上/下各 50、回启动区 +100；三次尝试、可跳过（跳过该障碍 0 分、
  不得回程奖励）、自动/手动同分、允许固定动作、世界位姿真值只可用于生成指令。
- Part2 判分：六类 1300 分；不计时；每障碍至多 4 次；只记最高难度；仅 RL、不得写固定动作；
  hip/thigh ≤25 N·m、calf ≤60 N·m；"上述所有障碍均需要能够反复稳定越过"（当前只作为质量指标，
  见 §4 未决）。
- 最终成绩 = (Part1 + Part2) × 70% + 答辩 × 30%。

## 3. 本次迁移包含的判分修复（旧 eval 中与规则相符性的问题）

| # | 问题 | 旧实现 | 新实现 |
|---|---|---|---|
| 1 | **梅花桩几何错误** | 直接用 `map.xml` 的 24 桩 + 斜坡段；规则明说该图梅花桩参数有误 | 按队内确认重建：**无斜坡**，入口 0.8×0.8×0.2 m 平台，**3 根** 0.2 m 方桩单排、桩心距 0.6 m、平台-桩间隙 0.1 m，出口 0.8×0.8×0.2 m 平台；直接下 0.2 m 台阶到地面；20 个官方便利关里 plum=1 档 |
| 2 | **T 台阶 100 分不可达** | 顶部后首次触地就冻结证据 → 常见只有 50 | 上一条臂（1→2→3）+ 下**另外两条臂之一**（3→2→1）即 100；同臂上下=50；只上或只下=50；漏级=失败 |
| 3 | **blackW 轮接触 geom 无名（BL-6）** | 轮子碰撞几何没有名字 → judge 收不到 `ground_feet`，2 足触地规则无法判，5 个障碍无法计分 | 给四个轮碰撞 geom 加 name（只加属性，**官方模型物理零改动**；见 `demo/UPSTREAM.md`） |
| 4 | **判分采样 20 ms** | `eval_state_every=10`，规则"任何瞬间"的接触可能漏判 | 判分跑默认 `eval_state_every=1`（2 ms）；SPEC 写明 |
| 5 | 平台/梅花桩顶面判定上界 0.20 m 过松 | `wheel_z <= top + 0.20` | 收紧为 `top + 0.05`；下界不变 |
| 6 | Part2 15–20 s 计分超时 | 规则 Part2 不计时 | 只保留安全超时（120 s），在关卡 manifest 中注明 |
| 7 | `contract.json` 混合了曲线/规则/哈希/历史 | 机器表与文档双标准 | 删除；曲线/次数进 `eval/robocon_eval/rules.py`，人读规范进 `eval/SPEC.md`，场景数据进 `terrain/` |
| 8 | 旧仓 `--frozen`/多 lock | 环境锁碎片化 | 单仓三环境独立解析，只钉 MuJoCo 3.11 + 训练验证集 |

## 4. 旧 eval 中"更严但保留"的实现（规则允许范围内，队内合理性约束）

这些不是错误，已在 SPEC 中与官方规则**分开标注**：

- Part2 六类都要"上去 + hold + 从另一侧下来"（官方只写高台上平面/迎着坡面走过/越过）；
  队内原因：实际比赛要一次启动走完全部障碍，必须下得去。
- Part2 每关 15–60 s 超时仅作自动化安全帽（官方不计时）。
- Part1 木桥必须对侧出口、砂砾坑两条短边都合法、T 台阶异臂才 100、高墙四支撑点全跨墙、
  限高杆必须穿门洞；比官方表 2 严，方向正确。
- Part2 lane ±0.7 m、footprint、墙体 y-span、fall 阈值 75° 等为队内附加约束（注释标明人为合理性）。
- 沟壑的 z 代理（轮底相对平台顶下降 ≤0.103 m 才判失败）**比官方"不触地"更严**，保留。
- Part1 摔倒恢复容差 0.30 m / 0.52 rad：保留为测量余量；队内目标是全程不摔（更严口径）。

## 5. 未决 / 后续

1. **mjdata 裁判接口**：判分器最终应接收 `mjData`（接触对、几何位姿、支撑点）而不是只吃 trace，
   以便直接检查"任何可能冲突的接触"；本轮先保持现有逻辑，只把接口/SPEC 留好（高度杆几何判定
   同样等该接口，逻辑暂不改）。
2. **Part2 反复稳定性**：规则备注要求"反复稳定越过"，当前计分只要求单次成功；后续加稳定性指标
   （独立随机化 ≥N 次）作为提交门槛。
3. **Part2 连续全场 demo**：新增一次启动走完六类的非计分演示（单关独立计分保持不变）。
4. **MPC**：Part1 固定路线先做（速度层 MPC/路径跟踪，输出 vx/wz 给现有 RL）；Part2 规则仅 RL，
   另行研究（训练期 teacher/数据分析，不进正式计分环）。
5. 旧仓不需要改动/tag/归档；新仓不引用旧仓。

## 6. 迁移完成状态（2026-11-02）

- 新仓 `/home/bis/robocon-lean`，独立 `git init`；旧仓未做任何修改/打 tag。
- 规模：`demo/` 56 MB（含本地权重 ~11 MB，不入库）、`eval/` 1.6 MB、`train/` 43 MB（42 MB 为
  blackW STL mesh）、`terrain/` 292 KB、文档 4 篇 + 各目录 README。
- Part 2：20 个官方便利关（platform 4 / stairs 4 / slope 4 / wall 4 / trench 3 / plum 1），
  全部 `seconds=120` 安全帽；`terrain/gen_*.py` 重跑后文件 sha256 不变（可复现）。
- 验证结果：`eval` pytest **86 passed**；demo 契约/出生点/场景加载全绿（9 策略、20 关、Part1
  场景 202 geoms）；`train` import + 6/6 单测；`assets/verify.py` 20/20 哈希一致；
  Part1 运行时场景与官方 `scene_terrain.xml` 几何一致（148 primitive）。
- 未完成（后续）：`pixi` 环境首次安装 + colcon build + ROS 冒烟（需联网，未在本轮执行）、
  真实 rollout 判分复跑、Part1 MPC 落地、Part2 连续全场 demo、mjdata 裁判接口、
  Part2 稳定性指标。

## 7. 新增：速度跟随衔接模型（VT，2026-11-02）

迁移后新增一个独立的训练任务与验收工具链（旧仓的 `vt_flat/vt_pose` 是迁移前产物，未直接复用权重）：

- **训练任务**：`blackw-vt`（平地）与 `blackw-vt-edge`（0.05–0.20 m 平台边缘）；
  recipe 在 `train/src/alldog_mjlab/tasks/velocity/profiles/vt/`，命令/奖励/事件全部显式写在代码里，
  不受 env knob 影响，可复现。RecoveryLoco（点式足）鲁棒性配方的轮足适配见 `docs/VT-RECIPE.md`。
- **验收工具**：`tools/tracking/` — F1–F7 命令门禁、支撑矩形、转弯足迹、闭环位置环 Harness、
  RecoveryLoco 式鲁棒性探针、checkpoint 一键评估脚本。
- **功能完整性证据**：新仓代码直接跑通 `train → checkpoint → export（写 manifest）→
  PolicyRunner → 全部探针`；`train/scripts/checks/check_vt_env.py` 配置/构建冒烟；
  `train/tests/test_vt_command.py` 4 项契约测试；训练 XML 只给轮碰撞 geom 补 name（无物理改动）。
- **结果与已知限制**：见 `docs/VT-RECIPE.md` §4.1/§7。结论：F1/F2/F3/F5/F6/F7 全过（F4 vy 按方向
  仅诊断）；原地转漂移压到 0.05–0.09 m（目标 0.15）；90° 扫掠半径 ~0.57 m（静止 ~0.34 m）；
  负向低速转向仍有死区，彻底“零额外占地”需要 stepping-turn 步态（已列为后续项）。
