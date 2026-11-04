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
  不受 env knob 影响，可复现。RecoveryLoco（点式足）鲁棒性配方的轮足适配见 `docs/training/VT-RECIPE.md`。
- **验收工具**：`tools/tracking/` — F1–F7 命令门禁、支撑矩形、转弯足迹、闭环位置环 Harness、
  RecoveryLoco 式鲁棒性探针、checkpoint 一键评估脚本。
- **功能完整性证据**：新仓代码直接跑通 `train → checkpoint → export（写 manifest）→
  PolicyRunner → 全部探针`；`train/scripts/checks/check_vt_env.py` 配置/构建冒烟；
  `train/tests/test_vt_command.py` 4 项契约测试；训练 XML 只给轮碰撞 geom 补 name（无物理改动）。
- **结果与已知限制**：见 `docs/training/VT-RECIPE.md` §4.1/§7。结论：F1/F2/F3/F5/F6/F7 全过（F4 vy 按方向
  仅诊断）；原地转漂移压到 0.05–0.09 m（目标 0.15）；90° 扫掠半径 ~0.57 m（静止 ~0.34 m）；
  负向低速转向仍有死区，彻底“零额外占地”需要 stepping-turn 步态（已列为后续项）。

## 8. 复现验证（2026-11-02，pixi + 全场路线）

本轮把迁移后的仓库真正按 README 从零跑通，并修掉迁移过程中遗留的问题：

- **pixi 环境**：`pixi install -e eval`、`pixi install -e demo` 均成功（demo 2.7 GB，含 ROS 2
  Humble + MuJoCo 3.11 + CPU libtorch）；`pixi run -e eval eval-pytest`（86 passed）、
  `pixi run -e eval demo-tests`、`pixi run -e demo demo-build`、`robocon-eval --help` 全部验证。
  生成并提交单份 `pixi.lock`（不 per-env lock、不用 `--frozen`）。
- **构建修复**：`quadruped_ros2/CMakeLists.txt` 仍安装迁移时删掉的 `launch/` 目录 → 构建失败；
  已移除该安装项（唯一入口是 `rl_sar/launch/rl_sim_all.launch.py`）。
- **运行时修复**：`rl_sdk.cpp` 用旧的 `ws/src/rl_sar` 层级硬编码 robot.yaml 根；改为从
  `CMAKE_CURRENT_SOURCE_DIR` 向上搜索 `robots/<name>/robot.yaml`。`part1_auto_route.py` 的
  `_hold` 墙钟分支递归调用自己（旧仓同款 latent bug）→ 改为 `time.sleep`。
- **路线能力**：
  - 新增 `stop_on_fail`（launch 参数 / `PART1_ROUTE_STOP_ON_FAIL`，默认开）：第一段 FAIL 即
    停止路线并关闭 launch，不再卡在掉落点等外层 timeout；
  - 障碍段 PASS/FAIL 改为查询 **live `Part1JudgeV2`**（与 eval replay 同一份代码/常量），
    路线只保留排序；`PART1_ROUTE_JUDGE=0` 可退回本地 precheck；
  - high_wall 从“过 y=-6 即 PASS”改为 judge 的 support 跨越规则（四轮各自跨 y=-7 且 x 在墙宽内）；
  - 新增 `robots/blackW` 之外的 `PolicyTrack` 绑定：`LB + 十字右` 切 `vt_track_v1`，
    `transfer_policy:=vt_track_v1` 走 launch；
  - 录像：`PART1_RECORD_PATH`/`PART1_RECORD_PLAN_PATH` + `tools/video/replay_route_video.py`
    生成 800×620@25fps 的路线回放视频。
- **实测一致性 / 路线结果**（同一段路线、同一策略）：
  - run7（旧 reach 判据）：route 日志 high_wall PASS（support 4/4）、t_stairs PASS、
    bridge_blocks PASS；judge replay 同一份 trace 判 high_wall success、t_stairs success、
    bridge_blocks **fail wrong_exit_side**，总分 **200**。→ 证明旧 PASS 与 judge 不一致。
  - 已改为 live `Part1JudgeV2` 门控（run8）：high_wall judge=success、t_stairs judge=success，
    bridge_blocks 在路线停止时 judge 仍 `running` → 按 non-success 处理并触发 `stop_on_fail`；
    launch 三个进程 clean exit（sim_node/rl_sim/route 都打印 finished cleanly），不再等 timeout。
  - 录像：run7 的 `/eval_state` 渲染为 `artifacts/route/vt_track_route.mp4`
    （800×620@25fps，205 s，含路线/目标/四轮矩形叠加；命令见 `docs/OPERATIONS.md` §4.1）。
  - 路线自身仍卡在 bridge A 东台固定差速原地转（30–180°/40 s + 大漂移），已在
    `stop_on_fail` 下安全停止；这是路线/固定动作问题，不是衔接模型问题。
- **已知路线问题（非衔接模型）**：bridge A 东台的固定动作差速原地转在此环境只转
  ~30–180°/40 s 且漂移大（`spin_fixed`），换新模型 `rotate_hold` 也失败；路线在
  `stop_on_fail` 下会停在该段并给视频/日志，不阻塞其它段验收。

## 9. v2 奖励与路线交接改进（2026-11-02）

- **路线交接缺陷**（用户观察）：bridge A 东台到达后旧顺序是“旧专家发负/零速度刹车 → stop →
  切 vt_track”，旧专家在零指令下回溜 ~0.2 m 造成平台点位偏移。已改为距离目标 ~0.25 m 处
  **边前进边切换**到速度模型，由新模型主动刹车并闭环回 `(5.70, -0.45)`；不再让旧模型发零指令。
- **点式足相关奖励删得过多**：轮足没有摆动相，但“四落足点跟随/共面/不扭”的约束必须保留。
  v2 新增 `wheel_plane_level`（机身坐标四轮心 z 展布）与 `wheel_rotation_follow`（纯 yaw 时
  四轮心速度对齐刚体旋转场），并把 `body_ang_vel` 从 −0.05 加到 −0.10、`track_angular` 权重
  2.5→3.0。实现坑：reward 函数默认 `SceneEntityCfg` 不会被 manager 解析（`geom_ids=slice`），
  会静默索引全部 geom 使两项奖励为 0；已在 params 显式传 asset_cfg 并加单测。
- **基线测量**（`tools/tracking/probe_wheel_follow.py`）：旧 `vt_track_v1` 纯 yaw ±0.3 的
  `follow_rel_err=0.63/0.69`、四轮 z 展布 0.031/0.022 m；目标 ≤0.35 / ≤0.015 m。
- **v2 训练中发现的额外缺陷**：
  1. reward 函数默认 `SceneEntityCfg` 不会被 manager 解析（`geom_ids=slice(None)`），
     `wheel_plane_level`/`wheel_rotation_follow` 静默变 0（vt_v6 首次启动）；已显式传 params
     并加单测。
  2. 平面奖励若全程生效，会抑制平移速度（vt_v7_600 平移比 0.63，v1 为 0.78）；已 mask 到
     纯 yaw / 站立（用户要求针对旋转跟随，不是直线速度）。
  3. `wheel_rotation_follow` 若以“指令 wz”为目标，会把负向低 wz 的 under-turn 双重惩罚并
     鼓励更慢的转；改为以**实际机身 yaw 速率**为目标，只罚轮子相对机身的滞后/打滑，
     指令跟踪仍由 `track_angular_velocity` 负责。
  4. `probe_wheel_follow.py` 同时报告 `follow_rel_err_cmd`（对指令）与
     `follow_rel_err_actual`（对实际机身旋转），后者才是“四轮跟随机身”的直接指标。
- **v2 结果与路线验证**：最终交付 `vt_track_v2 = vt_v7 model_600`（从零 900 it 中选 600 it）：
  四轮平面 z 展布 0.031/0.022 → **0.0036/0.0023 m**，90° 足迹 0.572 → **0.520 m**，
  四轮 follow(cmd) 0.633/0.689 → 0.615/0.636；代价是支撑矩形平移比 0.781 → 0.631（偏保守）。
  vt_v7_final（900 it）足迹 0.556、位置漂移 0.063 m 但 follow 更差；vt_v8（follow 用实际 yaw）
  学会“不转”退化；vt_v9（mask 平面 + 指令 follow）平移 0.688、follow 0.84，均不如 vt_v7_600。
- 路线 handover 日志新增 `bridge A east handover: pre=... post=... drift=...`：旧顺序回溜/偏移 →
  第一版修复全速切换冲过头 drift 1.70 m → 当前“提前 0.25 m + 先慢行 0.12 m/s + 切换 + v2 回位”
  drift 0.23–0.28 m；v2 平移保守导致最后回位仍差 ~0.25 m，下一轮给 v2 速度补偿。
- 路线高墙 precheck 存在 run-to-run 方差（一次 supports_crossed=3/4 side_bypass），
  `stop_on_fail` 会安全停止；这条与衔接模型无关，是 support 几何/朝向的路线问题。


## 10. 锁轮 12-DoF 点足原型（2026-11-02）

- 新增 `blackw-vt-locked` task + `get_blackw_locked_robot_cfg` + `LockedWheelLegAction`：
  轮关节 PD 锁 0、12 腿 DoF 步行；观测/动作保持 16 维/57×6，部署端 `PolicyRunner` / `rl_sdk`
  增加 `lock_wheels` 模式（轮 action 忽略，轮 q=0 位置 PD）。
- 奖励：RecoveryLoco 核心 + `feet_air_time`/`foot_clearance`/`foot_slip`/`stance_joint`
  （hip/thigh 加权，抑制落足点向外/前后扩展）+ `wheel_lock`/`wheel_axis_level` + `gait_phase`/`foot_lift`。
- 实测：v1/v2 不迈步（轮锁滑行），v3 原地踏步，v4/v5 加 trot 相位后仍只有三足跳/单轮常抬，
  `vx=0.3` 实际 ~0.14 m/s，`error_vel_xy/yaw≈0.57/0.81`；**未达到 F 门禁/交接精度**。
  因此最终交付仍为滚轮 `vt_track_v2`；锁轮任务作为后续研究入口保留，需更长训练/步态课程。
