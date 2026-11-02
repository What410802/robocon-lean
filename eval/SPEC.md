# 评测标准（SPEC，2026-11-02 新仓 v0.2）

本文件是**人读单一标准**；机器可读的曲线/次数/分值在
`eval/robocon_eval/rules.py`（不再有 `contract.json`）。实现与测试见
`eval/robocon_eval/` 与 `eval/tests/`；规则依据与原文核对见 `docs/RULES-BASIS.md`。

**口径分层**（全文表格都标注来源）：

- **官方** = 队内赛规则 PDF（`~/BiS.d/tmp/队内赛.pdf`，内嵌官方 25th V1.0 表 2/表 3）
  与官方地形/场地资产；
- **队内** = 本队为"一次启动走完全部障碍"而增加的**更严合理性子约束**，不是官方原文；
- **操作化** = 把官方语义变成可执行判据的测量口径（如代理量、hold 时间），可能比官方更严。

冲突时：官方 > 本规范 > 代码现状。任何语义修改必须同步 `rules.py` + 本文件 + 测试。

---

## 1. 范围

| 赛项 | 内容 | 满分 | 来源 |
|---|---|---|---|
| Part 1 足式障碍赛 | 8 个障碍，手动遥控/允许固定动作，3 次整场尝试 | 900 | 官方成功定义 + 队内赛制 |
| Part 2 越障能力测试 | 6 类障碍 × 官方难度，仅 RL，每关最多 4 次 | 1300 | 官方图/表 + 队内场景 |
| 总分 | (Part1 + Part2) × 70% + 答辩 × 30% | — | 队内赛 |

判分只服务队内赛（纯仿真）；全国赛的单场 210 s / 犯规罚分不混用。本模式不判答辩。

## 2. 官方规则 vs 队内更严约束

| 项 | 官方 | 队内实现 | 性质 |
|---|---|---|---|
| Part1 T 台阶分值 | 表 3：上/下 75/50（自动/遥控） | 上/下各 50，两侧都过 100 | 队内确认口径（2026-10-27） |
| Part1 砂砾坑 | 从 1 m 短边**进入或离开** | 进入**和**离开都必须是不同短边 | 队内更严 |
| Part1 木桥 | 从一侧平台经桥面到另一侧 | 入口/出口侧相对、桥面接触 ≥3 帧、两侧平台有接触证据 | 队内更严 |
| Part1 高墙 | 上方跃过或攀爬通过 | 4 个支撑点投影都必须在墙 x 跨度内跨墙平面 | 队内更严（纯化） |
| Part1 限高杆 | 任何姿态从下方穿过、横杆不掉 | 必须从门洞 x 范围穿过；高度暂用 base-z 代理 | 队内 + 操作化 |
| Part1 T 台阶同臂上下 | 未规定 | 同臂上下 = 50；只上/只下 = 50；漏级 = 失败 | 队内（2026-10-27） |
| Part2 终端态 | 越过/通过障碍 | 六类都要求"上去 → hold → 从另一侧下去回到正常地面" | 队内（一次启动全程） |
| Part2 车道 | 无 | base/支撑点 \|y\| ≤ 0.7 m + 0.02 m 容差，0.6 s 宽限，越界即 `footprint_forbidden` | 队内合理性 |
| Part2 高墙 | 跃过或攀爬 | 每个轮子跨墙平面时 y 必须在墙实际跨度内；越侧即失败；不加 z 阈值 | 队内（训练 transfer 一致） |
| Part2 沟壑 z 代理 | "不触地" | 轮底相对平台顶下降 > `min(0.5·depth, 0.103)` 即失败（当前 0.103 m） | 队内更严（保留，见 §8） |
| Part2 计时 | 不计时 | 不计分；每关 `seconds=120` 只是**安全帽**（防挂死） | 操作化 |
| Part2 稳定性 | "反复稳定越过" | 当前**不计分**（质量指标，见 §7） | 未来项 |
| Part2 策略 | 仅 RL，不得写固定动作 | 代码 `part:=2` 禁用固定动作键 | 官方 |
| 传感器 | IMU/编码器/相机/雷达；禁止 body_lin_vel、heightmap、foot force/height、body height 等 | 训练/部署只用允许量；世界位姿真值仅可生成指令 | 官方（人工核查） |
| 力矩 | hip/thigh ≤25 N·m，calf ≤60 N·m | 训练/部署/评测三方一致 | 官方（人工核查） |

## 3. Part 1 判定与计分

### 3.1 硬件与赛制

- 场地：`demo/robots/<robot>/scene_part1.xml`（官方几何 148 primitive，与
  `terrain/official/scene_terrain.xml` 一一对应；仅补语义命名与限高杆可动体）。
- 机型：black（12 DoF 足端）/ blackW（16 DoF 轮端），判据语义相同，支撑点类型不同。
- 3 次整场尝试；**恢复不消耗**尝试，开始/主动重跑/操作跳过消耗；跳过障碍该障碍 0 分
  且不触发回程奖励。
- 失败在尝试内可重试：离开区域后重新武装，成功为最终态。
- 计时在返回启动区且机器人稳定后停止（线速度 ≤0.15 m/s、倾角 ≤10°、持续 0.5 s）。
- 恢复姿态容差 0.30 m / 0.52 rad：**测量余量**（队内目标是全程不摔）。

### 3.2 八个障碍的成功定义（官方表 2 + 队内细化）

| 障碍 | 成功（判据要点） | 主要失败 |
|---|---|---|
| 直角绕杆 slalom | 任一圆形必达区出发，按序 S 形绕过每根竖杆，经拐角必达区到另一侧必达区 | 漏杆/乱序/未过拐角/非 S 形 |
| 砂砾碎木坑 gravel_pit | 从一条合法 1 m 短边进入、从另一条合法短边离开，坑面有接触证据 | 长边/缺口边进出、2+ 支撑点触地 |
| 高墙 high_wall | 支撑点全部越过墙平面，且每个支撑点跨墙时 x 在墙实际跨度内 | 侧面绕行、未全部越过 |
| 限高杆 height_bar | 在门洞 x 范围内从下方连续穿过（base-z ≤ 横杆高度代理），横杆未掉落 | 横杆掉落、未从门洞穿过 |
| T 台阶 t_stairs | 一条臂 1→2→3 上、**另一条臂** 3→2→1 下（每条臂每级都有接触证据） | 2+ 支撑点触地、漏级 |
| 斜坡 slope | 坡面语义区内沿 3 m 长边投影累计 ≥1.0 m（方向不限） | 距离不足、2+ 支撑点触地 |
| 木桥 A bridge_blocks | 平台→方墩桥面→对侧平台连续通过（≥3 帧桥面接触、两侧平台均有证据） | 掉桥、同侧进出、2+ 触地 |
| 木桥 B bridge_bands | 同上（窄带桥面） | 同上 |

### 3.3 计分

- 每障碍 100 分；T 台阶：异臂上下 100 / 同臂上下 50 / 单向 50；其余 0/100。
- 8 项**全部满分**且返回启动区：+100 回程奖励（任何部分分/跳过都不触发）。
- `attempt_score = Σ 障碍分 + return_bonus`（0..900）。
- 最终分 = 非跳过尝试的最高分；同分取**最快有效时长**（`scoring.part1_session_summary`）。

## 4. Part 2 判定与计分

### 4.1 任务目录

任务 = (类别, 官方难度)。20 个官方关卡（高台 4 / 楼梯 4 / 斜坡 4 / 高墙 4 / 沟壑 3 /
梅花桩 1），场景由 `terrain/` 生成：`demo/scenes/part2/part2_official_levels.json` +
`demo/scenes/part2/generated/part2_off_<key>.xml`。judge 参数唯一来源是该 manifest
的 `judge` 块（评测代码不另存阈值）。

### 4.2 官方评分曲线（`rules.py` 同源，档位间线性、floor）

| 类别 | 档位 → 分数 |
|---|---|
| 高台 high_platform | 10/20/30/40 cm → 10/50/100/200 |
| 楼梯 stairs | 10/15/20/30 cm → 10/50/100/200 |
| 斜坡 slope | 20/30/35/45° → 10/50/100/200 |
| 高墙 high_wall | 10/20/30/40 cm → 0/50/100/200 |
| 沟壑 trench | 10 cm=0，30 cm=300（20 cm = 150） |
| 梅花桩 plum | 通过 = 200 |

类别分 = 成功过的最高难度对应分；总分为六类之和（满分 1300，floor）。

### 4.3 终端态（"越障"而不是"站上障碍"）

每关独立场景、出生点 x=0、障碍在 x=1.8 m 外、通道中线 y=0。成功 = 同时满足：

1. **到达/上障碍证据**：四轮到达目标支撑面并保持 `hold_s`（0.2 s；判据再加 0.1 s 稳定余量）；
2. **离开/下障碍证据**：平台下台、楼梯下完、斜坡下坡、高墙/沟壑/梅花桩到另一侧正常地面；
3. **正常状态**：回到正常地面后再稳定 hold；只在障碍上停住不算成功。

各类别终端（判据参数见 manifest）：

| 类别 | 成功终端 |
|---|---|
| 高台 | 四轮在台顶（`wheel_z ∈ [top-0.06, top+0.05]`）hold → 越过远边、四轮回到台后地面（z≤0.20）hold |
| 楼梯 | 四轮到顶平台（`goal_x_min` 后）→ 下完末级、四轮回到 `descend_end_x+0.10` 之后地面并 hold |
| 斜坡 | 四轮在坡顶（rise_z 带）hold → 继续前进下坡，四轮回到 `exit_x_min` 后地面 hold |
| 高墙 | 每个轮子跨 `wall_x` 时 y 在墙跨度内（±0.7 m）→ 全部越墙、回到墙后地面（`wall_x+0.16` 之后）hold |
| 沟壑 | 跨槽 → 到远侧平台（四轮在平台顶）→ 下完出口坡，四轮回到 `exit_ramp_end` 后地面（z≤0.08）hold |
| 梅花桩 | 跨桩区（轮底不低于 `z_cross_min=0.05`）→ 到出口平台（0.2 m 带）→ 从出口平台远边下到正常地面，四轮回到 `exit_ramp_end` 后地面 hold |

**梅花桩几何（owner 2026-11-02 确认，非旧 map.xml）**：无斜坡；入口平台 0.8×0.8×0.2 m
（x 1.8–2.6）、3 根 0.2 m 方桩单排（桩心 x 2.8/3.4/4.0，边 2.7–4.1，间隙 0.4 m）、
平台-首/末桩间隙 0.1 m、出口平台 0.8×0.8×0.2 m（x 4.2–5.0），直接从出口平台下 0.2 m
台阶到地面。manifest 中 `exit_ramp_end = 5.0`（保留了历史字段名，语义为出口平台远边）。

### 4.4 判分证据与失败原因

- judge 输入（`Part2OfficialJudge.update(t, base, tilt_deg, wheel_x, wheel_y, wheel_z)`）：
  躯干位姿/倾角、四轮**几何最低点**世界坐标；不读策略内部/训练 reward。
- 失败原因枚举：`fell:*`（倾角 >75° 或 base z <0.15 m）、`footprint_forbidden`（车道越界）、
  `wall_side_bypass`、`dropped:*`（梅花桩轮底 <0.05 m）、`gap:wheel_in_slot:*`、
  `missing_support_evidence`（四轮证据缺失 → fail-closed），以及超时/未完成诊断原因
  （`not_exited_platform`、`not_descended` 等）。
- fall 优先于车道/越界类失败；成功前车道宽限 0.6 s 仅用于避免擦碰误判。

## 5. 尝试 / 采样 / 超时

| 项 | 值 | 说明 |
|---|---|---|
| Part1 整场尝试 | 3 | 恢复不消耗；开始/重跑/跳过消耗 |
| Part2 每关尝试 | 4 | 成功一次即该关完成；只记最高成功难度 |
| 判分采样 | **2 ms/帧** | 判分跑 `eval_state_every:=1`；trace 用 2 ms 帧（规则"任何瞬间"） |
| 兼容采样 | 20 ms/帧 | 旧录制仍可 replay：judge 用 `t` 差分推进 hold，不按帧数 |
| Part2 计时 | 不计分 | 关卡 `seconds` 只是安全帽（≥120 s），不是 15–20 s 计分上限 |
| 物理/控制/策略 | 2 ms / 5 ms / 20 ms | 官方 MJCF + `robot.yaml`；三环境一致 |
| 确定性 | 固定环境 pin | MuJoCo 3.11、CPU torch；同输入必须同判据结果 |

`policy_runner` 记录 20 ms（策略周期）帧；部署栈录制才用 2 ms。replay 对两者都给同一判定。

## 6. 人工合规核查（非自动判分）

以下无法从 trace 自动证明，发布成绩前**人工核查并记录**：

1. **传感器/真值**：输入仅 IMU、关节编码器、相机（本队主用深度相机；雷达不用）；无
   body_lin_vel、heightmap、foot force/height、body height、特权地形扫描；世界真值只用于生成指令。
2. **Part2 仅 RL**：无固定动作脚本参与正式跑分；`part:=2` 固定动作键禁用已由 demo 验证。
3. **力矩**：hip/thigh ≤25 N·m，calf ≤60 N·m（训练/部署/评测一致）。
4. **Part1 遥控/自动**：手动与固定动作同分，无额外加分。
5. **场景/权重哈希**：policy.pt、config.yaml、scene XML、manifest sha256 记录在 trace/产物 meta。

## 7. 已知缺口 / 路线图

- **Part2 "反复稳定越过"**：官方备注要求，当前计分只记单次成功；后续用独立随机化 ≥N 次
  失败率作为提交门槛（`rules.py` 未混入 4 次计分尝试）。
- **mjdata 级裁判接口**：未来 judge 直接读 `mjData`（接触对/几何位姿/支撑点），用于
  "任何瞬间"接触与限高杆几何净空；本轮保持 trace 输入与现有逻辑不变（接口已在 SPEC 留位）。
- **Part1 高度杆 / 高墙**：仍用 base-z 代理（`not_under_bar` 用横杆高度）；mjdata 几何净空
  是计划项，改口径需升版本并重跑全部测试（当前不改，避免分数漂移）。
- **梅花桩/沟壑 z 代理**：比官方"不触地"更严（保留）；接触分类（槽底/槽壁）留作 v9 诊断。
- **Part2 trace 倾角**：部署 `/eval_state` 当前不含 tilt/quat 时 replay 退化为 base-z 跌倒检测；
  recorder 增补 tilt 后无需改判据。

## 8. 资产 / 溯源 / 可复现

| 输入 | 位置 | 校验 |
|---|---|---|
| 队内赛规则 PDF | `~/BiS.d/tmp/队内赛.pdf` | 2026-10-20，4 页；表 2/3 为图片 |
| Part2 官方地形 | `terrain/official/map.xml`（= terrain.zip 内 map.xml） | sha256 `57531fe2…` |
| Part1 官方场地 | `terrain/official/scene_terrain.xml` | sha256 `730ccebd…` |
| Part2 关卡 manifest | `demo/scenes/part2/part2_official_levels.json` | 由 `terrain/gen_part2_levels.py` 生成 |
| Part2 单关场景 | `demo/scenes/part2/generated/` | 每关一个 XML，include 官方 blackW |
| Part1 判分场景 | `demo/robots/<robot>/scene_part1.xml` | 与官方几何一一对应，仅命名/可动杆 |
| 旧仓冻结证据 | 旧仓（只读，不再引用） | 旧 contract/FREEZE 仅作追溯 |

复现要求：同一规范版本 + 同输入哈希 + 同环境 pin（MuJoCo 3.11、CPU torch）必须产生相同
成功/失败/分数/原因；不稳定则标 `flaky` 并定位，不得静默取最好值。

## 9. 快速对照（实现入口）

| 内容 | 代码 |
|---|---|
| 曲线/次数/分值/合规常量 | `eval/robocon_eval/rules.py` |
| 分数聚合（floor、best-of） | `eval/robocon_eval/scoring.py` |
| Part1 事件判据 + 尝试协议 | `eval/robocon_eval/part1/{judge,course,geom_tags}.py` |
| Part2 终端判据 + 关卡元数据 | `eval/robocon_eval/part2/{v5_judge,official}.py` |
| trace replay | `eval/robocon_eval/{trace,trace_part2_v5}.py` |
| 独立策略 runner | `eval/robocon_eval/policy_runner.py` |
| CLI | `eval/robocon_eval/cli.py`（`run` / `replay-part1` / `replay-part2` / `score`） |
| 测试 | `eval/tests/`（`pixi run -e eval eval-pytest`） |
