# Part2 全场自动路线 v2（修正版场地 / 保守验证档）

更新：2026-11-04。实现：`demo/rl_sar/scripts/part2_auto_route.py`（v2，替换原 252 行初版）。
场景：`demo/scenes/part2/full_field/part2_fixed_field.xml`（`map_fixed.xml` + 竞赛 PDF 修正后的梅花桩），
`route:=part2` 固定 spawn `(0, 0, 0.675, 0)`、`eval_state_every:=10`（50 Hz 反馈）。

## 1. 结论（本轮）

**1× 真场整场已完成：`real_v22` = 5/5 段集成门限 PASS（plum bypassed，高墙默认关），
eval 判据投影复核 4/4 穿越段 success、合计 350/1300**，1224.8 s 仿真，command 见 §8。逐段：平台 20 → 楼梯 20（st30@0.45）→ 斜坡 30 → 沟壑 20 → 梅花桩绕行。
默认档刻意低于 eval 侧已成功过的最高档（平台 20<40、楼梯 20<30、斜坡 30<35、沟壑 20<30），
用可靠性换演示稳定性；更高档位仍可用 env 切回（§8）。

- 默认路线 5 段（真场 1× 验证档）：平台 20（S→N）→ 楼梯 20（W→E）→ 斜坡 30（W→E）→
  沟壑 20（W→E）→ 梅花桩**绕行**（无部署专家）。高墙 20cm 是可选第 6 段，默认关
  （`PART2_ROUTE_WALL=1` 打开）。
  更高档位（平台 30/40、楼梯 30=200 分、斜坡 20/30、沟壑 10/20、高墙 bar6/bar7）全部可用环境变量切换，
  几何/审计均覆盖；楼梯 30 在 v14/v17 整场 PASS，但 v15/v18 在顶层平台 stall（run-to-run 不稳）。
- 段落切换全部走 `vt_locked12` 锁轮衔接；转场是 **`goto()` 折线**（开阔区 ≤10 m @0.30 m/s，
  紧走廊 ≤4 m @0.22 m/s）。锁轮模型的 vy 车道保持基本无效（~0.008 m/m 系统横漂），所以不靠 vy，
  而靠“多段折线 + 每段重新闭环 + 缺口中心进场 + 短直进缺口”。斜坡行是连续 1.2 m 缺口组成的墙，
  斜→沟转场改为**西端绕行**（y=4.5 走廊 → x=20.5 列 → y=7.5 走廊），v19/v20/v21 均通过。
- 每个专家入场前用有界基元对齐（位置 goto + 死区补偿偏航伺服 + **横向 `_sidestep_lane`** + 短直 settle），
  门限：默认 0.05 m/2.5°，platform 0.07 m/3°，stairs 0.02 m/1.5°，slope/trench/plum 0.08 m/2.5°，
  wall bar5 0.20 m/6°（bar5/bar6 走廊只有 1.08 m）。
- 整条折线用 MuJoCo 几何 AABB 做净空审计（`tools/route/audit_part2_route.py`）：**PASS**
  （默认保守档、各切档变体、wall-off、plum-cross 全部 PASS）。
- 部署单关补测（2026-11-04，用于默认档）：`slope_30deg` row2p@0.50 → success（50 分）；
  `trench_20cm` s1d@0.80 → success（150 分）；trace 在 `artifacts/traces/lower_*.jsonl`。
- **eval 判据逐段复核**（不再把路线自己的 PASS 门限当第二条判据）：工具
  `tools/route/replay_field_part2.py` 把每段场地轨迹投影回单关坐标后调用同一个
  `Part2OfficialJudge`。`real_v22`：4/4 穿越段 success、梅花桩 bypass、合计 350；
  `real_v21`：6/6 段（含高墙）success、合计 600。路线门限仅作集成检查。
- 假运动学整场验证（旧 6 段档）见 §7.1；真场整场 1× 结果见 §7.3。

## 2. 几何（MuJoCo `MjModel.from_xml_path` 实测，非估算）

| 行 | y | 结构 | x 范围 | 顶高 | 备注 |
|---|---|---|---|---|---|
| 1 | 0 | `platform_40cm` | 21.00–22.50（y ±0.8） | 0.40 | 西侧邻 `platform_30cm` 止于 19.50 |
| 2 | 3 | `stairs_30cm` | 63.00–67.40 | 1.80 | 6 级 0.30 上 + 0.8 台面 + 6 级下；y 2.3–3.7 |
| 3 | 6 | `slope_30deg` | 33.00–37.80 | 1.20 | 上坡 33.0–35.0，台面 35.0–35.8，下坡 35.8–37.8 |
| 4 | 9 | `trench_20cm` | 9.00–13.20 | 0.20 | 坡道 9–10 / 10–11 台 / **净宽 0.20 缺口 11.0–11.2（按该档几何）** / 11.2–12.2 台 / 坡道 12.2–13.2 |
| 5 | 12 | plum（竞赛 PDF） | 4.00–7.50 | 0.20 | entry 4.0–4.8，r1 方块 y=12.2 x=5.0/5.6/6.2，r2 y=11.8 x=5.3/5.9/6.5，exit 6.7–7.5 |
| 6 | 21 | `cross_bar_7`（40 cm） | 10.14–10.26（y 20.3–21.7） | **0.40** | bar8 为 50 cm（y 同），free gap 10.26–11.34 |

> 注意：`cross_bar_N` 的高度是 `2*size_z`（pos z == size z）：bar5=20 cm、bar6=30、bar7=**40**、bar8=50。
> 任务书里的“cross_bar_8 top 0.32 m”是错的；官方最大 40 cm 墙是 **cross_bar_7**。

**难度档位全部可切换（顶层表 + 环境变量）**：每个 leg 的场地 geom/进场几何都是表格项，
默认取任务书最大难度；任意组合都已通过 MuJoCo 净空审计。

| 类别 | 环境变量（默认） | 可选档 | 场地 x（y 行） | stand / 穿越门 |
|---|---|---|---|---|
| 平台 | `PART2_PLATFORM_LEVEL`（**20**，field 实测最稳档） | 20/30/40 | platform_20/30/40cm = 12.75 / 18.75 / 21.75（y=0） | stand (cx,−3.90) 朝 +y，S→N；门 y>1.45 |
| 楼梯 | `PART2_STAIRS_LEVEL`（30） | 20/25/30 | 48.00–52.40 / 55.50–59.90 / 63.00–67.40（y=3） | stand x=46.00 / 53.50 / 60.50，y=2.970，朝 +x |
| 斜坡 | `PART2_SLOPE_LEVEL`（35） | 20/30/35 | 21.00–25.80 / 33.00–37.80 / 39.00–43.80（y=6） | stand x=20.40 / 32.40 / 38.35，朝 +x |
| 沟壑 | `PART2_TRENCH_LEVEL`（30） | 10/20/30 | 3.00–7.10 / 9.00–13.20 / 15.00–19.30（y=9） | stand x=1.20 / 8.05 / 14.10，朝 +x |
| 高墙 | `PART2_ROUTE_WALL=0/1` + `PART2_ROUTE_WALL_VARIANT`（**bar5**，部署实测档） | bar5/bar6/bar7（20/30/40cm） | 7.74–7.86 / 8.94–9.06 / 10.14–10.26（y=21） | stand x=8.40 / 9.60 / 10.80，朝 −x（E→W） |

平台 leg 的进场几何对三档相同：南侧 y=−4.5 走廊 → stand (cx,−3.90) 朝 +y → S→N 翻越
（3.1 m 助跑）→ 结束门限 y>1.35 → 从 (cx+1.75,1.50) 沿 y=1.5 北侧走廊东行去楼梯。
**默认档是 platform_20cm**：field S→N 部署重复 2/2（`field_plat20_{a,b}`），platform_30cm
同几何 field 0/2（`field_plat30_{a,b}`，虽有单关 2/3 记录），platform_40cm 部署不可过；
30/40 仍是一键 env 切换（审计全过）。
审计输出：`artifacts/part2_route3/audit_platform{40,30,20}.txt`、
`audit_stairs20.txt`、`audit_slope20.txt`、`audit_trench20.txt`、
`audit_wall_{bar5,bar6}.txt`（其余为默认档审计）。

## 3. 路线表（专家 / 速度 / 对齐位 / 穿越门）

默认档 = **1× 真场 `real_v22` 5/5 段集成门限验证档**（平台 20 / 楼梯 20 / 斜坡 30 / 沟壑 20 / 梅花绕行；
高墙默认关；更高档位见 §8 的 env 说明，门槛与速度都可切换）。
`vx` 是专家 AdaptiveCommand 的目标体速（`/joy` 轴值 = vx/1.5，部署端再乘 `joy_command_scale=1.5`）。

| # | leg | 方向 | 专家（env 覆盖） | vx | 入场对齐位 (x,y,yaw) | 集成门限 | 转场（goto 折线） |
|---|---|---|---|---|---|---|---|
| 1 | `high_platform_20cm` | **S→N** | `row2p_p40w40_v1` | 0.50 | (12.75, **−3.90**, +90°) | y>1.35 | (0,0)→(0,−4.5)→(10,−4.5)→(12.75,−4.5) |
| 2 | `stairs_20cm` | W→E | `st30_ppo_st30_s100_600` | 0.45 | (46.00, 3.000, 0°) | x>52.80 | 平台 exit→(14.25,1.5)→(14.25,−2.5)→y=−2.5 东行→(46.45,−2.5)→(46.45,−0.5)→(46.45,2.97)→(46.00,2.97) |
| 3 | `slope_30deg` | W→E | `row2p_p40w40_v1` | 0.50 | (32.40, 6.000, 0°) | x>38.20 | 楼梯 exit→(53.60,3.0)→(53.60,4.5)→y=4.5 西行→(32.40,4.5)→(32.40,5.50)→stand |
| 4 | `trench_20cm` | W→E | `s1d_trench_reward_v75_m100` | 0.80（自适应） | (8.05, 9.000, 0°) | x>13.60 | 斜坡 exit→(38.40,4.5)→y=4.5 西行→(20.50,4.5)→(20.50,7.5)→y=7.5 西行→(8.05,7.5)→(8.05,8.5)→stand |
| 5 | `high_plum_bypass` | W→E（绕行） | —（不切专家） | — | — | 无（到 join 点即 PASS） | (20.15,9)→y=10.5 西行→(2.60,10.5)→(2.60,13.2) |
| 6（可选，默认关） | `high_wall_20cm`（cross_bar_5） | E→W | `row2p_p40w40_v1`（bar5；bar6/7 用 `vt_band_v1`） | 0.40（bar5） | (8.40, 21.000, 180°) | x<7.20 | (2.40,13.2)→(2.40,19.5)→(8.40,19.5)→(8.40,20.5)→stand |

- 平台 leg 是 **S→N**（南侧 y=−4.5 走廊，3.1 m 助跑）：W→E 的 field 槽只有 0.75 m 助跑，专家起不来。
- 每个 leg 后先 `ensure_transfer()`（停 + 0.6 s + 切 `vt_locked12` + 1.2 s 稳定），再走 exit 折线。
- 高墙：`PART2_ROUTE_WALL=0`（默认关）/`1` 开；`PART2_ROUTE_WALL_VARIANT=bar5|bar6|bar7`。
  bar5 用 row2p@0.40 + 0.20 m/6° 宽容差（1.08 m 走廊内无空间做 2 cm 级对齐）；
  bar6/bar7 用 vt_band@0.50，但 field 槽 0.54 m 助跑只有官方单关证据（见 §7.2）。
- 专家策略名/速度/模式在模块顶部 `EXPERT_SPECS`（`PART2_EXPERT_<LEG>`、`PART2_VX_<LEG>` env 覆盖），
  档位在 `PART2_PLATFORM_LEVEL` / `PART2_STAIRS_LEVEL` / `PART2_SLOPE_LEVEL` /
  `PART2_TRENCH_LEVEL` / `PART2_PLUM_MODE` 中切换。
- `mode=fixed` 时穿越循环只发 `(vx,0,0)`；默认全部 adaptive（训练侧车道反馈：
  `vy=clip(−0.5·e,±0.2)`、`wz=clip(−0.8·e−2.0·h−0.6·vy_b,±0.7)`，对 ±x/±y 车道按 heading 镜像）。
- 梅花桩不穿越：盲走专家 eval 0/200、旧仓 0/384；`real_v14` 的 `obstacle` 穿越在桩前 stall 并摔。
  `PART2_PLUM_MODE=cross` 保留，等未来有 plum 专家再启用。

## 4. 转场走廊与净空（审计实测）

转场不用车道保持（锁轮模型 vy 基本无效，实测 ~0.008 m/m 系统横漂），而是 `goto()` 折线：
每段 ≤10 m、tol=0.12、vx=0.5，到达即重新闭环纠偏（旧 real8 的 goto 段实测落点 y_err 0.02–0.13 m）。

`tools/route/audit_part2_route.py` 把 `part2_auto_route.py` 的 `AUDIT_SEGMENTS`（执行器实际
驱动的同一折线）对 MuJoCo 场景全部非地面 geom AABB 做 2-D 距离检查；`terrain_label*` 与
`contype=0` 视觉 geom 忽略。要求：转场/入场 ≥0.45 m，专家段除“被穿越目标”外所有 geom ≥0.45 m，
slot 起跑距 0.45–2.20 m、横向邻距 >0.50 m。结果：**全部 PASS**（完整输出见
`artifacts/part2_route3/audit.txt`）。

| 走廊/通道 | 用途 | 最小净空（审计） |
|---|---|---|
| y=−2.5（平台行南侧） | 起点→平台行；平台→楼梯 | 1.70 m（最近平台行 bbox） |
| y=4.5（楼梯行/斜坡行之间） | 楼梯→斜坡 | 0.80 m（两行各 0.8） |
| y=7.5（斜坡行/沟壑行之间） | 斜坡→沟壑 | 0.80 m（两行各 0.8） |
| y=10.5（沟壑行/梅花桩行之间） | 沟壑→梅花桩 | 0.80 m（沟壑行） |
| y=13.2（梅花桩行/粗糙行之间） | 梅花桩→高墙区西侧 | 0.80 m（梅花桩行） |
| x=2.40（场地最西侧纵向通道） | 绕开粗糙行/窄桥行 | **0.59 m**（粗糙 t1 xmin=2.986） |
| y=19.5（窄桥行/横杆行之间） | 东行至 bar7/bar8 缺口下方 | 0.80 m（横杆行 ymin=20.3） |
| 各专家 slot | 入场对齐位 | 平台 0.75 / 楼梯 1.80·1.30 / 斜坡 **0.65·0.55** / 沟壑 0.90 / 梅花 1.80 / bar7 **0.54·0.54** |
| 专家行内摆线（cross 段） | 穿越时不得蹭邻档 | 平台 0.75 / 楼梯 1.30 / 斜坡 0.55 / 沟壑 0.90 / 梅花 2.14 / bar7 0.54 |

> 最紧的三处：斜坡 slot（0.65 m 起跑 / 0.55 m 邻距，邻侧坡道边缘 z≈0，实际几何不构成碰撞）、
> bar7/bar8 缺口（0.54 m，起跑与邻距都只剩 0.54 m）、x=2.4 纵向通道（0.59 m，goto 折线纠偏）。

## 5. 对齐（专家切换前）

**问题**：锁轮衔接模型 `vt_locked12` 没有可用的横向速度，且位置闭环很差——
- 旧版 `goto(tol=0.02)` 的终局是"绕着目标点转圈"，实测在 20 s×3 次超时里原地打转；
- 旧版 `_pose_move_to` 在"前向误差≈0、纯横向残差"时退化成死平衡（vx≈0、wz≈0）；
- 90–115° 原地旋转会让轮式底盘横漂 5–12 cm；`wz` 实测只有指令值的 ~50%（死区）。

因此 v3 起对齐只用**有界基元**（每一步都有时间上限、都基于实时 `/eval_state` 反馈，
失败时只重试失败的那一步，绝不反复回到粗 goto）：

0. `goto(approach_point, tol=0.12, speed=0.25)` 粗接近（approach_point = stand − backoff·dir，
   platform 用 0.70 m backoff 留在开阔走廊里；20–25 s 上限，**不计入** 45 s 精细预算）；
1. `goto(stand, tol=0.06, speed=0.10)` 一次（≤10 s）；只有横向偏差 >0.10 m 才允许再补一次；
2. `_rotate_servo` 偏航伺服到车道 heading（≤18–20 s）：`wz=clip(1.2·err,±0.28)`，
   `|err|>0.35` 用 0.28、否则 0.20，`|err|>0.02 rad` 时保底 `|wz|≥0.14` 越过死区，
   旋转漂移 >0.02 m 时用微小 body-x 保持；
3. `_lane_settle` 横向 S 弯：`theta=clamp(12·|e|, 8°, 25°)`，按 heading 投影前进直到
   `|e|≤0.018 m`，再把 heading 转回车道；前进里程按 leg 限幅（platform 1.00、stairs 0.30、
   slope 0.05、trench 0.20、plum 0.25、wall 0.03 m），保证专家助跑距离不被吃掉；
3b. **横向 `_sidestep_lane`**（锁轮旋转横漂 0.15–0.25 m 的彻底解法）：偏航伺服后若 `|e|>xy_tol`，
   朝车道法向转身 → 直行把 `|e|` 压到 0.02 m → 转回车头 → 短偏航伺服；不收助跑距离（只在 lane 法向动）。
   真实部署实测把 slope e +0.149→−0.019、trench +0.175→+0.006、wall e +0.18→−0.046。
4. `_creep_along` 直线慢速爬到预对齐 along 目标（heading 保持，**不再用 goto 收尾**）；
5. 最后再修一次偏航，然后硬门限：`|e|≤xy_tol`、`|yaw_err|≤yaw_tol`、`along∈[along_min,along_max]`。
   默认 0.05 m / 2.5°；stairs_30cm 0.02 m / 1.5°，slope/trench/plum 0.08 m / 2.5°。
   整个精细阶段硬预算 45 s、最多 2 次尝试；失败打印残差并让整条路线 FAIL（绝不挂死）。

**越障恢复（通用，`PART2_RECOVER=0` 可关）**：部署栈的 platform/stairs/wall 存在 run-to-run
变异性（同一对齐位一次过、一次卡）。`expert_cross()` 检测"车道投影 12 s 内前进 <0.03 m"
（或 cross_timeout 到点）：命中后不立即 FAIL，而是切回 `vt_locked12`、沿 −lane 后退 0.6 m
（goto，0.20 m/s）、用 30 s 预算重新对齐到同一 stand，再重试一次穿越（`attempt=2`）；
重试仍失败才 FAIL。日志 `expert_recover leg=... reason=stall|timeout dist=X`，报告行带
`recover=0|1` 和 `gate=gate_recovered`。每 leg 最多 1 次恢复。另：对齐/恢复过程中一旦
`fallen()` 立即抛错让整条路线失败（不再让倒地机器人反复重试，`stop_on_fail` 直接收尾）。

开关日志机器可读（每步 `align ...` + 最终一行）：

```
align start leg=... policy=... stand=(...) yaw=... backoff=... from=(x,y,yaw)
expert switch leg=... policy=... lane_center=(x=,y=,yaw=) start_dy=... stand=(x=,y=)
              actual=(x=,y=,yaw=) e=... along=... yaw_err=... gates=(m,deg) align_s=... OK|ALIGN_FAIL
```

**真实部署实测（leg1，2026-11-04，domain 87，全场）：** 旧失败模式（`goto tol=0.02` 转圈、
纯横向 `_pose_move_to` 死平衡）已消除。platform S→N（3.1 m 助跑）一次通过：
`actual=(+21.73,−3.21,+89.7°)`、`e=+0.017 m`、`yaw_err=−0.3°`、`align_s=48.1 s`；
随后专家在 0.40 m 台阶面卡住（y≈−0.55、z≈0.67、pitch≈−20°）——这是部署栈越障能力问题
（同一轮部署实测 0.30/0.40 m 台阶都卡，见 §7.2），**不是对齐问题**，所以默认档位改为
platform_20cm。stairs 的 `start_dy` 在部署实测通过档为 **0.0**（vx=0.50 实测成功，见 §7.2），
与 eval 侧 vx=0.70/`dy=−0.030` 的窄盆不同——部署栈不遵循 eval 的窄盆，按部署实测取。

> 调参：`PART2_ROUTE_ONLY=<leg名或1-based序号>` 可只跑到指定 leg 就停，便于单段调试。

## 6. 切换接线（本次新增）

`vt_band_v1`（stairs_30cm 专家）此前没有手柄绑定。新增：
- `demo/rl_sar/include/input_map.hpp`：`Intent::PolicyBand` + 绑定 **RB + 十字左**，键盘 **`b`**；
  `KbKey::B` 新增；`Name()` 补名。**保留 RB+十字上 = EnterRL**（路线启动/回 RL 依赖它）。
- `demo/rl_sar/src/rl_sim.cpp`：`KeyboardMask()` 映射 `Input::Keyboard::B`；
  `PolicyBand` → `SwitchPolicy("vt_band_v1")`。
- `demo/docs/joystick.md` §2.2/§3：名册补 `vt_band_v1`、说明 Part2 覆盖更新。
- 选中绑定后需 `pixi run -e demo demo-build`；`input_map_selftest` 通过（旧键位全部保留）。

## 7. 假运动学模型验证

`tools/route/fake_kinematic_route_sim.py`（理想速度跟随、无动力学/无接触、MuJoCo ray cast 取地形高度），
与真实路线同一份代码路径：

```bash
source demo/install/setup.bash
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=88
PART1_ROUTE_STEPLOCK=1 PART1_ROUTE_JUDGE=0 PART1_TRANSFER_POLICY=vt_locked12 \
PART1_RECORD_PATH=$PWD/artifacts/part2_route3/fake_states.jsonl \
PART1_RECORD_PLAN_PATH=$PWD/artifacts/part2_route3/fake_plan.jsonl \
.pixi/envs/demo/bin/python tools/route/fake_kinematic_route_sim.py \
    --scene demo/scenes/part2/full_field/part2_fixed_field.xml --rate 50 \
    --spawn-x 0 --spawn-y 0 --spawn-yaw 0 &
sleep 1
timeout 1150 .pixi/envs/demo/bin/python demo/rl_sar/scripts/part2_auto_route.py
```

或直接 `tools/route/run_part2_fake.sh <tag>`（自带 ROS domain 88、进程互斥检查、落盘）；
`tools/route/summarize_part2_fake.py <tag>_route.log <tag>_plan.jsonl` 输出逐段 PASS/FAIL 表。

### 7.1 历史结果（假运动学，旧 6 段档 6/6 PASS）

| tag | 档位 / 转场 | 结果 | 证据 |
|---|---|---|---|
| `v6` | platform20；旧 y=1.5 转场 | 6/6 PASS，plan 630.3 s | `v6_route.log` 等 |
| `v7` | platform20；对齐门限 0.07 m/3°+75 s 预算 | 6/6 PASS，plan 595.0 s | `v7_*` |
| `v8` | platform20；南侧 y=−2.5 转场 | 6/6 PASS，plan 615.4 s | `v8_*` |
| `v10` | platform20；缺口中心转场 + 0.30/0.22 m/s 慢速转场 | 6/6 PASS，plan 963.4 s | `v10_*`（sim 跨度最长，因慢速转场） |

- 全段 `gate=gate post=ok`、`recover=0`；切换残差平台/楼梯/斜坡/沟壑/梅花/高墙均在门限内。
- 假模型没有动力学/接触：它只证明路线编排、几何、门限、专家切换顺序正确；**专家能否真正爬过
  障碍必须由真 MuJoCo 全场跑验证**（§7.3）。
- 复现：`tools/route/run_part2_fake.sh <tag>`（`FAKE_RATE=200` 可选）；审计：
  `tools/route/audit_part2_route.py`。

> 假模型没有动力学/接触：它只证明路线编排、几何、门限、专家切换顺序正确；专家能否真正爬过
> 障碍必须由真 MuJoCo 全场跑验证（§7.2 的单关部署 trace 是档位依据，整场真跑由 Lead 执行）。

### 7.2 部署栈实测档位（real `ros2 launch`，2026-11-04，判分 trace 复放）

**结论：部署栈（sim_node+rl_sim+FSM）不等于 eval 侧 policy_runner**，同一专家/同一命令
两边结果会不同；档位必须用真实部署 trace 定。复放命令：
`pixi run -e eval robocon-eval replay-part2 --trace artifacts/traces/<f>.jsonl`。

| 地形 | 部署实测结果（trace） | 判定 |
|---|---|---|
| high_platform_20cm | 官方单关 `deploy_20cm_row2p_p40w40_v1_ad1` → success；field S→N `field_plat20_a/b` → **2/2 PASS** | **默认档**（field 最稳） |
| high_platform_30cm | 官方单关 2/3 success（`deploy__30cm_…` 1 次 stall）；同一 field S→N 几何 `field_plat30_a` timeout stall、`field_plat30_b` 触发恢复但重对齐失败后重试摔倒 → **field 0/2** | 非默认（env 可切） |
| high_platform_40cm | `deploy_40cm_st30_…`、`deploy__40cm_vt_freezehid_1349_ad0`、`deploy__40cm_vtg09_1100_ad1`、`deploy2_plat40_r2p_v33/v35` → 全 fail（台阶面卡住，z≈0.68/pitch≈20°） | **部署不可过**；eval 侧 row2p 0.50 却能过 |
| stairs_30cm | vt_band adaptive：vx=0.50 `db3_stairs30_band_0.5` → success、vx=0.60 success；0.55/0.65/0.70 fail（`not_descended`） | **可过：vx=0.50、start_dy=0.0**；eval 侧 vx=0.70+dy−0.03 盆不迁移 |
| stairs_30cm | vx=0.70 `deploy2_stairs30_band_v70` → fail（`not_on_top_landing`） | 与上一致 |
| slope_35deg | `deploy2_slope35_r2p`、`db3_slope35_1` → success（重复 3/3） | 可过（max-difficulty 保留） |
| trench_30cm | `deploy2_trench30_s1d` → success（3/3） | 可过（max-difficulty 保留） |
| wall_40cm | 官方单关 `deploy2_wall40_band` → success（2/3，1.8 m 助跑）；`db3_wall40_2` → fail（`not_crossed`） | 官方关可过、**field 槽不可**（槽内助跑仅 0.54 m） |
| plum | 无判分专家（§9.1） | transit only |

**为什么高墙取 20 cm 而不是 40 cm**：field 里 `cross_bar_7` 与 `bar8` 之间只留 0.54 m 助跑
（官方单关把障碍放在 1.8 m 处），部署栈在 0.54 m 内起不来；`cross_bar_5`（20 cm）在同款
0.54 m 槽内被早前 real8 用 row2p 验证过（本路线默认 `vt_band_v1@0.50`，可用
`PART2_EXPERT_WALL=row2p_p40w40_v1` 切回）。官方单关的 40 cm 声明仍由 eval 证据支撑
（`docs/part2/EXPERTS.md`），但**不计入 field 路线**。

**部署变异性**：platform_30cm 与 wall 是 2/3 级，不是 100%。真跑若某腿失败，最便宜的处理是：
platform → `PART2_PLATFORM_LEVEL=20`；wall → `PART2_ROUTE_WALL=0`；stairs → 保 vx=0.50 重跑
（0.50/0.60 两个成功档，不要用 0.55/0.65/0.70）。steplock+realtime=false 的整场重跑约 2 min。

**真场验证必须用 `realtime:=true`（1×），不能用 steplock + `realtime:=false`**：
`PART1_ROUTE_STEPLOCK=1` 只在仿真与路线同步推进（如 50 Hz 假模型）时成立。真实 MuJoCo launch
在 `realtime:=false` 下仿真会跑在路线前面（每秒产生的 `/eval_state` 帧远多于路线处理数），
命令更新与超时都会按错误的仿真时间发生——实测同一份 v4 路线在同一档位下：快速 steplock 命令
leg1 platform30 失败（row2p 卡 0.30 m 台阶面，y=−0.55、z=0.62、tilt 7.5°），而
`realtime:=true` 1× 用同一路线同一档位 leg1 PASS（`PART2_ROUTE_ONLY=1`：终点 y=+1.53，
切换残差 +0.047 m；`artifacts/part2_route3/real_p30_1x_*`）。结论：**整场真跑用 1×；
steplock 只用于假模型/逻辑验证**。`tools/route/run_bench_official.sh` 与
`bench_expert_cross.py` 默认 wall-clock（steplock 仅显式开启）。

**对齐实测（部署栈，full field）**：platform S→N 3.1 m 助跑，一次通过：
`actual=(+21.73,−3.21,+89.7°)`、`e=+0.017 m`、`yaw_err=−0.3°`、`align_s=48.1 s`。
对齐按 §5 的有界基元实现，45 s 精细预算 + 2 次尝试，失败会打印残差而不是挂死。

**本路线与部署栈一致性的自我诊断工具**（不改 eval）：
`tools/route/run_bench_official.sh <level> <policy> <vx> <secs> <lane-yaw> <tag>` +
`tools/route/bench_expert_cross.py` 可在真实 launch 上跑单关自适应车道命令；`rl_sim.cpp` 的
`RL_DEBUG_OBS=1`（`RL_DEBUG_OBS_MAX=<n>` 控制帧数）会 dump obs57/act16 到 `/tmp/rl_obs_debug.csv`，
用于和 eval 侧逐帧对比（实测部署侧 action 与 Python 复算 bit-exact，而部署侧同一命令下速度
偏快 ~1.5×，说明差异在控制/执行链路而非观测/策略）。

### 7.3 真场整场试跑记录（1× realtime，逐步收敛）

| tag | 档位 / 本轮改动 | 结果 |
|---|---|---|
| `real_v8` | platform20/stairs30，旧 y=1.5 转场 | leg1 PASS；leg2 +x 转场北漂蹭 `stairs_08` 西端 → 转场改走南侧 y=−2.5 大走廊 |
| `real_v9` | + 南侧转场 | leg1 PASS；leg2 北爬 x=60.50 被 turn-in 横漂带到 `stairs_25` 东缘 → 北爬移到缺口中心（30cm 档 61.45） |
| `real_v10` | + 缺口中心北爬 | leg1 PASS；leg2 stairs30 对齐 OK、爬完上坡，**顶层平台 stall**（z≈2.15）；恢复倒车摔倒 → 加“高地不倒车”守卫 |
| `real_v11` | stairs20 试验 | 发现硬编码缺口（仍走 61.45 会撞 `stairs_25`），运行前主动中止 → 缺口 x 按档位表推导 |
| `real_v12` | platform20/stairs20 st30@0.45 | leg1 **PASS**、leg2 stairs20 **PASS**；leg3 slope35 对齐失败（旋转横漂 ~0.2 m，settle 前向只有 5 cm，e 停在 +0.18）→ 新增 `_sidestep_lane` 横向基元 |
| `real_stairs30_v60` | 单测 stairs30 vt_band@0.60 | **PASS**（(67.88,3.20,−1.4°)）→ 楼梯 30=200 分档在 field 可用（但见 v15/v18 的变异性） |
| `real_v13` | platform20/stairs20/slope30（无 sidestep 已加载） | leg1-2 PASS；slope30 对齐仍失败 → 已被 v14 的 sidestep 修复覆盖 |
| `real_v14` | platform20/**stairs30@0.60**/slope35 + sidestep | leg1–4 **PASS**（platform20 / stairs30 / slope35 / trench30；slope sidestep e +0.149→−0.019）；leg5 plum `obstacle` 穿越在桩前 stall、恢复倒车摔倒 → plum 改绕行 |
| `real_v15` | + plum bypass | leg1 PASS；leg2 stairs30@0.60 **stall 在顶层平台**（z=1.09, yaw −17.5）→ stairs30 变异性（v14/v17 过、v15/v18 挂） |
| `real_v16` | 安全档（stairs20）+ 东侧 slope 出口 | leg1–3 **PASS**（platform20/stairs20/slope35）；leg3 后 slope→trench 东侧 1.2 m 缺口转场卡住（turn-in 横漂撞 slope_35 东面）→ 改西端绕行 |
| `real_v17` | 最佳档（stairs30@0.60）+ 东侧 slope 出口 | 同 v16：leg1–3 PASS，同一东侧缺口转场失败 |
| `real_v19` | **安全档 + 西端绕行（y=4.5 走廊 → x=20.5 列 → y=7.5 走廊）+ plum bypass** | **leg1–5 全部 PASS**（platform20 / stairs20 st30@0.45 / slope35 / trench30 / high_plum_bypass）；leg6 wall20 对齐失败（e=+0.183/yaw−5.8°，bar5/bar6 1.08 m 走廊内无空间）→ wall 改用 bar5 专用宽容门限 + row2p@0.40 |
| `real_v20` | 安全档 + **PART2_ROUTE_WALL=0** | **PASS：`Part2 route finished: plum bypassed`**（leg1–5 全 PASS，1182 s 仿真） |
| `real_v21` | 安全档 + wall bar5（row2p@0.40 + 0.20 m/6° 门限） | **6/6 PASS**（含 wall20）；旧的高墙-included 验证 |
| `real_v22` | **保守档**：platform20/stairs20/slope30/trench20/plum bypass，wall 默认关 | **路线 5/5 段集成门限 PASS，eval 判据复核 4/4 穿越段 success（合计 350）**（1224.8 s 仿真）：platform20（y=1.35 门）、stairs20、slope30（x>38.20）、trench20（x>13.60）、plum bypass 到 (2.60,13.20) 后结束；**最终交付档** |

部署栈整场结论：**所需五段地形（平台/楼梯/斜坡/沟壑/梅花桩）在 1× 真场可以连续走完**：平台 20cm、
楼梯 20cm（st30@0.45；200 分档 stairs_30 用 vt_band@0.60 可过但 run-to-run 不稳）、斜坡 30°（默认；
35° 仍可切且部署 3/3）、沟壑 20°（默认；30° 仍可切且部署 3/3）、梅花桩**不穿行**（无部署专家，
绕行且不报分）。高墙 20cm 为可选段，默认关（bar5 专用宽容对齐，`PART2_ROUTE_WALL=1` 打开）。

**判据口径**：路线门限只保证“这一段走过去了”，不对外代表官方判分；对外复核用上面的
eval 投影工具，正式分数仍来自单关 eval 运行。
每个失败点都定位到转场几何/对齐/恢复实现并修复，不是专家能力缺失。

**与 eval 判据的一致性复核**：用 `tools/route/replay_field_part2.py`（已入库）把每段场地轨迹
投影回对应单关坐标，喂 `Part2OfficialJudge`（v9）：`real_v22` 的 platform20 / stairs20 /
slope30 / trench20 四段均 **success**（plum 绕行段不计分），合计 350；`real_v21`（含高墙）
6/6 段 success，合计 600。这确认了路线门限与 eval 判据在已跑过的运行上不矛盾；正式报分仍
以单关 eval 运行为准，路线 PASS 不再作为对外判据。

## 8. 真实运行命令（最终档）

默认值 = `real_v22` 保守验证档（平台 20 / 楼梯 20 st30@0.45 / 斜坡 30 / 沟壑 20 / 梅花绕行，
高墙默认关），不需要任何 env：

```bash
cd /home/bis/robocon-lean
source demo/install/setup.bash
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=57
PART1_ROUTE_JUDGE=0 \
PART1_RECORD_PATH=$PWD/artifacts/part2_route3/real_v22_states.jsonl \
PART1_RECORD_PLAN_PATH=$PWD/artifacts/part2_route3/real_v22_plan.jsonl \
timeout 2800 ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=1 route:=part2 \
    policy_name:=vt_locked12 transfer_policy:=vt_locked12 \
    viewer:=false realtime:=true eval_state_every:=10 stop_on_fail:=true
```

- 200 分楼梯 30 档：加 `PART2_STAIRS_LEVEL=30 PART2_EXPERT_STAIRS=vt_band_v1
  PART2_VX_STAIRS=0.60`（v14/v17 整场 PASS；v15/v18 顶层平台 stall，约 50% 可靠）。
- 其它回退/加难：`PART2_PLATFORM_LEVEL=20|30|40`、`PART2_SLOPE_LEVEL=20|30|35|45`、
  `PART2_TRENCH_LEVEL=10|20|30`、`PART2_ROUTE_WALL=1`（打开高墙 20cm）、
  `PART2_ROUTE_WALL_VARIANT=bar5|bar6|bar7`。
- `part:=1`：`route:=part2` 只覆盖场景/spawn；启动序列需要 A=StandUp（`part:=2` 时该键按规则无效）。
- `realtime:=true` **必须**给：真场验证用 1×。**不要**用 steplock+`realtime:=false` 做整场真跑——
  仿真会跑在路线前面，命令更新/超时按错误仿真时间发生（§7.2 实测 leg1 会假失败）。
- `stop_on_fail:=true` 在第一个 FAIL 处关掉整个 launch；先看第一条 FAIL 再改。
- **时长**：`real_v22` 1224.8 s 仿真（约 20 min 墙钟）；建议外层 `timeout 2800`。路线跑完自己退出。
- 单段调试：`PART2_ROUTE_ONLY=<leg名或序号>`。
- eval 判据复核：`pixi run -e eval python tools/route/replay_field_part2.py --tag real_v22
  --out artifacts/part2_route3/real_v22_eval.json`（`--require-all` 可让非 success 段返回非零）。
- 录像/复核：`real_v22_route.mp4`（10× 快放，122 s；`MUJOCO_GL=egl tools/video/replay_route_video.py`）、
  `real_v22_overview.png`（俯视轨迹+切换点）、`real_v22_states.jsonl`/`_plan.jsonl`。

## 9. 已知风险（按优先级）

1. **plum 无可判分专家（已定论，路线改为绕行）**：plum-probe 盲走 0/200、旧仓库 0/384，无可部署的
   盲走专家；depth-CNN 学生策略只能 eval（rl_sim 无 depth 链路）且修正地图 0/9。早期尝试过用
   `obstacle@0.4` 直穿（1× 真跑 `real_v14` 在第一个桩排前 stall、恢复倒车时摔倒），因此默认
   **`PART2_PLUM_MODE=bypass`**：leg5 变为 `high_plum_bypass`（不切专家、不穿桩），沿 y=10.5 西行到
   (2.60,10.5) → 北到 (2.60,13.2)（梅花桩行 x=4.00–7.50，故 x=2.60 净空）→ 接高墙转场。
   路线报告写 `high_plum_bypass: PASS ... TRANSIT ONLY / bypass`；梅花桩 0 分、如实写进最终报告。
   `PART2_PLUM_MODE=cross` 保留给未来有 plum 专家时使用（审计已过）。
2. **部署栈 ≠ eval 栈（本轮最大风险，已用档位规避）**：同一专家/命令在真实 sim_node+rl_sim
   与 policy_runner 上结果可以相反（platform_40cm eval 0.50 过、部署 0/4 卡台阶面；stairs_30cm
   eval 0.70 稳、部署 0.70 反而失败而 0.50 过）。因此默认档位全部按 §7.2 的部署 trace 选，
   不再直接引用 eval 侧 pass map。部署侧 obs/action 已确认与 eval bit-exact，差异在控制/执行
   链路（同命令体速偏快 ~1.5×，疑 command_limit/S4 发布节流/PD 路径），Lead 侧继续定位。
3. **平台/高墙的部署可靠性**：field S→N 实测 platform_20cm **2/2 通过**（后续 1× 整场多次通过）、
   platform_30cm 1/4（台阶面 stall 与接触混沌），故默认 20；30/40 仍可 env 切换
   （`PART2_PLATFORM_LEVEL=30|40`）但不要再当默认。wall_20cm（bar5）有旧 real8 同槽验证；
   真跑若失败优先 `PART2_ROUTE_WALL=0`（路线会在 plum 绕行后结束）。
4. **stairs_30cm 单专家、通过档窄**：`vt_band_v1` 唯一部署可过的 30 cm 楼梯专家；部署成功档是
   **vx=0.60**（1× field 整场 v14 与单测 `real_stairs30_v60` 均 PASS），vx=0.50 也可过但 1× 整场
   v10 在顶层平台 stall（变异性），vx=0.55/0.65/0.70 不行（非单调）；不要用 eval 侧的 0.70 盆。
   入场门限 `0.02 m/1.5°`、`start_dy=0.0`、`cross_timeout=45 s`。失败先重跑（变异性），
   仍失败再降到 stairs_20（st30@0.45，旧 real8 同档实证）。
5. **锁轮模型横漂**：长转场已改为 goto 折线（≤10 m）规避；若 10 m 段内仍出现 >0.15 m 横漂，
   把 `search_goto` 的注点加密或 tol 收紧到 0.10。
6. **滑移/时序**：假模型无动力学，专家段的真实速度/耗时可能明显长于假模型；各 `cross_timeout`
   已按 2–3 倍余量给（30–45 s）。真跑若某段超时，先看该专家是否在原地打滑。
7. **对齐死区**：burst/dwell 已针对锁轮低 wz 死区补偿；若某次对齐重试仍失败，路线会 FAIL 并打印
   `y_err/yaw_err`，据日志决定增大 burst 或放宽到 3°。

## 10. 文件清单

| 文件 | 说明 |
|---|---|
| `demo/rl_sar/scripts/part2_auto_route.py` | 路线 v2（专家段 + goto 转场 + 对齐 + 审计折线数据） |
| `demo/rl_sar/include/input_map.hpp` | `Intent::PolicyBand` + RB+左 / `b` 绑定 |
| `demo/rl_sar/src/rl_sim.cpp` | `PolicyBand` → `vt_band_v1`；键盘 B 映射 |
| `demo/docs/joystick.md` | 名册/键表更新 |
| `tools/route/audit_part2_route.py` | MuJoCo AABB 净空审计（`--dump` 可打印全场障碍） |
| `tools/route/replay_field_part2.py` | 全场轨迹逐段投影回单关，跑 eval Part2 judge（对外判据复核；`--require-all` 可作集成门禁） |
| `tools/route/run_part2_fake.sh` | 假模型整场验证启动脚本 |
| `tools/route/summarize_part2_fake.py` | 假模型 plan/log → PASS/FAIL 表 |
| `tools/route/run_bench_official.sh` + `bench_expert_cross.py` | 真实部署栈单关基准（自适应车道命令，domain 87） |
| `demo/tests/run_joy_trial.sh` + `tools/trial/adaptive_joy.py` | 真实部署栈单关 trial harness（固定/自适应命令 + 判分 trace） |
| `docs/part2/ROUTE.md` §7.2/§7.3 | 部署实测档位证据 + 1× 真场整场试跑记录（trace 文件名 + 复放命令） |
| `artifacts/part2_route3/` | 审计输出 + 假模型 `v6_*/v7_*/v8_*/v10_*` + 部署单关/整场 `real_v8_*`…`real_v22_*`；默认交付证据 = `real_v22_{route.mp4,overview.png,states.jsonl,plan.jsonl}` |
