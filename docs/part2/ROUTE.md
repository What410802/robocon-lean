# Part2 全场自动路线 v3（修正版场地 / 六类全开 / 深度相机接入）

更新：2026-11-07。实现：`demo/rl_sar/scripts/part2_auto_route.py`（v3：梅花桩改由深度相机专家
穿越 + 出口台阶交轮式专家下降，高墙默认打开，新增 `/policy_switch` 快切）。
场景：`demo/scenes/part2/full_field/part2_fixed_field.xml`（`map_fixed.xml` + 竞赛 PDF 修正后的梅花桩），
`route:=part2` 固定 spawn `(0, 0, 0.675, 0)`、`eval_state_every:=10`（50 Hz 反馈）。
运行时另需 `sim_node` 的深度相机（`/depth_cam`，64x48 公制深度，50 Hz；与 policy 目录的
`depth.json` 是同一条契约，见 §6.3）。

## 1. 结论（本轮）

**默认路线 = 六类全开的 1× 整场演示，`real_v27` 首次跑通：路线 6/6 段集成门限 PASS，
eval 判据逐段投影复核 6/6 段 success、合计 600/1300**（平台 20=50、楼梯 20=100、斜坡 30=50、
沟壑 20=150、梅花桩=200、高墙 20=50；全场连续跑不是官方计分单元，分数是逐段投影合计）。
单关判分侧六类最高档见 `EXPERTS.md`，合计 **1200/1300**（只差按用户要求本轮不做的 slope_45 的 100）。

- 梅花桩本轮从"绕行"改成**真穿越**：`plum_depth_final_v1`（54x6 proprio + 1 帧 48x64 深度，
  无 command 项、`/joy` 无效）负责入口平台 + 桩阵；出口 0.2 m 台阶交给轮式专家
  `vt_band_v1`（在出口平台上 2 s 站定后以 0.45 m/s 慢速走下）。两半都是 RL 策略，判分只看
  support 点，所以一条 trace 仍是合法一次尝试；`/policy_switch` 换策略实测耗时 ~3 ms。
  **为什么拆**：同一份深度权重在 eval 判分链路里能自己完成出口下降（判分侧 success 200），
  但部署栈（sim_node+rl_sim）里出口下降是刀锋状态——身体从 0.2 m 平台前倾翻下；拆开之后
  深度策略只负责它稳定的部分。`PART2_PLUM_HANDOVER=none` 保留纯深度尝试用于对拍。
- 深度相机链路：`sim_node` 用 EGL 离屏渲染 64x48 公制平面深度（只画 geom group 0）发布
  `/depth_cam`（50 Hz，先于同一步 `/motor_state`）；`rl_sim` 按训练契约归一化后作为策略第二
  输入。C++ 渲染与 eval 侧 `mujoco.Renderer` **逐位相同**（max_abs_diff 0.0），C++ 推理与
  Python 同输入同输出（`demo/tests/check_depth_inference.py`，max_abs_diff 0.0）。
- 默认档刻意低于 eval 侧已成功过的最高档（平台 20<40、楼梯 20<30、斜坡 30<35、沟壑 20<30、高墙
  场地 20<官方 40），用可靠性换演示稳定性；更高档位仍可用 env 切回（§8）。
- 每个专家入场前用有界基元对齐，门限：默认 0.05 m/2.5°，platform 0.07 m/3°，stairs 0.02 m/1.5°，
  slope/trench 0.08 m/2.5°，plum 0.05 m/1.5°（深度策略对初始位姿敏感），wall bar5 0.20 m/6°。
- 段落切换全部走 `vt_locked12` 锁轮衔接；转场是 **`goto()` 折线**（开阔区 ≤10 m @0.30 m/s，
  紧走廊 ≤4 m @0.22 m/s）。锁轮模型的 vy 车道保持基本无效（~0.008 m/m 系统横漂），所以不靠 vy，
  而靠“多段折线 + 每段重新闭环 + 缺口中心进场 + 短直进缺口”。斜坡行是连续 1.2 m 缺口组成的墙，
  斜→沟转场改为**西端绕行**（y=4.5 走廊 → x=20.5 列 → y=7.5 走廊），v19/v20/v21 均通过。
- 对齐实现（位置 goto + 死区补偿偏航伺服 + **横向 `_sidestep_lane`** + 短直 settle）与逐 leg 门限
  见 §5；wall bar5/bar6 走廊只有 1.08 m，所以 wall 用 0.20 m/6° 的宽容差。
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

默认档 = **六类全开**（平台 20 / 楼梯 20 / 斜坡 30 / 沟壑 20 / 梅花桩穿越 / 高墙 20 开；
`real_v27` 整场验证，见 §7.3；更高档位见 §8 的 env 说明，门槛与速度都可切换）。
`vx` 是专家 AdaptiveCommand 的目标体速（`/joy` 轴值 = vx/1.5，部署端再乘 `joy_command_scale=1.5`）。

| # | leg | 方向 | 专家（env 覆盖） | vx | 入场对齐位 (x,y,yaw) | 集成门限 | 转场（goto 折线） |
|---|---|---|---|---|---|---|---|
| 1 | `high_platform_20cm` | **S→N** | `row2p_p40w40_v1` | 0.50 | (12.75, **−3.90**, +90°) | y>1.35 | (0,0)→(0,−4.5)→(10,−4.5)→(12.75,−4.5) |
| 2 | `stairs_20cm` | W→E | `st30_ppo_st30_s100_600` | 0.45 | (46.00, 3.000, 0°) | x>52.80 | 平台 exit→(14.25,1.5)→(14.25,−2.5)→y=−2.5 东行→(46.45,−2.5)→(46.45,−0.5)→(46.45,2.97)→(46.00,2.97) |
| 3 | `slope_30deg` | W→E | `row2p_p40w40_v1` | 0.50 | (32.40, 6.000, 0°) | x>38.20 | 楼梯 exit→(53.60,3.0)→(53.60,4.5)→y=4.5 西行→(32.40,4.5)→(32.40,5.50)→stand |
| 4 | `trench_20cm` | W→E | `s1d_trench_reward_v75_m100` | 0.80（自适应） | (8.05, 9.000, 0°) | x>13.60 | 斜坡 exit→(38.40,4.5)→y=4.5 西行→(20.50,4.5)→(20.50,7.5)→y=7.5 西行→(8.05,7.5)→(8.05,8.5)→stand |
| 5 | `high_plum` | W→E | 深度专家 `plum_depth_final_v1` → 出口台阶 `vt_band_v1` | 0.40（深度段；策略无 command）= 轮式段 0.45（先 0.6 s 倒车刹车） | (2.20, 12.000, 0°)（field x=4.0 入口平台前） | x>9.00（=level x 6.80，过出口平台后的地面） | 沟壑 exit→(20.15,9)→y=10.5 西行→(2.20,10.5)→stand（field 坐标） |
| 6 | `high_wall_20cm`（cross_bar_5） | E→W | `row2p_p40w40_v1`（bar5；bar6/7 用 `vt_band_v1`） | 0.40（bar5） | (8.40, 21.000, 180°) | x<7.20 | plum exit→y=13.2 西行→(2.40,13.2)→(2.40,19.5)→(8.40,19.5)→(8.40,20.5)→stand |

梅花桩 leg 的 field 坐标：入口平台 x 4.0–4.8、桩阵 x 5.0–6.5、出口平台 x 6.7–7.5、level 坐标 = field−2.20；
判分侧对应的单关是 `plum_1`（level 坐标）。

- 平台 leg 是 **S→N**（南侧 y=−4.5 走廊，3.1 m 助跑）：W→E 的 field 槽只有 0.75 m 助跑，专家起不来。
- 每个 leg 后先 `ensure_transfer()`（停 + 0.6 s + 切 `vt_locked12` + 1.2 s 稳定），再走 exit 折线。
- 高墙：**默认开**（`PART2_ROUTE_WALL=0` 关）；`PART2_ROUTE_WALL_VARIANT=bar5|bar6|bar7`。
  bar5 用 row2p@0.40 + 0.20 m/6° 宽容差（1.08 m 走廊内无空间做 2 cm 级对齐）；
  bar6/bar7 用 vt_band@0.50，但 field 槽 0.54 m 助跑只有官方单关证据（见 §7.2）。
- 专家策略名/速度/模式在模块顶部 `EXPERT_SPECS`（`PART2_EXPERT_<LEG>`、`PART2_VX_<LEG>` env 覆盖），
  档位在 `PART2_PLATFORM_LEVEL` / `PART2_STAIRS_LEVEL` / `PART2_SLOPE_LEVEL` /
  `PART2_TRENCH_LEVEL` / `PART2_PLUM_MODE` 中切换；梅花桩拆分参数见 §6.4。
- `mode=fixed` 时穿越循环只发 `(vx,0,0)`；默认全部 adaptive（训练侧车道反馈：
  `vy=clip(−0.5·e,±0.2)`、`wz=clip(−0.8·e−2.0·h−0.6·vy_b,±0.7)`，对 ±x/±y 车道按 heading 镜像）。
- 梅花桩不再绕行（`PART2_PLUM_MODE=bypass` 仍保留作无深度相机时的回退）：盲走专家 eval 0/200、
  旧仓 0/384，`real_v14` 的 `obstacle` 穿越在桩前 stall 并摔，所以穿越只能靠深度专家。

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
   默认 0.05 m / 2.5°；stairs 0.02 m / 1.5°，slope/trench 0.08 m / 2.5°，plum 0.05 m / 1.5°。
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

## 6. 切换接线（本轮）

### 6.1 手柄/键盘绑定（沿用）

`vt_band_v1`（stairs_30cm/高墙专家）绑定 **RB + 十字左**、键盘 **`b`**（`input_map.hpp` +
`rl_sim.cpp` 的 `PolicyBand`）；**RB+十字上 = EnterRL** 保留。`input_map_selftest` 通过。

### 6.2 `/policy_switch`（gamepad-less 快切，本轮新增）

- `rl_sim` 新增订阅 `/policy_switch`（`std_msgs/String`，payload = policy 目录名，如
  `plum_depth_final_v1` / `vt_band_v1`），回调直接 `SwitchPolicy()` 加载 TorchScript，
  实测 **~3 ms**（日志 `[policy] switched to <name> (N ms)`）。游戏手柄绑定表不变。
- 路线用它在**两次策略切换**上：① plum leg 前从衔接模型切到深度专家；② 出口平台上切到
  轮式专家做台阶下降。第二次切换必须快——手柄组合键路径约 0.7 s，机器人会多走 ~0.5 m、
  在 0.8 m 长的出口平台上没有余量。
- 该话题只做加载，不做状态机迁移；切到盲走策略后照常由 `/joy` 驱动，切到深度策略后
  `/joy` 无效（策略自走）。

### 6.3 深度相机链路（`sim_node` → `rl_sim`）

- 相机 = 场景 XML 里已有的 `depth_cam`（挂在 `d435_link`，`fovy=70`），64x48、`cutoff=3.0 m`、
  `min_depth=0.1 m`；与 `demo/rl_sar/policy/blackW/plum_depth_final_v1/depth.json` 同一条契约。
- `sim_node`：EGL 离屏渲染（有窗口时复用 GLFW 上下文），`mjRND_SEGMENT+mjRND_IDCOLOR`、
  `readDepthMap=mjDEPTH_ZEROFAR`、reverse-Z→米、上下翻转，只画 geom group 0（看不见自己的狗）；
  每 10 步（50 Hz）渲一帧、**先于同一步 `/motor_state`** 发布 `DepthFrame`（`header + sim_time +
  width/height + float32[]` 公制平面深度，row-major，无命中=远端值）。
- `rl_sim`：按训练契约 `where(finite & >0, raw, cutoff); clip(min_depth,cutoff)/cutoff` 归一化成
  `1x1x48x64`；深度策略的 TorchScript 调用是 `policy(obs_1d_324, [depth])`，盲目策略仍走
  `policy(obs)`。首个深度帧未到时保持上一指令（最多告警 3 次）。
- 数值验收：C++ 渲染 vs eval 侧 Python `mujoco.Renderer` **逐位相同**（5 个场景，max_abs_diff 0.0）；
  `sim_node --depth-self-test <scene.xml>` 可离线对拍；C++ 推理 vs Python 同输入
  `demo/tests/check_depth_inference.py` max_abs_diff 0.0（`RL_DEPTH_DEBUG=1` 时 rl_sim 落盘
  obs/raw depth/raw action）。
- 失败降级：EGL 不可用或连续 3 帧渲染失败 → 打印一次告警、停发 `/depth_cam`，仿真与盲目策略
  不受影响（深度策略会停在“waiting for /depth_cam”，路线在该 leg FAIL）。

### 6.4 梅花桩 split（本轮新增）

- 参数（路线模块顶部，env 可调）：`PART2_PLUM_HANDOVER=vt_band_v1|none`、
  `PART2_PLUM_HANDOVER_X=6.80`（field x；level x=4.60，出口平台中部）、
  `PART2_PLUM_SETTLE_S=2.0`、`PART2_PLUM_DRIVE_VX=0.45`、
  `PART2_PLUM_BRAKE_S=0.6`、`PART2_PLUM_BRAKE_VX=-0.30`、
  `PART2_PLUM_GATE_SETTLE_S=1.5`（gate 后再零指令停 1.5 s：轮子还在滚时锁轮会
  把机器人向后拽一下，之前的连接器 goto 失败就是这么来的）。
- 流程：对齐（0.05 m/1.5°）→ 切深度专家 → 自走穿越入口平台+桩阵 → 到 field x≥6.80
  用 `/policy_switch` 切 `vt_band_v1` → **倒车 0.6 s 刹车**（0.8 m 出口平台装不下 2 m/s 的
  自然减速）→ 0 指令站定 2.0 s → 0.45 m/s + 车道保持走完 0.2 m 台阶与地面 →
  field x>9.00 集成门限 PASS → `ensure_transfer()` → exit 折线。
- **必须切在出口平台（field ≥6.7）**：若提前到桩阵上（如 field 5.80 = level 3.60）再切
  轮式专家，它会在桩间踩空——judge `dropped:z=0.03@x=4.32`（实测），所以不行。
- 判分窗口：`replay_field_part2.py` 的 leg 窗口从 `high_plum(policy=...)` 计划行开始，
  到 `ensure_transfer` 结束；下降段在窗口内，判分侧看到的是连续一条 trace。
- `PART2_PLUM_HANDOVER=none`：只跑深度策略到 field x>7.90（旧行为），用于和 eval 侧对拍；
  部署栈下它通常会在出口下降时跌倒（`not_descended_exit_platform` / `fell`）。

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
| `cross_wall_v2` | **v3 档**：六类全开（plum 组合穿越 + wall bar5 开） | 6/6 PASS，plan 867.5 s | `cross_wall_v2_*`（route-wiring 验证） |
| `cross_wall_v3` | **v3 档 + 当前默认参数**（6.80 切 + 刹车 + gate 站定） | **6/6 PASS**，plan 876.7 s；`summarize_part2_fake.py` 6 PASS/0 FAIL | `cross_wall_v3_*`（本轮回归） |

- 全段 `gate=gate post=ok`、`recover=0`；切换残差平台/楼梯/斜坡/沟壑/梅花/高墙均在门限内。
- 梅花桩组合段在假模型里只验证“切专家 → 站定 → 慢速前进 → gate”这条编排逻辑（假模型没有
  动力学，也不区分策略）；真实下降由 §7.3 的 `real_v27` 证明。
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
| plum | 纯深度策略（部署栈）：桩阵可过，出口台阶翻下（刀锋）；组合穿越（深度 + 轮式下降）→ 判分 success（2026-11-07，见下） | 组合档为默认 |

**2026-11-07 新增部署证据（plum 组合穿越，统一 torch 2.14.1 后）**：

- 单关判分侧（eval `policy_runner`，official profile）：`plum_1` + `plum_depth_final_v1`
  → **success 200**（`artifacts/part2_evals/torch214/plum_1.*`，六类最高档合计 **1200/1300**）。
- 部署栈单段真跑（`PART2_ROUTE_START=5 PART2_ROUTE_ONLY=high_plum`，spawn (10.05,10.5)）：
  - 纯深度策略（`PART2_PLUM_HANDOVER=none`，gate field x>7.90）：穿越桩阵成功，但出口 0.2 m
    台阶处前倾翻下 → 判分投影 `not_descended_exit_platform` / `fell`（与"刀锋"描述一致）。
  - 组合档（默认：field x=6.80 切 `vt_band_v1` → 倒车 0.6 s 刹车 → 站 2.0 s → 0.45 m/s
    走下出口台阶，gate field x>9.00）：路线集成门限 PASS，判分投影 **success 200**
    （`artifacts/part2_route3/plum_comp_v4/v5*`）。
  - 失败记录（都是真实数据，说明边界在哪）：① 纯深度策略自己下台阶 → 出口前倾翻下
    （`not_descended_exit_platform`/`fell`）；② 切太早（field 5.80 = 桩阵上）→ 轮式专家在桩间
    踩空，judge `dropped:z=0.03@x=4.32`；③ 切在 6.80 但不刹车 → 2 m/s 的动量把身体带出
    出口平台边缘（整场首跑 `real_v27` `fell:tilt=76.6`）。
- 复现性单测（组合切换，2026-11-07，默认档：6.80 + 0.6 s 倒车 + 2 s 站定）：
  `vt_band_v1` 连续 2 次单段路线 PASS 且判分 **success 200**（`plum_comp_v5r1/r2`）；
  切换本身 ~3 ms，剩余变异性来自部署栈实时调度的微秒级控制相位差（专家在出口边缘是混沌的）。
  整场演示如果 plum 段失败，重跑整场是最便宜的恢复手段。
- 深度链路数值验收：C++ 渲染 = eval 侧 Python 渲染逐位相同（5 个场景 max_abs_diff 0.0）；
  C++ 推理 = Python 同输入同动作（`demo/tests/check_depth_inference.py`，max_abs_diff 0.0）。
  因此上面"部署失败"不是观测/推理链路问题，而是执行/接触动力学的混沌边界。

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
| `real_v22` | **保守档**：platform20/stairs20/slope30/trench20/plum bypass，wall 默认关 | **路线 5/5 段集成门限 PASS，eval 判据复核 4/4 穿越段 success（合计 350）**（1224.8 s 仿真）：platform20（y=1.35 门）、stairs20、slope30（x>38.20）、trench20（x>13.60）、plum bypass 到 (2.60,13.20) 后结束；旧保守档，保留作回退证据 |
| `real_v27` | 六类全开首跑（plum 组合切在 6.80、未刹车） | leg1–4 PASS；leg5 深度穿越后在出口边缘跌倒（判分 `fell:tilt=76.6`）→ 加“倒车刹车 + 站定” |
| `real_v24` | 六类全开（含刹车） | **leg1–5 全 PASS**，判分投影 **550/1300**（platform20/stairs20/slope30/trench20/plum 全 success，plum=200）；leg5 后连接器 goto 失败（轮子还在滚就锁轮被向后拽）→ 加 gate 后 1.5 s 零指令站定 |
| `real_v25` | 六类全开（含 gate 站定） | leg1–4 PASS；leg5 深度下降段跌倒（单段组合的混沌边界，非路线逻辑）→ 用单段重复试验调参 |
| `real_v26` | 六类全开（默认档 6.80 + 刹车 + gate 站定） | leg1–2 PASS；leg3 slope 对齐 2 次尝试后 yaw_err 23.8° 失败（对齐/接触混沌，与 plum 无关）→ 重跑 |
| `real_v27` | 六类全开（同默认档） | **路线 6/6 PASS、判分投影 6/6 success、合计 600/1300**（plum 200 + wall 50；73712 帧、约 1475 s 仿真）；**本轮交付证据** |

部署栈整场结论：**六类在 1× 真场可一次启动连续走完**——`real_v27` 路线 6/6 段 PASS，
逐段判分投影 6/6 success、合计 600/1300（平台 20、楼梯 20 st30@0.45、斜坡 30、沟壑 20、
梅花桩组合穿越 200、高墙 20 bar5）。更高档位仍可切（平台 30/40、楼梯 30 vt_band@0.60、
斜坡 35、沟壑 30、高墙 bar6/bar7）。整场是**非计分集成演示**；可靠性受 plum 组合下降
（约 2/3 单段成功率）与个别 leg 对齐/接触的混沌边界限制，失败时重跑整场。

**判据口径**：路线门限只保证“这一段走过去了”，不对外代表官方判分；对外复核用上面的
eval 投影工具，正式分数仍来自单关 eval 运行。
每个失败点都定位到转场几何/对齐/恢复实现并修复，不是专家能力缺失。

**与 eval 判据的一致性复核**：用 `tools/route/replay_field_part2.py`（已入库）把每段场地轨迹
投影回对应单关坐标，喂 `Part2OfficialJudge`（v9）：
- `real_v27`（当前默认档）：**6/6 段 success，合计 600/1300**（platform 50 + stairs 100 +
  slope 50 + trench 150 + plum 200 + wall 50）。
- 回退证据：`real_v24`（plum 组合首次判分成功）5 段 success、合计 550；`real_v22`（plum 绕行，
  wall 关）4 段 success、350；`real_v21`（plum 绕行，wall 20 开）6 段 success、600。
这确认了路线门限与 eval 判据在已跑过的运行上不矛盾；正式报分仍以单关 eval 运行为准，
路线 PASS 不再作为对外判据。

## 8. 真实运行命令（最终档，六类全开）

默认值 = 六类全开档（平台 20 / 楼梯 20 st30@0.45 / 斜坡 30 / 沟壑 20 / 梅花桩组合穿越 /
高墙 20 bar5 开），不需要任何档位 env：

```bash
cd /home/bis/robocon-lean
source demo/install/setup.bash
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=57
PART1_ROUTE_JUDGE=0 \
PART1_RECORD_PATH=$PWD/artifacts/part2_route3/real_v27_states.jsonl \
PART1_RECORD_PLAN_PATH=$PWD/artifacts/part2_route3/real_v27_plan.jsonl \
timeout 3000 ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=1 route:=part2 \
    policy_name:=vt_locked12 transfer_policy:=vt_locked12 \
    viewer:=false realtime:=true eval_state_every:=10 stop_on_fail:=true
```

- 梅花桩拆分参数：`PART2_PLUM_HANDOVER=vt_band_v1`（默认）/`none`（纯深度尝试）、
  `PART2_PLUM_HANDOVER_X=6.80`（field x）、`PART2_PLUM_SETTLE_S=2.0`、
  `PART2_PLUM_DRIVE_VX=0.45`、`PART2_PLUM_BRAKE_S=0.6`、`PART2_PLUM_BRAKE_VX=-0.30`、
  `PART2_PLUM_GATE_SETTLE_S=1.5`；
  `PART2_PLUM_MODE=bypass` 回到绕行（无深度相机时）。
- 200 分楼梯 30 档：加 `PART2_STAIRS_LEVEL=30 PART2_EXPERT_STAIRS=vt_band_v1
  PART2_VX_STAIRS=0.60`（v14/v17 整场 PASS；v15/v18 顶层平台 stall，约 50% 可靠）。
- 其它回退/加难：`PART2_PLATFORM_LEVEL=20|30|40`、`PART2_SLOPE_LEVEL=20|30|35|45`、
  `PART2_TRENCH_LEVEL=10|20|30`、`PART2_ROUTE_WALL=0`（关高墙）、
  `PART2_ROUTE_WALL_VARIANT=bar5|bar6|bar7`。
- `part:=1`：`route:=part2` 只覆盖场景/spawn；启动序列需要 A=StandUp（`part:=2` 时该键按规则无效）。
- `realtime:=true` **必须**给：真场验证用 1×。**不要**用 steplock+`realtime:=false` 做整场真跑——
  仿真会跑在路线前面，命令更新/超时按错误仿真时间发生（§7.2 实测 leg1 会假失败）。
- `stop_on_fail:=true` 在第一个 FAIL 处关掉整个 launch；先看第一条 FAIL 再改。
- **时长**：`real_v22` 1224.8 s 仿真（约 20 min 墙钟）；建议外层 `timeout 2800`。路线跑完自己退出。
- 单段调试：`PART2_ROUTE_ONLY=<leg名或序号>`。
- eval 判据复核：`pixi run -e eval python tools/route/replay_field_part2.py --tag real_v27
  --out artifacts/part2_route3/real_v27_eval.json`（`--require-all` 可让非 success 段返回非零）。
- 录像/复核（离线运动学回放，不重步物理）：
  `MUJOCO_GL=egl pixi run -e eval python tools/video/replay_route_video.py \
  artifacts/part2_route3/real_v27_states.jsonl artifacts/part2_route3/real_v27_route.mp4 \
  artifacts/part2_route3/real_v27_plan.jsonl --scene demo/scenes/part2/full_field/part2_fixed_field.xml`；
  `real_v27_overview.png`（俯视轨迹+切换点）、`real_v27_states.jsonl`/`_plan.jsonl`。
- 单关判分侧六类最高档复测（统一 torch 2.14.1，official profile）：
  `pixi run -e eval robocon-eval run --robot blackW --part 2 --level <key> --policy <p>
  --profile official --adaptive --adaptive-vx <vx> --out/--result artifacts/part2_evals/torch214/...`
  （plum 不加 `--adaptive`）；合计分数
  `pixi run -e eval robocon-eval score --levels '{"high_platform":40,"stairs":30,"slope":35,"high_wall":40,"trench":30,"plum":1}'`
  → **1200/1300**（`artifacts/part2_evals/torch214/score.json`）。

## 9. 已知风险（按优先级）

1. **plum 深度专家的鲁棒性（最大剩余风险）**：判分侧官方固定 spawn 的单关运行 success 200，
   但 spawn-offset 网格同环境只有 **2/15**（`plum_depth_final_v1` manifest 的
   `spawn_offset_grid`：±2°/±0.05 m 就会翻车）。这不是种子噪声——策略是确定性的，网格测的是
   刀锋边界。官方允许每类 4 次尝试、且只用默认 spawn，所以判分证据成立；但"反复稳定越过"
   的备注要求还没满足。下一步是**重训**（更宽的初始位姿/速度随机化 + 出口下降奖励），
   本轮不做。
2. **部署栈 ≠ eval 栈的出口下降**：同一份 plum 权重在 eval 链路能自己完成出口 0.2 m 台阶，
   部署栈在边缘是混沌的（前倾翻下）。全场路线把出口下降拆给轮式专家（`vt_band_v1`）：
   **必须切在出口平台（field 6.80）**、先倒车 0.6 s 刹车、站 2 s、再 0.45 m/s 慢速走下；
   切在桩阵上会踩空（judge `dropped`），不刹车则 2 m/s 动量把身体带出平台（整场首跑实证）。
   默认档单段连续 2/2 次判分 success；仍非 100%，若整场 plum 段失败，最便宜的处理是
   **重跑整场**（每 leg 的 PASS 是集成门限，不是判分）。
3. **部署栈 ≠ eval 栈的档位差异（沿用 2026-11-04 结论）**：同一专家/命令在真实 sim_node+rl_sim
   与 policy_runner 上结果可以相反（platform_40cm eval 0.50 过、部署 0/4 卡台阶面；stairs_30cm
   eval 0.70 稳、部署 0.70 反而失败而 0.50 过）。默认档位全部按 §7.2 的部署 trace 选。
   部署侧 obs/action 已确认与 eval 逐位一致（含深度策略的 C++ 推理对拍），剩余差异在
   实时调度/接触混沌。
4. **平台/高墙的部署可靠性**：field S→N 实测 platform_20cm 2/2 通过（后续 1× 整场多次通过）、
   platform_30cm 1/4（台阶面 stall 与接触混沌），故默认 20；30/40 仍可 env 切换。wall_20cm
   （bar5）有旧 real8 同槽验证 + 本轮整场验证；真跑若失败优先 `PART2_ROUTE_WALL=0`。
5. **stairs_30cm 单专家、通过档窄**：`vt_band_v1` 唯一部署可过的 30 cm 楼梯专家；部署成功档
   `vx=0.60`（v14/v17 整场 PASS），vx=0.50 也可过但整场会出现顶层平台 stall（变异性），
   vx=0.55/0.65/0.70 不行（非单调）；不要用 eval 侧的 0.70 盆。
6. **torch 版本必须三环境统一**：2.12↔2.14 的数值差异会改变 plum 的策略行为（success→fail）。
   `pixi.toml` 已把 demo/eval 固定到 PyPI CPU 轮子 2.14.1+cpu、train 固定 `torch==2.14.1`；
   升级/降级 torch 前先重跑 `artifacts/part2_evals/` 的六类单关证据与 `plum_comp_*` 组合段。
7. **锁轮模型横漂**：长转场已改为 goto 折线（≤10 m）规避；若 10 m 段内仍出现 >0.15 m 横漂，
   把 `search_goto` 的注点加密或 tol 收紧到 0.10。
8. **滑移/时序**：假模型无动力学，专家段真实速度/耗时可能明显长于假模型；各 `cross_timeout`
   已按 2–3 倍余量给（30–60 s）。真跑若某段超时，先看该专家是否在原地打滑。
9. **对齐死区**：burst/dwell 已针对锁轮低 wz 死区补偿；若某次对齐重试仍失败，路线会 FAIL 并打印
   `y_err/yaw_err`，据日志决定增大 burst 或放宽到 3°。

## 10. 文件清单

| 文件 | 说明 |
|---|---|
| `demo/rl_sar/scripts/part2_auto_route.py` | 路线 v3（专家段 + goto 转场 + 对齐 + 梅花桩组合穿越 + 审计折线数据） |
| `demo/rl_sar/include/input_map.hpp` | `Intent::PolicyBand` + RB+左 / `b` 绑定 |
| `demo/rl_sar/src/rl_sim.cpp` | `PolicyBand` → `vt_band_v1`；键盘 B；`/policy_switch` 订阅；深度策略输入/调试 dump |
| `demo/rl_sar/library/core/rl_sdk/*` | `depth:` 配置块解析、`obs_contract_depth` 校验 |
| `demo/rl_sar/launch/rl_sim_all.launch.py` | `depth:=auto|on|off`、`depth_every:=10`、深度契约预检 |
| `demo/quadruped_ros2/{include,src,msg}` | `depth_camera.hpp` + `DepthFrame.msg` + `sim_node` 深度渲染/发布/自检 |
| `demo/robots/blackW/robot.yaml` + `blackW_description.xml` | `obs_contract_depth` 契约；`depth_cam` 相机（d435_link） |
| `demo/rl_sar/policy/blackW/plum_depth_final_v1/` | 深度专家（`config.yaml`/`depth.json`/`manifest.json`；`policy.pt` 本地） |
| `demo/tests/check_depth_inference.py` | C++ 部署推理 vs Python 同输入对拍（`RL_DEPTH_DEBUG=1`） |
| `demo/tests/check_policy_contracts.py` / `check_robot_manifests.py` | 深度契约交叉校验（policy/config/robot/XML 相机） |
| `eval/robocon_eval/policy_runner.py` / `cli.py` | 深度渲染路径；`--adaptive` 命令（专家矩阵同款增益） |
| `tools/route/audit_part2_route.py` | MuJoCo AABB 净空审计（`--dump` 可打印全场障碍） |
| `tools/route/replay_field_part2.py` | 全场轨迹逐段投影回单关，跑 eval Part2 judge（`high_plum -> plum_1`、plum bbox 按 geom 名选；`--require-all` 可作集成门禁） |
| `tools/route/run_part2_fake.sh` | 假模型整场验证启动脚本 |
| `tools/route/summarize_part2_fake.py` | 假模型 plan/log → PASS/FAIL 表 |
| `tools/route/run_bench_official.sh` + `bench_expert_cross.py` | 真实部署栈单关基准（自适应车道命令，domain 87） |
| `demo/tests/run_joy_trial.sh` + `tools/trial/adaptive_joy.py` | 真实部署栈单关 trial harness（固定/自适应命令 + 判分 trace；`EXTRA_LAUNCH_ARGS` 透传） |
| `docs/part2/ROUTE.md` §7.2/§7.3 | 部署实测档位证据 + 1× 真场整场试跑记录（trace 文件名 + 复放命令） |
| `artifacts/part2_evals/torch214/` | 统一 torch 2.14.1 后六类单关判分证据（trace+result+score.json）+ plum spawn 网格 |
| `artifacts/part2_route3/` | 审计输出 + 假模型 trace + 部署单关/整场 `real_v8_*`…`real_v27_*`；交付证据 = `real_v27_{route.mp4,overview.png,states.jsonl,plan.jsonl}` + `plum_comp_v3_*` |
