# Part1 一键自动整场路线（操作者手册）

> 目标：**打开即可用**——机器人自动走完 Part1 全场；操作者手柄最多只需要「开始/复位」。
> 本入口是演示/回归用途（减少手柄操作不确定性）；比赛操作手仍以
> [`joystick.md`](joystick.md) 的手动流程为准。

## 1. 启动（一条命令）

```bash
pixi run -e demo demo-part1
# 无窗口：pixi run -e demo demo-part1 viewer:=false
```

要点：

- `route:=part1`：自动启动 `part1_auto_route` 虚拟手柄节点（发布 `/joy`），自动完成
  A 站立 → RB+十字上进 RL → 逐段走路线。**不需要任何手柄操作**。
- `stop_on_fail:=true`（默认）：第一段路线预检 FAIL 即停止路线并关闭整个 launch（不再卡在
  掉落点等超时）；排查时传 `false` 保留旧行为。
- launch 默认 `realtime:=true`，保留原路线的墙钟控制，按 1× 运行；
  Part1 现场演示应保持此参数，不使用 `realtime:=false` 全速运行。
- MuJoCo 窗口默认开启；传 `viewer:=false` 可无窗口实时运行。
- 真机/带手柄场景：`joy:=true`；自动路线节点与手柄共存（它就是一个 `/joy` 发布者），但自动运行
  期间不要摇杆。
- 判分 trace：launch 的 `eval_state_every=auto` 在自动路线中默认用 10，在其它场景中用 1；
  判分时显式传 `eval_state_every:=1`（每 2 ms 一帧）。
- `demo-part1` 是实体运动演示预设：默认 `PART1_ROUTE_JUDGE=0`，使用原路线 precheck；
  显式设为 `1` 可启用原版 live judge。直接 `ros2 launch` 的 live judge 默认和正式判分器保持原样。

## 2. 路线流程（节点自动执行）

起点 (3.7,-9.0) → 每段先由衔接模型（默认 `vt_locked12`；`transfer_policy:=...` 可显式换模型）低速闭环对线/落位，再切专家过障：

| # | 段 | 用的模型/动作 |
|---|---|---|
| 1 | 高墙 high_wall | row2p 0.5 m/s |
| 2 | T 梯（南臂上/北臂下） | st30 0.45 m/s（x 保持 1.8） |
| 3 | 短坡 → 桥 A 西台 | row2p 0.4 m/s；台上定心 + 原地转 +x |
| 4 | 积木桥 A（西台→东台） | row2p（失败自动重试 3 种入口） |
| 5 | 桥 A 东台定心 → 桥 B 窄带 | 台上相对转 -90° → `bridge_drive` 固定动作 0.32 |
| 6 | 长坡直行向南 | row2p（**退出条件含 z<0.54**：必须到平地才交还） |
| 7 | 限高杆 | `low_bar_drive` 固定动作 |
| 8 | 砂砾坑 L 形 | 主办方 `obstacle` 北进东臂；衔接模型定向到角点、顺时针转西并西出 |
| 9 | S 弯 7 关键帧 | 衔接模型低速绕杆 |
| 10 | 返回起点 | 衔接模型 |

当前 `vt_locked12` 在进入策略时保存四个实测轮角，持续保持该目标角度和零目标轮速。
当前默认入口恢复 v55/v67 阶段保留源码，与 v56/v72 冻结快照哈希一致，SHA256 `c556739c…`。
砂砾坑入口仍为 `x=5.70`，专家循迹 `x_hold=5.95`，门限 `y<-11.95`；
随后保持交接航向、用带正负号的位置修正到 `y=-12.55`，再顺时针转西。
桥 A 恢复原换模门限，桥 B 恢复以 `x=5.65` 为中心的原对线，低杆恢复原 `(5.70,-8.20)` 路点。
桥 B 外侧木带中心 `5.35` / `5.95` 的几何中线确为 `5.65`。

本次新的桥 A 延后交接、桥 B 精对线、低杆有界纠偏与四轮上坑即换模仍未解决，
已从默认入口撤回；实验源码与失败记录保存在 `artifacts/route_fix/part1_final_calibration_v88_*`。
默认只保留一套 Part1 自动路线，实验副本不供现场快捷入口选择。

**边缘恢复（自动）**：每段落位前，节点检查四轮着地面。若在平地上四轮分处两个高度面
（轮面差 >6 cm 且车体倾角 <10°——倾角大说明在坡/梯上，是几何分层不触发），自动切
`vt_edge_v1`（边缘恢复专家），朝多数平面的轮心质心行驶至四轮回同一平面后切回衔接模型。
日志关键字：`wheel-off-edge detected` / `edge recovery done`。结束日志含 `wheel recoveries: N`。

## 2.1 路线 PASS 与 eval 判分的关系（2026-11-02 对齐）

路线日志里的 `xxx: PASS/FAIL` 是**路线排序用的门控**；官方分仍以 `eval/` 的 Part1 judge 对
2 ms trace 回放为准。**启用 `PART1_ROUTE_JUDGE=1` 时障碍段 PASS/FAIL 直接查询 live `Part1JudgeV2`**（与
`robocon-eval replay-part1` 同一份代码、同一套常量），路线里只剩“怎么走/何时切策略”的
sequencing；仍属于路线侧的只有：平台原地转、停走、落位这类非计分动作。已知差异：

| 段 | 路线动作/排序 | eval judge（`eval/robocon_eval/part1/judge.py`） |
|---|---|---|
| high_wall | row2p 0.5 m/s 沿墙宽直行（support 证据取自四轮世界坐标） | support 全跨、x 在墙宽内；无 support 时退化为 base-z/墙接触证据 |
| t_stairs | st30 沿 x=1.8 车道到北侧臂外 | 南臂上 + 北臂下（不同臂）、逐级 step 证据、从北侧 exit |
| bridge_blocks | row2p 沿 y≈0 车道过桥到东台 | 逐块 support/exit 证据；仅到达东台还未完成出口判据 |
| bridge_bands / slope / height_bar / gravel | 固定动作或专家走到下一路点即 PASS | band/坡/杆/坑的支撑与边界事件、部分分/失败条件 |
| slalom | 7 个关键帧到几个报几个（不强制全过） | 7 根杆 + corner + 起终点圆、顺序和方向 |
| return_start | goto 回起点即可 | 8 个障碍全成功后的回起点 bonus（100 分） |
| attempts/fall | 不建模 | 3 次 attempt、fall 暂停/恢复、跳过、计时 |

因此：

- 启用 live judge 后障碍段路线 PASS 与其一致；但路线仍有“走法”自由（例如绕路/切车），
  live judge 在无 support 数据或不同采样下可能略有差异，**最终仍以 trace 回放为准**。
- `demo-part1` 默认关闭 live judge，以原路线 precheck 演示实体运动；这不代表正式判分全通过。
- 回放方法（与路线同时录一份 judge trace）：

```bash
# 终端 A（录 2 ms/50 Hz trace；路线内部只记原始 /eval_state 供视频用）
python demo/tests/record_eval_trace.py --part 1 --robot blackW \
    --scene $PWD/demo/robots/blackW/scene_part1.xml --seconds 900 \
    --out artifacts/route/part1_judge_trace.jsonl
# 终端 B
PART1_ROUTE_JUDGE=1 pixi run -e demo demo-part1 viewer:=false eval_state_every:=1
# 结束后
pixi run -e eval robocon-eval replay-part1 --trace artifacts/route/part1_judge_trace.jsonl
```

后续若要把路线预检进一步对齐 judge，只需把更多段的 `report(...)` 证据改用
`eval/robocon_eval/part1/judge.py` 的 support/zone 常量与事件（high_wall 已改为该方式）；
不建议在路线里复制判分逻辑，判分仍以独立 replay 为准。

## 3. 手柄按键（全部可选）

| 按键 | 作用 |
|---|---|
| A | 站立（GetUp）——自动路线自己会按；手动接管时用 |
| RB+十字上 | 进 RL（EnterRL）——同上 |
| Start / Back+X 等 | 任何按键都会作为 `/joy` 输入；自动运行期间**不要按**（会插入指令） |
| Back+十字上 | 手动切 `vt_edge_v1`（排障用） |
| LB+十字左 | 手动切回 `vt_pose_v1` |

「复位」= 关掉 launch 重新跑同一条命令（仿真场景从头初始化）。

## 3.5 转向所需平地范围（实测）

原地转 90° 时四轮扫掠足迹（静止矩形 0.53×0.42 m）：

| 模型 | 90° 足迹半径 | 扫掠包围盒 | 相对静止的额外占用 |
|---|---|---|---|
| vt_pose_v1 | 0.52-0.64 m | ~1.04×0.97 m | 每侧最多 +0.34 m |

该 `vt_pose_v1` 探针需要约 0.65 m 半径（≈1.1×1.1 m 净空），超过桥 A 西台的
1.0×1.0 m 尺寸。这是旧策略在此处的已知限制；不能直接作为其它策略的转向足迹。
当前保留 `locked12_full40` 视频对应的短坡上西台、台上转向、桥 A 接桥 B 的原路线。

## 4. 预期结果与时间线

- **当前默认已恢复 v55/v67 阶段保留源码**，不包含此次失败校准。恢复后未再跑物理整场。
- [v88 实验记录](../../artifacts/route_fix/part1_final_calibration_v88_evidence.json) 退出状态 1：
  墙、楼梯、桥 A 通过，桥 B 精对线超门限；v87 同处失败，v85 / v86 两桥通过后低杆前对线失败。
  实验源码和 13 项检查副本均保留在同目录，未默认启用。延后桥 A 换模、桥 B 精对线、低杆纠偏、
  四轮上坑即换模等本次请求仍需后续完成。
- 调整前的 [v67 整场](../../artifacts/route_fix/part1_final_v67_evidence.json)：
  开窗口、1× 实时完成，全部路线门限 PASS、0 次恢复、退出状态 0。
  共 13109 帧、262.16 s，最终车身 `(3.561, -8.939, 0.557)`，已返回起点。
  这是关闭 live judge 的实体运动演示，未据此宣称正式得分。
- 此前 [v55 真锁轮 + 砂砾坑局部修正整场](../../artifacts/route_fix/part1_fixed_wheels_v55_evidence.json)
  同样全部门限 PASS、0 次恢复、退出状态 0；13346 帧、266.90 s，终点
  `(3.562, -8.925, 0.557)`，保留为历史证据。
- 已有视频 `artifacts/route/locked12_full40_operator.mp4` 及 `locked12_full40_states.jsonl` / `locked12_full40_plan.jsonl`
  证明原路线完成实体运动并回到起点；记录长度约 338 s，计划中衔接策略为 `vt_locked12`。
  原始启动命令和全部环境变量未保存。这份历史视频早于此次真锁轮修复，仍是当时实体运动成功的证据。
- 2026-11-07 真锁轮修复前的 1× 无窗口基线复跑也完成：所有路线门限通过、0 次恢复，
  退出状态 0，最终车身 `(3.533, -8.895, 0.557)`，记录约 354 s。
  [复跑证据](../../artifacts/route_fix/part1_baseline_recheck_evidence.json) 保存命令、状态及日志路径；
  此次使用实体运动预设，未启用 live judge。
- 终端逐段打印 `PASS/FAIL`；结束打印 `part1_auto_route finished (wheel recoveries: N)`。
- 历史 full40 视频 trace 用原版 judge 回放为 500 分，不能作为正式全通过证据；
  实体运动演示和严格判分分别验证，正式判分器未为演示修改。
- 现场运行时避免并发物理仿真或训练，以减少 50 Hz 控制环抖动；一次只跑一个 launch 实例，
  避免 `/joy` 与 `/eval_state` 串台。
- 现场扰动和机器负载仍影响重复性；原路线拓扑保留，本次仅修正锁轮部署与砂砾坑局部控制。
- 单段重跑（调试用）：`PART1_ROUTE_ONLY=t_stairs`（可选 `slope_hold` / `bridge_A_rel` /
  `height_bar` / `bridge_bands_east` 等，见脚本头部）。

## 5. 常见问题

| 现象 | 原因/处理 |
|---|---|
| 想按 1× 现场演示 | 保持默认 `realtime:=true`；Part1 原控制器使用墙钟节拍 |
| `no /eval_state` 报错 | 检查 `route:=part1` 是否生效；自动路线默认启用反馈，不需要额外参数 |
| 某段 FAIL | 默认立即停止并返回失败；调试传 `stop_on_fail:=false` 可继续，最终仍返回失败 |
| 想换衔接模型 | `pixi run -e demo demo-part1 transfer_policy:=<policy_name>`（默认 `vt_locked12`；也接受 `PART1_TRANSFER_POLICY`） |

## 6. 依赖

- `rl_sar/policy/blackW/` 下需要：`vt_locked12`、`vt_edge_v1`、`row2p_p40w40_v1`、
  `st30_ppo_st30_s100_600`、`obstacle` 与固定动作 `behaviors/{low_bar_drive,bridge_drive}.yaml`；
  权重按 `config.yaml`+`manifest.json` 本地补齐（不入库）。
