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
| 7 | 限高杆 | `low_bar_drive` 降姿后静止精对正，再固定动作通过 |
| 8 | 砂砾坑 L 形 | 主办方 `obstacle` 北进东臂；衔接模型定向到角点、顺时针转西并西出 |
| 9 | S 弯 7 关键帧 | 衔接模型低速绕杆 |
| 10 | 返回起点 | 衔接模型 |

当前 `vt_locked12` 在进入策略时保存四个实测轮角，持续保持该目标角度和零目标轮速。
当前只有一套默认整场路线。桥 A 在车身 `x>5.55` 且两只后轮参考点 `x>5.25` 后交接，
避免后轮仍处于最后一个积木与东台之间的空隙时换模。
桥 B 外侧木带内缘为 `x=5.45 / 5.85`，中线为 `x=5.65`；进场和固定姿态跨越均带横向位置反馈。
[实际接触核查](../../artifacts/route_fix/bridge_b_contact_audit/README.md) 区分轮体参考点和真实接触点：
旧 v67 右前轮在内缘有侧面接触；v90 和 v95 已消除侧碰，v95 四轮均接触顶面，
但右轮仍贴内缘，未确认充分余量。
砂砾东臂中心为 `x=5.8361`，入口恢复已到达坑面的 `(5.84,-10.20)` 路点（10 cm 容差）
与南向转正；连续失败的有界对线辅助已撤回。
专家只负责爬上东臂；四轮均实际接触坑面后通过相对 `policy_switch` 话题立即切回衔接模型，
省去旧专家零指令滑行阶段，不再额外等车身达到固定 y 门限。
随后衔接模型沿南向车道前进到 `y≈-12.50`，再顺时针转西；当前 lookahead 从 0.45 调至 0.30，
该小修已完成 v99 局部与 v100 整场验证。
限高杆保留原路点 `(5.70,-8.20)` 与正南目标；切入低姿态后短脉冲差速纠偏，
停稳 0.3 s 后再次确认偏角小于 1°才前进。原锁轮对准约有 3°残差，降姿态还可增加约 6°；
小差速指令受轮胎静摩擦影响，不能在行驶中及时消除。
[v102 / v103 局部实测](../../artifacts/route_fix/part1_lowbar_alignment_v102_v103_evidence.json)
过杆偏角为 1.28° / 1.12°，无记录到的机器人与杆柱接触。
早期 v85–v88 的窄容差桥 B / 低杆进场精对线未重新启用；
历史实验保存在 `artifacts/route_fix/`，不作为现场快捷入口提供。

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

- **当前源码 v104 整场已完成**：[完整证据](../../artifacts/route_fix/part1_lowbar_full_v104_evidence.json)。
  开窗口、1× 实时，全部路线预检通过、返回起点、退出 0、0 次恢复；13190 帧、263.80 s，
  末帧 `(3.5585,-8.9372,0.5571)`，源码 SHA256 `16da364e…` 与默认入口一致。
  限高杆低姿态精对正后偏角 0.54°，实际过杆偏角 0.69°，没有记录到机器人碰梁或柱，横杆中心高度保持不变。
  [同次实跑限高杆片段](../../artifacts/route_fix/part1_lowbar_full_v104_cli.mp4) 保留 t=123–134 s，场景外仅显示 `> CLI:`。
  沙砾坑首次满足四轮上坑判据与换模同在 t=137.84 s；下个记录帧 t=137.86 s 已应用锁轮 MIT。
  这是关闭 live judge 的实体演示，未据此宣称正式满分或广泛扰动下鲁棒。

- **上个已提交版本 v100 整场已完成**：[完整证据](../../artifacts/route_fix/part1_gravel_full_v100_evidence.json)。
  开窗口、1× 实时，全部 15 条路线预检通过、返回起点、退出 0、0 次恢复；14029 帧、280.58 s，
  末帧 `(3.5720,-8.9211,0.5558)`，该次源码 SHA256 `3bf5db2c…`。
  四轮上坑判据首次满足与换模均为 t=154.56 s；t=154.58 s 的首个后续记录帧已为四轮锁轮 MIT，
  四个轮角目标随后 0.8 s 均固定。车身同期南移 18.03 cm、东移 2.96 cm，是锁轮下制动与姿态收敛。
  角点西转出现的坑接触均为顶面，接触点最小余量 8.50 cm；悬空左后轮的完整水平投影最薄仅 2.11 mm，
  同帧轮底仍在坑内，不能据此宣称鲁棒余量充分。live judge 关闭，未宣称正式满分。
  [同次整场的砂砾连续片段](../../artifacts/route_fix/part1_gravel_full_v100_gravel_cli.mp4)
  保留 t=151.46–183.08 s，场景外仅显示 `> CLI:`。
- [v96 / v97 砂砾局部实跑与 CPU 几何审计](../../artifacts/route_fix/part1_gravel_v96_v97_evidence.json)：
  原对准器恢复后，两次均完成上坑、交接、角点、西行、下台，退出状态 0。v96 从更偏西/南的
  `(5.42,-10.37)` 出生，v97 从实测限高杆出口 `(5.5257,-10.2225)` 出生；两次均共用整场 `run_gravel()`。
  四轮锁轮 MIT 在交接后的首个 20 ms 记录帧生效，四个目标轮角在随后 0.8 s 都固定；
  车身同期南移 24.36 / 24.06 cm，东移 3.27 / 0.14 cm，最大倾角 6.46° / 5.15°。
  这段是锁轮衔接模型下制动，不是继续给专家前进指令。
- 两次交接与角点西转出现的坑面接触均为顶面；角点实际接触点最小外边距 9.40 / 9.21 cm。
  悬空左后轮完整圆柱水平投影最薄只有 1.05 / 0.88 cm；轮体参考点的 12 cm 余量不能替代此值。
  以上为几何接触候选，不含接触力，也不代表广泛扰动下的鲁棒性。
- [v98 开窗口 1× 整场](../../artifacts/route_fix/part1_gravel_full_v98_evidence.json) 完成全部路线预检并回到起点，
  退出 0、0 次恢复、13795 帧、275.9 s，末帧 `(3.5643,-8.9378,0.5562)`。
  该次已删除额外的 y 门限与重复入口转正，采用 0.45 lookahead；live judge 关闭，未宣称正式满分。
  [角点审计](../../artifacts/route_fix/gravel_contact_audit/v98/summary.json) 中实际承重点最小余量 7.99 cm，
  悬空左后轮上缘水平投影外伸 2.96 mm；该轮最低点仍在坑内、当帧无坑接触，不能解释为承重点跌出。
- 随后仅将角点 lookahead 从 0.45 改为 0.30，[v99 局部审计](../../artifacts/route_fix/gravel_contact_audit/v99/summary.json)
  对应砂砾全段通过、退出 0，最薄实际接触点 / 完整轮胎投影余量改善至 10.00 / 1.72 cm。
  该小修随后在上述 v100 整场通过；该局部对比不代替独立的整场证据。
- 较早 v90 / v94 / v95 的入口有界纠偏失败，已删除该辅助，保留原 `require + rotate_hold`。
  v93 未改动的西北坡失败；v94 增加的桥 B 进场转向使左前轮侧碰内缘，亦已撤回。
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
- 现场扰动和机器负载仍影响重复性；原路线拓扑保留，本次修正桥梁交接、车道反馈与砂砾局部控制。
- 单段重跑（调试用）：`PART1_ROUTE_ONLY=t_stairs`（可选 `slope_hold` / `bridge_A_rel` /
  `height_bar` / `bridge_bands_east` / `gravel` 等；`height_bar` 与 `gravel` 均复用整场同一个函数）。

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
