# Part2 全场自动路线

现场入口：

```bash
pixi run -e demo demo-part2
# 无窗口仍按 1× 实时运行：
pixi run -e demo demo-part2 viewer:=false
```

首次运行先执行 `pixi install -e demo` 和 `pixi run -e demo demo-build`，并补齐本地策略权重。
启动器自动 source 工作区、选择 Part2 场景和 RL `part=2`、站立并进入 RL；长直线使用
`flat`，转向与精对线使用 `vt_locked12`。Part2 只允许学习策略过障，固定动作仍被禁用。一个进程退出或第一段失败时，
launch 会关闭整场；窗口默认跟随机器人，`Ctrl+F` 切固定视角。

## 唯一的路线定义

[`part2_route_plan.py`](../../demo/rl_sar/scripts/part2_route_plan.py) 是路线几何的唯一来源：
`LEGS` 给出有序障碍、精确 geom 名称、入口、朝向、出口和转场点，`AUDIT_SEGMENTS` 由它生成。
[`part2_auto_route.py`](../../demo/rl_sar/scripts/part2_auto_route.py) 执行计划，
[`audit_part2_route.py`](../../tools/route/audit_part2_route.py) 审计同一份数据。
`tools/live_demo.sh` 和 pixi task 仅启动统一 launch，不另存一套路线。
Part2 的线段跟随和时间同步放在 [`part2_motion.py`](../../demo/rl_sar/scripts/part2_motion.py)
的 `Part2Motion` 中；Part1 默认已恢复此前整场通过阶段保留的路线与墙钟节拍，
本次未稳定的入口校准实验已撤回，恢复后未重新跑物理整场。

场景为 `demo/scenes/part2/full_field/part2_fixed_field.xml`，其地形来自 `map_fixed.xml`。
`full_field:=true` 是原官方整场的人工查看入口，不是第二套自动路线；单关 `level:=...` 是独立评测入口。

当前唯一计划执行 **33 段**，逐类按难度递增跨越；40 cm 平台为明确记录的临时跳过项：

| 类别 | 实际跨越的全部规格 | 方向与几何说明 |
|---|---|---|
| 平台（6 段） | 5 / 10 / 15 / 20 / 25 / 30 cm | 西→东；沿 `y=0` 按实体排列连续跨越，同一专家保持前进，整排结束后再北转 |
| 楼梯（9 段） | 每级 4 / 6 / 8 / 10 / 12 / 15 / 20 / 25 / 30 cm | 西→东；每档 6 级上、顶平台、6 级下；顶高为单级高度的 6 倍，最高 1.8 m |
| 斜坡（7 段） | 5 / 10 / 15 / 20 / 25 / 30 / 35° | 西→东；斜坡单侧水平长 2 m，35° 顶高约 1.4004 m |
| 沟壑（3 段） | 10 / 20 / 30 cm 缝宽 | 西→东；沟壑两侧平台顶高是 30 cm，难度名称不是平台高度 |
| 梅花桩 | 比赛 20 cm 桩阵，6 个错列方桩 | 西→东；`y=12` 的入口平台、6 桩、出口平台；地图其它高桩练习行不属于该比赛桩阵 |
| 高墙（7 段） | 4 / 8 / 12 / 16 / 20 / 30 / 40 cm | 西→东；精确实体为 `cross_bar_1` 至 `cross_bar_7`，场地图中没有 10 cm 墙 |

这些是**整场实体的规格**，不能直接拿独立评测关卡名称替代。旧路线每类只挑一档，
遗漏低档位；此前 25 段计划又错误地在平台 20 cm、楼梯 20 cm、坡 30°、沟宽 20 cm、
墙 20 cm 截断。当前表按实际模型尺寸复核，覆盖所选最高档及全部较低实体。
规则最高档依次为平台 40 cm、楼梯单级 30 cm、坡 45°、沟宽 30 cm、墙 40 cm；
斜坡保留此前用户明确的“本轮不做 45°”约定，选到 35°，也不额外跨越 40° 练习坡。
地图中超过规则上限的其它练习实体保留，不属于当前要求。

**40 cm 平台仍是规则要求与当前能力缺口。** 当前部署在连续行进、单段重置和加长助跑中
仍未稳定通过；[v77 长助跑记录](../../artifacts/route_fix/part2_platform40_runup_v77.log)
已经爬上但随后摔倒。用户确认允许本次演示暂时跳过该档后，唯一计划用
`SKIPPED_TIERS["platform"][40]` 写明原因；场景实体和正式判据均保留，不把跳过计为通过。
执行段数从完整所选范围的 34 减为 33，后续仍需解决平台 40 cm，而非重新定义规则上限。

## 转场和过障的根因修复

旧转场每次重新追逐目标点，经过路点或横漂后会把目标判断成身后，形成明显的大角度回转。
旧 Part2 对线还显式执行“转向垂直车道→横移→转回来”，就是平地行走途中接近 90°
再回头的一个直接来源。当前衔接沿固定线段反馈，使用有界航向修正；对线改为最多 25°
的浅转向，在明确的跑道余量内修正横向误差。转场拐角需要的转向仍保留。

转场先合并同一直线上的细分路点，仅保留真实拐角。长度至少 4 m 的直线先用锁轮策略定向，
再切 `flat` 前进；距拐角 1.2 m 时制动并交还锁轮策略，完成最后接近和转向。
当前 `flat` 前进指令为 0.50 m/s；[v62 两段实测](../../artifacts/route_fix/part2_fast_transit_v62_evidence.json)
的匀速段中位速度约 0.74 m/s，含加减速的均速为 0.53 / 0.58 m/s，相对各自测量直线的最大
横偏为 3.1 / 1.0 cm。这是两段走廊探针，不能据此宣称精确跟随 0.50 m/s 或整场鲁棒性已验证。
[v81 自适应走廊探针](../../artifacts/route_fix/part2_flat_adaptive_v81_evidence.json) 在 21.48 s 内
前进投影 14.508 m（均速 0.675 m/s），相对入口冻结线的最大横偏 7.2 cm，随后完成首档楼梯，
退出状态 0；指令 0.50 m/s 仍不等于实际精确速度。

旧出口判断还把米制位移传给角度 `wrap()`，使相隔约 `2π` m 的错误位置也可能满足门限。
当前过障检查在该障碍的局部车道中直接计算米制前进量、横向偏差和方向，绑定该段精确实体，
不把越过身后或旁边障碍视为完成。对线、过障均有限时；失败会打印段名、位姿和残差。

当前按档默认配置如下；新增高档位仍需当前整场动力学验证，历史单关成功不代替该验证：

| 实体 | 专家 | 默认前进速度 |
|---|---|---|
| 平台 5–30 cm | `row2p_p40w40_v1` | 0.50 m/s |
| 楼梯 4–15 cm / 20 cm | `st30_ppo_st30_s100_600` | 0.45 / 0.50 m/s |
| 楼梯 25 / 30 cm | `vt_band_v1` | 0.60 m/s |
| 坡 5–30° / 35° | `row2p_p40w40_v1` | 0.70 / 0.50 m/s |
| 沟宽 10 / 20 cm | `row2p_p40w40_v1` | 0.70 m/s |
| 沟宽 30 cm | `s1d_trench_reward_v75_m100` | 0.80 m/s |
| 高墙全部档位 | `row2p_p40w40_v1` | 0.50 m/s |

`PART2_EXPERT_<类别>` / `PART2_VX_<类别>` 可显式覆盖对应类别配置，不改变实体清单。
对 row2p 的坡、沟壑以及 st30 的移动指令补偿小转向响应弱的问题，
指令仍限制在训练的 ±0.70 rad/s 内。
高墙整排沿用旧 `vt_mode_v2` 成功录像在障碍阶段的等价配置 `row2p@0.50`，
避免在短间隙内切换。`vt_band_v1@0.50` 墙探针未成功，已撤回作为当前墙默认配置。
20 cm 沟壑跨缝有北向侧漂，移动反馈参考取物理中线南侧 6 cm，提前保留余量；入口、出口、
车道越界与支撑判据仍使用原物理中线 `y=9`，不会随反馈参考移动。
同排相邻档位使用同一专家时，保持移动反馈，只检查实测衔接门限，
不在间隙内停车或原地转向；越界就停止并明确报错。楼梯 25→30 cm 是例外：在两档之间
3.1 m 的间隙停车并重置专家，重新满足 30 cm 楼梯的精确入口门限。楼梯 20→25 cm、沟宽 20→30 cm
需要更换专家，因此仍检查实际切换位姿，不能按同专家连续通过处理。
原地转向和低速修正曾使楼梯专家回退、沟壑专家转圈。每排最后一档保留落地后的停车检查。
楼梯 25/30 cm 保留 120 s 过障限时、60 s 停滞窗口：已知官方成功轨迹曾有约 56 s 的慢爬，
旧 12 s 停滞门限会提前终止它；这只调整运行时等待，不改变通过判据或允许高处倒车。
平台沿排列方向连续通过，前五个出口与下一档入口重合；30 cm 平台最后门限为 `(19.85, 0)`，
允许停车超越不超过 0.55 m，再经 `(20.15, 1.5)` 进入楼梯转场走廊，绕开未通过的 40 cm 平台。
梅花桩使用 `plum_depth_final_v1`，在出口平台交给 `vt_band_v1` 刹车、站稳并下降。
两段都是学习策略；深度专家不读取速度指令，进入后不能靠零指令刹停，因此必须显式切换策略。
入口最后一个转场点为 `(1.2, 11.8)`：在车道南侧完成必要的右转，补偿锁轮衔接模型转向时的
北向漂移，再沿浅角度前进并对齐 `(2.2, 12.0)`。已经经过的入口路点不再回头追逐；入口门限仍为
横偏不超过 6 cm、航向误差不超过 1.5°。出口默认在实测 `x≥6.85` 时切换：先保持前进
0.40 m/s 共 0.16 s，再以 −0.30 m/s 制动 0.6 s、零指令站定 2 s，随后以 0.45 m/s 下降。
切换、制动和站定期间保持上述固定输入；这几段仍逐帧检查支撑、倒地和车道，下降阶段恢复车道反馈。
桩阵中若轮底低于原判据的 5 cm 门限，或缺失支撑点遥测，立即卸载自走策略并报错，
不能再仅因车身到达出口就打印成功。

## 运行和调试

默认窗口与无窗口均为 1× 实时。需要全速回归时：

```bash
pixi run -e demo demo-part2 viewer:=false realtime:=false
```

launch 在实时和全速模式中均按仿真时间同步 50 Hz 控制，避免演示和回归采用不同的控制时钟。
`eval_state_every` 独立控制记录频率：快捷入口默认 10（50 Hz），判分 trace 显式传
`eval_state_every:=1`（每 2 ms 一帧），不会加快路线控制周期。
梅花桩现场验证应保留 `realtime:=true`：策略加载耗费墙钟时间，全速仿真在此期间仍会前进，
使自走深度专家的旧动作持续更长的仿真时间。时间同步不能消除这个切换延迟，
因此全速模式的梅花桩结果不能直接代替 1× 切换验证。

记录同一次运行的机器人状态和路线切换：

```bash
PART1_RECORD_PATH=$PWD/artifacts/part2_route/states.jsonl \
PART1_RECORD_PLAN_PATH=$PWD/artifacts/part2_route/plan.jsonl \
pixi run -e demo demo-part2
```

调试同一计划的子段：

```bash
PART2_ROUTE_START=stairs_04cm PART2_ROUTE_ONLY=stairs_04cm \
pixi run -e demo demo-part2 viewer:=false
```

`START` 从指定段开始，launch 从该段的 `transit[0]` 或 `stand` 推导默认出生点；
`ONLY` 在指定段后停止。两者也接受从 1 开始的序号；显式 `spawn_*` 仍可覆盖。
子段测试不代表从原点走完全部路线。旧几何选择变量 `PART2_*_LEVEL`、`PART2_PLUM_MODE`、
`PART2_ROUTE_WALL` / `_VARIANT` 已移除，设置后会报错，以免旧 shell 环境悄悄换路线。
专家/速度参数仍可调，不能改变跨越实体清单。

```bash
# 不依赖 ROS 的实际地形/覆盖/净空回归
pixi run -e eval python demo/tests/check_part2_route.py
pixi run -e eval python demo/tests/check_part2_replay.py
pixi run -e eval python tools/route/audit_part2_route.py
# Part2 衔接控制器的横漂、拐角与越过路点回归
pixi run -e demo python demo/tests/check_route_controller.py
pixi run -e demo python demo/tests/check_part2_controller.py
# 已录 trace 的运动学回放，不重新计算物理
MUJOCO_GL=egl pixi run -e eval python tools/video/replay_route_video.py \
    artifacts/part2_route/states.jsonl artifacts/part2_route/route.mp4 \
    artifacts/part2_route/plan.jsonl \
    --scene demo/scenes/part2/full_field/part2_fixed_field.xml --cli-only
```

`--cli-only` 生成简版演示视频，画面外只有 `> CLI:` 指令标注。
同一次完整运行的录制、样帧和局部预览命令见 [视频工具说明](../../tools/video/README.md)。

## 证据与剩余限制

路线 `PASS` 是集成门限，不是官方得分。独立评测的最高档成功不能证明部署栈中
每个低档位都能稳定通过。旧 `artifacts/part2_route3/real_v27_*` 只有每类一档，
不能作为新 33 段路线的覆盖证据。

当前录像可通过支撑点判据逐段复核：

```bash
pixi run -e eval python tools/route/replay_field_part2.py \
    --states artifacts/part2_route/states.jsonl --plan artifacts/part2_route/plan.jsonl \
    --out artifacts/part2_route/eval.json --require-all
```

回放器从同一计划选择精确实体并读取真实支撑点。有独立官方关卡的档位使用该官方 profile 的坐标
投影；其它中间档位从实际地形生成判据，标为 `field_geometry`，**不计官方分**。分数每类只取
最高成功档，不能累加同类低档位。整个结果标为场地演示投影，不代表官方独立关卡成绩。
`--require-all` 默认要求全部 33 段都有完整成功记录，空文件或缺段必失败；子段调试可显式传
`--expected-leg <段名或序号>`（可重复）。

同排高墙使用同一专家连续行进时，路线出口门限可能早于判据所需的 0.3 s 稳定保持。
回放仅允许相邻、已闭合、无衔接策略插入的同专家高墙段观察最多额外 0.5 s：原出口时必须已经
四轮越墙且稳定保持正在累计，只能补足这一次连续保持，不能在尾段获得越墙证据或重新开始保持。
尾段每帧必须有完整支撑点和有效接触记录，并在首次接触下一堵目标墙**之前**截止；遥测缺失、
保持中断或采样间断均不补判成功。报告保留原始 `window`，另列 `observation_end`、`tail_s`
和截止原因 `tail_stop`。官方判据及其中的稳定时间不变，此观察规则不用于其它障碍类别。

实际几何、覆盖与净空回归读取当前唯一计划及明确跳过项；真实动力学记录在 `artifacts/route_fix/`。
以下物理记录按当时计划解读，旧 25 段不能直接作为当前新增 8 段的证据。
`v56` 对应的旧计划已冻结为 `artifacts/route_fix/part2_full_live_v56_route_plan.py`，
仅用于历史溯源，启动入口始终读取当前唯一计划。

| 记录 | 路线集成门限 | 支撑点回放结果 |
|---|---|---|
| `part2_full_live_v84` | 从原点通过平台 5/10/15/20 cm 四段；25 cm 平台前轮上台、后轮卡在前沿，持续无前进后触发停滞，未倒地；整场失败 | [严格支撑点回放：前四段 success，25 cm 平台 `not_on_platform`](../../artifacts/route_fix/part2_full_live_v84_partial_eval.json)；[运行日志](../../artifacts/route_fix/part2_full_live_v84.log)；未完成 25 cm 平台及后续段 |
| `part2_full_live_v83` | 前 14 段集成门限通过，30 cm 楼梯入口停车对齐后横偏仍为 19.3 cm；退出状态 1 | [所选前 14 段 14/14 success](../../artifacts/route_fix/part2_full_live_v83_partial_eval.json)；未进入 30 cm 楼梯，整场失败 |
| `part2_full_live_v82` | 从原点连续完成平台六档、楼梯至 25 cm 共 14 段；30 cm 楼梯前横偏 12.2 cm、偏航 5.4° 超门限，未进入该障碍 | [14/33 success，19 段缺记录](../../artifacts/route_fix/part2_full_live_v82_eval.json)；完整回放检查失败，未放宽对齐门限 |
| `part2_full_live_v79` | 新 33 段计划从原点连续通过平台 5–30 cm 六段，随后在楼梯前西行走廊拐点未对准而退出；整场失败 | [平台六段 6/6 success](../../artifacts/route_fix/part2_full_live_v79_eval.json)；仅这六段的支撑点回放，不代表全部 33 段 |
| `part2_platform40_runup_v77` | 40 cm 平台加长助跑单段重试，爬上后摔倒 | 失败；经用户确认列入 `SKIPPED_TIERS`，不计成功 |
| `part2_full_live_v56` | **旧 25 段从原点连续完成，开窗口 1×，25/25 PASS、退出状态 0**；[全程 CLI 视频](../../artifacts/route_fix/part2_full_live_v56_cli.mp4) 长 1220.40 s，单次 trace，无拼接 | [24/25 success](../../artifacts/route_fix/part2_full_live_v56_eval.json)，末段 20 cm 墙 `not_stable_after_crossing`；含梅花桩 success 200，场地投影合计 550；不放宽稳定判据 |
| `part2_platform_x_v53` | 新西→东走向连续通过 4 档平台，使用固定轮角的衔接实现，整排出口转场完成、退出码 0 | [4/4 success](../../artifacts/route_fix/part2_platform_x_v53_eval.json)；该次为全速回归，整场现场验证仍使用 1× |
| `part2_rows_v38` | 连续通过 7 档楼梯、6 档斜坡及首档沟壑，次档沟壑前因横漂停止 | [14/14 success](../../artifacts/route_fix/part2_rows_v38_eval.json)，随后将低档沟壑默认专家改为 row2p |
| `part2_slopes_v33` | 补偿小转向响应后，6 档斜坡、2 档沟壑连续通过 | [8/8 success](../../artifacts/route_fix/part2_slopes_v33_eval.json) |
| `part2_trenches_v39` | row2p 连续通过 10、20 cm 沟壑 | [2/2 success](../../artifacts/route_fix/part2_trenches_v39_eval.json) |
| `part2_walls_v31` | 5 档高墙通过 | [5/5 success](../../artifacts/route_fix/part2_walls_v31_observation_eval.json)；16 cm 段使用上述连续稳定观察尾段 0.06 s，成功早于下一墙首次接触 0.22 s |
| `part2_plum_v25` | 旧集成门限到达出口 | [去重后的支撑点回放失败](../../artifacts/route_fix/part2_plum_v25_dedup_eval.json)：切换前左后轮已掉落；新运行时检查会拒绝该记录对应的支撑状态 |
| `part2_plum_live_v37` | 1× 实时测试越过桩阵，出口切换后摔倒 | [支撑点回放失败](../../artifacts/route_fix/part2_plum_live_v37_eval.json)：`fell:tilt=76.5`，不能归因于全速加载延迟 |
| `part2_plum_live_v49` | 1× 实时单段通过入口、桩阵、出口下降和后续两个转场点，退出码 0；采用上述南侧入口和 6.85 m 切换 | [1/1 success，200 分投影](../../artifacts/route_fix/part2_plum_live_v49_eval.json)；未改变正式支撑判据 |
| `part2_full_v40` | 历史南→北平台走向：从原点连续完成前 19 段，梅花桩入口横漂超门限后停止 | [19/25 success](../../artifacts/route_fix/part2_full_v40_eval.json)，含全部 4 档平台、7 档楼梯、6 档斜坡及 2 档沟壑；不能代替新西→东平台方向验证 |

`v83` 加入 30 cm 楼梯前停车对齐后，仍因横偏 19.3 cm 超过 10 cm 门限而终止，
偏航已回到 −0.5°；不能将朝向修正视为位置对齐成功。该次使用的 0.60 m 前进纠偏预算已耗尽。
当前入口停车点仍为 `(60.45, 3.0)`，将 `settle_forward` 增至 1.10 m、`along_max` 增至 1.60 m；
车身中心最迟到达 `x=62.05`，距 `x=63.0` 的楼梯前沿仍有 0.95 m。
入口横偏 10 cm、航向误差 2.5° 的精度门限保持不变。
`v84` 已包含这个预算调整，但在 25 cm 平台停止，**尚未物理验证新的 30 cm 楼梯对齐预算**。
`v82`/`v83` 的前 14 段成功与 `v84` 的前四段来自不同运行，不能拼接为一次连续完成证据。

**当前 33 个执行段尚无从原点连续完成并通过全部支撑点回放的记录。** 即使之后全部完成，
也不能宣称规则档位全通过，因为 40 cm 平台仍跳过。旧 25 段的 `v56` 已完成
实体运动，但严格回放仍有末墙稳定性失败；子段成功、静态净空或合成模型通过均不能代替新版
整场验证。上述回放也只是场地投影，不是官方独立关卡成绩。
`v56` 已在固定轮角实现修正后的 1× 整场中复核梅花桩成功；增加上游高档障碍后的入口状态仍需重验。

梅花桩鲁棒性尚未通过本次路线修改解决。现有深度权重对入口位姿/速度和出口下降敏感；
本任务不再训练梅花桩专家，继续使用现有权重验证入口对齐与出口切换。不能据一次成功承诺
所有现场运行都会完成。
