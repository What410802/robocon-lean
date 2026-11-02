# Part1 一键自动整场路线（操作者手册）

> 目标：**打开即可用**——机器人自动走完 Part1 全场；操作者手柄最多只需要「开始/复位」。
> 本入口是演示/回归用途（减少手柄操作不确定性）；比赛操作手仍以
> [`joystick.md`](joystick.md) 的手动流程为准。

## 1. 启动（一条命令）

```bash
source demo/install/setup.bash   # 仓库根目录，先 pixi run -e demo demo-build
ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=1 \
    policy_name:=vt_pose_v1 route:=part1 viewer:=false realtime:=true
```

要点：

- `route:=part1`：自动启动 `part1_auto_route` 虚拟手柄节点（发布 `/joy`），自动完成
  A 站立 → RB+十字上进 RL → 逐段走路线。**不需要任何手柄操作**。
- **`realtime:=true` 必须显式传**（launch 默认 `false` 是全速回归模式，会把墙钟节拍的路线控制器
  甩开：启动睡眠的几秒墙钟里仿真已跑百余秒，机器人零指令漂移出场）。
- 要看 MuJoCo 窗口就去掉 `viewer:=false`（窗口默认开启，`realtime:=true` 保持）。
- 真机/带手柄场景：`joy:=true`；自动路线节点与手柄共存（它就是一个 `/joy` 发布者），但自动运行
  期间不要摇杆。
- 判分 trace：`eval_state_every` 默认 1（每个 2 ms 物理帧一帧）。`route:=part1` 只在调用方显式
  传 `eval_state_every:=0` 时才把它抬到 10（50 Hz 位姿反馈）；判分跑保持默认 1。

## 2. 路线流程（节点自动执行）

起点 (3.7,-9.0) → 每段先由 `vt_pose_v1` 低速闭环对线/落位，再切专家过障：

| # | 段 | 用的模型/动作 |
|---|---|---|
| 1 | 高墙 high_wall | row2p 0.5 m/s |
| 2 | T 梯（南臂上/北臂下） | st30 0.45 m/s（x 保持 1.8） |
| 3 | 短坡 → 桥 A 西台 | row2p 0.4 m/s；台上定心 + 原地转 +x |
| 4 | 积木桥 A（西台→东台） | row2p（失败自动重试 3 种入口） |
| 5 | 桥 A 东台定心 → 桥 B 窄带 | 台上相对转 -90° → `bridge_drive` 固定动作 0.32 |
| 6 | 长坡直行向南 | row2p（**退出条件含 z<0.54**：必须到平地才交还） |
| 7 | 限高杆 | `low_bar_drive` 固定动作 |
| 8 | 砂砾坑 L 形 | 主办方 `obstacle`（北进、坑中转西、西出） |
| 9 | S 弯 7 关键帧 | vt_pose_v1 低速绕杆 |
| 10 | 返回起点 | vt_pose_v1 |

**边缘恢复（自动）**：每段落位前，节点检查四轮着地面。若在平地上四轮分处两个高度面
（轮面差 >6 cm 且车体倾角 <10°——倾角大说明在坡/梯上，是几何分层不触发），自动切
`vt_edge_v1`（边缘恢复专家），朝多数平面的轮心质心行驶至四轮回同一平面后切回 `vt_pose_v1`。
日志关键字：`wheel-off-edge detected` / `edge recovery done`。结束日志含 `wheel recoveries: N`。

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

**结论：任何现有模型原地转 90° 都需要约 0.65 m 半径（≈1.1×1.1 m 净空）。**
桥 A 西台只有 1.0×1.0 m——零余量，转向必然压边。这是几何约束不是模型缺陷；在 stepping-turn
步态出现前，**路线设计必须避免在 1.0 m 台上原地转**，或先在台下转好朝向再上台。180° 转向同样慎用。

## 4. 预期结果与时间线

- 全程约 **7-8 分钟**（1x 实时，全场 ~420 s 仿真）。
- 终端逐段打印 `PASS/FAIL`；结束打印 `part1_auto_route finished (wheel recoveries: N)`。
- 已验证：高墙 / T 梯 / 短坡连续通过；桥 A 西台 90° 定心转向是当前卡点（见 §3.5），
  该段失败后路线继续但后续积木桥入口质量劣化；斜坡/积木桥/砂砾坑在同构闭环里 5/6 通过。
- **运行前必须独占 CPU**：并行的 MuJoCo 探针/训练导出会抢资源，50 Hz 控制环抖动导致站立不稳、
  首段超时；一次只跑一个 launch 实例（`/joy` 与 `/eval_state` 会串台）。
- S 弯与返程为已知弱项（vt_pose_v1 低速绕杆精度不足）。
- 单段重跑（调试用）：`PART1_ROUTE_ONLY=t_stairs`（可选 `slope_hold` / `bridge_A_rel` /
  `height_bar` / `bridge_bands_east` 等，见脚本头部）。

## 5. 常见问题

| 现象 | 原因/处理 |
|---|---|
| 机器人开场就乱跑 | 忘了 `realtime:=true`（全速模式）——加参数重跑 |
| `no /eval_state` 报错 | 显式传了 `eval_state_every:=0` 但 route 未生效（例如节点没起）；默认 1 不需要额外参数 |
| 某段 FAIL 后卡住 | 节点会继续下一段；若位姿混乱导致 `arrive ... failed` 中止，直接重跑 |
| 想换衔接模型 | `PART1_TRANSFER_POLICY=<policy_name> ros2 launch ...`（默认 vt_pose_v1） |

## 6. 依赖

- `rl_sar/policy/blackW/` 下需要：`vt_pose_v1`、`vt_edge_v1`、`row2p_p40w40_v1`、
  `st30_ppo_st30_s100_600`、`obstacle` 与固定动作 `behaviors/{low_bar_drive,bridge_drive}.yaml`；
  权重按 `config.yaml`+`manifest.json` 本地补齐（不入库）。
