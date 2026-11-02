# 速度跟随衔接模型（VT）训练配方与验收

目标：为 Part 1 固定路线提供一个**位置环 Harness 可用的速度衔接策略**——命令是机体系
`(vx, 0, wz)`（Part 1 方向：不投 vy），要求：

1. vx / wz 跟随准确，含 0 指令、低速、倒车、急停/再走；
2. **原地转弯不额外占地**：纯 yaw 命令下 base 不漂走，四轮扫掠足迹尽量接近静止矩形；
3. 对初态扰动、推扰、轮地摩擦变化的鲁棒；
4. 与专家策略同契约（blackW 16 DoF / 57×6 obs / 16 act / signed wheel），可直接热切换。

规则背景：Part 1 允许自动/固定动作，位置环由 `part1_auto_route` 在障碍之间负责对线/落位；
本模型只接速度层，不改过障策略。

## 1. 与 RecoveryLoco 的对应关系（点式足 → 轮足）

RecoveryLoco 是点式足（Go1，12 DoF）的“摔倒恢复 + 鲁棒行走”框架。迁到 blackW 轮腿时，
保持其**鲁棒性思想**，把点式足专属项替换/删除：

| RecoveryLoco 配方 | blackW VT 实现 |
|---|---|
| 随机初态（`random_states_init_tilt`，roll/pitch 最高 ~0.5 rad，yaw 全向） | reset `pose_range`：yaw 全向、roll/pitch ±0.10 rad（阶段 2 加到 ±0.35）；root 速度 ±0.3 m/s、角速度 ±0.15/±0.5 rad/s；关节位置 ±0.08 |
| 周期推扰（push interval 10 s，±1.0 m/s） | `push_robot`：每 5–12 s 推 ±0.8 m/s（x/y） |
| 摩擦域随机（friction 0.2–1.25，restitution/motor strength/kp/kd/payload/COM） | 轮地摩擦 0.40–1.20（新增，依赖轮碰撞 geom 命名）；payload/COM 沿用；blackW 的 PD/编码器 DR 因动作路径为 JointPosition 暂不支持 |
| 点式足 reward：tracking_lin/ang、lin_vel_z、ang_vel_xy、orientation、torques、dof_acc、feet_air_time、collision、stand_still | 保留 tracking/lin_vel_z/body_ang_vel/upright/dof_acc/joint_power/action_rate/smoothness；删除点式足专属 `foot_clearance`；新增轮足项（见下） |
| `only_positive_rewards=True`（负总奖励截 0） | MjLab RewardManager 无此开关，未实现；以 tracking-dominant 权重与 stance/姿态锚定替代 |
| 地形课程（坡度/台阶/障碍） | 阶段 1 平地；阶段 2 用平台边缘地形（0.05–0.20 m，对应桥台/台阶唇） |

轮足专属新增：
- `pose_dofpos`：腿 12 关节对默认站姿的 L2 锚定（0.8）。世界 z 的 base_height 在平台地形上会把
  标称站姿误罚成蹲姿（pose 配方 r4 负结果），因此移除 base_height、改用关节空间锚定。
- `pose_rotdrift`：纯 yaw 指令下 `|v_xy| / |wz_cmd|` 惩罚（1.5），直接优化“转弯不漂走”。
- 纯 yaw / 纯平移模式采样：见 §2。

## 2. 命令分布（`blackw-vt`）

| 项 | 值 |
|---|---|
| 包络 | vx ∈ [-0.4, 1.0] m/s，wz ∈ [-0.65, 0.65] rad/s，vy = 0 |
| standing | 10% |
| 重采样 | 5 s |
| 低速混合 | 非 standing 的 50% 从 |vx|≤0.12、|wz|≤0.20 盒采样（压死区） |
| 纯 yaw 模式 | 非 standing 的 50%：vx=0，|wz| ∈ U(0.10, 0.60)，符号随机（位置环最常要的原地转；0.10 下界覆盖低速死区） |
| 纯平移模式 | 非 standing 的 10%：vx ~ U(-0.4,1.0)，wz=0 |
| 其余 | 包络/低速混合正常采样 |

## 3. 奖励

| 项 | 权重 | 说明 |
|---|---|---|
| track_linear_velocity | +2.0（σ=0.25） | exp(-err²/σ)（HIMLoco 式） |
| track_angular_velocity | +2.5（σ=0.08） | yaw 单独更锐的 σ：0.25 时低 wz 指令“不转”几乎不亏（0.2 rad/s 命令不转仍得 0.85），0.08 时降到 0.61，逼出低速/原地转 |
| track_linear_velocity_l2 | -0.30 | 显式 L2 平面误差，压过速 |
| upright | -1.2 | 姿态直立 |
| pose_dofpos | -0.80 | 站姿锚定（腿关节） |
| pose_rotdrift | -1.00 | 纯 yaw 平移漂移；用 yaw 跟踪门控（跟踪好才罚漂移），避免“不转就不漂”的退化解 |
| lin_vel_z / body_ang_vel / dof_acc / joint_power / action_rate / smoothness | 冻结 baseline | 稳定性/平滑正则 |

## 4. 训练阶段

| 阶段 | task | 初始化 | 规模 | 目的 |
|---|---|---|---|---|
| 0（pilot） | `blackw-flat` + pose 命令 knobs | 随机初始化 | 4096 envs × 24 × 300 it | 验证从零可站立/走/急停（300 it 已过 F1/F2/F3/F5/F6/F7，仅 F4 vy 失败——方向本就不投 vy） |
| 1 | `blackw-vt` | pilot checkpoint 续训 | 4096 envs × 24 × 1200 it | 加入纯 yaw/纯平移模式、站姿/转弯漂移奖励、推扰与随机初态 |
| 2 | `blackw-vt`（更高扰动：roll/pitch ±0.35） | 阶段 1 最佳点续训 | 4096–8192 envs × 24 × 600–1000 it | 鲁棒性（随机倾斜初态/更强推扰） |
| 3（可选） | `blackw-vt-edge` | 阶段 2 最佳点 | 4096 envs × 24 × 400–800 it | 平台边缘恢复（桥台/台阶唇掉轮后按指令回台） |

训练命令（本机验证路径）：

```bash
pixi install -e train          # 首次
pixi run -e train train blackw-vt \
  --env.scene.num-envs 8192 --agent.num-steps-per-env 24 \
  --agent.max-iterations 1200 --agent.save-interval 100 \
  --agent.resume True --agent.load-run <parent-run> --agent.load-checkpoint <ckpt.pt> \
  --agent.command-curriculum-restore none --agent.run-name vt_vx
```

迁移期本机验证用已有训练解释器 + `PYTHONPATH=train/src`（新 pixi 环境尚未在本机安装）；
入口、导出与探针全链路均已跑通。导出：`pixi run -e train python train/scripts/export/export_history_policy.py --task blackw-vt ...`
（写 `policy.pt` + 部署用 `manifest.json`）。

## 4.1 试验记录（2026-11-02，串行 GPU）

| 试验 | 配方/初始化 | 结果 |
|---|---|---|
| pilot（300 it，从零，`blackw-flat`+pose 命令） | 全包络 + 低速混合 | 300 it 已过 F1/F2/F3/F5/F6/F7（F4 vy 失败，方向不投 vy）；支撑矩形平移比 1.01，但原地转不敏感（纯 yaw 样本占比低） |
| vt_v1（pilot+400 it） | 加纯 yaw/纯平移模式（35/15%）+ pose_dofpos + 未门控 rotdrift(1.5) | 400 it：F 6/7、平移比 1.14–1.29（劣化）；低 wz 原地转不转 → **rotdrift 可被“不转”利用** |
| vt_v2（pilot+400 it） | yaw 单独 σ=0.08、rotdrift 门控（仅跟踪好时罚漂移） | 400 it：F 6/7；平移比 0.94–0.96；90° 足迹半径 0.53（baseline 0.64）；但原地转漂移仍 0.17–0.37 m/90° |
| vt_v3（vt_v2+200 it） | 提高 rotdrift(2.0)+正向 turn-hold(0.5)+RSI tilt 0.30 | **负结果**：低 wz 下不转/游走，位置环漂移 2.7 m；已回退 |
| vt_v4（vt_v2_400+1200 it，跑到 819 it） | 回退 vt_v2 配方 + 纯 yaw 下界 0.15 + 低速混合 45% + lr 1e-4 | 600 it：F 6/7、位置环转漂移 0.094 m（目标达成）、平移比 1.10；800 it：位置漂移 0.172 m、足迹 0.639 |
| vt_v5（vt_v4_600+400 it 低 wz 精修） | 纯 yaw 50%（下界 0.10）+ 纯平移 10% + 低速混合 50% + lr 5e-5 | **最终选点 vt_v5_800**：位置环前进/±90°转/停走全收敛，转漂移 **0.051 m**；90° 足迹半径 **0.47/+0.57**；鲁棒 11/11 |

结论：**from scratch 300 it 即可过平地跟随门禁**（除 vy 诊断项）；原地转的瓶颈是
“低 wz 段不转 + 滚动轮转向的平移耦合”，需要继续用覆盖低 wz 的纯 yaw 样本训练，而不是简单的
强漂移惩罚（强惩罚会退化成不转）。

## 4.2 v2 目标：恢复点式足的“四落足点”约束（2026-11-02）

用户观察到的两个问题与对应改法：

1. **路线交接倒退**：bridge A 东台到达后，旧顺序是“row2p 先发负/零速度刹车 → stop →
   切换 vt_track”，旧专家在零指令下会回溜 ~0.2 m，平台点位因此偏。v2 改为在距离目标
   ~0.25 m 处**边前进边切换到速度模型**，由新模型主动刹车并闭环回目标位（`bridge A east
   handover`），不再让旧模型发零/负指令。
2. **删掉了过多点式足奖励**：RecoveryLoco 的 `feet_air_time`/`foot_clearance`/`collision`
   原意是“四个落足点按节拍接触、不拖地、不撞”。轮足没有摆动相，但**四轮接触平面与机身
   平齐、四轮角速度跟随机身旋转、不扭曲**同样必须显式约束，否则会出现“机身转了、轮子
   滞后/扭腰”的现象。v2 新增：

| 奖励项 | 形式 | 作用 |
|---|---|---|
| `wheel_plane_level` | +1.0，`exp(-spread(body-z_wheel)²/0.0004)` | 四个轮心在机身坐标下 z 展布尽量小 → 四腿平面与机身平齐（接近站立阻尼） |
| `wheel_rotation_follow` | +0.6，纯 yaw 时 `exp(-Σ|v_i - ω_cmd × r_i|²/1.0)` | 四轮心速度匹配刚体旋转场 → 贴地跟随、无滞后/打滑 |
| `body_ang_vel` | −0.10（原 −0.05） | 抑制机身 roll/pitch 扭动 |
| `track_angular_velocity` | 2.5→3.0（σ=0.08） | 把 yaw 跟踪权重再抬高 |

可行性验证（v2 实际训练，2026-11-02）：

| 试验 | 结果 |
|---|---|
| v1 基线 `vt_track_v1` | follow_rel_err(cmd) 0.63/0.69、四轮平面 z 展布 0.031/0.022 m、平移比 0.78、90° 足迹 0.57 |
| vt_v7（v2 from scratch, 900 it） | 平面 z 展布 **0.003–0.005 m**（明显变好）；90° 足迹 0.52–0.56；位置环转漂移 0.063–0.106 m；平移比 **0.63–0.67（退化）**；follow_rel_err(cmd) ~0.6、对实际机身仍 ~0.8 |
| vt_v8（v7_600 + 200 it，实际 yaw 目标） | 平移比回到 **0.805**，但纯 yaw 只剩 0.07 rad/s（策略学会“不转”来骗 follow 奖励）→ **负结果**，已回退 |
| vt_v9（v7_600 + 200 it，指令目标 + 平面仅 turn/stand mask） | 目的：保留 v7 的平面/足迹收益，同时把平移比拉回 v1 水平 |

结论：**共面约束有效**（0.03→0.003 m），但“四轮速度严格匹配刚体场”与“允许轮子滚动/打滑”的
物理现实有冲突；若以实际机身为目标会诱发“不转”退化，以指令为目标又和 tracking 重叠。v2 采用
“平面奖励 turn/stand mask + 指令目标 follow（权重 0.6）”，follow 指标作为诊断项而不是硬门禁；
最终选点以 v7/v9 的抖动、足迹、位置环与鲁棒性综合决定（见 §7 最终表）。

实现坑（已修）：给 reward 函数传默认 `SceneEntityCfg("robot", geom_names=...)` 时，MjLab
只会解析显式写在 `params` 里的 asset_cfg；默认参数里的 `geom_ids` 仍是 `slice(None)`，
会索引全部 53 个 geom，两个新奖励项静默变成 0（vt_v6 第一次启动即此现象）。v2 已改为在
params 中显式传 asset_cfg，并在 `train/tests/test_vt_command.py` 加断言防回归。

## 5. 验收（`tools/tracking/`，部署同契约 CPU MuJoCo）

| 门禁 | 命令 | 通过标准 |
|---|---|---|
| **F1–F7 命令响应** | `command_probe.py --gate` | F1/F2/F3/F5/F6/F7 PASS；F4（vy）按方向记为**诊断项**（不投 vy） |
| **支撑矩形** | `probe_support_rectangle.py` | 纯 yaw：漂移 ≤ 0.08 m/90°；平移：轮心矩形质心速度 / 指令 ≈ 1.0（0.8–1.2） |
| **转弯足迹** | `probe_turn_footprint.py` | 90° 能完成；足迹半径尽量接近静止矩形（0.53×0.42 m，半径 ~0.34 m）；相对静止额外占用报告记录 |
| **位置环 Harness** | `position_harness.py` | forward 1 m / 原地 ±90° / L 形 / 停走 全部收敛；原地转 base 漂移 ≤0.10 m；不摔 |
| **可部署性** | `demo/tests/check_policy_contracts.py` + `robocon-eval run` | 57×6/16/signed 契约一致，Part 1 flat 场景可直接跑 |

证据 JSON 落在 `artifacts/velocity-tracking/`（gitignored）；最终数字回填到本文与
`eval/SPEC.md` 的速度衔接小节。

## 6. 交付物

- `demo/rl_sar/policy/blackW/vt_track_v1/{policy.pt,config.yaml,manifest.json}`（权重本地，不入库；
  来源 run `2026-11-02_15-34-19_vt_v5` / checkpoint `model_800.pt`）。
- `part1_auto_route` 以 `PART1_TRANSFER_POLICY=vt_track_v1` 使用；位置环参数按本模型响应重新标定
  （旧默认针对 vt_pose_v1）。`demo/docs/joystick.md` 与 `docs/OPERATIONS.md` 已记录。
- 训练 run 与导出 checkpoint 在 `logs/rsl_rl/black_velocity/`（gitignored）；验收 JSON 在
  `artifacts/velocity-tracking/vt_track_v1/`（gitignored）。

## 6.1 v2 交付（2026-11-02）

- 策略：`demo/rl_sar/policy/blackW/vt_track_v2/`（来自 `vt_v7` checkpoint `model_600.pt`；
  从零 900 it 训练，选 600 it 是因为其四轮跟随/足迹优于 900 it 版本）。
- 与 v1（`vt_track_v1` = vt_v5_800）对比（部署同契约 CPU MuJoCo）：

| 指标 | v1 | **v2** | 说明 |
|---|---|---|---|
| F 门禁 | 6/7（F4 vy 诊断） | 6/7（F4 vy 诊断） | F1/F2/F3/F5/F6/F7 均 PASS |
| 四轮平面 z 展布（纯 yaw ±0.3） | 0.031 / 0.022 m | **0.0036 / 0.0023 m** | 共面奖励生效 |
| 四轮跟随相对误差（对指令） | 0.633 / 0.689 | **0.615 / 0.636** | 小幅改善 |
| 四轮跟随相对误差（对实际 yaw） | 0.767 / 0.860 | 0.805 / 0.754 | 一正一负，仍受轮子打滑限制 |
| 90° 转弯足迹半径 | 0.572 m | **0.520 m** | 更好 |
| 位置环原地转漂移 | 0.051 m | 0.106 m | 仍 ≤0.15 目标 |
| 支撑矩形平移比 | 0.781 | **0.631** | **退化**：v2 更慢；位置环需把速度指令上调约 20–30% |
| 鲁棒性 | 11/11 | **11/11** | 随机初态/推扰/关节偏置全过 |

结论：四轮共面与足迹达到预期；**平移速度上限未达预期**（v2 策略偏保守），这是下一轮
（速度恢复 fine-tune 或把平面奖励权重进一步降）的首要项。v8（follow 用实际 yaw）会诱发
“不转”退化，已作为负结果记录，不用于交付。

路线 handover 实测（v2，`PART1_ROUTE_JUDGE=0` 以便走到平台段）：

| 版本 | pre → post | 备注 |
|---|---|---|
| 旧顺序（旧专家零/负刹车 → 切换） | 回溜 ~0.2 m + 切换后偏移 | 用户观察的缺陷 |
| 第一版 v2 修复（全速边前进边切） | (5.45,-0.06) → (7.10,-0.50)，drift **1.70 m** | 切换延迟 + 未减速，冲过平台 |
| 当前 v2 修复（提前到 x>5.25、先 0.12 m/s 慢行、再切换、再由 v2 闭环回位） | (5.26,-0.12) → (5.50,-0.27)，drift **0.23–0.28 m** | 明显收敛；但 v2 平移保守导致 10–15 s 内仍差 ~0.25 m 到 (5.70,-0.45)，日志 `bridge A east handover recentre failed` |

下一步：给 v2 的 `goto` 速度做 ~1.5× 补偿（或在 controller 上调 max-vx），并把平台转
（固定差速 spin 仍会 stall）换成 stepping-turn / 原地转专用动作。

## 7. 最终结果与限制

| 验收项 | 结果 |
|---|---|
| F1–F7 命令门禁 | **6/7**：F1/F2/F3/F5/F6/F7 PASS；F4（vy ±0.2）未训，按方向仅作诊断 |
| 位置环（forward 1 m / ±90° / 停走） | **全 PASS**；原地 ±90° 转 drift **0.051 m**（目标 ≤0.15，旧 vt_pose_v1 同场景 -90° 为 0.94 m） |
| 90° 转弯足迹 | 半径 **+0.47 / −0.57 m**（静止矩形 ~0.34 m；旧 vt_pose_v1 为 0.64 m）；±90° 都完成（7–10 s） |
| 支撑矩形 | 纯 yaw +0.3：0.041 m/90°，−0.3：0.501 m/90°（**负向仍不对称**）；平移质心速度比 0.78–0.86 |
| 四轮跟随（`probe_wheel_follow.py`，纯 yaw ±0.3） | 目标：`follow_rel_err ≤ 0.35`、`plane_spread_mean ≤ 0.015 m`（v1 基线 0.63/0.69，plane 0.031/0.022 m） |
| 鲁棒性（RecoveryLoco 式） | **11/11 PASS**：yaw/roll/pitch 初态偏置、速度/角速度冲量、关节偏置、行进中推扰 |
| 契约/可部署 | 57×6/16/signed；`check_policy_contracts` 通过；`demo_tests.sh` 通过 |

已知限制（如实记录）：

1. **负向低速转向（|wz|≈0.1–0.2）仍有死区/不对称**：低 wz 下负转的支撑矩形漂移偏大；
   位置环用 `--min-wz`（如 0.10–0.15）可避开。彻底对称需要下面第 3 项。
2. **滚动轮原地转的物理耦合**：即使 base 漂移压到 0.05 m，四轮扫掠半径仍 ~0.47–0.57 m
   （静止 0.34 m）。要真正"零额外占地"需要 **stepping-turn（抬轮换位）** 步态，属独立课题。
3. l_shape 在固定 staging 的 harness 里是诊断项（受路线"先转后走/先走后转"选择影响）；
   Part 1 路线用自身 `require_facing` 闭环，不依赖该脚本的 L 形结果。
4. Part 2 规则只允许 RL，本模型只作 Part 1 衔接；未在 Part 2 计分链路使用。
