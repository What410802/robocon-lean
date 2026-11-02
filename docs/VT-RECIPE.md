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

## 7. 最终结果与限制

| 验收项 | 结果 |
|---|---|
| F1–F7 命令门禁 | **6/7**：F1/F2/F3/F5/F6/F7 PASS；F4（vy ±0.2）未训，按方向仅作诊断 |
| 位置环（forward 1 m / ±90° / 停走） | **全 PASS**；原地 ±90° 转 drift **0.051 m**（目标 ≤0.15，旧 vt_pose_v1 同场景 -90° 为 0.94 m） |
| 90° 转弯足迹 | 半径 **+0.47 / −0.57 m**（静止矩形 ~0.34 m；旧 vt_pose_v1 为 0.64 m）；±90° 都完成（7–10 s） |
| 支撑矩形 | 纯 yaw +0.3：0.041 m/90°，−0.3：0.501 m/90°（**负向仍不对称**）；平移质心速度比 0.78–0.86 |
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
