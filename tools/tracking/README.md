# tools/tracking/ — 速度跟随 / 位置环 Harness

这些脚本都在**部署同契约**的 CPU MuJoCo 上运行（`eval/robocon_eval/policy_runner.PolicyRunner`：
50 Hz 推理、5 ms PD、2 ms 物理、命令限幅），因此测的是实际会部署的行为，不依赖 ROS。

| 脚本 | 作用 |
|---|---|
| `command_probe.py` | 三轴速度命令响应 + **F1–F7 门禁**（vx sweep / 零指令 / 倒车 / vy / yaw / 急停再走 / 不摔） |
| `probe_support_rectangle.py` | 四轮支撑矩形：纯 yaw 的质心漂移（m/90°）、平移时矩形质心速度比 |
| `probe_turn_footprint.py` | 原地转 90/180° 的四轮扫掠足迹（半径/包围盒/相对静止额外占用） |
| `probe_pose_control.py` | 低速 `vx/vy/wz` 响应曲线与死区检查（位置环标定用） |
| `position_harness.py` | 闭环位置环：到点（forward / 原地 ±90° / L 形 / 停走），统计收敛时间、最终误差、原地转漂移 |
| `robustness_probe.py` | RecoveryLoco 式鲁棒性：随机 yaw/roll/pitch 初态、速度/角速度冲量、关节偏置、行进中推扰（11 例） |
| `probe_wheel_follow.py` | 纯 yaw 时四轮接触点跟随：轮心速度 vs 刚体旋转场（相对误差/RMSE）、机身坐标下四轮平面 z 展布、实际 yaw 速率 |

## 常用命令

```bash
cd ~/robocon-lean
export PYTHONPATH=$PWD/eval MUJOCO_GL=egl

# F1-F7 门禁（--part 2 时需要 --level；--scene 默认 demo/robots/blackW/smoke_flat.xml）
.pixi/envs/eval/bin/python tools/tracking/command_probe.py \
    --policy vt_track_v1 --part 1 --gate \
    --out artifacts/velocity-tracking/vt_track_v1.json

# 转弯足迹 / 支撑矩形 / 位置环
.pixi/envs/eval/bin/python tools/tracking/probe_turn_footprint.py \
    --policies vt_pose_v1,vt_track_v1 --out artifacts/tracking/turn.json
.pixi/envs/eval/bin/python tools/tracking/probe_support_rectangle.py \
    --policies vt_pose_v1,vt_track_v1 --out artifacts/tracking/support_rect.json
.pixi/envs/eval/bin/python tools/tracking/position_harness.py \
    --policies vt_pose_v1,vt_track_v1 --out artifacts/tracking/position.json
.pixi/envs/eval/bin/python tools/tracking/robustness_probe.py \
    --policies vt_pose_v1,vt_track_v1 --out artifacts/tracking/robustness.json
```

`--policy` 可写 `demo/rl_sar/policy/<robot>/` 下的目录名，也可直接写带 `config.yaml` 的导出目录。

## 验收口径

见 `docs/training/VT-RECIPE.md` §5。要点：F1/F2/F3/F5/F6/F7 必须过；F4（vy）只作诊断（本方向不投 vy）；
原地转支撑矩形漂移 ≤0.08 m/90°、位置环原地转 base 漂移 ≤0.10 m；转弯足迹相对静止矩形的额外占用
如实报告，目标是把额外占用压到最小（轮腿滚动原地转的物理下限由 `probe_turn_footprint.py` 记录）。
