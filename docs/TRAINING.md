# 训练与导出

训练代码与任务定义在 `train/`（mjlab 1.6 + RSL-RL）；本文件只给上层流程与改造入口，
细节以 `train/README.md` 为准。

## 1. 环境

```bash
pixi install -e train          # GPU：torch 2.14 CUDA + mjlab 1.6 + mujoco-warp 3.11
pixi shell -e train
python -c "import mjlab, mujoco; print(mjlab.__version__, mujoco.__version__)"
```

`train/` 只从 `terrain/official/part2_terrain_spec.json` 读取地形参数（单一来源）；改官方地图
时先跑 `terrain/gen_part2_spec.py` 再训练。

## 2. 任务与 profile

| 任务 | 机器 | 用途 |
|---|---|---|
| `blackw-*` velocity | blackW 16 DoF | Part2 六类 / Part1 过障 |
| `black-*` velocity | black 12 DoF | 历史对照（Part1 备用） |
| `profiles/part1` | 公共 | Part1 地形/奖励覆盖 |
| `profiles/part2` | 公共 | Part2 六类、成功谓词、课程 |

观测/动作契约以 `demo/robots/<robot>/robot.yaml` 为准：50 Hz 推理、200 Hz 控制、
obs 57×6、action 16；导出时转成部署侧 frame-major。

## 3. 训练 → 导出 → demo

```bash
# 训练（示例，具体参数见 train/README.md；不要求每次迁移都重训）
python -m alldog_mjlab.train --task blackw --num_envs 4096 ...

# 导出 TorchScript（policy.pt + config.yaml + manifest.json）
python -m alldog_mjlab.utils.export_policy ...

# 放入 demo 运行时（权重不入库）
cp -r <export>/demo/rl_sar/policy/blackW/<name>/
```

## 4. 改造入口（迁移后计划）

- **算法**：`train/src/alldog_mjlab/algorithms/`（保留核心 PPO/HIM；研究分支未迁移）。
- **奖励/终止**：`tasks/velocity/profiles/{part1,part2}/`；成功谓词必须与 `eval/` 判分口径同步
  （改判分时先改 `eval/robocon_eval/rules.py`/judge 与 SPEC，再改训练侧）。
- **地形/课程**：`terrain/gen_*.py` 是唯一来源；训练只读 spec，不再各自维护地形副本。
- **域随机化**：`utils/knobs.py` + task 配置；保持"仿真参数在 `train/`、判分只看状态"的边界。

## 5. 权重管理

- 所有 `*.pt`/`onnx` 都在 `.gitignore`；新仓不带历史权重，按需重训/导出。
- 策略目录只提交 `config.yaml` + `manifest.json`；`policy.pt` 本地放置。
