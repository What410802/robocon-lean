# alldog_mjlab — RoboCon 2026 训练侧（robocon-lean/train）

本目录是 `robocon-lean` 的训练子项目：用 **MjLab 1.6.0 + RSL-RL 5.4.2** 训练
**black**（12 DoF 四足）与 **blackW**（16 DoF 轮足）的速度跟踪 / Part 1、Part 2 越障策略，
并导出部署用 TorchScript。只保留比赛需要的 **PPO / HIM 主线**；深度相机、BC 蒸馏、
PUMA foothold、Extreme-Parkour 等研究分支已在迁移时移除（来源与取舍见
[UPSTREAM.md](UPSTREAM.md)）。

地形尺寸的唯一来源是仓库根的 `terrain/`（Part 2 spec：`terrain/official/part2_terrain_spec.json`，
plum：`terrain/gen_plum.py`）。本目录**不再**保存地形 spec 副本，也不手写障碍尺寸。

---

## 1. 环境

训练环境由仓库根 `pixi.toml` 的 `train` environment 提供（Python 3.12、
`mjlab==1.6.0`、`rsl-rl-lib==5.4.2`、CUDA torch、`alldog-mjlab` editable 安装）：

```bash
pixi install -e train          # 首次/依赖变更后
pixi run -e train train --help # 之后所有命令都走 pixi run -e train
```

- 不要用 `pip install` / `uv sync` 手工装依赖；mjlab / rsl-rl / mujoco-warp 的版本组合
  锁定在仓库根 `pixi.toml`，升级会改变数值与 checkpoint 格式。
- 并行训练需要 NVIDIA GPU；默认 `num_envs=4096`，显存不足时用
  `--env.scene.num-envs 1024` 降低。纯 CPU 只适合 tiny 冒烟测试。
- `train` / `play` 是 MjLab CLI；`export` 是本仓库 `pyproject.toml` 注册的入口。

## 2. 目录结构

```text
src/alldog_mjlab/
├── robots/black/            # black（12 DoF）MJCF / joint·actuator 身份 / 固有参数
├── robots/blackw/           # blackW（16 DoF）MJCF + 官方 STL mesh（xmls/assets/）
├── tasks/velocity/
│   ├── black/               # black 任务装配（观测/动作/reward/termination/terrain/DR）
│   ├── blackw/              # blackW 任务装配（官方 Part1/Part2 地形任务、HIM 集成）
│   ├── common/              # 公共 config / rl_cfg / runner / curriculum / course
│   └── profiles/
│       ├── part1/           # Part 1 竞赛 proxy 地形与课程
│       ├── part2/           # Part 2 官方地形、judge 镜像 predicates/rewards
│       └── pose/            # blackW flat 任务的 pose-edge / 低速指令扩展（核心 flat 任务在用）
├── algorithms/him/          # HIM estimator / HIMPolicy / HIMPPO / warm start
└── utils/                   # action/observation 契约、knobs、导出 CLI
scripts/checks/              # 确定性自检（env cfg dump、地形、spawn、friction）
scripts/export/              # blackW 的 history / HIM TorchScript 导出
tests/                       # Part 2 wall predicate 单测（CPU）
```

## 3. 任务列表

`black`（12 DoF，45 维单帧 obs / 12 维 action）：

| Task ID | 内容 |
|---|---|
| `black-flat` | 平地速度跟踪，PPO |
| `black-rough` | 复杂地形速度跟踪，PPO |
| `black-gravel-flat` / `black-gravel-free` | 碎石/自由石块变体（S1b 专项） |
| `black-basic-obstacle` | Part 1 基础障碍（坡面/台阶） |
| `black-competition` / `black-competition-hist` | Part 1/2 竞赛 proxy 地形（单帧 / 6 帧 history） |
| `black-flat-him` / `black-rough-him` | HIM（history-biased estimator）算法 |

`blackW`（16 DoF，57 维单帧 obs / 16 维 action，官方 signed wheel contract）：

| Task ID | 内容 |
|---|---|
| `blackw-flat` | 平地速度跟踪 |
| `blackw-vt` | **速度跟随衔接（Part 1 位置环）**：pose 包络 + 纯 yaw/纯平移模式 + stance/转弯漂移奖励 + 推扰/随机初态/轮摩擦 DR（配方与验收见 `docs/VT-RECIPE.md`） |
| `blackw-vt-edge` | `blackw-vt` + 0.05–0.20 m 平台边缘地形（桥台/台阶唇交接边缘鲁棒） |
| `blackw-vt-locked` | **真 12-DoF 锁轮衔接**：45×6 obs / 12 act，轮关节不属于策略 IO、由轮锁 PD 保持；RecoveryLoco 式接触/步态 + 动态支撑框（见 `docs/VT-RECIPE.md` §4.3/§8） |
| `blackw-official` | Part 1/2 官方难度混合地形（主力任务） |
| `blackw-plum` | Part 2 plum（按规则 PDF 图的 6 桩两排、无坡道布局） |
| `blackw-{stairs,platform,slope,trench,wall}-exact` | 单障碍官方地形专项 |
| `blackw-platform-easy` | 低难度平台专项（课程起点） |
| `blackw-flat-him` / `blackw-official-him` / `blackw-platform-{exact,easy}-him` | HIM 变体 |

> `*-exact` / `blackw-official` / `blackw-plum` 的地形尺寸全部读取
> `terrain/official/part2_terrain_spec.json` 与 `terrain/plum/plum_spec.json`；
> plum 按规则 PDF 图重建为 **6 桩两排、无坡道**（详见 `terrain/README.md`）。

## 4. 训练示例（每型机器人一条；本文档不执行训练）

`black`（平地 PPO，先跑常用 baseline）：

```bash
pixi run -e train train black-flat --agent.logger tensorboard
```

`blackW`（官方 Part 1/2 混合地形主力任务）：

```bash
pixi run -e train train blackw-official --agent.logger tensorboard
```

常用覆盖参数（CLI 显式参数优先于任务默认值）：

```bash
--env.scene.num-envs 1024     # 环境数量（显存不足时降低）
--agent.max-iterations 2000   # 训练迭代数；resume 时表示“再训练多少”
--agent.seed 43               # 随机种子
--agent.save-interval 100     # checkpoint 间隔
--agent.run-name mytag        # run 目录后缀 → <时间戳>_mytag
```

续训 / HIM（与旧仓一致，只是前缀从 `uv run` 换成 `pixi run -e train`）：

```bash
# black：从最新 flat run 续训 rough（full resume）
pixi run -e train train black-rough --agent.resume True --agent.load-run '.*_flat$'

# blackW：官方地形任务从已有 checkpoint 继续
pixi run -e train train blackw-official --agent.resume True --agent.load-run '<run>'

# black：HIM 从已训练 PPO checkpoint 初始化（warm start，不是 resume）
pixi run -e train train black-flat-him --agent.warm-start True --agent.load-run '.*_flat$'

# blackW：HIM 任务（官方 TorchScript warm start 走 RC_HIM_OFFICIAL_WARMSTART）
pixi run -e train train blackw-official-him --agent.resume True --agent.load-run '<him-run>'
```

训练默认写入 `logs/rsl_rl/<experiment_name>/<时间戳>_<run_name>/`（`logs/` 已被
仓库根 `.gitignore` 忽略）。默认 logger 是 wandb；训练通常用
`--agent.logger tensorboard`，或先 `wandb login`。

## 5. 回放

```bash
pixi run -e train play blackw-official \
    --checkpoint-file 'logs/rsl_rl/black_velocity/<run>/model_500.pt' \
    --viewer viser
```

`--viewer viser` 打开网页界面（可拖速度指令滑条）；`--viewer native` 需要桌面
DISPLAY。`play` 只加载 actor，不使用 critic。

## 6. 导出策略并交给 demo/

### black（12 DoF / 45 维单帧）

```bash
pixi run -e train export --task-id black-flat
pixi run -e train export --task-id black-rough \
    --load-run '<run>' --checkpoint model_500.pt --output-dir ~/models/black
```

默认流程：task → 最新同 stage run → 最新 checkpoint → `<run>/exported/policy.pt`（CPU 导出）。
产物契约：TorchScript、仅 actor、输入 `float32 [1, 45]`、输出 `float32 [1, 12]`（raw action）、
文件名 `policy.pt`。

### blackW（16 DoF / 57×6 history）

blackW 的部署契约是 **frame-major、newest→oldest** 的 342 维 history，需要专用导出脚本
（会重排 actor 第一层线性层输入列并写 `manifest.json`）：

```bash
# 普通 PPO / history 策略
pixi run -e train python scripts/export/export_history_policy.py \
    --task blackw-official --robot blackW \
    --checkpoint logs/rsl_rl/black_velocity/<run>/model_500.pt \
    --out ~/models/blackW/official --device cpu

# HIM 策略（官方 signed wheel contract）
pixi run -e train python scripts/export/export_him_policy.py \
    --task blackw-official-him \
    --checkpoint logs/rsl_rl/black_velocity/<run>/model_500.pt \
    --out ~/models/blackW/official_him --device cpu
```

### 拷贝到 demo/

演示 runtime 从 `demo/rl_sar/policy/<robot>/<policy-name>/` 读取策略：

1. 把导出的 `policy.pt` 与 `manifest.json` 放到
   `demo/rl_sar/policy/blackW/<policy-name>/`（black 的历史策略目录用 `best.pt`，
   新导出统一用 `policy.pt`；导出脚本已写 `manifest.json`）；
2. 同目录补一份 `config.yaml`：`num_of_dofs/num_observations/observations/
   observations_history/model_name/action_scale/wheel_indices` 等字段必须与
   `manifest.json` 以及 `demo/rl_sar/policy/base.yaml` 的机器人常量一致
   （blackW：16 DoF、57 维单帧、history 6、signed wheel scale `FL/RL +10, FR/RR -10`）；
3. 运行 demo（Part 2 单关场景由 `terrain/` 生成）：

```bash
pixi run -e demo demo-build        # 首次构建 ROS 2 runtime
ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=2 \
    level:=plum_1 policy_name:=<policy-name> viewer:=false realtime:=true
```

`policy.pt` 与训练 checkpoint `model_*.pt` 用途不同，不能互换。

## 7. 自检 / 测试

```bash
# 全部任务的 env cfg 快照（CPU，无需 GPU）
pixi run -e train python scripts/checks/dump_env_cfgs.py --out /tmp/env_cfgs.json

# Part 1 竞赛地形 / Part 2 单体地形 / 摩擦自检
pixi run -e train python scripts/checks/check_competition_terrain.py
pixi run -e train python scripts/checks/check_tier_geometry.py
pixi run -e train python scripts/checks/check_world_friction.py

# Part 2 predicate 单测（CPU）
pixi run -e train python -m unittest discover -s tests
```

`scripts/checks/check_spawns.py` 用 MuJoCo 校验 `demo/scenes/part2/part2_official_levels.json`
里每个 spawn 的支撑高度/穿透，可用 eval 环境运行（只需 mujoco/numpy）：

```bash
pixi run -e eval python train/scripts/checks/check_spawns.py
```

## 8. 与 terrain/ 的耦合（单一来源）

- `tasks/velocity/profiles/part2/terrain.py` 与 `predicates.py` 直接读取
  `terrain/official/part2_terrain_spec.json`（含 sha256 的官方 map 派生数据）；
  plum 的尺寸来自 `terrain/gen_plum.py` 生成的 `part2_terrain_spec.json`
  `categories.plum` 段。
- 修改地形尺寸：编辑 `terrain/gen_part2_spec.py` / `terrain/gen_plum.py` 的常量后重新生成，
  **不要**在本目录手改数字，也不要再复制一份 spec。
- 重新生成命令见 `terrain/README.md`。

## 9. 关键契约（不要随手改）

`black`：

```text
policy joint order : FL → FR → RL → RR，每腿 hip → thigh → calf
actor observation  : 单帧 45 维（PPO，无 history、无 running normalization）
action             : 12 维，q_target = q_default + 0.25 × raw_action
physics dt         : 0.002 s（官方 MJCF），control 0.005 s，decimation 4 → policy 50 Hz
nominal PD         : Kp = 50，Kd = 1.2
```

`blackW`：

```text
joint order        : FL,FR,RL,RR × hip,thigh,calf,wheel（wheel 为速度目标）
actor observation  : 57 维单帧（command3 + ang_vel3 + gravity3 + dof_pos16 + dof_vel16 + actions16）
                     history 6 → 342 维（HIM / history 策略）；部署帧序 frame-major newest→oldest
action             : 16 维 raw action；leg 位置 scale 0.25；wheel scale 官方 signed（FL/RL +10, FR/RR -10，
                     PPO 训练契约为未签名 +10，导出时由 action_contract 标注）
physics dt         : 0.002 s；control 0.005 s（200 Hz）；policy 0.02 s（50 Hz）
nominal PD         : legs Kp=50/Kd=1.2；wheel Kp=0（速度控制），Kd=1
```

joint order / observation layout / term 顺序属于**部署契约**；改动前先与部署/demo 维护者确认。

## 10. 常见问题

| 现象 | 处理 |
|---|---|
| 训练一开始就要求 wandb 登录 | 加 `--agent.logger tensorboard`，或先 `wandb login` |
| CUDA out of memory | 降低 `--env.scene.num-envs`（如 1024 / 512） |
| `play` 弹不出窗口 | 无桌面会话时用 `--viewer viser`，用浏览器打开显示的地址 |
| 任务没注册 / 找不到 task id | 确认在仓库根执行、`pixi install -e train` 已装好；`pixi run -e train train --help` 查看可用 task |
| Part 2 地形尺寸与 judge 不一致 | 不要手改 `terrain.py` 常量；重新跑 `python terrain/gen_part2_spec.py` / `terrain/gen_part2_levels.py` |
