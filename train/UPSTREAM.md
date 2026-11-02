# 上游来源与许可说明

本目录（`robocon-lean/train/`）是 RoboCon 2026 队内赛训练侧子项目，由以下来源派生：

| 来源 | 版本/标识 | 说明 |
|---|---|---|
| `N-W-wolf/nwwolf_mjlab_training_release`（上游 mjlab 训练基座） | `24c7f16`（上游已不再前进） | 本目录 `src/alldog_mjlab` 的初始快照；原仓未声明开源许可证，**本仓仅作为队内私有仓库使用**，不对外分发 |
| 本队旧任务仓工作线 `legacy/ppo` | `04ae722` | 通用补丁、判据镜像、审计修复、导出/工具、PPO 线 |
| 本队旧任务仓工作线 `legacy/him` | `e977ecb` | HIM 算法守卫、官方 warm start、blackW HIM 集成、安全终止 |

迁移到单一仓库时的取舍（`robocon-lean`，2026-11-02）：

- **保留**：velocity 任务的 black / blackW 主线、common 配置、profiles `part1` / `part2` /
  `pose`（`pose` 仍被 `blackw-flat` 使用）、robots `black` / `blackw`（含官方 mesh）、
  utils 契约与 knobs、HIM 算法（`algorithms/him/`）、blackW 的 history/HIM 导出脚本、
  核心自检脚本与 predicate 单测。
- **移除**：研究/实验分支 —— 深度相机（`camera_depth`、`depth_envs`、`depth_rl_cfg`、
  `him_depth_rl_cfg`、privileged teacher）、BC 蒸馏（`bc_ppo`、`depth_warm_start`、
  `gated_depth_cnn`、`guarded_ppo`）、PUMA foothold（`foothold_envs`、
  `profiles/part2/foothold.py`）、Extreme-Parkour 教师（`plum_ep_envs`、
  `fine_scan`、`strict_contact`、`ep_rewards`、`teacher_predicates`）、
  实验配置/实验输出/`__pycache__`，以及研究期脚本与文档。
- **地形单一来源**：旧 `train/scripts/tools/gen_official_terrain_spec.py` 迁到
  `terrain/gen_part2_spec.py`；训练侧不再持有 `official_terrain_spec.json` 副本，
  `profiles/part2/terrain.py` / `predicates.py` 直接读取
  `terrain/official/part2_terrain_spec.json`。part2 plum 使用队内确认的
  no-ramp 3-pile 布局（`terrain/gen_plum.py`），不再沿用旧 `map.xml` 的 ramp/双排布局。
- **权重/历史**：本目录**不包含**任何训练权重或实验 artifacts；部署策略由
  `demo/rl_sar/policy/<robot>/<policy-name>/` 管理，训练 checkpoint 只落在 `logs/`
  （已被根 `.gitignore` 忽略）。

主办方材料（官方 `map.xml` 地形、Part 1 `scene_terrain.xml`、blackW 模型/mesh、比赛 PDF）：
来源为比赛官方材料，仅用于本队训练与比赛，随 `terrain/`、`demo/robots/blackW/` 入库；
哈希与出处见 `terrain/README.md` 与 `assets/manifest.json`。上游引用与等价性证据的旧文档
在旧任务仓 `docs/migration/phase1-training-equivalence.md`（本仓不再携带）。
