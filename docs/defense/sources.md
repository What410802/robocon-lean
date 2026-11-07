# 事实来源与引用口径

结论冻结于2026-11-08。代码验收基线为 `e2d7818`；材料构建不训练模型、不重跑物理。本表中的本仓路径相对仓库根目录。

| 主讲页 | 事实 | 主要来源 | 解释边界 |
|---|---|---|---|
| 1–2 | 七分钟备赛答辩；Part1允许固定动作、Part2仅RL；导航真值只计算速度指令 | 原PDF `/home/bis/BiS.d/tmp/队内赛.pdf` 第1、3、4页；`docs/RULES-BASIS.md` | 答辩占30%，比赛占70%；未据此虚构成绩 |
| 2、9 | 16DoF轮足和12DoF锁轮分工；固定轮角目标 | `README.md`；`demo/rl_sar/library/core/rl_sdk/rl_sdk.cpp`；`demo/rl_sar/src/rl_sim.cpp`；`demo/rl_sar/policy/blackW/vt_locked12/` | 网络真12DoF，部署有16维兼容适配；有限PD力矩不会使实际速度瞬间归零 |
| 3 | 支撑无关进度奖励曾导致跳跃；0/178、0/4000；站21s累计+11.8，当时终止奖励+10 | `train/src/alldog_mjlab/tasks/velocity/profiles/plum/ep_rewards.py` 开头及 `tracking_yaw()`；`teacher.py` | 历史过程诊断；累计回报与终止奖励不同，不能称严格收益消融；当前成功奖励常量500采用dt缩放 |
| 3 | EP奖励迁移、前进速度门控、非轮触地终止 | 同上；`teacher.py::_strict_contact` | 训练约束比部署judge的非轮碰撞条件更严格；不是给网络增加接触传感器 |
| 4 | VT v1/v2四项定量对比 | `docs/training/VT-RECIPE.md` §6.1 | 历史滚轮VT，同部署探针；v2不只共面一项变化，不能叫严格单项消融；不是当前locked12数据 |
| 5 | 早期预测1764维扫描+冻结教师，最终深度64latent+324本体→动作 | `train/src/alldog_mjlab/tasks/velocity/profiles/plum/student.py`、`depth.py`；`evidence/plum_final_manifest.json` | 两条不同分支；当前actor无fine-scan，critic保留训练特权观测 |
| 5 | 轮心扫描±0.10m，相机前方约0.51–3m | `plum/depth.py` 模块注释；`fine_scan.py`、`depth_camera.py` | 是表示取舍的诊断；没有宣称严格控制变量证明前瞻信息单独带来某成功率增益 |
| 5 | BC数据1,228,800帧，动作RMSE0.12514 | `evidence/bc_depth_r4.json`；`train/scripts/plum/bc_depth.py` | 动作拟合误差，不是闭环成功率 |
| 5、备份C | 1024环境×32步、1200迭代、lr3e-5、model1150 | `evidence/plum_stage2_{agent,env}.yaml`；`evidence/plum_final_manifest.json` | 注册s0/s1/s2不等于串行完成过三个阶段；最终模型不是冻结教师 |
| 6 | 训练311/316，标准初态单关success200；15格2/15 | `evidence/plum_final_manifest.json` | dy∈{−.05,0,.05}m，yaw∈{−3,−2,0,2,3}°；固定确定性局部网格，非随机独立成功率；不与旧25格5/25混用 |
| 7 | 历史高台40、墙40 sidecar成功200 | `evidence/platform40_sidecar.json`、`wall40_sidecar.json`、`historical_adaptive_campaign.json` | adaptive基准vx=.5，per-frame指令回放；HUD停在running，依据sidecar。campaign判据v7.6，sidecar本身未存版本 |
| 7 | 六桩v56深度专家+轮式专家组合，本段success200 | `evidence/part2_v56_eval.json`、`part2_v56_provenance.json`；视频源时间1054.00–1062.44s | v9六桩；旧25段整场严格24/25，末墙失败；本段成功不代表当前33段整场完成 |
| 8 | v100过杆3.077°，v104过杆0.689°；降姿后停稳复核 | `evidence/part1_v104.json`；`demo/rl_sar/scripts/part1_auto_route.py` | 不同运行对照；v104无需主动脉冲，主动修正证据v102/v103；50Hz无接触记录不等于每2ms绝无碰撞 |
| 9 | v104首次支撑/切换意图137.84s，下一记录帧137.86s看到锁轮MIT | `evidence/part1_v104.json` 的 `gravel_handover` | 采样20ms；不声称切换零延迟或实际速度瞬时为0 |
| 9–10 | 当前唯一33段路线，resume_before恢复，timeout_scale=0 | `docs/part2/ROUTE.md`；`demo/rl_sar/scripts/part2_route_plan.py` | 当前执行含中间档；恢复是子段验证，不能当从原点连续通过 |
| 10 | Part1完整263.8s、0恢复、预检通过，judge关闭 | `evidence/part1_v104.json` | 已验证源码哈希对应；未认证900分或广泛初态鲁棒性 |
| 10 | 当前Part2整场未完成；40cm平台跳过，45°坡无成功 | `docs/part2/ROUTE.md`、`EXPERTS.md` | v82/v83连续前14段成功；v84前4段后卡25cm台；不同试跑不能拼接 |

## PDF答辩要求原文

第4页：“赛后需要制作简易的ppt或者文档（不要求形式好看，展现内容即可），在7分钟内简述自己的备赛过程。包括：奖励调整，论文阅读/复现，训练过程中的机制的琢磨，做过的对比实验等，讲出你认为值得说的部分即可。”

本文对应：奖励调整在第3页，对比实验在第4/6页，论文借鉴与训练机制在第5页及备份B/C，系统部署与过程结果在第2、7–10页。

## 论文与复现范围

- Schulman et al., *Proximal Policy Optimization Algorithms* (2017). <https://arxiv.org/abs/1707.06347>。当前训练主线的算法基础。
- Long et al., *Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response* (2023). <https://arxiv.org/abs/2312.11460>。仓内有HIM代码与历史训练分支；最终depth/VT并非HIM。
- Cheng, Shi, Agarwal, Pathak, *Extreme Parkour with Legged Robots* (ICRA 2024). <https://arxiv.org/abs/2309.14341>；<https://github.com/chengxuxin/extreme-parkour>。迁移奖励、课程和教师学生思路，不宣称原论文性能复现。
- 训练基座 `N-W-wolf/nwwolf_mjlab_training_release`，部署基座 `fan-ziqi/rl_sar`；具体版本、来源与使用范围见 `train/UPSTREAM.md`、`demo/UPSTREAM.md`。

## 没有用于主讲的历史汇总

用户提供的“19/20关有成功记录”属于不同策略/命令的历史独立尝试并集，旧梅花桩还使用长桩几何，不能代表当前修正六桩地图覆盖。Part1历史8段中T台阶仅50分，绕杆是手动单关；旧接触标签/判据也有限制。主讲没有把这些拼成连续整场或正式满分。

本仓另有六类官方profile单关1200/1300汇总，但它也不是实际ROS路线或稳定重复通过的成绩，本稿不将它用作主结果。

媒体已从本地 `artifacts/` 复制到本目录并有SHA清单，换机不依赖原始绝对路径；摘要中仍保留原路径便于追溯。未打包庞大状态trace，不能声称仅凭此展示包即可重新进行全部物理/官方判分重放。
