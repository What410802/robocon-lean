# 事实来源与引用范围

修订日期2026-11-08；验收代码基线 `e2d7818`。本次只修订答辩材料，不训练、不重跑物理、不提交。

## 答辩要求

`/home/bis/BiS.d/tmp/队内赛.pdf` 第4页：

> 赛后需要制作简易的ppt或者文档（不要求形式好看，展现内容即可），在7分钟内简述自己的备赛过程。包括：奖励调整，论文阅读/复现，训练过程中的机制的琢磨，做过的对比实验等，讲出你认为值得说的部分即可。

对应：论文2–3页；平地训练与实验4–5页；通用专家及高台配方6–7页；系统衔接合并为第8页；梅花桩第9页；交付与收获10–11页。主讲参数有保存配置，细节留在12–16页与下列索引。

## 论文与参考仓

| 文献 / 图 | 本地材料 | 本项目采用的部分 |
|---|---|---|
| Long et al., *Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response*, 2024，<https://arxiv.org/abs/2312.11460>，PDF第4页Fig.2 | `~/Legged_OpenSource/uw-himloco`、`_analysis/repo-external-picks.md`；本仓 `train/src/alldog_mjlab/algorithms/him/{policy,estimator,spec}.py` | 算法移植与历史训练。编码器、速度MSE、prototype swap及估计器/PPO独立更新。配置 `evidence/him_historical_agent.yaml`。 |
| Nahrendra, Yu, Myung, *DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep Reinforcement Learning*, 2023，<https://arxiv.org/abs/2301.10602>，PDF第3页Fig.2 | `~/Legged_OpenSource/Dreamwaq` 轮足衍生源码、`_analysis/repo-external-picks.md` | 阅读CENet显式速度、隐上下文、下一观测重建与KL；没有blackW复现原论文性能的证据。该本地仓为yusongmin1衍生版本。 |
| Cheng, Shi, Agarwal, Pathak, *Extreme Parkour with Legged Robots*, 2023 / ICRA2024，<https://arxiv.org/abs/2309.14341>，PDF第5页Fig.2 | `~/Legged_OpenSource/extreme-parkour`、`_analysis/repo-parkour-nwwolf.md` | 迁移特权教师→视觉学生、任务奖励与课程。论文GRU/航向预测与本项目CNN/历史堆叠有差异；后期学生继续PPO。 |
| RecoveryLoco | `~/Legged_OpenSource/RecoveryLoco`、`_analysis/repo-wheelleg-recovery.md` | 阅读初态、动力学扰动与恢复训练思路；不作为完整复现或直接贡献对照。 |
| Schulman et al., *Proximal Policy Optimization Algorithms*, 2017，<https://arxiv.org/abs/1707.06347> | 本仓rsl_rl及保存agent.yaml | 主要训练算法，clip与GAE等取值按实际保存配置引用。 |

三张论文架构图从原论文PDF裁切，图号、URL、裁切坐标与SHA见 `figures/manifest.json`。原图署名保留，图内机器人与成绩属于原作者。我们的网络结构、流程和图表由本项目绘制。训练/部署基座参考范围见 `train/UPSTREAM.md`、`demo/UPSTREAM.md`。

## 平地：第4–5、12页

依据 `docs/training/VT-RECIPE.md`、保存的 `evidence/vt_v5_{env,agent}.yaml`，当前策略manifest和运行时adapter。

- 16DoF actor：57维×6历史=342，MLP512/256/128→16；12腿位置残差+4轮速度。
- VT v5：4096×24；从v4/model_600续训，本阶段max_iterations=400，选择model_800。保存lr0.001 adaptive、clip0.2、γ0.99、λ0.95、5epochs×4batches、entropy0.01。
- vx[-0.4,1.0]、wz[-0.65,0.65]、vy0，5s重采样，站立10%；非站立中的pure_yaw_frac0.5、pure_vx_frac0.1，纯转向幅度0.1–0.6；剩余混合样本中low_frac0.5，低速|vx|≤0.12、|wz|≤0.2。两种“50%”不表示整个集合的两个独立半数。
- 奖励linear exp(−e²/0.25)×2；yaw exp(−e²/0.08)×2.5；linearL2−0.3；upright−1.2；poseleg−0.8；rotationdrift−1，带yaw跟踪门控。回合20s。
- 历史VT v1→v2同探针：轮高展布3.10→0.36cm，90°足迹半径0.572→0.520m，平移跟随0.781→0.631，转漂0.051→0.106m；多项修改的对照，不是单因素消融，也不是最终locked12的指标。
- locked12：342→270输入裁剪，16→12输出裁剪，critic243；1024×64，微调预算200，最终保留model_0裁剪初始化。保存训练配置为 `evidence/locked12_*`，不是说已部署200轮微调结果。
- 运行时轮PD保持切换时q，目标dq0，kp80/kd3；训练时轮锁目标0。PD为有限增益，不保证数学上的绝对静止。
- 当前长直线仍保留官方 `flat`；自研VT与锁轮实验不能被表述为官方平地权重也是自己训练的。

## 通用越障与高台：第6–7、13页

保存源为旧umbrella `logs/rsl_rl/black_velocity/2026-10-29_22-59-42_numrows31_mixed/params`，复制为 `evidence/platform40_{agent,env}.yaml`；部署 `row2p_p40w40_v1/manifest.json` 复制为 `platform40_manifest.json`。该manifest指向model_750，恢复自st30_native_s100_600/model_600。保存max_iterations=900是预算，不将manifest历史计划字段当成训练完成量。

- 512×24；31行，d=k/30，初始等级覆盖0–30，4类型混合（不等同于4列）。高台/墙0.10+0.30d；坡20+25d；6级楼梯总高0.6+1.2d，单级0.10+0.20d。
- 40cm高台row30，30cm高台row20，20cm楼梯row15，35°坡row18。必须按真实几何映射，不能按放置序号猜难度。
- 关键奖励：linear+1/yaw+0.5，σ0.25；progress+8/climb+10/clear+50；相对地形高度目标0.43、权重−4；竖直速度−2、angxy−0.05、upright−0.2、dofacc−2.5e−7、power−2e−5、动作差与平滑各−0.01。
- 终止：倾角1.05rad、横向0.7m、越界margin0.8m、完成、数值异常；20s超时。模型训练采用当时判据，后续出口判据版本变化见下文。
- PPO：MLP512/256/128、lr0.001 adaptive、clip0.2、γ0.99、λ0.95、5epochs×4batches、entropy0.01。
- 通用row2p可展示高台40cm、高墙40cm、坡35°；楼梯30cm是vt_band，沟壑30cm是s1d，不是单一模型全包。各类别档位是历史能力证据，不是同一整场统一成绩。
- 主讲40cm视频为 `vt_mode_v2`（障碍mode=row2p）、adaptive基准vx0.5；sidecar判定success/200。源路径见media manifest。HUD末帧可能仍running，判定按sidecar与campaign。

## 系统衔接与交付：第8、10、16页

路线、adaptive速度指令、策略调度与支撑条件见demo控制器与 `docs/part2/ROUTE.md`。世界位姿真值仅生成导航速度指令，不直接输入越障actor；部署本体/相机、训练教师/critic的信息分别处理。Part1允许固定动作，Part2过障采用学习策略。

Part1 v104 `evidence/part1_v104.json`：263.80s、0恢复、返回起点，live judge关闭，不是独立认证900分。限高杆视频来自该次片段。Part2第10页片段来自旧v56六桩与出口交接，同段支撑点回放success200；依据 `part2_v56_eval.json`、`part2_v56_provenance.json`。片段不表示当前整场已全关连续满分。`resume_before` 与 `timeout_scale:=0` 是演示复现功能。

## 梅花桩：第9、14–15页

**第9页视频为早期长桩场地 route2_v3**，与当前六桩不同。原视频 `artifacts/plum-strict/p1/videos/judge_plum_route2_v3_robot.mp4`；18s，HUD `plum_1 JUDGE=success`，随包 `evidence/plum_early_verdict.json`、`plum_early_manifest.json`、`plum_early_report.md` 保留来源。

视频模型：depth1×48×64→CNN预测1764 fine scan，与342本体拼接→冻结教师MLP512/256/128→16。教师model_1600；原始数据259200帧，DAgger r1为96env×1000=96000帧，20epochs、batch1024、AdamW；扰动出生dy±10cm/yaw±5°、关节状态与深度dropout/noise/delay。scan验证RMSE0.0250543m。r2的115200帧属于route2_v4，不归于该视频。

**第14–15页为后续六桩直接深度模型**：

- depth1×48×64→CNN16/32/64、kernel5/3/3、stride2、globalavg→64；本体54×6=324；拼接388→MLP512/256/128→16。actor无command、无fine scan；critic247仅训练用。
- 早期教师/引导课程固定桩高20cm及桩顶20×20cm，以填隙控制间隙0/.21/.2575/.305/.3525/.40m；当前窗口40回合，≥50%升、≤5%降，每调用至多4条，冷却64调用。最终stage2所有行difficulty_range=[1,1]，未逐级推进；出生x/y±.1m、yaw±5°、初速度0。
- EP迁移15项奖励与阈值详见 `training_recipe.json`。总奖励为max(0,0.02×Σwᵢrᵢ)；前进+1.5，航向+.5且v>.25375门控；碰撞−10且>.1N，边缘−1且<.05m；success500自身dt缩放贡献+10，不是总回报必等于10。诊断静止21s+11.8与通关终止项+10提示奖励漏洞，不是两条受控成功/失败总回报对照。
- 非轮接触>1N、倾角1.05rad、轮底<5cm、lane.7m提前结束；24s超时。严格完成要求下完出口、四轮回地、机身高正常并保持20步=0.4s。接触塑形0.1N与终止1N不同，训练倾角与judge75°不同。
- BC5×245760=1228800帧，动作验证RMSE.125141；1024×32、1200迭代预算=39321600步，选model1150。lr3e−5fixed、clip.2、γ.99、λ.95、5×4、entropy.01、std.1。预算不是实测耗时。
- 训练记录311/316、标准初态success200、15格dy/yaw诊断2/15是不同分布。网格不是随机成功率，不与历史25格拼接。初态鲁棒性仍是后续工作。

## 媒体与历史证据

当前主讲视频：platform40 **6.94s**、plum_gait_early **18s**、delivery_reel **19.48s**，合计 **44.42s**。delivery_reel由lowbar与plum_handover原速顺序拼接；其余旧合集与wall40保留为素材，不在主讲播放。文件、来源、截取区间、SHA见 `media_manifest.json`，本地媒体已复制，构建不依赖源artifacts。

历史19/20关证据混合不同模型、命令档、判据与旧桩几何，不作为本稿统一成功率。45°坡无success证据；沟壑30cm v7.6成功使用fixed vx0.8；楼梯30cm主要为adaptive成功trace。v7.5严格沟壑出口，v7.6严格梅花桩出口。Part1部分旧接触类视频的BL-6 replay限制留在历史审计文档中。

原记录用于追溯，本展示包不包含全部状态trace。主讲简要表达技术内容，以上说明供问答核对。
