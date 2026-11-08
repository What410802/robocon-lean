# 四足／轮足运控文献：研究脉络与思想讲解

调研日期：2026-11-08。用途：为面向同组队员的 **15 分钟备赛分享**准备材料。本轮交付文献清单、方法比较和讲解素材，尚未重写现有 HTML 或讲稿。

建议用三个问题组织论文：**机器人缺什么信息？用什么训练信号补足？补足以后，策略究竟获得了什么能力？** “编码器 → 隐变量 → PPO”只是许多方法共有的外形，不能代替对方法的解释。

本轮整理 29 条论文、预印本和研讨会稿记录，分为基础、状态估计与适应、视觉跑酷、浙大系列、任务与控制对照、同作者拓展。它们不是 29 项独立创新，也不表示你读过或本仓实现了其中全部工作。完整标题、作者、原文版本和链接见 [论文目录](#论文目录)及 [papers.json](papers.json)。

## 先看研究关系

“同族”依据论文作者、单位和项目页；“思想关联”依据方法与引用。下面的箭头表示适合顺着阅读的研究关系，不表示逐行继承代码。

| 研究群体 | 可连起来阅读的工作 | 连续研究的问题 |
|---|---|---|
| KAIST Urban Robotics Lab，Hyun Myung 团队 | DreamWaQ → DreamRiser；DreamWaQ → DreamWaQ++ | 从本体历史理解接触环境；将这套表示用于恢复；再补充外部感知 |
| 上海 AI Lab OpenRobotLab，Jiangmiao Pang 团队 | HIM → H-Infinity Locomotion Control | 前者改进运动响应表示，后者改进抗扰训练分布；不是简单的网络升级 |
| CMU Deepak Pathak 与 UC Berkeley Jitendra Malik 等合作 | RMA → Egocentric Vision → Extreme Parkour；Deep WBC / Legs as Manipulator 为相关支线 | 从动力学适应走向视觉控制、敏捷运动和全身协调 |
| 浙大控制科学与工程相关团队，Jun Wu / Qiuguo Zhu | Key Estimation → PIE；PIE → MOVE；Terrain Reconstruction / START；PIE → PUMA；MOVE → LF2WB | 估什么、怎样融合视觉、怎样记忆盲区、怎样保留稀疏几何、怎样给落足先验 |
| ETH RSL，Marco Hutter 团队及合作者 | 2019 sim-to-real → 2020 特权教师与本体学生 → 2022 鲁棒感知；另有大规模并行、ANYmal Parkour、Risky Terrains、DTC | 训练效率、感知可靠性、导航与运动组合、精确落足 |
| 其他需单列的方法 | Robot Parkour Learning；MIT Rapid Locomotion；ASL 等合作的 Blind Stair Climbing | 各有作者合作网络；在这份材料中作为独立方法对照，不宣称它们没有其他同族工作 |

几个归属尤其值得保留：

- **DreamWaQ 与 HIM 方法接近，作者团队不同。** DreamWaQ 项目页明确列出 KAIST Urban Robotics Laboratory；HIM 首页列出 OpenRobotLab / Shanghai AI Laboratory，另有浙大、清华作者。
- **同在 KAIST 也未必同组。** Concurrent Training 来自 Jemin Hwangbo 的 RAILAB；DreamWaQ 来自 Hyun Myung 的 Urban Robotics Lab。
- **Extreme Parkour 的 Cheng 支线跨实验室。** Extreme Parkour、Deep WBC、Legs as Manipulator 属于其与 Pathak 等人的合作；Visual WBC 的论文单位为 UC San Diego、合作导师为 Xiaolong Wang。共同作者能串起阅读，不应因此写成一个实验室的统一系列。[作者主页](https://chengxuxin.github.io/)
- **HIM 有确证的改题。** 同一个 [arXiv:2312.11460](https://arxiv.org/abs/2312.11460)，[v1](https://arxiv.org/abs/2312.11460v1) 题为 *Hybrid Internal Model: A Simple and Efficient Learner for Agile Legged Locomotion*，v3 改为 *Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response*。PIE 参考文献使用前一个标题；这不是两篇不同的方法。
- **Walking with Terrain Reconstruction 与 START 应并排读。** 两者共同主要作者及 TR-Net、GRU、U-Net 精修、I-E Estimator、AdaSmpl 等机制高度重合。本轮按同一研究支线处理；尚未找到明确的版本声明来断言它们是“正式改名”或“扩展版”。

## 一张方法地图

读表时，先区分三件事：actor 在部署时得到什么、训练时额外有什么监督、隐变量被要求保留什么。critic 能看到真值，不等于 actor 部署时也能看到真值。

| 方法 | 部署的主要输入 | 表示的训练信号／中间信息 | 关键选择 |
|---|---|---|---|
| RMA | 本体状态、动作历史 | 拟合由真实环境参数编码得到的 latent | 两阶段；先训练能使用环境编码的策略，再学从历史估计编码 |
| Concurrent Training | 本体及关节历史 | 速度、足高、接触概率的真值监督 | 策略训练期间就使用估计结果，适应估计误差 |
| DreamWaQ | 本体历史 | 速度 MSE；下一观测重建；VAE 的 KL | 显式估速度，隐式编码剩余上下文；另有 AdaBoot |
| HIM | 本体历史 | 速度监督；历史与后继状态的 prototype 交换预测 | 学对控制有用的响应表示，不要求重建整张地形图 |
| Egocentric Vision / Extreme Parkour | 深度＋本体，含时序记忆 | scandot 教师动作；EP 还蒸馏航向 | 把几何真值教师迁到低成本视觉传感器 |
| PIE | 深度历史＋本体历史 | 速度、足部离地高度、地形图重建、下一本体观测重建、KL | 同一训练阶段学习多模态估计和控制 |
| MOVE | 前向深度＋本体历史 | 前视重建＋与特权周围表示的对齐 | 处理有限视野、多方向运动与三维障碍 |
| Terrain Reconstruction / START | 深度历史＋本体历史 | 先显式重建局部高程图，再供策略编码 | 将小桩顶、边缘与脚下盲区变成清楚的空间表示 |
| PUMA | 深度缓冲＋本体历史 | 预测自我坐标中的落足先验、速度与环境 latent | 用软先验引导姿态和前足接触；PAS＋多 critic |
| DTC | 规划参考、机器人状态和所需地形／估计信息 | 优化器给出参考轨迹，RL 学跟踪 | 将落足决策与鲁棒执行分别处理 |

这里的“一阶段”通常指**不先单独训练一个完整教师，再蒸馏学生**。它不意味着没有辅助监督、没有特权标签、没有课程，或不在训练早期混用真值。反过来，两阶段方法也不一定只能做动作模仿：RMA 拟合的是环境 latent，Extreme Parkour 的视觉阶段还涉及航向。

## 核心精读：六篇各讲清一个问题

这六篇构成准备材料的骨架。15 分钟展示中不需要逐篇讲完，可按后文建议选其中的比较。

### A3 · DreamWaQ：没有相机，“想象地形”是什么意思？

[原文](https://arxiv.org/abs/2301.10602) · ICRA 2023 · Nahrendra、Yu、Myung。

关节编码器和 IMU 给出的单帧观测不足以直接判断接触条件：相同的身体姿态可能来自平地、打滑或脚被挡住。历史里包含“之前做了什么、身体随后怎么动”的线索。DreamWaQ 用 CENet 从历史得到两部分：有物理意义的机身线速度估计，以及没有逐维物理标签的环境上下文 `z`。

**关键不是增加了一个 latent，而是规定它怎样学。** 速度用仿真真值做 MSE；上下文通过预测下一帧本体观测学习，并用 β-VAE 的 KL 项约束。可以用简化式理解：

$$
L_{CE}=\|\hat v-v\|^2+\|\hat o_{t+1}-o_{t+1}\|^2+\beta D_{KL}(q(z\mid o^H)\|p(z)).
$$

如果一种接触条件会改变随后的运动响应，预测任务就鼓励编码器保留这种条件的相关信息。原文还用 **AdaBoot** 根据回合奖励的变异系数调节训练时使用估计速度的概率，兼顾早期学习与估计误差适应。

**讲给队员的说法：**“它不需要把摩擦系数和地形形状逐项识别出来，而是从身体的运动反馈中，提取足够帮助下一步控制的上下文。”

“地形想象”不代表无视觉就能预见尚未接触的深沟，也不是部署时做多步轨迹规划的世界模型。主图可用 **Fig.1** 看整体、**Fig.2** 解释 CENet。当前展示已有 Fig.2 裁图。

### A4 · HIM：为什么不继续重建下一帧？

[原文](https://arxiv.org/abs/2312.11460) · ICLR 2024，预印本 2023 · Long 等。

HIM 也从历史输出显式速度与隐表示，但选择了另一种表示目标：不要求 latent 逐维复原下一观测，而是让历史编码与**后继本体状态编码**在表示空间中具有一致的簇分配。后继状态已经包含身体对环境和动作的响应，因此可作为学习“接触条件怎样影响运动”的训练依据。

两个 encoder 分别处理历史与后继状态，归一化特征与可学习 prototypes 比较，再用 Sinkhorn 产生批次中的平衡软分配，做交换预测。这样既鼓励两种视角对应，也防止所有样本都输出同一个常量。速度另外用监督损失训练。

**与 DreamWaQ 最有价值的对比：** 两者部署都能只用本体历史；差异在于“什么算一个好的上下文”。DreamWaQ 用重建误差和 KL 约束；HIM 用批次中的 prototype 分配一致性来组织表示。无需在讲稿中把二者描述成“有没有记忆”的差别。

**讲给队员的说法：**“同一种环境作用下，历史应该能预示随后身体会出现怎样的响应；HIM 学的是这种响应的表示，而不是一份完整环境参数表。”

主图用 **Fig.2**。不要把 latent 称为严格定义、能直接读出数值的“稳定性指标”，也不要将原文开放环境演示推广为任意地形保证。原文附录配置与 released code 存在差异，例如 prototype 数；本仓历史配置采用 **32 个**，不能把论文表格参数直接当本仓配方。

### B2 · Extreme Parkour：不只是一张教师—学生框图

[原文](https://arxiv.org/abs/2309.14341) · ICRA 2024，预印本 2023 · Cheng、Shi、Agarwal、Pathak。

它要解决的首先是**动作如何学出来**。跳沟、上高台和斜面换向，不宜被每个时刻的固定机身速度目标约束。论文用地形上的路点生成目标方向，奖励世界坐标速度在该方向上的投影，并在指令速度处截断：

$$
r_{track}=\min(\langle v,\hat d\rangle,v_{cmd}).
$$

因此允许策略利用不同姿态和速度变化取得进展；同时用触地靠边惩罚等减少危险落脚，配合难度课程。在教师阶段，scandots 提供便宜、明确的地形几何，ROA 处理动力学适应。

第二个问题才是**怎样让相机驱动这些动作**。视觉阶段用 CNN–GRU 从深度提取时序信息，以 DAgger 的方式在学生访问的状态上取得教师动作标签；还学习预测路点给出的航向。航向预测早期不准时混用教师方向，避免输入分布突变。

**讲给队员的说法：**“先让机器人借助可靠几何学会怎样过障，再让深度图接管几何和方向信息；奖励负责允许动作出现，蒸馏负责把信息换成能部署的传感器。”

主图用 **Fig.2**，讲述时分别指出奖励、课程、动作蒸馏和航向蒸馏。scandot 是地形高度采样，不是相机图，也不等同于本仓轮边 fine scan 的覆盖范围。本仓采用了相关训练思路和奖励，但最终 CNN＋本体历史学生没有论文同款视觉 GRU 和航向预测器。

### C2 · PIE：“隐式—显式”到底有哪两层？

[原文](https://arxiv.org/abs/2408.13740) · RA-L 2024 · Luo、Li、Yu、Wang、Wu、Zhu。

PIE 将深度历史和本体历史分别编码，用 Transformer 融合，再用 GRU 留住当前视野外的信息。表示学习与 PPO 在同一训练阶段进行；actor 使用能部署的观测与估计，critic 可以使用特权状态。

论文所谓双层含义是：第一层，既通过**地形重建**利用外部视觉，又通过**预测下一本体观测**学习身体与环境相互作用的结果；第二层，输出同时包含**有物理标签的量**（速度、足部离地高度）和**编码后的量**（地形 latent、其余上下文）。上下文使用 VAE 的 KL 正则。不要把所有“隐式”都理解成“没有视觉”，也不要把所有 latent 都理解成“没有监督”。

**与 Extreme Parkour 的区别：** PIE 的重点是训练期间就让可部署感知参与控制，并用多个辅助目标帮助表示形成；它没有先训练一个完整 scandot actor 再蒸馏视觉 actor 的独立流程。这是另一种处理表示学习与控制相互依赖的方案，不能仅凭“一阶段”就断言所有任务都优于蒸馏。

**讲给队员的说法：**“相机告诉它前面可能有什么，身体反馈检验这种理解是否可靠；训练同时要求它看懂地形、估对关键状态，并解释随后身体怎么动。”

主图用 **Fig.2**。地形 decoder 在这里提供表示监督；应与 START 把重建图作为控制中间输入的做法区分。原文也明确承认 2.5D 地形理解的局限，不能据此推出可以处理头顶横杆等完整三维场景。

### C5 · START：为什么重新引入显式高程图？

[原文](https://arxiv.org/abs/2512.13153) · RA-L，2025 年 12 月接收、2026 年卷期 · Yu 等。

梅花桩和窄梁让视觉表示面临不同要求：小范围桩顶和边缘占图像比例不大，却决定能不能支撑；只把整张深度图压成一个向量，可能保住整体地形类别，却丢掉关键边界。相机还看不到已到脚下、尤其后足附近的区域。

START 用 **TR-Net** 从深度和本体特征重建机器人坐标中的局部高程图。GRU 保存以前看见的地形，先用 MLP 生成粗图，再用 U-Net 精修边缘和表面；粗图用 MSE、精修图用 L1 监督。随后策略对重建图编码，与本体历史融合，继续做显式状态和隐上下文估计。

这里的图不仅是训练时的辅助输出，还是**视觉与动作之间真正经过的中间表示**。训练初期重建不准，AdaSmpl 会按学习情况混用真值图与预测图；部署仅用预测图。边缘惩罚、地形随机化和课程也都是方法的一部分。

**讲给队员的说法：**“端到端训练不妨碍中间量有明确物理意义。稀疏落足最贵的信息是哪里能踩、边缘在哪里，所以它专门把这些几何信息留下来。”

主图用 **Fig.2**，必要时补 **Fig.3** 的重建和精修链。它与本仓早期“深度 → 轮边 scan → 冻结教师”都使用几何中间量，但地图覆盖、时序记忆、联合训练和动作网络的可更新性不同，不能称为同一实现。

### C6 · PUMA：落足信息能否引导动作，而不规定整条轨迹？

[原文](https://arxiv.org/abs/2601.15995) · RA-L，2026 年 7 月接收 · Wang 等。

PUMA 的切入点是：只奖励机身前进，策略还得自己探索哪里值得接触；若完整规定每只脚轨迹，又会约束高动态动作。论文从几何支撑区域构造候选锚点，形成自我坐标中的前足距离、当前与后续方向误差等极坐标先验，并从深度与本体历史估计这些量，输入 actor。

**它仍有落足奖励，但目标不是严格跟踪完整脚轨迹。** 距离、朝向和稀疏到达奖励共同引导接近有用接触区域，策略保留通过姿态变化、蹬墙等完成任务的自由。训练用 PAS 将 actor 的落足输入从真值逐渐换为预测，另以多 critic 分别学习任务、落足和正则奖励组的回报，再合并优势更新策略。

**讲给队员的说法：**“给策略一个下一步值得接触哪里的提示，比只说向前走更具体；同时保留身体怎样配合这个接触的选择空间。”

主图用 **Fig.2**。它的一阶段训练不等于无需人工设计：候选支撑锚点、先验表示和奖励都含任务知识。也不能将它讲成无条件自行发现全部落足点的通用规划器。本仓迁移记录提到曾有 PUMA foothold 研究分支，但当前交付策略没有充分证据可归为 PUMA 复现。

## 用什么实验支撑这些解释？

架构图说明信息怎样流动，对照实验才帮助判断某个设计是否有用。准备讲稿时，每篇保留一项最贴近设计动机的证据即可；以下数字和结论属于原论文实验，并非本仓复测。

| 论文 | 优先读的实验 | 能支持什么，边界在哪里 |
|---|---|---|
| DreamWaQ | Fig.5 对比绊脚时的速度估计；Table III 比较有无 AdaBoot | 上下文辅助估计在非正常接触时有价值；原文仿真随机行走的存活时间占比从无 AdaBoot 的约 90.7% 提升到约 95.2%。这不是实机所有地形的成功率。 |
| HIM | Fig.5 去掉速度／隐表示的输入或学习目标，并与直接回归比较 | 历史、辅助目标和 actor 使用这些信息都要分别检查；仅把历史拼进 PPO 不等于复现 HIM。作者的实验支持其设置下的收益，不能证明 latent 每一维都有稳定的物理含义。 |
| Extreme Parkour | Table 2 的 NoInner 与 NoClear；Table 3 的航向蒸馏对照 | NoInner 在高台上明显受损；NoClear 的总前进指标略好，但平均踩边指标约 0.08，高于完整方法的约 0.03。优化前进与降低危险落脚是两个目标，不能只挑一个分数。 |
| PIE | Table III 去掉地图、速度、足高与下一观测目标；Fig.3 增加相机误差 | 不同辅助量的作用不同。下一帧预测在标准随机化范围内收益较小，在更强视觉误差下差异扩大；适合讲“身体反馈如何补视觉”，不能概括成每个场景都有同样大的提升。 |
| START | Fig.5 重建误差；Fig.6 / Fig.7 任务表现和落足分布 | 分别去掉本体输入、GRU、U-Net、AdaSmpl、边缘惩罚和地形课程，区分地图质量与控制训练的贡献。对 PIE 的比较在此主要是同环境重训的 START w/o TR-Net，不能当作所有实现与硬件上的全局排名。 |
| PUMA | Table II 先验／PAS／多 critic 对照；Table IV 实际落足误差 | 严格落足跟踪器的落足误差更小，但在蹬墙类任务上成功率较低；说明对于这些启发式候选点，接触提示与姿态自由度可能比精确追点更有用。结论依赖候选点的质量，并不否定精确规划。 |

## 其他论文怎样放进这张地图

### 基础：RL 运控为什么在工程上变得可用？

**F1 · Learning agile and dynamic motor skills（2019）**：将学习得到的执行器模型纳入仿真，并结合动力学随机化，使仿真训练的策略能在 ANYmal 上执行运动与恢复。值得讲的是：策略输出正确不够，执行器延迟、力矩响应和闭环执行也必须可信。不要把它概括成“第一个用了 PPO 的四足”。[原文](https://arxiv.org/abs/1901.08652)

**F2 · Challenging Terrain（2020）**：教师在仿真中利用地形与接触等特权信息学运动，再将能力迁给仅有本体历史的学生；自适应课程将训练放在有学习价值的地形上。它是“训练时信息丰富，部署时传感有限”的重要背景。输出运动结构也与今天常见的纯关节目标策略有差异，不宜直接用本仓 actor 图代替原方法。[原文](https://arxiv.org/abs/2010.11251)

**F3 · Learning to Walk in Minutes（CoRL 2021）**：核心贡献包括单 GPU 大规模并行、适合并行 PPO 的训练配置，以及按表现升降难度的 game-inspired curriculum。应借它解释“同一批机器人为什么处在不同难度”，无需逐项念 PPO 参数。论文使用 Isaac Gym；本仓当前使用 MjLab / MuJoCo-Warp，继承的是训练范式而非相同仿真器。[原文](https://arxiv.org/abs/2109.11978)

### 状态估计、适应与恢复

**A1 · RMA（RSS 2021）**：先将摩擦、载荷、电机强度等环境参数编码为 latent，与策略一起训练；再冻结基础策略，从状态和动作历史拟合这个 latent。部署时“适应”主要是前向推理随历史变化，不是现场梯度下降改权重。它解释了为何“一个固定策略＋一个随接触变化的上下文”能产生不同动作。图 **Fig.2** 最清楚。[原文](https://arxiv.org/abs/2107.04034)

**A2 · Concurrent Training（RA-L / ICRA 2022）**：历史观测估计速度、足高和接触概率，估计器用仿真真值监督；actor 与估计器并行训练，actor 直接面对估计误差。与 RMA 不同，它不以拟合特权环境编码为主要目标，也没有先完成一个 oracle actor 再换输入的独立阶段。与 DreamWaQ 的对照是：显式量已经很有用，为何还需要无逐维标签的上下文？主图 **Fig.2**。[原文](https://arxiv.org/abs/2202.05481)

**A5 · DreamWaQ++（T-RO 2026，预印本 2024）**：同组从盲行走扩展到障碍预见。外感知以点云进入 PointNet 分支，配合置信过滤、短时外感知对齐与记忆、MLP-mixer 多模态融合；保留速度与重建监督，并增加表示对齐和技能多样性目标。相机或 LiDAR 提供的数据在框架中可以统一为点云，它不是简单给 DreamWaQ 接一层深度 CNN。相机失效时的退化行为值得讨论，但不等于失效时仍可安全越过所有地形。图 **Fig.2**。[原文](https://arxiv.org/abs/2409.19709)

**A6 · DreamRiser（RSS 2023 workshop）**：把 DreamWaQ 的本体地形上下文用于复杂地形上的起身，目标同时重视机身恢复和足部稳定支撑，并随机初始姿态、地形难度。适合解释“恢复后能继续运动”比“短暂翻正”更有意义。它是独立恢复策略，可与 DreamWaQ 切换，不要与 RecoveryLoco 视为同一项目；也不要把 workshop 稿标成 RSS 主会论文。[原文](https://arxiv.org/abs/2306.12712)

**A7 · Learning H-Infinity Locomotion Control（CoRL 2024）**：HIM 同组的另一条线，改变的是扰动产生方式。固定随机推力未必打中策略弱点，因此训练一个依赖状态的 disturber 找到有效外力；再用 H∞ 启发的性能代价与外力强度约束稳定对抗优化。它适合与“随便加大随机推力”比较，不宜解释成 HIM 更换了 latent loss。图 **Fig.2**。[原文](https://arxiv.org/abs/2404.14405)

### 同时期跑酷：相似视频背后的不同训练解法

**B1 · Egocentric Vision（CoRL 2022）**：RMA 作者合作脉络中的视觉前作，也是 Extreme Parkour 的直接方法背景。scandot 教师先用便宜几何学运动，视觉学生再借时序记忆从前向深度推断脚下地形。分阶段还有计算动机：大量 RL 样本都渲染深度很贵。主图 **Fig.3**。不能因论文集在 2023 出版就将会议年改为 2023。[原文](https://arxiv.org/abs/2211.07638)

**B3 · Robot Parkour Learning（CoRL 2023）**：探索困难是核心。先允许带代价的软约束穿障，使策略获得“有可能过去”的经验；再用真实硬约束微调技能，最后将多种技能蒸馏成一个深度视觉策略。对应的是**探索课程**，不是部署时允许穿障。与 EP 比较时，应突出软／硬物理课程和技能合并，而不是只比较最终高台高度。[原文](https://arxiv.org/abs/2309.05665) · [正式论文集](https://proceedings.mlr.press/v229/zhuang23a.html)

**B4 · ANYmal Parkour（Science Robotics 2024）**：分为环境重建、导航和低层运动技能。导航策略根据几何与技能能力选择跳跃、爬升、下台或下蹲及其目标；低层执行技能。它适合支持“整个系统可由多个层次组成”的讨论。与本仓路线切换存在结构上的可比性，但本仓手工路线和策略调度不能称作复现了其学习式导航。主图 **Fig.2**。[原文](https://arxiv.org/abs/2306.14874)

**B5 · Robust Perceptive Locomotion in the Wild（Science Robotics 2022）**：将本体与带噪地图观测融合成 recurrent belief，并学习在视觉不可靠时降低其影响。其外感知输入来自机器人中心的 elevation map，与 EP 直接从深度到动作的部署链不同。最值得讲的是“看见的表面未必能支撑身体”，例如植被、雪、反光地面，身体接触反馈必须修正感知。训练图 **Fig.5**、belief 结构 **Fig.6**。[原文](https://arxiv.org/abs/2201.08117)

### 浙大系列：从估计内容到空间表示和记忆

**C1 · Key Estimation（IROS 2024）**：在 Wukong-IV 人形运动任务中，结合显著性分析与对照实验检查哪些估计变量真正影响策略。实验强调速度估计的重要性，地形估计也帮助复杂地形适应。它提供“为什么显式估这些量”的思路；结论依赖任务、奖励、机体和观测，不能当作轮足梅花桩各输入重要性的直接证据。图 **Fig.2** 架构、**Fig.3** 分析。[原文](https://arxiv.org/abs/2403.05868)

**C3 · MOVE（ICRA 2025，预印本 2024）**：PIE 中 Luo、Li、Wu、Zhu 的连续工作，把问题拓展到三维障碍、侧向／后向运动和受阻视觉。PS-Net 的可部署编码器融合本体、前向深度并用 GRU 记忆；另一训练用编码器看到更完整的周围信息。前视重建提供清楚监督，pseudo-siamese 表示对齐学习两者共享的特征。特权 cube map 包括前、上、下、左、右，**不包括后向**。不能把推断未知环境讲成真正看见了背后。图 **Fig.2 / Fig.3**。[原文](https://arxiv.org/abs/2412.03353)

**C4 · Walking with Terrain Reconstruction（arXiv 2024，v2 为 2025）**：与 START 并排阅读的早期公开记录，已有本体辅助记忆、局部图重建、粗图＋精修、I-E Estimator、AdaSmpl 和稀疏地形课程。这些机制已存在，因此展示时优先用 START 这一条讲完整，再把 C4 放脚注；不把这些共同机制当成 START 相比 C4 新增加的创新。图 **Fig.2 / Fig.3**。[原文](https://arxiv.org/abs/2409.15692)

**C7 · Look Forward to Walk Backward / LF2WB（ICRA 2026）**：同组 Luo、Li 等将“前视相机如何支持后退”单独研究。先在向前运动时写入地形记忆，后退时检索；delta-rule 记忆沿相关子空间选择性擦除与写入，训练可分块并行，部署保持固定大小状态与每步计算。输出局部高程图及速度供 actor 使用。它针对**曾看过、现在看不见**的地形；不是凭前向相机无中生有地看见未知后方。有效记忆时域没有形式保证，论文也报告偶发失败。图 **Fig.2**。[原文](https://arxiv.org/abs/2603.03138)

### 单列对照：任务定义与控制接口有时比网络更关键

**D1 · Risky Terrains（IROS 2024）**：将精确落足问题设为到达目标的导航任务，让策略自行安排速度；结合课程、RND 探索与对称数据增广，先训练 generalist，再微调 specialist。重要限制是实机实验使用**真值地图和动捕状态估计**，作者明确不以感知为研究重点。因此不能用它的梅花桩视频证明已解决低成本前视感知。图 **Fig.2**。[原文](https://arxiv.org/abs/2311.10484)

**D2 · DTC（Science Robotics 2024）**：模型规划器提供足点与运动参考，RL 学习在模型误差、打滑和软地面上可靠跟踪。它把“哪里落脚”的几何规划与“如何稳健执行”组合起来，提醒我们端到端自由探索不是唯一选择。参考轨迹包含足端等运动信息，与本仓向速度策略发 `vx / wz` 的接口不同。[原文](https://arxiv.org/abs/2309.15462)

**D3 · Blind Stair Climbing with Legged and Wheeled-Legged Robots（ICRA 2024）**：用位置目标代替逐时刻速度跟踪，并加入 stairs mode 的布尔输入；非对称训练让部署时不需外感知。可对照“上台阶本来就要先减速、抬身、再前进，固定速度误差是否惩罚了必要动作”。实机展示为双轮平衡机器人 Ascento 的 15 cm 台阶；不同四足及轮足平台主要在仿真验证，不能笼统写成全部平台实机验证。图 **Fig.2 / Fig.4**。[原文](https://arxiv.org/abs/2402.06143)

**D4 · Rapid Locomotion（RSS 2022）**：MIT 团队用自适应指令课程逐步扩展能掌握的速度区域，结合动力学适应实现快速运动。它与 RMA 共享适应思想，但不是相同作者系列。用于本仓分享时，最有意义的迁移思考是：训练采样分布决定策略擅长哪些命令；我们需要的是低速转向与精对齐，不能因为它高速出色就照搬命令分布。图 **Fig.2** 只能看接口，理解课程需读方法。[原文](https://arxiv.org/abs/2205.02824)

### Cheng 的相关工作：动作空间、任务组合和训练稳定性

**E1 · Deep Whole-Body Control（CoRL 2022）**：同时学习腿与机械臂的统一策略，ROA 缩小特权编码与可部署历史估计之间的差异；Advantage Mixing 利用动作与任务奖励的关系缓解局部最优。它解释了 EP 中引用的 ROA 从哪里来。统一策略不等于把所有回报混成一个标量就自然学好协调。图 **Fig.2**。[原文](https://arxiv.org/abs/2210.10044)

**E2 · Legs as Manipulator（ICRA 2023）**：通过坡度课程学会爬墙，再训练一条前腿操作、其余腿支撑的技能；用从演示得到的行为树组合较长任务。最适合支持“技能内部可用 RL，技能组合可用结构化系统”的讨论。它并不是 EP 的另一个名字，也不是简单复制 EP 的视觉跑酷网络。图 **Fig.2**。[原文](https://arxiv.org/abs/2303.11330)

**E3 · Visual Whole-Body Control（CoRL 2024）**：低层控制身体速度与末端目标，高层根据视觉提出任务命令；为高层使用特权教师—视觉学生训练。其机械臂部分采用末端命令与 IK，不应概括为所有关节都由同一个直接动作头生成。它是 Cheng 与 UCSD Xiaolong Wang 等人的相关合作，适合作为“相似积木可以组成不同控制层级”的备查材料。图 **Fig.3**。[原文](https://arxiv.org/abs/2403.16967)

## 把论文转回本仓，哪些话有证据？

本轮查阅了 train 的 HIM 实现、plum 教师／两类视觉学生、蒸馏与 BC 脚本、部署 manifest，以及当前展示证据。部分总览文档保留早期迁移时的描述；判断是否存在深度学生应以当前代码与 manifest 为准。这里不重跑已验收任务。

| 文献或方法 | 本仓能支持的关系 | 讲解时的措辞 |
|---|---|---|
| HIM | 有 estimator、prototype swap、独立估计器优化、PPO 集成及历史配置 | “移植并训练过 HIM 分支”；不能把全部已交付策略统称为 HIM |
| DreamWaQ | 有阅读和本地轮足衍生仓参考；未见本仓完整原论文复现证据 | “用于理解显式速度＋隐上下文的设计”；用户本地 Dreamwaq 仓是衍生实现 |
| Extreme Parkour | 有教师—视觉学生、奖励及课程迁移；后期学生 BC 后继续 PPO | “借用了训练机制与部分奖励思路”；说明没有同款 GRU／航向蒸馏 |
| PIE / START / PUMA | 是本轮补充的方法对照；迁移记录出现 PUMA 旧研究分支 | “可用于解释方案取舍”；不能据此补写成当时已经采用的核心算法 |
| RecoveryLoco | 本地仓包含恢复、GRU、多 critic 等训练实现，README 致谢 HIMLoco | “参考过开源训练配方”；目前没有核实到与该仓明确对应的正式论文，不放进论文计数 |

直接证据入口：

- [HIM estimator](../../../train/src/alldog_mjlab/algorithms/him/estimator.py)：速度 MSE、prototype、Sinkhorn、swap loss；[policy](../../../train/src/alldog_mjlab/algorithms/him/policy.py)：当前观测＋估计输入 actor，PPO 不向 estimator 回传梯度。
- [几何中间量学生](../../../train/src/alldog_mjlab/tasks/velocity/profiles/plum/student.py)：深度预测 fine scan，接冻结教师。早期长桩视频的具体历史版本以 [sources.md](../sources.md) 和随包证据为准，不能与当前文件维度混用。
- [直接深度 actor](../../../train/src/alldog_mjlab/tasks/velocity/profiles/plum/depth.py)：1 帧深度＋本体历史，critic 使用特权信息；[部署 manifest](../../../demo/rl_sar/policy/blackW/plum_depth_final_v1/manifest.json)：当前交付模型来源。
- [上游与迁移取舍](../../../train/UPSTREAM.md)、[训练说明](../../../train/README.md)：区分代码来源、研究分支、后来补回的实现。

最值得发展成讲稿的是这个问题：**为什么后来从预测轮边 scan 改成直接深度 actor？** 代码中的诊断指出，fine scan 集中在轮周围，而相机看到前方；强制先重建轮边 scan，可能丢掉视觉特有的提前量。后期选择直接特征融合，是本项目的具体取舍。START 则提示另一个可能方向：如果显式重建覆盖前方、脚下和后方，并带时序记忆，中间图仍可能有价值。两者并不矛盾。

“冻结教师的学生永远不能超过教师”也不宜当作数学定理。更准确的是：固定教师动作映射且只优化模仿目标，训练没有直接鼓励学生在任务回报上探索更好动作；学生偶然在个别场景更好，并不违背这一点。

## 15 分钟分享中的取舍建议

文献讲解建议占 **约 4 分钟，穿插在实现故事里**，其余时间用于任务效果、训练取舍和实验。若把 29 篇论文讲成单独综述，会挤掉队内赛要求中的奖励调整、训练机制和对比实验。

| 出现场景 | 讲哪一个问题 | 画面 | 讲述预算 |
|---|---|---|---:|
| 平地／本体历史之前 | 历史为何有用，DreamWaQ 与 HIM 的训练目标怎样不同 | 两篇原图，逐次显示；口头只追踪编码器、监督、actor 三条关系 | 65 s |
| 越障与教师学生之前 | 动作如何先学出来，为什么再换成视觉 | EP Fig.2；奖励投影式放讲稿，必要时指图解释 | 55 s |
| 梅花桩实现之后 | 中间表示应保留哪些几何信息 | START Fig.2，对照我们两代学生的结构 | 65 s |
| 方法拓展与讨论 | 一阶段估计、落足先验还有哪些路线 | PIE 或 PUMA Fig.2 **二选一** | 35 s |
| 转场 | 从论文设计问题落回本项目实验 | 一句话承接 | 20 s |

准备时精读六篇，台上重点解释 4–5 篇即可。MOVE、LF2WB、DreamWaQ++、DTC 和其他支线放备查；它们的价值是帮助回答“还能怎么做、为什么这次没这样做”，不是增加展示页数量。架构图给观众定位，**缺口、训练信号、增量和局限写进讲稿**。

## 易混点与引用原则

1. **历史堆叠 ≠ HIM。** 普通 MLP 也可吃历史；HIM 的特殊之处包括显式估计与 representation objective。
2. **隐变量 ≠ 摩擦系数标签。** 除非训练目标明确定义，不能逐维赋予物理意义；聚类图也不足以证明完全解耦。
3. **下一帧预测 ≠ 多步规划。** DreamWaQ / PIE 的重建目标与部署时滚动预测、搜索动作是两回事。
4. **两种 KL 要分清。** VAE 中是 latent 分布的正则；PPO 中可能涉及策略更新距离或学习率调节。
5. **训练时特权信息有多个入口。** critic 输入、估计监督、临时真值替换、教师 actor 输入应分别说明。
6. **深度图、scandot、局部高程图、轮边 fine scan 的视野不同。** 名字都叫“地形信息”不能说明可相互无损替换。
7. **不能横比不同论文的最高障碍数字当算法排名。** 机器人尺寸、力矩、起始条件、感知、地形与成功判据都可能不同。
8. **年份区分预印本、接收和卷期。** Robot Parkour Learning 是 CoRL 2023；EP 是 ICRA 2024；DreamWaQ++ 的当前版本是 T-RO 2026；START 的 PDF 写 2025 年 12 月接收，正式卷期为 2026 年。
9. **同名作者检索会混入无关工作。** Shixin Luo 的作者查询包含视觉生成和通信论文；本材料只在单位、合作者和研究内容吻合时归为 PIE 系列。

本轮优先阅读论文全文的架构、方法、实验设置和局限，作者／实验室关系再用项目页交叉核对。原文下载只用于临时查阅，仓库不保存完整 PDF；[papers.json](papers.json) 保存 arXiv ID、读取版本、对应版本链接和 PDF SHA256。现有展示用图的出处可查 [sources.md](../sources.md)；新图目前只给图号建议，尚未制作或替换展示素材。

同族核验与正式发表补充来源：

- [DreamWaQ 项目页](https://sites.google.com/view/dreamwaq)、[DreamRiser 项目页](https://sites.google.com/view/dreamriser)、[DreamWaQ++ 项目页](https://dreamwaqpp.github.io/)。
- [HIM 官方仓说明](https://raw.githubusercontent.com/InternRobotics/HIMLoco/main/README.md)、[HIM 项目页](https://junfeng-long.github.io/HIMLoco/)、[H-Infinity 项目页](https://junfeng-long.github.io/HINF/)。
- [RMA 项目页](https://ashish-kmr.github.io/rma-legged-robots/)、[Egocentric Vision 项目页](https://vision-locomotion.github.io/)、[Extreme Parkour 项目页](https://extreme-parkour.github.io/)、[Cheng 作者主页](https://chengxuxin.github.io/)。
- 浙大系列各论文首页单位与作者、PIE / START / PUMA / LF2WB 的方法及参考文献；[MOVE 正式 DOI](https://doi.org/10.1109/ICRA55743.2025.11127928)、[START 正式 DOI](https://doi.org/10.1109/LRA.2025.3645649)、[Key Estimation 正式 DOI](https://doi.org/10.1109/IROS58592.2024.10801648)。
- [Blind Stair Climbing 正式 DOI](https://doi.org/10.1109/ICRA57147.2024.10610069)、[Robot Parkour Learning 正式论文集](https://proceedings.mlr.press/v229/zhuang23a.html)。

## 论文目录

下列 ID 对应正文及 [papers.json](papers.json)；星号是准备阶段优先精读，并非全部都要进入主讲。完整作者以 JSON 为准。

| ID | 完整题名与原文 | 首作／研究群体 | 发表记录 | 阅读顺序 |
|---|---|---|---|---|
| F1 | [Learning agile and dynamic motor skills for legged robots](https://arxiv.org/abs/1901.08652) | Jemin Hwangbo 等；ETH RSL / Intel | Science Robotics 2019 | 背景对照 |
| F2 | [Learning Quadrupedal Locomotion over Challenging Terrain](https://arxiv.org/abs/2010.11251) | Joonho Lee 等；ETH RSL / Intel / KAIST RAILAB | Science Robotics 2020 | 背景对照 |
| F3 | [Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://arxiv.org/abs/2109.11978) | Nikita Rudin 等；ETH RSL / NVIDIA | CoRL 2021（论文集出版于2022） | 背景对照 |
| A1 | [RMA: Rapid Motor Adaptation for Legged Robots](https://arxiv.org/abs/2107.04034) | Ashish Kumar 等；CMU Pathak / UC Berkeley Malik | RSS 2021 | 背景对照 |
| A2 | [Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion](https://arxiv.org/abs/2202.05481) | Gwanghyeon Ji 等；KAIST RAILAB / Jemin Hwangbo | RA-L 2022 / ICRA 2022 | 背景对照 |
| A3 ★ | [DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep Reinforcement Learning](https://arxiv.org/abs/2301.10602) | I Made Aswin Nahrendra 等；KAIST Urban Robotics Lab / Hyun Myung | ICRA 2023 | 核心精读 |
| A4 ★ | [Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response](https://arxiv.org/abs/2312.11460) | Junfeng Long 等；Shanghai AI Laboratory OpenRobotLab / Jiangmiao Pang | ICLR 2024（预印本2023） | 核心精读 |
| A5 | [DreamWaQ++: Obstacle-Aware Quadrupedal Locomotion With Resilient Multi-Modal Reinforcement Learning](https://arxiv.org/abs/2409.19709) | I Made Aswin Nahrendra 等；KAIST Urban Robotics Lab / Hyun Myung + collaborators | T-RO 2026（预印本2024） | 拓展备查 |
| A6 | [Robust Recovery Motion Control for Quadrupedal Robots via Learned Terrain Imagination](https://arxiv.org/abs/2306.12712) | I Made Aswin Nahrendra 等；KAIST Urban Robotics Lab / Hyun Myung | RSS 2023 Workshop on Experiment-oriented Locomotion and Manipulation Research | 拓展备查 |
| A7 | [Learning H-Infinity Locomotion Control](https://arxiv.org/abs/2404.14405) | Junfeng Long 等；Shanghai AI Laboratory OpenRobotLab / Jiangmiao Pang | CoRL 2024 | 拓展备查 |
| B1 | [Legged Locomotion in Challenging Terrains using Egocentric Vision](https://arxiv.org/abs/2211.07638) | Ananye Agarwal 等；CMU Pathak / UC Berkeley Malik | CoRL 2022（论文集出版于2023） | 背景对照 |
| B2 ★ | [Extreme Parkour with Legged Robots](https://arxiv.org/abs/2309.14341) | Xuxin Cheng 等；CMU Pathak | ICRA 2024（预印本2023） | 核心精读 |
| B3 | [Robot Parkour Learning](https://arxiv.org/abs/2309.05665) | Ziwen Zhuang 等；Shanghai Qi Zhi / Stanford / ShanghaiTech / CMU / Tsinghua collaboration | CoRL 2023 | 背景对照 |
| B4 | [ANYmal Parkour: Learning Agile Navigation for Quadrupedal Robots](https://arxiv.org/abs/2306.14874) | David Hoeller 等；ETH RSL | Science Robotics 2024（预印本2023） | 背景对照 |
| B5 | [Learning robust perceptive locomotion for quadrupedal robots in the wild](https://arxiv.org/abs/2201.08117) | Takahiro Miki 等；ETH RSL / Intel / KAIST RAILAB | Science Robotics 2022 | 背景对照 |
| C1 | [Toward Understanding Key Estimation in Learning Robust Humanoid Locomotion](https://arxiv.org/abs/2403.05868) | Zhicheng Wang 等；Zhejiang University / Jun Wu / Qiuguo Zhu | IROS 2024 | 拓展备查 |
| C2 ★ | [PIE: Parkour with Implicit-Explicit Learning Framework for Legged Robots](https://arxiv.org/abs/2408.13740) | Shixin Luo 等；Zhejiang University / Jun Wu / Qiuguo Zhu | RA-L 2024 | 核心精读 |
| C3 | [MOVE: Multi-skill Omnidirectional Legged Locomotion with Limited View in 3D Environments](https://arxiv.org/abs/2412.03353) | Songbo Li 等；Zhejiang University / Jun Wu / Qiuguo Zhu | ICRA 2025（预印本2024） | 背景对照 |
| C4 | [Walking with Terrain Reconstruction: Learning to Traverse Risky Sparse Footholds](https://arxiv.org/abs/2409.15692) | Ruiqi Yu 等；Zhejiang University / Jun Wu / Qiuguo Zhu | arXiv 2024，当前v2为2025 | 背景对照 |
| C5 ★ | [START: Traversing Sparse Footholds with Terrain Reconstruction](https://arxiv.org/abs/2512.13153) | Ruiqi Yu 等；Zhejiang University / Jun Wu / Qiuguo Zhu + collaborators | RA-L：2025年12月接收，2026年2月卷期 | 核心精读 |
| C6 ★ | [PUMA: Perception-driven Unified Foothold Prior for Mobility Augmented Quadruped Parkour](https://arxiv.org/abs/2601.15995) | Liang Wang 等；Zhejiang University / Jun Wu / Qiuguo Zhu + Zhe Sun collaborators | RA-L：2026年7月接收 | 核心精读 |
| C7 | [Look Forward to Walk Backward: Efficient Terrain Memory for Backward Locomotion with Forward Vision](https://arxiv.org/abs/2603.03138) | Shixin Luo 等；Zhejiang University / Jun Wu / Qiuguo Zhu + collaborators | ICRA 2026 | 拓展备查 |
| D1 | [Learning Agile Locomotion on Risky Terrains](https://arxiv.org/abs/2311.10484) | Chong Zhang 等；ETH RSL | IROS 2024（预印本2023） | 背景对照 |
| D2 | [DTC: Deep Tracking Control](https://arxiv.org/abs/2309.15462) | Fabian Jenelten 等；ETH RSL | Science Robotics 2024（预印本2023） | 背景对照 |
| D3 | [Reinforcement Learning for Blind Stair Climbing with Legged and Wheeled-Legged Robots](https://arxiv.org/abs/2402.06143) | Simon Chamorro 等；ETH Autonomous Systems Lab / Mila and collaborators | ICRA 2024 | 背景对照 |
| D4 | [Rapid Locomotion via Reinforcement Learning](https://arxiv.org/abs/2205.02824) | Gabriel B Margolis 等；MIT Improbable AI | RSS 2022 | 拓展备查 |
| E1 | [Deep Whole-Body Control: Learning a Unified Policy for Manipulation and Locomotion](https://arxiv.org/abs/2210.10044) | Zipeng Fu 等；CMU Pathak | CoRL 2022 | 拓展备查 |
| E2 | [Legs as Manipulator: Pushing Quadrupedal Agility Beyond Locomotion](https://arxiv.org/abs/2303.11330) | Xuxin Cheng 等；CMU Pathak / UC Berkeley | ICRA 2023 | 拓展备查 |
| E3 | [Visual Whole-Body Control for Legged Loco-Manipulation](https://arxiv.org/abs/2403.16967) | Minghuan Liu 等；UC San Diego / Xiaolong Wang | CoRL 2024 | 拓展备查 |
