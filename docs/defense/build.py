#!/usr/bin/env python3
"""Build editable PPTX, offline HTML, PDF and previews from the same layout.

python -m pip install -r requirements.txt
python -m playwright install chromium
python build.py
No robot simulation or model inference is run.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path

from PIL import Image, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
W, H = 1600, 900
BG, CARD, LINE = '#0b1423', '#142236', '#29405b'
WHITE, MUTED, CYAN, ORANGE = '#eef5ff', '#a6bbd3', '#58dfd1', '#ffbc78'
RED, GREEN = '#eb8f94', '#58dfb5'
FONT = 'Noto Sans CJK SC'
FONT_FILE = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
SLIDES = []


def wrap_text(text, width, size):
    font = ImageFont.truetype(FONT_FILE, size, index=2)
    lines = []
    for paragraph in text.split('\n'):
        line = ''
        for char in paragraph:
            if line and font.getlength(line + char) > width:
                lines.append(line)
                line = char
            else:
                line += char
        lines.append(line)
    return '\n'.join(lines)


class Slide:
    def __init__(self, title, section, source, time='', backup=False):
        self.title, self.section, self.source = title, section, source
        self.time, self.backup, self.elements = time, backup, []
        self.rect(0, 0, W, H, BG, radius=0)
        self.text(64, 32, 1200, 35, 'blackW  /  ' + section, 22, CYAN, bold=True)
        self.text(64, 88, 1480, 80, title, 51, WHITE, bold=True)
        self.rect(64, 177, 1472, 2, LINE, radius=0)
        self.text(64, 831, 1390, 48, source, 18, MUTED)
        self.text(1460, 835, 76, 35, f'{len(SLIDES)+1:02d}', 24, CYAN, bold=True)
        SLIDES.append(self)

    def rect(self, x, y, w, h, fill=CARD, radius=16, border=None):
        self.elements.append(dict(kind='rect', x=x, y=y, w=w, h=h, fill=fill, radius=radius, border=border))

    def text(self, x, y, w, h, text, size=29, color=WHITE, bold=False, align='left'):
        text = wrap_text(text, w, size)
        estimated = len(text.split('\n')) * size * 1.33
        if estimated > h + 9:
            raise ValueError(f'Text exceeds layout on slide {len(SLIDES)}: {text!r} needs {estimated:.0f}px, has {h}')
        self.elements.append(dict(kind='text', x=x, y=y, w=w, h=h, text=text, size=size, color=color, bold=bold, align=align))

    def image(self, x, y, w, h, file):
        self.elements.append(dict(kind='image', x=x, y=y, w=w, h=h, file=file))

    def video(self, x, y, w, h, file, poster):
        self.elements.append(dict(kind='video', x=x, y=y, w=w, h=h, file=file, poster=poster))

    def label(self, x, y, text, color=CYAN, w=600):
        self.text(x, y, w, 40, text, 25, color, True)

    def arrow(self, x, y, w=60):
        self.text(x, y, w, 60, '→', 44, CYAN, True, 'center')

    def card(self, x, y, w, h, label, body, color=CYAN):
        self.rect(x, y, w, h)
        self.label(x+24, y+22, label, color, w-48)
        self.text(x+24, y+79, w-48, h-95, body, 27)


def make_slides():
    s = Slide('从论文思路到轮足越障策略', 'RoboCon 队内赛 · 备赛答辩', 'blackW · 论文阅读与复现、平地运动、越障训练、系统集成。', '0:00–0:15 · 15 秒')
    s.text(64, 225, 750, 92, 'blackW 轮足机器人', 56, WHITE, True)
    s.text(64, 359, 710, 166, '先做好平地运动\n再建立通用越障能力\n最后用视觉处理稀疏落足', 37)
    s.rect(64, 649, 710, 130)
    s.text(90, 674, 658, 83, '论文依据 → 训练实验 → 可演示系统', 30, CYAN, True)
    s.image(830, 216, 706, 594, 'media/plum_handover.jpg')

    s = Slide('理论起点：历史观测能补足什么？', '01 / 论文阅读与算法复现', '原图：Long et al., HIM, arXiv:2312.11460, Fig.2；Nahrendra et al., DreamWaQ, arXiv:2301.10602, Fig.2。', '0:15–0:50 · 35 秒')
    s.label(64, 211, 'HIM · 显式速度 + 对比隐表示', w=718)
    s.rect(64, 264, 718, 367, '#ffffff')
    s.image(74, 275, 698, 346, 'figures/him_fig2.png')
    s.label(822, 211, 'DreamWaQ · 速度估计 + β-VAE', w=714)
    s.rect(822, 264, 714, 207, '#ffffff')
    s.image(835, 277, 688, 181, 'figures/dreamwaq_fig2.png')
    s.text(844, 496, 670, 130, '历史编码 → 速度与地形上下文\n下一观测重建 + KL 正则\n从运动响应中理解接触环境', 30)
    s.card(64, 663, 718, 139, 'HIM 移植与训练', '速度 MSE + prototype 对比学习', CYAN)
    s.card(822, 663, 714, 139, 'DreamWaQ 论文与源码借鉴', '理解历史状态估计与非对称训练', CYAN)

    s = Slide('从参考方法中提炼训练机制', '02 / 视觉越障与鲁棒性', '原图：Cheng et al., Extreme Parkour, arXiv:2309.14341, Fig.2；参考：RecoveryLoco，本地开源阅读笔记。', '0:50–1:10 · 20 秒')
    s.rect(64, 215, 565, 586, '#ffffff')
    s.image(78, 228, 537, 560, 'figures/extreme_parkour_fig2.png')
    s.card(675, 224, 861, 240, 'Extreme Parkour · 分阶段学越障', '特权地形辅助教师学动作\n深度学生学习可部署的感知与决策\n课程、前进奖励和落足约束一起设计')
    s.card(675, 490, 861, 224, 'RecoveryLoco · 扩大状态覆盖', '借鉴初态、摩擦与外力扰动的思路\n把恢复能力纳入训练分布\n用于思考平地模型的转向与抗扰')
    s.text(675, 751, 850, 45, '取舍：历史理解“脚下”，视觉观察“前方”。', 31, CYAN, True)

    s = Slide('平地模型：先把速度指令学准确', '03 / 本体策略结构与训练', '配置：evidence/vt_v5_{agent,env}.yaml；训练过程与命令采样：docs/training/VT-RECIPE.md。', '1:10–2:00 · 50 秒')
    s.card(64, 225, 390, 180, '本体观测 · 六帧历史', '57 × 6 = 342 维\nIMU、关节、指令、动作')
    s.arrow(458, 278, 62)
    s.card(525, 225, 530, 180, 'PPO actor · MLP', '342 → 512 → 256 → 128\nELU · 历史提供短时运动信息')
    s.arrow(1060, 278, 62)
    s.card(1127, 225, 409, 180, '16DoF 动作', '12 腿位置残差\n+ 4 轮速度')
    s.card(64, 443, 718, 250, '指令采样就是训练课程', 'vₓ ∈ [−0.4, 1.0] m/s\nωz ∈ [−0.65, 0.65] rad/s\n增加纯转向与低速混合样本\n每 5 s 换指令，10% 站立')
    s.card(822, 443, 714, 250, '关键奖励 · VT v5', '线速度：+2 exp(−eᵥ² / 0.25)\n角速度：+2.5 exp(−eω² / 0.08)\n姿态 −1.2 · 腿姿偏差 −0.8\n旋转漂移 −1，带转向跟踪门控')
    s.rect(64, 709, 1472, 94)
    s.text(89, 728, 1420, 66, '4096 环境 × 24 步 / 迭代   ·   从 v4 继续训练，选 model_800   ·   回合 20 s', 30, CYAN, True)

    s = Slide('平地实验：共面、跟随与精对齐的取舍', '04 / 对比实验与 12DoF 锁轮', '同部署探针对照：VT-RECIPE.md §6.1；锁轮结构与权重选择：locked12 manifest、运行时轮关节 PD。', '2:00–2:45 · 45 秒')
    s.label(64, 216, 'VT v1 → v2 · 同一部署探针', w=820)
    for x,txt,w in [(84,'指标',400),(500,'v1',155),(691,'v2',170)]:s.text(x,282,w,45,txt,28,CYAN,True)
    rows=[('转弯四轮高度展布','3.10 cm','0.36 cm'),('90° 足迹半径','0.572 m','0.520 m'),('平移速度 / 指令','0.781','0.631'),('原地转漂移','0.051 m','0.106 m')]
    for i,(label,a,b) in enumerate(rows):
        y=346+i*77;s.rect(64,y,824,66)
        s.text(84,y+13,405,44,label,27)
        s.text(500,y+13,180,44,a,27,MUTED)
        s.text(691,y+13,177,44,b,27,CYAN if i<2 else ORANGE,True)
    s.text(64, 693, 824, 102, '共面性改善，速度与转向也要复核。\n低速指令覆盖不足，会放大对齐误差。', 30, CYAN, True)
    s.card(936, 225, 600, 225, '精对齐使用 12DoF 锁轮', '删除轮状态与轮动作通道\n45 × 6 = 270 → MLP → 12\n从 16DoF 权重裁剪初始化')
    s.card(936, 477, 600, 188, '轮关节 PD · 切换时捕获角度', 'q* = q切换时刻，dq* = 0\nkp = 80，kd = 3')
    s.rect(936, 695, 600, 108, '#343142')
    s.text(959, 710, 553, 83, '微调后低速过冲变差，\n最终保留初始化权重用于演示。', 28, ORANGE, True)

    s = Slide('通用越障：从速度跟随走向任务奖励', '05 / 基础专家与专项专家', '通用 row2p 保存配置；楼梯、沟壑采用对应专项权重。图中为已训练出的能力档位。', '2:45–3:35 · 50 秒')
    s.label(64, 212, '先保留运动能力，再增加“过障”的学习信号', w=1472)
    s.card(64, 275, 456, 210, '运动底座', '线速度 +1 · 角速度 +0.5\n相对地形机身高 0.43 m\n姿态、功耗与动作平滑约束')
    s.arrow(523, 326, 58)
    s.card(585, 275, 450, 210, '任务塑形', '沿赛道净进度 +8\n爬升高度 +10\n有效通过 +50')
    s.arrow(1040, 326, 58)
    s.card(1103, 275, 433, 210, '结束一回合', '倾角 >1.05 rad / 越界\n完成障碍 / 数值异常\n20 s 超时截断')
    s.label(64, 499, '通用策略打底，必要时保留专项能力', w=1472)
    capabilities=[('高台','40 cm','row2p'),('高墙','40 cm','row2p'),('斜坡','35°','row2p'),('楼梯','30 cm','vt_band'),('沟壑','30 cm','s1d')]
    for i,(name,value,model) in enumerate(capabilities):
        x=64+i*298;s.rect(x,559,280,162)
        s.text(x+18,574,244,42,name,27,MUTED,True)
        s.text(x+18,620,244,60,value,39,CYAN,True)
        s.text(x+18,682,244,34,model,21,MUTED)
    s.text(64, 756, 1460, 45, '共享本体 actor：342 → 512 → 256 → 128 → 16；混合地形训练后保留专项专家。', 29, CYAN, True)

    s = Slide('40 cm 高台：训练地形要覆盖目标高度', '06 / 视频与实际训练配方', '视频：vt_mode_v2 / row2p，adaptive vx基准0.5；训练配置与权重：evidence/platform40_*。', '3:35–4:15 · 40 秒（含 6.94 秒视频）')
    s.video(64, 242, 958, 314, 'media/platform40.mp4', 'media/platform40.jpg')
    s.label(64, 585, '31 级地形采样 · d = k / 30', w=958)
    for y,label,row in [(645,'10 cm',0),(691,'30 cm',20),(737,'40 cm',30)]:
        s.text(64,y,150,45,label,27,WHITE,True)
        s.rect(235,y+6,620,27,'#233c4b',radius=0)
        s.rect(235,y+6,max(12,620*row/30),27,CYAN,radius=0)
        s.text(877,y,145,45,f'row {row}',25,MUTED)
    s.card(1062, 225, 474, 242, '混合地形 + 精确难度覆盖', '高台 / 高墙：10–40 cm\n斜坡：20–45°\n六级楼梯：单级 10–30 cm\n初始等级覆盖 0–30')
    s.card(1062, 485, 474, 170, '实际训练规模', '512 环境 × 24 步 / 迭代\n楼梯权重续训 → model_750')
    s.card(1062, 664, 474, 139, '训练中的关键取舍', '进度+8 · 爬升+10 · 通过+50', CYAN)

    s = Slide('把已有专家串起来：系统与策略衔接', '07 / 通用能力先保底 · 30 秒', '导航只计算速度指令；执行与接触条件见 demo 路线控制器。Part 1 / Part 2 均已接入实时演示。', '4:15–4:45 · 30 秒')
    s.card(64, 230, 380, 175, '路线目标与世界位姿', '位置 / 航向误差\n按地形顺序选择目标')
    s.arrow(448, 281, 62)
    s.card(515, 230, 462, 175, '位置外环 · adaptive 指令', '距离近则减速，角差大先转向\n输出 vₓ、vᵧ、ωz')
    s.arrow(981, 281, 62)
    s.card(1048, 230, 488, 175, '策略调度与底层执行', '平地前进 → 入口对齐 → 过障\n支撑状态触发退出与接管')
    for x,label,body in [(64,'16DoF 平地','长直线与轮式前进'),(565,'12DoF 锁轮','低速转向与精对齐'),(1066,'地形专家','通用越障 / 专项能力')]:
        s.card(x,465,470,145,label,body)
    s.rect(64, 659, 1472, 144, '#173e42')
    s.text(93, 685, 1414, 100, '先用已有模型完成可运行系统，再补视觉难点。\n交接依据支撑与速度：例如砂砾坑上台即切平地，及时消除残余运动。', 33, CYAN, True)

    s = Slide('梅花桩：让深度信息引导稀疏落足', '08 / 早期良好步态 · 18 秒视频', '早期长桩场地 · route2_v3；视频及原训练报告已随包保存。', '4:45–5:40 · 55 秒（含 18 秒视频）')
    s.video(64, 215, 790, 592, 'media/plum_gait_early.mp4', 'media/plum_gait_early.jpg')
    s.label(901, 215, '对应视频的训练链路', w=635)
    s.card(901, 271, 635, 168, '1 / 特权 RL 教师', '细扫描 + 本体历史 → 16 维动作\n前进、碰撞和落足边缘共同约束')
    s.card(901, 439, 635, 168, '2 / 深度扫描蒸馏', '深度 CNN → 1764 维局部扫描\n拼接本体历史，输入冻结教师')
    s.card(901, 607, 635, 168, '3 / DAgger 补学生访问状态', '25.92 万原始帧 + 9.6 万补充帧\n20 epochs；随机位姿与深度扰动')

    s = Slide('交付结果：Part 1 与 Part 2 实时演示', '09 / 已通过项目验收', '片段：Part1 v104 低杆；Part2 v56 六桩与出口。对应原始记录与视频来源见 sources.md。', '5:40–6:30 · 50 秒（含 19.48 秒视频）')
    s.video(64, 215, 1010, 568, 'media/delivery_reel.mp4', 'media/delivery_reel.jpg')
    s.card(1110, 223, 426, 227, 'Part 1', '263.8 s 返回起点\n0 次恢复\n平地、专家与固定动作协同')
    s.card(1110, 477, 426, 243, 'Part 2', '逐档障碍与专项模型\n深度策略接入演示\n支持障碍前恢复和不限时运行')
    s.text(1110, 757, 426, 46, '模型能力 → 可运行的演示流程', 26, CYAN, True)

    s = Slide('这次备赛学到的三件事', '10 / 认识与后续方向', '后续：扩大入口状态覆盖、训练稳定下台，并持续用固定探针对照模型变化。', '6:30–6:50 · 20 秒')
    s.card(64, 232, 1472, 157, '1 / 信息结构决定模型选择', '历史状态帮助理解接触与动力学，前视深度帮助规划尚未接触的落点。')
    s.card(64, 418, 1472, 157, '2 / 奖励、采样与终止要一起设计', '低速转向靠指令覆盖，越障靠任务信号，课程要覆盖真正的目标难度。')
    s.card(64, 604, 1472, 183, '3 / 用实验决定保留什么', '共面性、速度和漂移一起比较；通用能力先组成系统，再迭代视觉与泛化。')

    s = Slide('平地训练细节与锁轮模型选择', '备份 A / 训练参数', '保存配置：vt_v5_{agent,env}.yaml、locked12_{agent,env}.yaml；VT-RECIPE.md；当前策略 manifest。', backup=True)
    s.card(64, 226, 718, 327, 'VT v5 · 实际保存参数', '4096 × 24；本阶段预算 400 迭代\n从 v4 model_600 续训，选择 model_800\nlr 0.001 adaptive；clip 0.2\nγ 0.99；λ 0.95；5 epochs × 4 batches\n非站立纯转向 50%；混合中低速 50%')
    s.card(822, 226, 714, 327, '12DoF · 结构与选择', '342→270 输入裁剪，16→12 输出裁剪\n重建 critic 为 243 维输入\n1024 环境 × 64 步，微调预算 200 迭代\n低速过冲变差，最终部署 model_0\n轮关节保持切换角度：kp 80 / kd 3')
    s.card(64, 589, 1472, 211, '训练与执行中的经验', '过强漂移惩罚会诱导“少转或不转”，因此转漂惩罚加跟踪门控，并补足低速指令。\n从零训练 / 命令分布调整 / 权重迁移分别记录；演示长直线仍保留官方提供的 flat 作为实用基线。\nVT 的共面改进与锁轮模型是两条实验线，不把 v1→v2 的指标直接归于最终锁轮模型。')

    s = Slide('40 cm 高台：完整配方与难度映射', '备份 B / 通用专家', 'evidence/platform40_env.yaml、platform40_agent.yaml、platform40_manifest.json 与 platform40_sidecar.json。', backup=True)
    s.card(64, 227, 718, 334, '地形与训练', '高台 / 高墙：0.10 + 0.30 × d\n坡度：20 + 25 × d；d = k / 30\n楼梯单级：0.10 + 0.20 × d\n高台40 cm → row30；楼梯20 cm → row15\n512 × 24，续训后选择 model_750')
    s.card(822, 227, 714, 334, '奖励权重', '线速度 +1 / 角速度 +0.5（σ=0.25）\n净进度 +8 / 爬升 +10 / 通过 +50\n高度 −4（目标相对地形 0.43 m）\n竖直速度 −2 / 水平角速度 −0.05\n姿态 −0.2 / 动作差与平滑各 −0.01')
    s.card(64, 598, 718, 201, 'PPO 实际取值', 'lr 0.001 adaptive；clip 0.2；γ 0.99\nλ 0.95；5 epochs × 4 batches\nentropy 0.01；MLP 512 / 256 / 128')
    s.card(822, 598, 714, 201, '终止与视频配置', '倾角 1.05 rad；横向偏移 0.7 m\n越界 margin 0.8 m；20 s 超时\n视频：adaptive 指令，vx 基准 0.5 m/s')

    s = Slide('两代梅花桩模型：扫描蒸馏到直接深度', '备份 C / 本项目网络结构', '早期 route2_v3 manifest/report；最终 plum/depth.py、plum_stage2_agent.yaml 与最终 manifest。', backup=True)
    s.label(64, 212, '视频对应的早期模型', w=1472)
    s.card(64, 268, 415, 175, '深度 CNN', '1 × 48 × 64\n→ 1764 维预测 fine scan')
    s.arrow(482, 310, 63)
    s.card(550, 268, 506, 175, '拼接扫描与本体', '1764 + 342 = 2106\n输入冻结 RL teacher')
    s.arrow(1060, 310, 63)
    s.card(1128, 268, 408, 175, 'MLP → 16 动作', '512 → 256 → 128\nscan 验证 RMSE ≈2.5 cm')
    s.label(64, 470, '后续六桩场地的直接深度模型', w=1472)
    s.card(64, 524, 415, 175, '深度与本体两支', 'CNN 16 / 32 / 64 → 64\n本体 54 × 6 → 324')
    s.arrow(482, 566, 63)
    s.card(550, 524, 506, 175, '融合特征 → actor', '388 → 512 → 256 → 128\n继续 PPO 优化视觉与动作')
    s.arrow(1060, 566, 63)
    s.card(1128, 524, 408, 175, '16 动作', '12 腿位置残差\n4 轮速度')
    s.rect(64, 722, 1472, 79)
    s.text(88, 739, 1424, 50, '仅训练：247 维特权观测 → critic；最终 actor 无 command、无特权扫描。', 28, MUTED)

    s = Slide('六桩训练：课程、奖励与训练规模', '备份 D / 最终深度模型', '配置与公式：evidence/training_recipe.json、plum_stage2_{agent,env}.yaml；奖励 dt 缩放后截负。', backup=True)
    s.card(64, 224, 475, 338, '地形与终止', '早期填隙课程：0 → 40 cm\n固定桩高与桩顶均为 20 cm\n最终 PPO：固定官方间隙\n出生 x/y ±10 cm，yaw ±5°\n接触 >1 N / 倾角 >1.05 rad\n轮底 <5 cm；最长回合 24 s')
    s.card(563, 224, 475, 338, '关键奖励', 'r = max(0, 0.02 × Σwᵢrᵢ)\n前进 +1.5 / 航向 +0.5\n航向门控：vₓ >0.254 m/s\n非轮碰撞 −10（>0.1 N）\n轮触边缘 −1（<5 cm）\n严格通过 +500；保持 20 步')
    s.card(1062, 224, 474, 338, '规模与优化', 'BC：1,228,800 帧\n动作验证 RMSE 0.125\nPPO：1024 环境 × 32 步\n1200 迭代预算 ≈3932 万步\n选择 model_1150\nlr 3e−5 固定；初始 std 0.10')
    s.rect(64, 604, 1472, 194)
    s.label(89, 624, '为何检查奖励分解与测试初态', w=1420)
    s.text(89, 679, 1420, 99, '发现静止航向收益后增加前进门控；同时将完整下台纳入终态。\n训练统计 311/316；标准初态可通过，位置×航向 15 格仅 2 格成功，后续重点扩大入口状态覆盖。', 29, MUTED)

    s = Slide('参考方法、观测信息与演示复现', '备份 E / 技术问答', '代码与来源索引：sources.md；原始证据与视频来源：evidence/、media_manifest.json。', backup=True)
    s.card(64, 227, 718, 320, 'HIM 在本仓的实现', '历史 342 → 编码器 128 / 64\n输出：3 维速度 + 16 维 latent\n当前 57 + 3 + 16 = 76 → actor\n速度 MSE + prototype swap 损失\n32 prototypes；估计器与 PPO 分开更新')
    s.card(822, 227, 714, 320, '运行与复现入口', 'Part 1：263.80 s，0 恢复，回起点\nPart 2：resume_before:=stairs_30cm\ntimeout_scale:=0 可取消超时\n支持从指定障碍前对齐点继续\n当前演示路线见 docs/part2/ROUTE.md')
    s.card(64, 587, 1472, 212, '资料口径与传感信息', 'HIM 为算法移植与历史训练；DreamWaQ / RecoveryLoco 为思路与源码参考；EP 为机制迁移。\n训练特权信息用于教师 / critic；部署使用本体与模拟相机，世界位姿仅用于生成导航速度指令。\n长桩旧视频、当前六桩及多模型单关分别保留原记录，具体判据与命令档见 sources.md。')

def rgb(value):
    return RGBColor.from_string(value.lstrip('#'))


def build_pptx(notes):
    prs = Presentation(); prs.slide_width=Inches(W/120); prs.slide_height=Inches(H/120)
    prs.core_properties.title='blackW：从论文思路到轮足越障策略'
    prs.core_properties.subject='RoboCon 队内赛 · 七分钟备赛答辩'
    prs.core_properties.author='blackW 项目'
    prs.core_properties.comments='Editable source: docs/defense/build.py; evidence and conditions included.'
    for i,s in enumerate(SLIDES):
        slide=prs.slides.add_slide(prs.slide_layouts[6])
        for e in s.elements:
            x,y,w,h=[Inches(e[k]/120) for k in ('x','y','w','h')]
            if e['kind']=='rect':
                kind=MSO_SHAPE.ROUNDED_RECTANGLE if e['radius'] else MSO_SHAPE.RECTANGLE
                sh=slide.shapes.add_shape(kind,x,y,w,h)
                if e['radius']:
                    sh.adjustments[0]=0.08
                sh.fill.solid();sh.fill.fore_color.rgb=rgb(e['fill']);sh.line.fill.background()
            elif e['kind']=='text':
                sh=slide.shapes.add_textbox(x,y,w,h);tf=sh.text_frame
                tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0
                tf.word_wrap=False
                for j,line in enumerate(e['text'].split('\n')):
                    par=tf.paragraphs[0] if j==0 else tf.add_paragraph()
                    par.text=line;par.font.name=FONT;par.font.size=Pt(e['size']*.6)
                    par.font.bold=e['bold'];par.font.color.rgb=rgb(e['color']);par.line_spacing=1.23
                    from pptx.oxml.xmlchemy import OxmlElement
                    ea=OxmlElement('a:ea');ea.set('typeface',FONT);par._p.get_or_add_pPr().get_or_add_defRPr().append(ea)
                    par.space_before=par.space_after=Pt(0)
                    if e['align']!='left':
                        from pptx.enum.text import PP_ALIGN
                        par.alignment={'center':PP_ALIGN.CENTER,'right':PP_ALIGN.RIGHT}[e['align']]
            elif e['kind']=='image':
                # Fit without distortion.
                im=Image.open(HERE/e['file']); ratio=min(e['w']/im.width,e['h']/im.height)
                ww,hh=im.width*ratio,im.height*ratio
                slide.shapes.add_picture(str(HERE/e['file']),Inches((e['x']+(e['w']-ww)/2)/120),Inches((e['y']+(e['h']-hh)/2)/120),width=Inches(ww/120),height=Inches(hh/120))
            elif e['kind']=='video':
                slide.shapes.add_movie(str(HERE/e['file']),x,y,w,h,poster_frame_image=str(HERE/e['poster']),mime_type='video/mp4')
        message=(notes[i] if i<len(notes) else '问答备份，不计入七分钟主讲。')
        slide.notes_slide.notes_text_frame.text=f'{s.time}\n{message}\n证据：{s.source}'
    prs.save(HERE/'blackW-defense.pptx')


def build_html():
    pages=[]
    for i,s in enumerate(SLIDES):
        nodes=[]
        for e in s.elements:
            style=f"left:{e['x']}px;top:{e['y']}px;width:{e['w']}px;height:{e['h']}px;"
            if e['kind']=='rect':
                nodes.append(f'<div class="element" style="{style}background:{e["fill"]};border-radius:{e["radius"]}px"></div>')
            elif e['kind']=='text':
                style+=f'font-size:{e["size"]}px;color:{e["color"]};font-weight:{700 if e["bold"] else 400};text-align:{e["align"]};'
                nodes.append(f'<div class="element text" style="{style}">{html.escape(e["text"])}</div>')
            elif e['kind']=='image':
                nodes.append(f'<img class="element" style="{style}object-fit:contain" src="{e["file"]}" alt="演示图像">')
            elif e['kind']=='video':
                nodes.append(f'<video class="element" style="{style}object-fit:contain;background:#000" controls preload="metadata" playsinline poster="{e["poster"]}" src="{e["file"]}"></video>')
        pages.append(f'<section class="slide" data-index="{i}" aria-label="{html.escape(s.title)}">'+''.join(nodes)+'</section>')
    data=[dict(title=s.title,time=s.time,backup=s.backup) for s in SLIDES]
    code='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>blackW · 七分钟备赛答辩</title>
<style>
@font-face{font-family:BlackWDefense;src:url("fonts/blackw-regular.woff2")}@font-face{font-family:BlackWDefense;src:url("fonts/blackw-bold.woff2");font-weight:700}
*{box-sizing:border-box}body{margin:0;background:#050a12;color:#eef5ff;font-family:BlackWDefense,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;overflow:hidden}
#stage{width:1600px;height:900px;position:absolute;transform-origin:top left}.slide{width:1600px;height:900px;position:relative;display:none;overflow:hidden}.slide.active{display:block}.element{position:absolute}.text{white-space:pre;line-height:1.23;overflow:visible}#toolbar{position:fixed;bottom:0;left:0;right:0;height:32px;background:#050a12;display:flex;align-items:center;justify-content:center;gap:18px;font:13px sans-serif;color:#a6bbd3}button{color:#eef5ff;background:#142236;border:1px solid #29405b;border-radius:4px;padding:3px 10px;cursor:pointer}a{color:#58dfd1}#hint{position:fixed;right:16px;top:12px;background:#142236;padding:8px 12px;border-radius:8px;font-size:14px;display:none}
@media print{@page{size:1600px 900px;margin:0}body{overflow:visible;background:#0b1423}#stage{position:static;transform:none!important;width:1600px;height:auto}.slide{display:block!important;break-after:page;page-break-after:always}#toolbar,#hint{display:none!important}video{visibility:hidden}img.pdf-poster{visibility:visible}}
</style><div id="stage">PAGES</div><div id="toolbar"><button onclick="go(n-1)">← 上一页</button><span id="counter"></span><button onclick="go(n+1)">下一页 →</button><button onclick="playVideo()">播放 / 暂停 V</button><button onclick="fullscreen()">全屏 F</button><span>方向键翻页 · 前11页主讲 · 后5页备份</span></div><div id="hint"></div>
<script>
const meta=METADATA; let n=0;const slides=[...document.querySelectorAll('.slide')];
function resize(){const k=Math.min(innerWidth/1600,(innerHeight-32)/900);const st=document.getElementById('stage');st.style.transform=`scale(${k})`;st.style.left=((innerWidth-1600*k)/2)+'px';st.style.top=((innerHeight-32-900*k)/2)+'px'}
function go(i){if(i<0||i>=slides.length)return;slides[n].querySelectorAll('video').forEach(v=>v.pause());n=i;slides.forEach((s,j)=>s.classList.toggle('active',j===n));document.getElementById('counter').textContent=`${n+1}/${slides.length} · ${meta[n].time||'问答备份'}`;location.hash=String(n+1)}
function playVideo(){const v=slides[n].querySelector('video');if(v){if(v.paused)v.play();else v.pause()}}
function fullscreen(){if(!document.fullscreenElement)document.documentElement.requestFullscreen();else document.exitFullscreen()}
window.addEventListener('resize',resize);window.addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();go(n+1)}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();go(n-1)}if(e.key.toLowerCase()==='v')playVideo();if(e.key.toLowerCase()==='f')fullscreen();if(e.key==='Home')go(0)});resize();go(Math.max(0,Math.min(slides.length-1,(Number(location.hash.slice(1))||1)-1)));
// For PDF and previews: the static poster is rendered above each video.
for(const v of document.querySelectorAll('video')){const im=document.createElement('img');im.src=v.poster;im.className='element pdf-poster';im.style.cssText=v.style.cssText+';object-fit:contain;pointer-events:none;display:none';v.after(im)}
</script></html>'''.replace('PAGES',''.join(pages)).replace('METADATA',json.dumps(data,ensure_ascii=False)).replace('前11页主讲 · 后5页备份',f'前{sum(not s.backup for s in SLIDES)}页主讲 · 后{sum(s.backup for s in SLIDES)}页备份')
    (HERE/'index.html').write_text(code)


def render():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport=dict(width=1600,height=932),device_scale_factor=1)
        page.goto((HERE/'index.html').as_uri());page.evaluate('document.fonts.ready')
        page.wait_for_function('Array.from(document.images).every(i=>i.complete)')
        page.add_style_tag(content='video{visibility:hidden}.pdf-poster{display:block!important}')
        for i,s in enumerate(SLIDES):
            page.evaluate(f'go({i})')
            page.locator('.slide.active').screenshot(path=str(HERE/'previews'/f'{i+1:02d}.png'))
        page.pdf(path=str(HERE/'blackW-defense.pdf'),width='1600px',height='900px',print_background=True,prefer_css_page_size=True)
        browser.close()
    imgs=[]
    font=ImageFont.truetype(FONT_FILE,22,index=2)
    from PIL import ImageDraw
    for i in range(len(SLIDES)):
        im=Image.open(HERE/'previews'/f'{i+1:02d}.png').convert('RGB');im.thumbnail((400,225));imgs.append(im)
    sheet=Image.new('RGB',(4*416,((len(SLIDES)+3)//4)*257),'#253448');dr=ImageDraw.Draw(sheet)
    for i,im in enumerate(imgs):
        x=(i%4)*416+8;y=(i//4)*257+8;sheet.paste(im,(x,y));dr.text((x,y+228),f'{i+1:02d}'+(' 备份' if SLIDES[i].backup else ''),font=font,fill='white')
    sheet.save(HERE/'overview.jpg',quality=90)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--no-render',action='store_true');args=ap.parse_args()
    make_slides()
    script=(HERE/'script.md').read_text();chunks=re.split(r'\n## ',script.split('\n---')[0])[1:]
    notes=[c.split('\n',1)[1].strip() for c in chunks]
    build_pptx(notes);build_html()
    if not args.no_render:render()
    media=json.loads((HERE/'media_manifest.json').read_text())
    main_videos={e['file'] for s in SLIDES if not s.backup for e in s.elements if e['kind']=='video'}
    video_budget=round(sum(v['duration_s'] for v in media if v['file'] in main_videos),2)
    report=dict(main_slides=sum(not s.backup for s in SLIDES),backup_slides=sum(s.backup for s in SLIDES),time_budget_s=410,video_budget_s=video_budget,accepted_code_commit='e2d7818',slides=[dict(number=i+1,title=s.title,time=s.time,backup=s.backup,videos=[e['file'] for e in s.elements if e['kind']=='video']) for i,s in enumerate(SLIDES)])
    (HERE/'presentation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('Built PPTX, HTML'+(', PDF, previews' if not args.no_render else ''))


if __name__=='__main__':main()
