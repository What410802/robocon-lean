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
            raise ValueError(f'Text exceeds layout on slide {len(SLIDES)+1}: {text!r} needs {estimated:.0f}px, has {h}')
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
        self.text(x+24, y+79, w-48, h-95, body, 29)


def make_slides():
    s = Slide('从单关越障到整场衔接', 'RoboCon 队内赛 · 备赛答辩', '依据：队内赛 PDF 第 4 页答辩要求；验收代码 e2d7818。', '0:00–0:25 · 25 秒')
    s.text(64, 217, 710, 94, 'blackW 轮足机器人', 57, WHITE, True)
    s.text(64, 343, 700, 164, '奖励是否对应有效越障\n视觉是否保留前瞻信息\n专家之间能否可靠交接', 37)
    s.rect(64, 612, 695, 118)
    s.text(90, 636, 640, 80, 'PPO 训练 · 深度感知 · 路线验证', 31, CYAN, True)
    s.image(828, 216, 708, 594, 'media/plum_handover.jpg')

    s = Slide('把导航、衔接、越障分开', '01 / 规则与系统', '规则：PDF 第 1、3 页；实现：README.md、demo/rl_sar/policy/WEIGHTS.json。', '0:25–1:00 · 35 秒')
    for x, title, body in [(64,'导航','世界位姿 → 速度指令\n唯一几何路线'),(576,'衔接','16 DoF：长直线\n12 DoF：锁轮精对线'),(1088,'越障','按地形选择 RL 专家\n梅花桩增加深度输入')]:
        s.card(x, 236, 448, 242, title, body)
    s.arrow(514, 316, 60); s.arrow(1026, 316, 60)
    s.card(64, 531, 710, 252, '规则决定策略边界', 'Part 1：允许 RL 与固定动作结合\nPart 2：越障使用 RL 策略', CYAN)
    s.card(806, 531, 730, 252, '训练信息与部署输入分离', '部署：IMU、编码器、真实渲染深度\n特权地形只用于训练；真值只算导航', ORANGE)

    s = Slide('奖励为何会鼓励原地不动？', '02 / 奖励诊断', '来源：train/.../profiles/plum/ep_rewards.py、teacher.py；历史诊断，不是受控单项消融。', '1:00–1:50 · 50 秒')
    s.card(64, 223, 470, 372, '现象', '只奖励向前位移\n跳过桩阵也有收益\n\n早期两轮：\n0 / 178、0 / 4000', ORANGE)
    s.card(558, 223, 470, 372, '诊断', '每步负总奖励截为 0\n静止也持续领航向奖励\n\n站立21秒累计：+11.8\n当时通关终止奖励：+10', ORANGE)
    s.card(1052, 223, 484, 372, '修改', '航向奖励加前进门控\n约束轮子边缘接触\n非轮部件触地则终止\n\n成功终态与判据对应', CYAN)
    s.rect(64, 641, 1472, 139)
    s.text(94, 662, 1400, 100, '参考 Extreme Parkour 的奖励与教师学生思路\n迁移到轮足和六桩地形，逐项检查奖励实际鼓励的行为。', 31)

    s = Slide('共面改善了，速度跟随却退化了', '03 / 对比实验', '来源：docs/training/VT-RECIPE.md §6.1；历史轮式 VT，同部署探针；与当前 vt_locked12 不同。', '1:50–2:35 · 45 秒')
    cols=[64,825,1155]
    for x,t in zip(cols,['指标 / 同一部署探针','VT v1','VT v2']): s.label(x+22,222,t,w=350)
    rows=[('四轮高度展布 · 正向纯转弯','3.10 cm','0.36 cm',CYAN),('90° 转弯足迹半径','0.572 m','0.520 m',CYAN),('平移速度 / 指令速度','0.781','0.631',ORANGE),('位置环原地转漂移','0.051 m','0.106 m',ORANGE)]
    for i,(label,a,b,c) in enumerate(rows):
        yy=281+i*92;s.rect(64,yy,1472,80)
        s.text(86,yy+17,700,52,label,30)
        s.text(847,yy+12,280,56,a,37,MUTED,True)
        s.text(1177,yy+12,320,56,b,37,c,True)
    s.text(64,696,1470,104,'取舍：长直线用 16 DoF 轮足；精对线用真 12 DoF 锁轮策略。\n不能只用一个几何指标判断整体性能。',32,CYAN,True)

    s = Slide('让深度信息直接服务动作', '04 / 论文借鉴与训练', '参考：Extreme Parkour, arXiv:2309.14341；实现：plum/student.py、depth.py；训练配置见 evidence/。', '2:35–3:30 · 55 秒')
    s.label(64,210,'早期成功分支 · 冻结教师',ORANGE,w=1300)
    s.rect(64,261,1472,87)
    s.text(88,281,1420,52,'深度 → 预测 1764 维局部扫描 → 冻结 RL 教师 → 动作',34,WHITE,True)
    s.text(64,370,1470,56,'局部扫描覆盖轮心 ±0.10 m；相机可见前方约 0.51–3 m。',30,MUTED)
    s.label(64,452,'最终部署 · 可继续学习的 CNN + MLP',CYAN,w=1300)
    s.rect(64,502,1472,100)
    s.text(88,520,1420,64,'48×64 深度 → 64 维特征  +  324 维本体历史 → 16 动作',34,CYAN,True)
    s.card(64,647,455,140,'1 / 行为克隆','1,228,800 帧教师数据')
    s.card(551,647,473,140,'2 / PPO 更新','1024 环境 × 32 步')
    s.card(1056,647,480,140,'3 / 判分选点','交付 model_1150')

    s = Slide('同一个模型，换初态就换了结论', '05 / 鲁棒性诊断', '数据：plum_depth_final_v1/manifest.json 的 15 格网格；不是随机采样成功率，不能与另一个25格汇总混用。', '3:30–4:15 · 45 秒')
    s.card(64,224,490,155,'训练环境记录','311 / 316 = 98.4%')
    s.card(64,405,490,155,'标准初态单关','存在 judge success / 200')
    s.rect(64,587,490,197)
    s.text(88,605,445,89,'2 / 15',65,ORANGE,True)
    s.text(88,700,445,60,'初态网格成功格数',29)
    gx,gy,cw,ch=773,337,139,113
    for j,t in enumerate(['−3°','−2°','0°','+2°','+3°']):s.text(gx+j*cw,gy-62,cw,46,t,28,MUTED,align='center')
    s.text(815,220,650,44,'航向偏移 →',25,CYAN,True)
    for i,t in enumerate(['−5 cm','0 cm','+5 cm']):
        s.text(580,gy+i*ch+28,173,55,t,28,MUTED,align='right')
        for j in range(5):
            ok=j==2 and i<2
            s.rect(gx+j*cw,gy+i*ch,cw-12,ch-13,'#194c45' if ok else '#342b38')
            s.text(gx+j*cw,gy+i*ch+22,cw-12,60,'通过' if ok else '失败',31,GREEN if ok else RED,True,'center')
    s.text(580,739,956,62,'横向偏移与航向组合；入口与出口仍敏感。',29,ORANGE,True)

    s = Slide('单关能力与组合交接的实跑证据', '06 / 视频 · 21.16 秒', '来源：历史 vtaudit 单关；旧25段v56的同次梅花桩片段。三段独立展示，不是一次连续通关。', '4:15–4:51 · 36 秒（含视频）')
    s.video(64,214,1080,607,'media/capability_reel.mp4','media/capability_reel.jpg')
    s.card(1173,216,363,221,'历史单关','40 cm 高台 / 高墙\nadaptive 基准 0.5\nsidecar 判分成功',ORANGE)
    s.card(1173,464,363,295,'六桩场地','深度专家穿桩\n→ 出口轮式专家\n→ 制动、站定、下台\n\n本段严格回放成功',CYAN)

    s = Slide('直立时对准，降姿后仍需复核', '07 / Part 1 · 低杆修复', 'v100 与 v104 是不同运行的前后对照；v104降姿后已在门限内。主动纠偏见v102/v103，非严格消融。', '4:51–5:36 · 45 秒（含11.04秒视频）')
    s.video(64,215,710,604,'media/lowbar.mp4','media/lowbar.jpg')
    s.label(821,231,'实际过杆时的航向误差',CYAN,w=680)
    s.text(821,295,710,100,'3.08°  →  0.69°',62,WHITE,True)
    s.text(821,400,710,48,'历史 v100                 验收 v104',27,MUTED)
    s.card(821,493,715,290,'修改后的动作顺序','直立进场 → 降低机身\n→ 低姿态对齐 → 零指令停稳复核\n→ 小于 1° 才前进',CYAN)

    s = Slide('交接要由实际状态触发', '08 / 路线与控制', '来源：part1_v104.json；README.md；docs/part2/ROUTE.md；锁轮目标固定不代表实际角速度瞬时为零。', '5:36–6:11 · 35 秒')
    s.card(64,224,710,222,'砂砾坑：四轮上台就切换','首次支撑条件：137.84 s\n切换意图：同帧；锁轮控制：下一帧',CYAN)
    s.card(806,224,730,222,'锁轮：捕获并保持实际轮角','固定 q_target；dq_target = 0\n有限力矩下允许实际短暂制动',CYAN)
    s.label(64,500,'Part 2：一套计划，逐档覆盖，便于复现',ORANGE,w=1400)
    s.rect(64,557,1472,219)
    s.text(92,578,1410,95,'恢复到某障碍前的对线点，再继续原路线\n超时可延长或关闭；精度、支撑和倒地检查仍保留',31)
    s.text(92,697,1400,48,'demo-part2 resume_before:=stairs_30cm timeout_scale:=0',26,CYAN,True)

    s = Slide('已交付的能力，与明确保留的缺口', '09 / 结果与下一步', '结论截至2026-11-08；独立单关、路线预检、严格支撑回放分别解释，未据视频宣称官方满分。', '6:11–6:45 · 34 秒')
    s.card(64,225,710,242,'Part 1 · 当前源码整场记录','263.80 s · 0 次恢复 · 返回起点\n路线预检全部通过\n关闭 live judge，未认证官方900分',CYAN)
    s.card(806,225,730,242,'Part 2 · 当前 33 段路线','尚无完整连续严格通过记录\n40 cm 平台部署仍跳过；45° 坡未解决\n梅花桩初态鲁棒性仍不足',ORANGE)
    s.rect(64,512,1472,272)
    s.label(92,538,'下一步：把交接分布纳入训练与验证',CYAN,w=1350)
    s.text(92,603,1400,148,'扩大入口位置、航向与速度随机化\n把出口下降纳入课程与终态奖励\n用预先定义的网格和真实整场轨迹复测',35,WHITE,True)

    s = Slide('规则、观测与控制边界', '备份 A / 问答', '依据：队内赛 PDF；README.md；plum_depth_final_v1/manifest.json；plum/depth.py。', backup=True)
    s.card(64,226,710,262,'部署网络可见','IMU、关节编码器、动作历史\n允许真实模拟的深度相机\n常规策略接收导航生成的速度指令')
    s.card(806,226,730,262,'训练期可用','教师局部扫描、critic特权地形\n只用于监督、回报与价值学习\n最终深度 actor 无 fine scan',ORANGE)
    s.card(64,530,1472,251,'Part 2 的学习策略组合','深度专家不读 command，不能用零摇杆刹停。出口改由另一学习策略完成。\n高层速度指令时序不等于固定关节动作；Part 2 不使用 Part 1 的固定姿态过障。')

    s = Slide('看过什么，迁移了什么', '备份 B / 论文与基座', '仅声明有源码/训练记录支撑的使用范围，不宣称原论文性能复现。', backup=True)
    s.card(64,218,1472,180,'PPO · Schulman 等，2017','Proximal Policy Optimization Algorithms · arXiv:1707.06347\n当前主力训练算法；非对称 actor / critic，历史观测与 CNN 输入。')
    s.card(64,410,1472,180,'HIM · Long 等，2023','Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response\narXiv:2312.11460 · 历史分支确实训练过；不把最终 depth、VT 统称 HIM。')
    s.card(64,602,1472,183,'Extreme Parkour · Cheng 等，2024','Extreme Parkour with Legged Robots · arXiv:2309.14341\n迁移奖励、课程和教师学生思路；机器人、仿真器、地形与最终训练方法不同。')

    s = Slide('梅花桩训练细节与失败选点', '备份 C / 复现实验', '来源：evidence/bc_depth_r4.json、plum_stage2_agent.yaml、plum_final_manifest.json；plum/depth.py。', backup=True)
    s.card(64,221,710,263,'行为克隆 → PPO','5 批数据，共 1,228,800 帧\nBC 验证动作 RMSE = 0.12514\n导入后将探索标准差重设为 0.10')
    s.card(806,221,730,263,'最终 PPO 运行','1024 环境 × 32 步，计划 1200 迭代\nlr = 3×10⁻⁵，clip = 0.2\nγ = 0.99，GAE λ = 0.95')
    s.card(64,530,1472,254,'用判分终态选模型，不只看训练曲线','交付 model_1150。1050 掉落，1100 倒地，1199 未下完出口。\ns0 / s1 / s2 是出生分布定义，不能据注册名称宣称完整串行训练过三个阶段。\n行为克隆误差是动作拟合指标，不是闭环越障成功率。',ORANGE)

    s = Slide('证据分层：哪些数字可以放在一起', '备份 D / 成绩口径', '来源：docs/part2/EXPERTS.md、ROUTE.md；evidence/part2_v56_eval.json、part1_v104.json。', backup=True)
    items=[('历史独立单关','平台40 / 墙40 有 success；多模型、多命令，不是当前整场成绩。'),('历史19/20覆盖','含旧版长桩几何；不能当当前六桩比赛地图的覆盖率。'),('旧25段路线 v56','集成门限25/25；严格支撑回放24/25，末墙稳定保持失败。'),('当前33段路线','v82/v83前14段严格成功；v84仅前4段后在25cm台卡住。'),('Part1 v104','完整实体运动与返回成立；live judge关闭，不能认证900分。')]
    for i,(label,body) in enumerate(items):
        y=218+i*114;s.rect(64,y,1472,102)
        s.text(86,y+21,340,57,label,30,CYAN,True)
        s.text(450,y+20,1058,67,body,28)

    s = Slide('低杆修复：诊断与证据的边界', '备份 E / 低姿态对齐', '数据：evidence/part1_v104.json；50Hz状态和接触采样，不能证明每个2ms物理步都无碰撞。', backup=True)
    s.card(64,221,710,256,'控制机制','小幅平滑转向会落入静摩擦死区\n改用短时差速脉冲 → 制动 → 复核\n停稳观察0.3秒；偏角<1°再前进')
    s.card(806,221,730,256,'主动纠偏与验收记录分别解释','v102 / v103：过杆误差1.28° / 1.12°\nv104：降姿后已对准，主要执行复核\n过杆误差0.69°，采样无杆柱接触',ORANGE)
    s.card(64,525,1472,258,'轮锁定是有限力矩的伺服目标','捕获切换时的非回绕轮角；q_target固定、dq_target=0，kp=80 / kd=3。\n力矩限幅时降低增益，不移动锁定目标；负载下实际轮角允许弹性误差。\nv104上台后记录的四轮目标角在停稳期间均保持不变。')


def rgb(value):
    return RGBColor.from_string(value.lstrip('#'))


def build_pptx(notes):
    prs = Presentation(); prs.slide_width=Inches(W/120); prs.slide_height=Inches(H/120)
    prs.core_properties.title='blackW：从单关越障到整场衔接'
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
                nodes.append(f'<img class="element" style="{style}object-fit:contain" src="{e["file"]}" alt="实跑画面">')
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
</style><div id="stage">PAGES</div><div id="toolbar"><button onclick="go(n-1)">← 上一页</button><span id="counter"></span><button onclick="go(n+1)">下一页 →</button><button onclick="playVideo()">播放 / 暂停 V</button><button onclick="fullscreen()">全屏 F</button><span>方向键翻页 · 前10页主讲 · 后5页备份</span></div><div id="hint"></div>
<script>
const meta=METADATA; let n=0;const slides=[...document.querySelectorAll('.slide')];
function resize(){const k=Math.min(innerWidth/1600,(innerHeight-32)/900);const st=document.getElementById('stage');st.style.transform=`scale(${k})`;st.style.left=((innerWidth-1600*k)/2)+'px';st.style.top=((innerHeight-32-900*k)/2)+'px'}
function go(i){if(i<0||i>=slides.length)return;slides[n].querySelectorAll('video').forEach(v=>v.pause());n=i;slides.forEach((s,j)=>s.classList.toggle('active',j===n));document.getElementById('counter').textContent=`${n+1}/${slides.length} · ${meta[n].time||'问答备份'}`;location.hash=String(n+1)}
function playVideo(){const v=slides[n].querySelector('video');if(v){if(v.paused)v.play();else v.pause()}}
function fullscreen(){if(!document.fullscreenElement)document.documentElement.requestFullscreen();else document.exitFullscreen()}
window.addEventListener('resize',resize);window.addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();go(n+1)}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();go(n-1)}if(e.key.toLowerCase()==='v')playVideo();if(e.key.toLowerCase()==='f')fullscreen();if(e.key==='Home')go(0)});resize();go(Math.max(0,Math.min(slides.length-1,(Number(location.hash.slice(1))||1)-1)));
// For PDF and previews: the static poster is rendered above each video.
for(const v of document.querySelectorAll('video')){const im=document.createElement('img');im.src=v.poster;im.className='element pdf-poster';im.style.cssText=v.style.cssText+';object-fit:contain;pointer-events:none;display:none';v.after(im)}
</script></html>'''.replace('PAGES',''.join(pages)).replace('METADATA',json.dumps(data,ensure_ascii=False))
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
    sheet=Image.new('RGB',(4*416,4*257),'#253448');dr=ImageDraw.Draw(sheet)
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
    report=dict(main_slides=10,backup_slides=5,time_budget_s=405,video_budget_s=32.20,accepted_code_commit='e2d7818',slides=[dict(number=i+1,title=s.title,time=s.time,backup=s.backup) for i,s in enumerate(SLIDES)])
    (HERE/'presentation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('Built PPTX, HTML'+(', PDF, previews' if not args.no_render else ''))


if __name__=='__main__':main()
