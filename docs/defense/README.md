# 队内赛答辩材料 · 修订版

主题：**blackW：从论文思路到轮足越障策略**。按《队内赛.pdf》第4页要求，讲述论文阅读/复现、训练机制、奖励调整与对比实验。系统已通过项目验收；本次只改答辩材料，**暂不提交**。

- [script.md](script.md)：中文主讲稿，附技术问答。
- [blackW-defense.pptx](blackW-defense.pptx)：可编辑图文，内嵌视频，主讲页含讲稿备注。
- [blackW-defense.pdf](blackW-defense.pdf)：静态版，视频显示海报。
- [index.html](index.html)：本地字体、视频齐备的离线展示版。
- [overview.jpg](overview.jpg)：16页总览。
- [sources.md](sources.md)：论文、训练、实验和视频来源。
- [figures/manifest.json](figures/manifest.json)：3幅论文原图的出处与SHA。
- [evidence/training_recipe.json](evidence/training_recipe.json)：训练参数索引，随包保存关键运行配置。
- [validation.json](validation.json)：构建、排版和媒体验证。

主讲 **11页，预算6分50秒**，含三处视频 **44.42秒**，留10秒余量；后5页为问答备份。视频播放时停讲，表格参数供查看，不逐项朗读。时间为排练预算。

## 展示架构与时间

按构建顺序组织：**论文依据 → 平地运动 → 通用越障 → 系统与衔接 → 梅花桩 → 演示与收获**。平地两页、通用专家与高台两页，系统和路线衔接合并为一页；梅花桩一页，以良好步态视频为主体，配简短训练说明。

| 页 | 内容 | 预算 |
|---|---|---:|
| 1 | 开场与备赛主线 | 15s |
| 2 | HIM / DreamWaQ：历史观测与隐状态 | 35s |
| 3 | Extreme Parkour / RecoveryLoco：训练机制启发 | 20s |
| 4 | 平地16DoF结构、低速指令采样与奖励 | 50s |
| 5 | 平地对比实验、12DoF锁轮与模型选择 | 45s |
| 6 | 通用越障：任务奖励、终止与专家分工 | 50s |
| 7 | 40cm高台视频与31级地形训练配方 | 40s，含6.94s视频 |
| 8 | **整体系统与路线衔接合并** | **30s** |
| 9 | 梅花桩早期良好步态与简要配方 | 55s，含18s视频 |
| 10 | Part 1 / Part 2 演示交付 | 50s，含19.48s视频 |
| 11 | 三点收获与后续方向 | 20s |

备份12–16页：平地训练细节；高台完整配方；两代梅花桩架构；六桩课程、奖励和训练规模；HIM、观测信息与演示复现。

主讲用自然语言讲训练思考，来源、版本和验证细节集中放在页脚、问答与来源文档中。

## 现场展示

浏览器打开 `index.html`，方向键翻页，`F`全屏，`V`播放/暂停当前页视频；不自动播放。换机时复制整个 `docs/defense/`，保留 `media/`、`figures/`、`fonts/`。

PPTX三处视频均内嵌，在第7、9、10页点击播放。PPTX字体为Noto Sans CJK SC，换机建议安装同字体；PDF已嵌入字体，离线HTML随包提供字体。HTML实际播放与文本边界、PDF和PPTX包结构均有检查，独立应用渲染记录见 `evidence/pptx_render_check.json`。Office/WPS视频播放未实测。

第9页是**早期长桩场地的route2_v3**，配方对应“深度预测扫描→冻结RL教师＋DAgger”；后续六桩直接深度模型放在备份。第10页为Part1限高杆与Part2六桩片段顺序展示。视频已复制到本目录，不依赖原 `artifacts/`；完整来源见 [media_manifest.json](media_manifest.json)。

## 修改和重建

在仓库根目录使用独立文稿环境：

```bash
uv venv /tmp/blackw-defense-venv
uv pip install --python /tmp/blackw-defense-venv/bin/python -r docs/defense/requirements.txt
/tmp/blackw-defense-venv/bin/python -m playwright install chromium
/tmp/blackw-defense-venv/bin/python docs/defense/subset_fonts.py
/tmp/blackw-defense-venv/bin/python docs/defense/build.py
/tmp/blackw-defense-venv/bin/python docs/defense/validate.py
```

`build.py::make_slides()`定义布局，`script.md`写入主讲页PPTX备注。PPTX、HTML、PDF共享布局；模型图、表格和采样图在PPTX中可编辑，论文图有署名。新增汉字后更新字体子集；需Linux安装的Noto CJK，许可见 `fonts/LICENSE.txt`。

图源已保存，无需下载论文。`prepare_media.py`仅在重新导入本地历史归档时使用；`make_delivery_reel.py`可从随包短片重建当前演示合集，`make_reel.py`保留旧合集生成入口。`evidence/`的历史绝对路径用于追溯，构建无需访问这些路径，也不运行训练或仿真。
