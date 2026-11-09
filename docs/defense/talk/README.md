# 十五分钟组内分享 · HTML 展示稿

面向同组队员，讲清楚**读论文的方法**与**实现过程中的取舍**。不复述成绩，主线是四个问题：

1. 没有相机时，脚下是什么接触？（DreamWaQ / HIM → 平地策略、锁轮专家）
2. 跳沟爬台的动作怎么学出来？（Extreme Parkour → 通用越障、40 cm 高台）
3. 单关专家怎么串成整场？（路线外环 + 热切换 + 交接动量）
4. 看路落脚时，视觉以什么形式进控制？（梅花桩两代 → START / PIE / PUMA）

## 文件

| 文件 | 内容 |
|---|---|
| [deck.html](deck.html) | 15 页主讲 + 4 页备查。自包含，只依赖同目录 `fonts/` 和仓库内 `figures/`、`media/` |
| [script.md](script.md) | 逐页主讲稿 + 四页问答备查，含逐页字数预算 |
| [build.py](build.py) | 重新子集字体（从 deck.html 与 script.md 提取字符）并渲染 `previews/` |
| [previews/](previews/) | 19 张 1600×900 渲染图，用于快速检查排版 |

## 现场

浏览器打开 `deck.html`。

- `→` / `空格` 下一页，`←` 上一页，`Home` 回封面
- `V` 播放 / 暂停当前页视频（不会自动播放）
- `F` 全屏

三段视频合计 44.4 秒：40 cm 高台 6.94 s、梅花桩第一代 18 s、交付片段 19.48 s。播放时停讲。

## 时间

讲稿正文 3,600 字，按 4.5 字/秒约 **13 分 20 秒**，加三段视频 44.4 秒共 **14 分 05 秒**，余约 55 秒机动。上限 15 分钟。

| 页 | 内容 | 预算 |
|---|---|---:|
| 01 | 封面 | 32 s |
| 02 | 四个问题 | 51 s |
| 03 | 三种“好上下文” | 38 s |
| 04 | DreamWaQ vs HIM | 63 s |
| 05 | 平地：指令分布即课程 | 94 s |
| 06 | 平地对比实验与锁轮 | 61 s |
| 07 | Extreme Parkour 与 DAgger | 95 s |
| 08 | 三层奖励与 31 级地形 | 53 s |
| 09 | 40 cm 实测 | 10 s（含 6.94 s 视频） |
| 10 | 系统串联与交接 | 55 s |
| 11 | 梅花桩第一代 | 91 s（含 18 s 视频） |
| 12 | 梅花桩第二代 | 69 s |
| 13 | START / PIE / PUMA | 48 s |
| 14 | 交付片段 | 28 s（含 19.48 s 视频） |
| 15 | 三点收获 | 57 s |

备查 A–D：29 篇怎么读、平地与锁轮参数、越障与六桩口径、移植范围与数值契约。

## 设计意图

- **页面只放证据，讲述放讲稿**。论文页只保留关键架构图 + 三条标注（缺什么 / 用什么信号 / 得到什么能力），实验页用大数字承担判断。
- **每页一个判断**。标题是结论句，不是栏目名；底部结论条是这一页唯一要记住的话。
- **图片全部原图口径**。论文图取自原文 PDF，署名与图号在页脚；本仓结构图用文字卡片绘制，不冒充论文原图。

## 重建

字体子集需要 `fontTools` + `brotli`，渲染需要 Playwright：

```bash
/tmp/blackw-defense-venv/bin/python docs/defense/talk/build.py
```

`build.py` 会先用 `deck.html` + `script.md` 的字符重新生成 `fonts/*.woff2`，再逐页截图到 `previews/`，同时检查文字横向溢出与 JS 报错。改了标题或新增汉字后必须重跑，否则字形会缺。

## 与旧材料的关系

`docs/defense/index.html` / `build.py` / `blackW-defense.pptx` 是**七分钟**版本，保留不动。本目录是十五分钟版，两套共用 `figures/`、`media/`、`evidence/` 与 `research/`。

口径与出处见 [../sources.md](../sources.md)、[../research/literature-review.md](../research/literature-review.md)、[../evidence/training_recipe.json](../evidence/training_recipe.json)。
