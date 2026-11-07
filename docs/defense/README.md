# 队内赛答辩材料

主题：**blackW：从单关越障到整场衔接**。依据《队内赛.pdf》第4页：七分钟内说明备赛过程，包括奖励调整、论文借鉴、训练机制和对比实验。比赛运行代码已先提交为 `e2d7818`；本目录只增加答辩材料，不更改策略或路线。

- [blackW-defense.pptx](blackW-defense.pptx)：可编辑文本、图形、表格，包含两段内嵌视频和逐页讲稿备注。
- [blackW-defense.pdf](blackW-defense.pdf)：静态提交/打印版，视频页显示海报画面。
- [index.html](index.html)：浏览器离线展示版，无 CDN；中文字体和视频随目录提供。
- [script.md](script.md)：对应10页主讲的中文讲稿及问答提示。
- [overview.jpg](overview.jpg)：15页整体预览。
- [sources.md](sources.md)：事实、实验、媒体的来源和引用边界。
- [media_manifest.json](media_manifest.json)：媒体原路径、SHA-256、截取区间、时长。
- [validation.json](validation.json)：构建和媒体验证结果。

主讲预算 **6分45秒**，包含两次视频 **32.20秒**，保留15秒余量。前10页主讲，后5页仅用于问答，不自动继续讲。预算不是口述计时实测；请按自己的语速排练一次，视频播放时停讲。

## 现场展示

直接用浏览器打开 `index.html` 即可。方向键翻页，`F`全屏，`V`播放/暂停当前页视频。视频不会自动播放；按画面上的播放键也可以。复制到其他电脑时保留整个 `docs/defense/` 目录，HTML、`media/` 和 `fonts/` 保持相对位置。

也可使用 PPTX（视频内嵌，无需连接源仓库），在第7/8页点击视频播放。PPTX字体为 Noto Sans CJK SC，换机如未安装会回退；以PDF或包含本地字体的HTML为版式基准。已验证PPTX包结构、可编辑对象、内嵌视频与备注，以及HTML实际播放和PDF渲染。另用临时解包的LibreOffice Impress实际渲染PPTX，15页总览及第6/8页全尺寸检查均正常；未在Office/WPS中实测视频播放。

第7页展示的是**三段独立证据**顺序播放，不是一次连续通过三个障碍。前两段为历史高台40cm、墙40cm独立单关；第三段来自旧25段v56中的六桩连续交接。第8页是当前源码Part1 v104全程中的低杆片段。原始短片分别保留在 `media/`，需要时可独立打开。

完整文件都已放在本目录；展示不依赖 `/home/bis/RoboCon/artifacts/` 或本仓本地大trace。`evidence/` 是精选的原始摘要副本，其中保存的绝对路径和旧判据版本属于历史证据，不能当作今天可重跑的入口。

## 修改和重建

在仓库根目录运行，使用独立文稿环境，不改机器人环境：

```bash
uv venv /tmp/blackw-defense-venv
uv pip install --python /tmp/blackw-defense-venv/bin/python -r docs/defense/requirements.txt
/tmp/blackw-defense-venv/bin/python -m playwright install chromium
/tmp/blackw-defense-venv/bin/python docs/defense/build.py
/tmp/blackw-defense-venv/bin/python docs/defense/validate.py
```

修改画面：编辑 `build.py` 的 `make_slides()`；修改讲稿：编辑 `script.md`，重建后写入PPTX备注。HTML/PDF/PPTX共享同一布局定义。加入新汉字后，可在安装Noto CJK字体的Linux机器上先运行 `subset_fonts.py`，更新随包字体子集；许可见 `fonts/LICENSE.txt`。

`prepare_media.py` 仅用于从本机历史归档重新导入视频，路径见脚本；日常展示与重建不需要执行。`make_reel.py` 使用已随包保存的三个源短片重建展示合集（不加速）。PDF生成不执行模型或物理仿真。

主讲计划如下，讲稿中有对应段落：

| 页 | 内容 | 预算 |
|---|---|---:|
| 1 | 目标与主线 | 25s |
| 2 | 规则、导航与策略拆分 | 35s |
| 3 | 奖励漏洞和诊断 | 50s |
| 4 | VT共面改进的代价 | 45s |
| 5 | 视觉路线与训练方法 | 55s |
| 6 | 初态网格鲁棒性 | 45s |
| 7 | 能力/交接视频 | 36s（含21.16s） |
| 8 | 低杆对齐与视频 | 45s（含11.04s） |
| 9 | 接触触发交接与统一路线 | 35s |
| 10 | 交付与未解问题 | 34s |

备份页覆盖观测合法性、论文来源、训练参数、判分口径、低杆控制机制。答辩重点是解释技术选择和证据，不宣称Part1官方900分或当前Part2整场成功。
