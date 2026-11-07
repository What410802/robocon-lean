# 轨迹渲染视频

`replay_route_video.py` 读取自动路线录下的机器人状态和计划，用 `mj_forward` 恢复姿态后渲染。
不重跑物理、不加载策略，也不会修正录下的动作。是否完成路线应检查同一次运行的日志和支撑点回放。

现场运行时同时保存两个文件：

```bash
PART1_RECORD_PATH=$PWD/artifacts/part2_route/states.jsonl \
PART1_RECORD_PLAN_PATH=$PWD/artifacts/part2_route/plan.jsonl \
pixi run -e demo demo-part2
```

保留默认 `realtime:=true`。一次只运行一场，现场仿真结束后再启动 GPU 视频渲染。
运行成功时终端显示当前计划的 33 段完成，快捷入口退出状态为 0；这表示路线集成门限完成。
需要支撑点验证时另用 `tools/route/replay_field_part2.py --require-all`，命令见
[Part2 路线说明](../../docs/part2/ROUTE.md)。

同一次运行若完成全部 33 段，可直接生成简版视频：

```bash
MUJOCO_GL=egl pixi run -e eval python tools/video/replay_route_video.py \
    artifacts/part2_route/states.jsonl artifacts/part2_route/part2_cli.mp4 \
    artifacts/part2_route/plan.jsonl \
    --scene demo/scenes/part2/full_field/part2_fixed_field.xml \
    --cli-only --preview artifacts/part2_route/part2_cli_preview.png
```

简版画面为 640×544、25 fps，保持 Part1 最终视频的场景尺寸、跟随相机和黑色顶栏，
画面外仅显示 `> CLI: ...`。没有标题、计时、遥测或俯视图。
按仿真时间重采样，2 ms 与 20 ms 录制频率均按 1× 播放；读取轨迹时逐行处理。
`--fps` 可改输出帧率。省略 `--cli-only` 可生成原有调试版；`--operator-only` 保留旧版标题格式。

先检查任意片段的样式（时间是 trace 中的绝对仿真秒数）：

```bash
MUJOCO_GL=egl pixi run -e eval python tools/video/replay_route_video.py \
    artifacts/route_fix/part2_plum_v25_states.jsonl artifacts/route_fix/part2_cli_style_sample.mp4 \
    artifacts/route_fix/part2_plum_v25_plan.jsonl \
    --scene demo/scenes/part2/full_field/part2_fixed_field.xml \
    --cli-only --start-time 104 --end-time 108 \
    --preview artifacts/route_fix/part2_cli_style_sample.png
```

上述现有样本渲染为 101 帧、4.04 s，仅用于确认视频格式；该次梅花桩记录并非正式判据成功证据。
整场视频使用完整原始 trace，不拼接各次运行的成功片段。

现有 [v56 全程简版视频](../../artifacts/route_fix/part2_full_live_v56_cli.mp4) 为旧 25 段计划的
开窗口 1× 记录，640×544、25 fps、30510 帧、20 min 20.4 s。原始 61020 帧状态按时间重采样，
全程来自同一次运行。集成门限 25/25、退出状态 0，
[支撑点回放 24/25](../../artifacts/route_fix/part2_full_live_v56_eval.json)：梅花桩成功，最后
20 cm 墙 `not_stable_after_crossing`。它不代表新版 33 段完成，也不是官方独立关卡全通过录像。
