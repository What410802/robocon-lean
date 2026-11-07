# tools/

| 文件 | 作用 |
|---|---|
| `build.sh` | colcon 构建 `demo/rl_sar` + `demo/quadruped_ros2`，产物在 `demo/build`、`demo/install` |
| `live_demo.sh` | Part1/Part2 现场演示启动器；只调用统一 launch，不定义路线。入口：`pixi run -e demo demo-part1` / `demo-part2` |
| `activate.sh` | source `demo/install/setup.bash`（非 demo env 时给 warning） |
| `demo_tests.sh` | 机器/策略契约/出生点自检 + Part1/Part2 场景 MuJoCo 加载冒烟 |
| `record_video.py` | 由 trace 重放同一策略并渲染带判分标注的复盘视频（需要 eval env + 场景） |
| `check_env.py` | 打印当前 pixi 环境解析到的关键版本 |
| `mpc/` | Part1 固定路线 MPC 的设计与代码（规划中，见 `mpc/README.md`） |
| `tracking/` | 速度跟随 / 转弯足迹 / 位置环 / 鲁棒性探针（见 `tracking/README.md`） |
| `video/` | [轨迹视频](video/README.md)：`replay_route_video.py --cli-only` 生成仅含 `> CLI:` 标注的简版 MP4，也支持调试叠加 |
| `route/audit_part1_bridge_contacts.py` | 从实测 trace 重建木桥 B 碰撞接触点，输出两侧窄带的内缘距离、接触法向、CSV 与分布图；不重新积分物理 |


第三方二进制资产清单在 `assets/`（`python assets/verify.py` 校验哈希）。

## 常用命令

```bash
pixi run -e demo demo-build
pixi run -e eval demo-tests
pixi run -e eval python tools/record_video.py --help
```

固定路线录像：

```bash
pixi run -e eval python tools/video/replay_route_video.py \
    artifacts/route/states.jsonl artifacts/route/route.mp4 artifacts/route/plan.jsonl
```
