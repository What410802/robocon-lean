# eval/ — 判分与回放

队内赛 Part 1 / Part 2 的独立判分侧：只吃 trace/JSON + MuJoCo CPU，不 import ROS、
不 import 训练仓。**人读标准 = [`SPEC.md`](SPEC.md)；机器可读标准 = `robocon_eval/rules.py`。**

## 环境

使用仓库 pixi 的 `eval` 环境（MuJoCo 3.11 + CPU torch + pytest）：

```bash
pixi run -e eval eval-pytest          # = python -m pytest eval/tests -q
```

没有 pixi 时（例如本机已有 eval 解释器）：

```bash
cd /home/bis/robocon-lean
PYTHONPATH=$PWD/eval /home/bis/RoboCon/.pixi/envs/eval/bin/python -m pytest eval/tests -q
```

## 命令

```bash
# 1) 判分：录制好的 JSONL trace -> judge 报告（JSON 到 stdout）
pixi run -e eval robocon-eval replay-part1 --trace artifacts/traces/part1_run.jsonl
pixi run -e eval robocon-eval replay-part2 --trace artifacts/traces/trench_30cm.jsonl
#    --levels <path> 可覆盖 manifest；trace meta 里声明的关卡参数会与官方 manifest 校验

# 2) 只算分：给定每类最高成功难度
pixi run -e eval robocon-eval score --levels '{"high_platform":30,"stairs":20,"plum":1}'
#    或 --levels-file <v5 result JSON / {category: level} JSON>

# 3) 策略闭环（MuJoCo + TorchScript，写 trace 再 replay 判分）
pixi run -e eval robocon-eval run --robot blackW --part 2 \
    --policy flat --level high_platform_10cm \
    --out artifacts/traces/high_platform_10cm.jsonl --result artifacts/traces/result.json
```

`run` 的退出码：`0` 成功/Part1 完成，`3` Part2 未成功，`4` 非有限物理（NaN/发散），`2` 契约错误。
权重未入库：需把 `policy.pt`（或 `best.pt`）+ `config.yaml` 放到
`demo/rl_sar/policy/<robot>/<name>/`；缺资源时相关测试自动 skip。

## trace 格式

一行 meta + 每帧一行 JSON（判分跑 2 ms/帧，`eval_state_every:=1`；旧 20 ms 录制也可回放）。
字段：`t, base, yaw, tilt_deg|quat, foot_tags, ground_feet, bar_z, support, contacts`。
Part2 的 meta 需带 `level`（官方关卡条目）与可选 `support_frame`（`wheel_lowest` = 轮底几何点，
默认 `wheel_center` = 轮心，replay 会减去 0.103 m 半径）。详见 `SPEC.md §5` 与
`demo/tests/record_eval_trace.py`。

## 测试与夹具

- `tests/test_rules_scoring.py` — 曲线/floor/尝试聚合/合规常量
- `tests/test_trace_replay.py` — trace 解析、golden 回放、2 ms/20 ms 兼容、伪造关卡拒绝
- `tests/test_part1_judge.py` — Part1 判据子集（接近地面/桥/限高杆/高墙/斜坡/尝试账本）
- `tests/test_t_stairs.py` — T 台阶回归（异臂 100、同臂 50、漏级失败、冻结时机）
- `tests/test_part2_v5_judge.py` — 高台/沟壑/高墙终端态、车道、跌倒、采样兼容
- `tests/test_part2_v5_plum.py` — 新梅花桩几何（3 桩、无斜坡、出口平台远边）
- `tests/test_course_events.py` — 整场尝试/暂停恢复/跳过/回程奖励
- `tests/test_policy_runner.py` — runner 契约（关节映射、obs 维度、场景/出生点、闭环 smoke）

夹具：`fixtures/golden/`（历史 trace）、`fixtures/course/`（操作事件流）、
`fixtures/legacy/part2_official_levels.json`（生成 manifest 缺失时的 bootstrap 回退；
正常路径是 `demo/scenes/part2/part2_official_levels.json`）。

## 结构

```
robocon_eval/
  rules.py          曲线/次数/分值/合规常量（单一机器可读标准）
  scoring.py        floor 曲线查找、Part1 尝试/整场聚合
  trace.py          Part1 trace 读取 + 接触分类 + replay
  trace_part2_v5.py Part2 trace replay（关卡校验 + 分数）
  policy_runner.py  独立 MuJoCo+TorchScript runner（部署契约对齐）
  part1/            judge（V2.1 事件判据）+ course（整场协议）+ geom_tags
  part2/            v5_judge（终端态判据）+ official（manifest 解析/校验）
  cli.py            run | replay-part1 | replay-part2 | score
```

判分语义改动流程：改 `rules.py`/judge → 更新 `SPEC.md` → 更新/新增测试 → 全量 pytest →
升降 `JUDGE_VERSION`。**禁止改代码不改规范或改规范不跑测试。**
