# terrain/ — 唯一地形来源（robocon-lean）

本目录是 Part 1 / Part 2 地形几何的**唯一来源**：官方地图/场景资产、训练与判分共用的
Part 2 尺寸 spec、plum 几何、以及生成训练 spec / 判分场景 / runtime manifest 的生成器。
`train/`、`eval/`、`demo/` 都从这里读取或由这里的脚本生成，任何一方都不再保存地形副本。

```text
terrain/
├── official/
│   ├── map.xml                    # 官方 Part 2 ability field（terrain.zip 原样）
│   ├── scene_terrain.xml          # 官方 Part 1 场景（原样）
│   └── part2_terrain_spec.json    # Part 2 尺寸 spec（由 gen_part2_spec.py 生成，训练用）
├── plum/
│   └── plum_spec.json             # plum 3-pile 几何 spec（由 gen_plum.py 生成）
├── gen_part2_spec.py              # map.xml(+gen_plum) -> official/part2_terrain_spec.json
├── gen_plum.py                    # plum 唯一几何源 -> plum/plum_spec.json
└── gen_part2_levels.py            # map.xml(+gen_plum) -> demo/scenes/part2/{generated,manifest}
generated（提交入库，便于直接运行 demo；由生成器可复现重建，勿手改）：
├── demo/scenes/part2/generated/part2_off_<key>.xml
└── demo/scenes/part2/part2_official_levels.json
```

## 1. 官方资产与哈希（provenance）

| 文件 | 来源 | sha256 | bytes |
|---|---|---|---|
| `terrain/official/map.xml` | 主办方 `terrain.zip` -> `terrain/map.xml`（官方 Part 2 ability field） | `57531fe29fe4c2c0a33b65c12b71ba2e169d733d3f7518fc95eec291512f4b32` | 187570 |
| `terrain/official/scene_terrain.xml` | 主办方 Part 1 场景 `scene_terrain (1).xml` | `730ccebd2c400720ca50eb000afc7cd0791f7484957c25ca18b9eb3816a4a6d0` | 28070 |
| `demo/scenes/part1/official/scene_terrain.xml` | 上面 Part 1 场景的 demo 副本（同一 sha256） | `730ccebd...`（同上） | 28070 |

- `map.xml` 是 Part 2 障碍尺寸的官方基准：`high_platform` / `stairs` / `slope` /
  `high_wall` / `trench` 五类障碍的几何全部从它解析，spec 里记录来源 sha256，
  `gen_part2_spec.py` 重新生成后可直接 `git diff` 审查任何尺寸变化。
- **plum 例外**：`map.xml` 里的旧 plum 场（1.2 m 上/下坡道 + 两排 12 颗、行距 0.4 m 的
  交错方块）已被队内确认的新布局替代，`part2_terrain_spec.json` 的 `categories.plum`
  来自 `terrain/gen_plum.py`，**不是**从 map.xml 解析的（见 §4）。
- 规则依据：`/home/bis/BiS.d/tmp/队内赛.pdf`（Part 2 允许自定义地图，只要障碍类型与
  官方 ability field 一致；本仓的 plum 3-pile 布局是队内确认的定制实现）。
- 尺寸单位一律为米（m），角度为度（deg，训练侧转换 rad）。

## 2. 生成器命令（都从仓库根运行；依赖 numpy，验证脚本另需 mujoco）

```bash
# 1) Part 2 训练 spec：解析 official/map.xml 五类官方障碍 + 覆盖 plum 段
python terrain/gen_part2_spec.py
#    输出 terrain/official/part2_terrain_spec.json（含 map.xml sha256；源尺寸变化可 diff）

# 2) plum 几何 spec（单一来源：常量 -> spec + MJCF boxes）
python terrain/gen_plum.py
#    输出 terrain/plum/plum_spec.json

# 3) 生成判分/演示用 20 个 official 单关场景 + runtime manifest
python terrain/gen_part2_levels.py
#    输出 demo/scenes/part2/generated/part2_off_<key>.xml
#         demo/scenes/part2/part2_official_levels.json
```

生成器同时做自检：类别数量必须为
`high_platform 4 / stairs 4 / slope 4 / high_wall 4 / trench 3 / plum 1 = 20`，
每关 `seconds` 必须是安全上限（120 s，**不是**计分时限），plum 必须是无坡道布局。

场景格式（`demo/scenes/part2/generated/`）：

- 机器人：`<include file="../../../robots/blackW/blackW_description.xml"/>`
  （相对 scene 文件解析到 `demo/robots/blackW/`）。MuJoCo 会把被 include 模型的
  `meshdir="assets"` 相对 **主 scene 文件目录**解析，因此生成器会保证
  `demo/scenes/part2/generated/assets -> ../../../robots/blackW/assets` 符号链接存在；
- 障碍：从 map.xml 原样复制 geom（含 mesh 顶点内联），只做整体平移，使障碍近端位于
  spawn 前方 1.8 m；场景文件自身不含额外物理改动；
- manifest 保留旧 schema/键：`schema_version/source_map/levels[]`，每关
  `key/category/level/unit/scene/source/obstacle_bbox/spawn/spawn_raw/spawn_default_pose/command/seconds/judge`；
  `scene` 只是文件名（相对 `demo/scenes/part2/generated/`）。

## 3. 如何新增/调整一个障碍关卡

1. **只在 `gen_part2_spec.py` / `gen_plum.py` 里改尺寸常量**（不要手改生成的 JSON，
   也不要在 `train/`、`eval/` 里硬编码）。官方五类障碍从 map.xml 解析，正常不需要改；
   plum 改 `gen_plum.py` 顶部“OWNER-CONFIRMED constants”。
2. 运行 `python terrain/gen_part2_spec.py` 与 `python terrain/gen_plum.py`。
3. 在 `gen_part2_levels.py` 的 `_specs()` 里登记新等级（category / level / unit /
   从 map 选择的 geom 前缀 / kind），并在 `_judge()`（或 plum 的 `_plum_judge()`）
   写好 judge dict。**只登记官方等级**；不要加 `plum_reg_*` 等非官方条目。
4. 运行 `python terrain/gen_part2_levels.py`，确认打印的计数与 judge 表；
   场景/关卡是生成物，不手工编辑。
5. 训练侧 `train/src/alldog_mjlab/tasks/velocity/profiles/part2/terrain.py` 与
   `predicates.py` 直接读 spec，所以尺寸改动会自动进入训练；predicates 的 judge 镜像
   常量（平台 x 区间、plum cube_x_range 等）也要在同一提交里同步核对。
6. 用 MuJoCo 加载验证（需要 `demo/robots/blackW/` 存在）：

```bash
/home/bis/RoboCon/.pixi/envs/eval/bin/python -c \
  "import mujoco; mujoco.MjModel.from_xml_path('demo/scenes/part2/generated/part2_off_plum_1.xml')"
```

## 4. Part 2 plum 3-pile 规则依据（队内确认，2026-11-02）

旧 `map.xml` 的 plum 是带 1.2 m 上下坡道、两排交错 12 颗方块的场地。队内赛改为
**无坡道 3-pile** 布局（owner 确认），judge 与训练统一使用该几何：

```text
入口平台  : 0.8 x 0.8 x 0.2 m，top z = 0.2 m；x ∈ [1.8, 2.6]，y ∈ [-0.4, 0.4]
3 颗方块  : 0.2 x 0.2 x 0.2 m，沿 +x 单排；中心 x = 2.8, 3.4, 4.0（中心间距 0.6 m），y = 0
出口平台  : 0.8 x 0.8 x 0.2 m，top z = 0.2 m；x ∈ [4.2, 5.0]，y ∈ [-0.4, 0.4]
间隙      : 入口平台远端(2.6) 与第一颗方块(2.7) 之间 0.1 m；
            最后一颗方块远端(4.1) 与出口平台近端(4.2) 之间 0.1 m
无坡道    : 机器人从地面直接上 0.2 m 入口平台，从出口平台 x=5.0 处下 0.2 m 回到地面
```

对应 judge dict（`gen_part2_levels.py::_plum_judge`）：

```json
{"kind": "plum", "top_z": 0.2, "cube_x_range": [2.55, 4.25], "goal_x_min": 4.2,
 "z_margin": 0.35, "z_cross_min": 0.05, "lane_half_width": 0.7, "hold_s": 0.2,
 "exit_ramp_end": 5.0, "exit_ground_tol": 0.08}
```

语义：base 在 `cube_x_range` 内时任何轮子低于 `z_cross_min` 即判失败（掉进间隙）；
踩上出口平台是证据而非成功；必须从出口平台远边（x=5.0，judge 里沿用旧键名
`exit_ramp_end`）下到地面、四轮在 `exit_ground_tol` 内稳定 `hold_s` 才算通过。
`terrain/gen_plum.py` 是唯一几何源，输出 `plum/plum_spec.json`（常量 + 派生 boxes）并
提供 `boxes()` / `mjcf_geoms()`（名字：`plum_entry`、`plum_r1_01..03`、`plum_exit`）
供关卡生成器与训练地形复用。

## 5. Part 1

Part 1 场景是主办方给的单一 `scene_terrain.xml`（`terrain/official/scene_terrain.xml`），
demo 直接使用 `demo/scenes/part1/official/scene_terrain.xml`，本目录不生成 Part 1 关卡。
