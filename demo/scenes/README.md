# scenes — 场景目录（不放机器人资产）

机器人模型、mesh、`robot.yaml` 全部在 `../robots/`；本目录只放 MuJoCo 场景 XML 与官方贴图。

```text
scenes/
├── part1/official/scene_terrain.xml   官方 Part1 地形原样（参考件，sha256 730ccebd…）
├── part2/
│   ├── official/map.xml               官方 Part2 整场地图（地形源 terrain/official/map.xml）
│   ├── part2_official_levels.json     20 个单关的运行时清单（terrain/ 生成，勿手改）
│   ├── generated/                     20 个单关场景 XML（terrain/ 生成，勿手改；判分用）
│   ├── full_field/                    官方整场查看场景 + 85 张难度标签纹理（人工驾驶，不判分）
│   ├── smoke_flat.xml / smoke_flat2.xml   无头冒烟用平地场景
│   └── assets -> ../../robots/blackW/assets        (软链接，见下)
```

## 运行时用哪个场景

| 用途 | 场景 |
|---|---|
| Part1 运行时 | `../robots/blackW/scene_part1.xml`（由官方 Part1 地形规范化 + geom 命名，`robot.yaml: scene_roots.part1`） |
| Part2 单关（判分/演示） | `part2/generated/part2_off_<level>.xml`（`robot.yaml: scene_roots.part2`，关卡清单 `part2_official_levels.json`） |
| Part2 官方整场人工查看 | `part2/full_field/part2_full_field.xml`（`full_field:=true`，**不用于判分**） |
| 无头策略冒烟 | `../robots/blackW/smoke_flat.xml`（纯平地） |

## 为什么有 `assets` 软链接

`blackW_description.xml` 里有 `<compiler angle="radian" meshdir="assets"/>`。MuJoCo 解析
`meshdir` 时以**主场景文件所在目录**为基准，因此任何“从别的目录 include 机器人描述”的场景
（`generated/`、`full_field/`、`part2/*.xml`）都需要在场景目录里放一个 `assets` 软链接指向
`../../(../)robots/blackW/assets`。软链接不是资产副本，`git` 只存链接本身；删除它会让
`MuJoCo 场景加载失败`。

`generated/` 另外保留了 `blackW_description.xml` 软链接（生成器兼容用）；场景 XML 自身用
`<include file="../../../robots/blackW/blackW_description.xml"/>`，两条路径指向同一份描述。

## 来源与修改

- `part1/official/scene_terrain.xml`：主办方 `scene_terrain (1).xml` 原样（sha256
  `730ccebd2c400720ca50eb000afc7cd0791f7484957c25ca18b9eb3816a4a6d0`），仅作参考；它 include 的
  `blackW_description.xml` 与 hfield 路径是旧目录布局，直接加载会失败，运行时请用
  `robots/blackW/scene_part1.xml`。
- `part2/official/map.xml`：主办方 `terrain.zip` 的 `map.xml` 原样（含内嵌 mesh 几何）。
- `generated/`、`part2_official_levels.json`：`terrain/` 生成器产出，`seconds` 只是 120 s 安全超时
  （规则不计时，没有计分时间上限）。
- `full_field/labels/`：主办方 85 张难度标签纹理，只服务整场查看。
