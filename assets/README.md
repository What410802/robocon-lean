# assets/

第三方二进制资产（机器人 STL、官方地图 XML）随仓库入库，清单与哈希在 `manifest.json`：

```bash
python assets/verify.py
```

- `demo/robots/blackW/assets/*.STL`：主办方 blackW 模型网格（官方 MJCF 引用）；
- `demo/robots/black/assets/*.STL`：black 模型网格；
- `terrain/official/map.xml`、`terrain/official/scene_terrain.xml`：主办方场景（哈希见下）。

约束：**不修改**这些文件的几何/材质数据；机器人 MJCF 只允许补 `name` 属性（判分接触分类用），
且必须在 `demo/UPSTREAM.md` 中注明"官方模型物理零改动"。

官方场景哈希：

| 文件 | sha256 |
|---|---|
| `terrain/official/scene_terrain.xml`（Part1） | `730ccebd2c400720ca50eb000afc7cd0791f7484957c25ca18b9eb3816a4a6d0` |
| `terrain/official/map.xml`（Part2） | `57531fe29fe4c2c0a33b65c12b71ba2e169d733d3f7518fc95eec291512f4b32` |

权重（`policy.pt` 等）与视频不入库，见 `.gitignore`；恢复/重训后放到 `demo/rl_sar/policy/`。
