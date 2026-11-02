# robots — 机器人唯一资产库

每个机器人一个目录，场景文件通过相对路径 include 这里的模型，**全仓不再复制第二份 mesh**。

```text
robots/
├── blackW/                 16 DoF 轮腿（Part1/Part2 主用）
│   ├── blackW_description.xml   官方描述 MJCF（仅 BL-6 给 4 个轮碰撞 geom 加了 name，物理零改动）
│   ├── robot.yaml               契约：关节顺序/映射、kp/kd、默认姿态、力矩上限、观测/动作、出生点
│   ├── scene_part1.xml          Part1 运行时场景（官方地形 + 命名 geom；include blackW_description.xml）
│   ├── smoke_flat.xml           纯平地冒烟场景
│   └── assets/                  18 个 STL（唯一一份；scenes/ 用软链接引用）
└── black/                  12 DoF 足式（历史对照）
    ├── black_policy.xml         描述 MJCF + 执行器/传感器（contract 用这个）
    ├── black.xml                同物理模型、无 actuator/sensor（上游参考模型）
    ├── robot.yaml
    └── scene_part1.xml
```

## robot.yaml 关键字段

`num_dofs`、`joint_names`、`joint_mapping`（策略关节序 ↔ 仿真执行器序）、`default_dof_pos`、
`torque_limits`、`command_envelope`（MIT 指令安全限幅）、`action_contract`（含轮速符号）、
`obs_contract`（57×6 / 45×6）、`spawn.part1`、`spawn.part2_source`、`scene_roots`。
`demo/tests/check_robot_manifests.py` 与 `demo/tests/check_spawns.py` 会 fail-loud 校验这些字段
和路径；策略侧由 `check_policy_contracts.py` 对照校验。

## BL-6（判分接触命名）

官方 blackW 的四个轮碰撞 geom 原本无名，判分器无法区分四个轮子。本仓只加了 name：
`FL_wheel_collision` / `FR_wheel_collision` / `RL_wheel_collision` / `RR_wheel_collision`
（属性仅 `name`，物理零改动；证明见 `../UPSTREAM.md` §3）。改 `blackW_description.xml` 时
**不要再动其它属性/几何**，否则与官方模型的物理一致性会被破坏。
