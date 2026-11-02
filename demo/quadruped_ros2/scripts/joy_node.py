#!/usr/bin/env python3
"""手柄节点（Python，任务第 3 项）：读游戏手柄，发 sensor_msgs/Joy，并订阅控制器的回程状态。

手柄从 **/dev/input/event*** 直接读内核事件（python-evdev），不经 SDL/pygame。
认 xpad（Xbox）布局，也认轴码位不同的第三方 HID 手柄（按设备自己的 AbsInfo 分配槽位）。

输出对齐主办者仓库 `quadruped_control/configs/input/gamepads.yaml` 的 `xbox` profile 的**规范化输出**
（8 个轴 / 12 个按钮，长度、顺序、**正负号**都一致；接口全文见 docs/joystick.md §3）：

    axes[0..7]     = 左摇杆 X、左摇杆 Y、LT、右摇杆 X、右摇杆 Y、RT、DPad X、DPad Y
    buttons[0..11] = A、B、X、Y、LB、RB、Back、Start、Guide、LS、RS、（第 12 个恒 0）

    正负号：摇杆 **上 / 左 = +1**；十字键 **上 / 右 = +1**；扳机松开 = −1、按到底 = +1。
    内核（evdev）是"上 / 左为负、十字键上为负"，所以摇杆四个轴与十字键竖轴在这里取反一次，
    下游（控制器、rl_sar、主办者的 gateway）就都不用再管设备的正负号。

按键含义由**控制器节点**决定（A 站立 / B 阻尼 / X 复位），手柄节点只管把原始输入发出去——
这样"手柄挂了"和"控制逻辑"是两件互不影响的事。

回程：订阅控制器的 quadruped_ros2/msg/ControlStatus（模式 / 斜坡进度 / 倾角 / 指令条数）并打印。
任务书要求"手柄节点需要和控制器节点互换消息"，这就是返程那一条。

用法（在仓库根目录）：

    ros2 run quadruped_ros2 joy_node
    ros2 run quadruped_ros2 joy_node --ros-args -p device:=/dev/input/event7
    ros2 run quadruped_ros2 joy_node --ros-args -p name:=        # 任何手柄都收

读 /dev/input/* 需要权限：把当前用户加进 `input` 组后重新登录（或临时 `sudo chmod a+r /dev/input/event*`）。
"""

from __future__ import annotations

import glob
import select
import sys
from pathlib import Path

import rclpy
from evdev import InputDevice, ecodes
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Joy

from quadruped_ros2.msg import ControlStatus

# 输出长度固定成 8 轴 / 12 按钮（与主办者仓库的 gamepads.yaml 一致，方便对照）
AXIS_COUNT = 8
BUTTON_COUNT = 12

# **Joy 的轴槽位**（xbox 布局，固定不动）：左摇杆 0/1、LT 2、右摇杆 3/4、RT 5、DPad 6/7。
# 每个轴具体落在哪个槽位、怎么归一化，由**设备自己的 AbsInfo**在连接时判定（见 build_axis_roles）：
# 内核给什么量程就用什么量程——xpad 是 ±32767 的摇杆 + 0..255 的扳机，而很多第三方 "HID gamepad"
# 是 0..255、静止在中点 128 的摇杆，把量程写死就会把摇杆读数压成 0（实测踩过，见 docs/joystick.md §7）。
STICK_SLOTS = (0, 1, 3, 4)
TRIGGER_SLOTS = (2, 5)
HAT_SLOTS = {ecodes.ABS_HAT0X: 6, ecodes.ABS_HAT0Y: 7}
# **输出的正负号**：每个槽位乘的系数（evdev → 规范化 /joy）。evdev 的约定是摇杆"上 / 左为负"、
# 十字键"上为负、右为正"（内核 drivers/input/joystick/xpad.c 与 drivers/hid/hid-input.c 都这么报）；
# 规范化输出是"上 / 左为正、十字键上 / 右为正"，与主办者 gamepads.yaml 的 xbox profile（摇杆 scale −1、
# 十字键经 pygame 的 get_hat）一致。扳机（2/5）与十字键横轴（6）不变号。
SLOT_SIGN = (-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0)
# sysfs/预筛用的"像手柄的轴"与"像手柄的键"：两者**同时**满足才算候选。
# 只看轴不够——触摸板也报 ABS_X/ABS_Y，会被当成候选、再因为没权限报成"手柄没权限"（实测踩过）；
# 触摸板只有 BTN_LEFT/BTN_TOOL_*，没有 BTN_A 这类游戏键。
GAMEPAD_AXIS_CODES = tuple(
    getattr(ecodes, n)
    for n in ("ABS_X", "ABS_Y", "ABS_Z", "ABS_RX", "ABS_RY", "ABS_RZ", "ABS_GAS", "ABS_BRAKE",
              "ABS_HAT0X", "ABS_HAT0Y")
)
GAMEPAD_BUTTON_CODES = tuple(
    getattr(ecodes, n)
    for n in ("BTN_A", "BTN_B", "BTN_X", "BTN_Y", "BTN_TL", "BTN_TR", "BTN_SELECT", "BTN_START",
              "BTN_MODE", "BTN_THUMBL", "BTN_THUMBR")
)
# 按钮顺序 = xpad 上报顺序；第 12 个（索引 11）没有对应键，恒 0，留给"长度固定"用
BUTTON_CODES = (
    "BTN_A",
    "BTN_B",
    "BTN_X",
    "BTN_Y",
    "BTN_TL",
    "BTN_TR",
    "BTN_SELECT",
    "BTN_START",
    "BTN_MODE",
    "BTN_THUMBL",
    "BTN_THUMBR",
)

# 名字匹配片段（小写）：内核 xpad 驱动给 360 手柄起的名字是 "Microsoft X-Box 360 pad"，
# 大写以后是 X-BOX 而不是 XBOX —— 只匹配 "xbox" 会漏掉实体手柄，所以两种写法都要；
# 第三方手柄常常只报 "…gamepad" / "…joystick"（实测这台是 "Zikway HID gamepad"），所以也收进来。
# 认哪台最终还要过 matches()（必须有 A 键 + 左摇杆两轴），所以放宽片段不会把键盘鼠标收进来。
DEFAULT_NAME_FRAGMENTS = ("xbox", "x-box", "xinput", "gamepad", "joystick")

# 读设备权限不足时给出的补救命令（写在报错里，省得再翻文档）
PERMISSION_HINT = (
    "读 /dev/input/* 需要权限：把当前用户加进 input 组后重新登录（或逐次 sudo chmod a+r /dev/input/event*）"
)

def sysfs_bits(path: str, capability: str) -> int | None:
    """把 sysfs 的能力位图读成一个大整数（`capability` 取 `abs` / `key` / …）。

    内核打印的是**多字**十六进制位图、**高位字在前**：第 i 个（从 0 数）字覆盖事件码
    `[64·(n-1-i), 64·(n-1-i)+63]`，所以按顺序左移拼成大整数后，第 c 位就是事件码 c。
    返回 None = 读不到 sysfs（设备刚 uinput 建出来、老内核等），调用方保守地"试着开一下"。
    """
    caps = Path("/sys/class/input") / Path(path).name / "device/capabilities" / capability
    try:
        words = caps.read_text().split()
    except OSError:
        return None
    bits = 0
    for word in words:  # 高位字在前
        bits = (bits << 64) | int(word, 16)
    return bits


def looks_like_gamepad(path: str) -> bool | None:
    """sysfs 里这个 event 设备像不像手柄：**至少有手柄类轴、且有游戏按键**（都不需要权限）。

    这不是手柄的完整判据（真判据在 `matches()`：A 键 + 左摇杆两个轴），只用来先把
    "一定不是手柄"的接口滤掉：它们也挂在 `/dev/input/event*` 上、往往同样不可读，挨个去开
    就会打出一串"手柄没权限"的假警报（实测：跑过 setup 脚本、手柄也认到了，开头却还在报
    event0 没权限）。两条实测教训：

    * 不能只写"绝对轴非 0"——手柄的**键盘接口** `/dev/input/event17` 就带一个 `ABS_VOLUME`；
    * 也不能只写"有手柄类轴"——**触摸板**报 `ABS_X/ABS_Y`，会被当成候选、再因为没权限
      报成"手柄没权限"（实测 `/dev/input/event5` = `… Touchpad`）。

    返回 None = 读不到 sysfs（设备刚 uinput 建出来、老内核等），调用方保守地"试着开一下"。
    """
    bits = sysfs_bits(path, "abs")
    if bits is None:
        return None
    if not any((bits >> code) & 1 for code in GAMEPAD_AXIS_CODES):
        return False
    keys = sysfs_bits(path, "key")
    if keys is None:
        return None
    return any((keys >> code) & 1 for code in GAMEPAD_BUTTON_CODES)


# 事件码 → 输出下标（按钮固定；轴由 build_axis_roles 在连接时按设备判定）
BUTTON_INDEX = {getattr(ecodes, name): i for i, name in enumerate(BUTTON_CODES)}


def build_axis_roles(dev: InputDevice) -> dict[int, tuple[int, str, float, float]]:
    """读设备的 AbsInfo，决定每个轴**落在哪个 Joy 槽位、按什么量纲归一化**。

    返回 `{事件码: (槽位, 类型, 中心, 半量程)}`，类型取 `stick` / `trigger` / `hat`。
    判定规则（不猜设备型号，只看内核给的量程与静止值）：

    * `ABS_HAT0X/Y` → 槽位 6/7，-1/0/1（正负号在 apply_event 里按 SLOT_SIGN 换）；
    * **双极性**（`min < 0`，或静止值在中点附近 ±25%）→ 当一个**摇杆轴**，按码位顺序填
      `STICK_SLOTS`：xpad 的 X/Y/RX/RY → 0/1/3/4，本机这台 Zikway 的 X/Y/Z/RZ → 0/1/3/4
      （它的 Z/RZ 是右摇杆，静止 128，不是扳机）；
    * **单极性**（静止在 0 一端）→ 当一个**扳机轴**，填 `TRIGGER_SLOTS`：`ABS_BRAKE`→2（LT）、
      `ABS_GAS`→5（RT）优先，其余按码位顺序（xpad 的 Z/RZ → 2/5）；
    * 量程为 0 的轴（描述里有、设备不报）与多出来的轴忽略，各打一行提示。
    """
    caps = dev.capabilities(absinfo=True)
    sticks: list[tuple[int, object]] = []
    triggers: list[tuple[int, object]] = []
    roles: dict[int, tuple[int, str, float, float]] = {}
    for code, info in sorted(caps.get(ecodes.EV_ABS, [])):
        if code in HAT_SLOTS:
            roles[code] = (HAT_SLOTS[code], "hat", 0.0, 1.0)
            continue
        span = info.max - info.min
        if span <= 0:
            continue
        center = (info.min + info.max) / 2.0
        if info.min < 0 or abs(info.value - center) <= 0.25 * span:
            sticks.append((code, info))
        else:
            triggers.append((code, info))

    for (code, info), slot in zip(sticks, STICK_SLOTS):
        roles[code] = (slot, "stick", (info.min + info.max) / 2.0, (info.max - info.min) / 2.0)
    preferred = {ecodes.ABS_BRAKE: 0, ecodes.ABS_GAS: 1}  # HID 里 brake = 左扳机、gas = 右扳机
    triggers.sort(key=lambda ci: (preferred.get(ci[0], 2), ci[0]))
    for (code, info), slot in zip(triggers, TRIGGER_SLOTS):
        roles[code] = (slot, "trigger", 0.0, float(info.max))
    return roles


def make_joy(frame_id: str, axes: list[float], buttons: list[int], stamp) -> Joy:
    msg = Joy()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.axes = list(axes)
    msg.buttons = list(buttons)
    return msg


class JoyNode(Node):
    def __init__(self) -> None:
        super().__init__("joy_node")
        self.declare_parameter("device", "")  # 空 = 自动找；给了就只认这个路径
        self.declare_parameter("device_dir", "/dev/input")  # 自动找时扫哪个目录
        self.declare_parameter("name", ",".join(DEFAULT_NAME_FRAGMENTS))  # 空 = 任何手柄都收
        self.declare_parameter("deadband", 0.08)  # 摇杆死区（LT/RT 与 DPad 不适用）
        self.declare_parameter("rate_hz", 100.0)  # 发布频率
        self.declare_parameter("topic", "joy")
        self.declare_parameter("status_topic", "control_status")

        self.device_path = self.get_parameter("device").value
        self.device_dir = self.get_parameter("device_dir").value
        name = self.get_parameter("name").value
        self.name_fragments = tuple(part.strip().lower() for part in name.split(",") if part.strip())
        self.deadband = float(self.get_parameter("deadband").value)
        rate_hz = float(self.get_parameter("rate_hz").value)
        joy_topic = self.get_parameter("topic").value
        status_topic = self.get_parameter("status_topic").value

        # /joy 用 reliable：它是低频"命令"而不是传感器流，也让 `ros2 topic echo /joy` 开箱可用
        # （控制器节点用同样的 reliable 订阅）
        self.publisher = self.create_publisher(Joy, joy_topic, 1)
        self.status_sub = self.create_subscription(
            ControlStatus, status_topic, self.on_status, QoSProfile(depth=1)
        )

        self.device: InputDevice | None = None
        self.axis_roles: dict[int, tuple[int, str, float, float]] = {}
        self.axes = [0.0] * AXIS_COUNT
        self.buttons = [0] * BUTTON_COUNT
        self.frame_id = "joy_disconnected"
        self.last_status_log = 0.0
        self.last_mode = ""

        self.create_timer(1.0 / rate_hz, self.poll)
        self.create_timer(1.0, self.try_connect)  # 没插 / 被拔掉时每秒重试一次
        self.try_connect()
        self.get_logger().info(
            f"手柄节点：{rate_hz:.0f} Hz 发布 /{joy_topic}；设备 = "
            + (self.device_path or "自动找（名字含 " + "/".join(self.name_fragments) + "）")
        )

    # ------------------------------------------------------------------ 设备
    def candidate_paths(self) -> list[str]:
        if self.device_path:
            return [self.device_path]
        # 用 glob 而不是 evdev.list_devices()：后者会先按"可读可写"过滤一遍，
        # 没权限时直接返回空表，于是只能报"没找到手柄"，看不出是权限问题。
        paths = sorted(glob.glob(f"{self.device_dir}/event*"))
        # 键盘 / 鼠标 / 电源键也挂在 /dev/input/event* 上，而它们跟手柄一样（在同一台机器上）
        # 往往只有 root:input 可读：直接挨个去开，会在日志里打出一串"手柄没权限"的假警报
        # （实测：跑过 setup 脚本、手柄也认到了，开头却还在报 event0 没权限）。
        # 先按 sysfs 的"像不像手柄"筛一遍（见 looks_like_gamepad）：键盘/鼠标/触摸板/Consumer
        # Control 直接跳过；判不了的（刚建的 uinput 设备等）留着"试着开一下"，不牺牲鲁棒性。
        # 一个手柄类轴都没有 = 这台机器现在就没有手柄，此时**不去开**那些设备，
        # 免得又把"键盘没权限"报成"手柄没权限"。
        verdicts = {path: looks_like_gamepad(path) for path in paths}
        return [p for p, v in verdicts.items() if v is True] + [p for p, v in verdicts.items() if v is None]

    def matches(self, dev: InputDevice) -> bool:
        caps = dev.capabilities()
        keys = set(caps.get(ecodes.EV_KEY, []))
        axes = {code for code, _ in caps.get(ecodes.EV_ABS, [])}
        # 必须像手柄：有 A 键 + 左摇杆两个轴（DPad 用 hat 还是轴不强求）
        if ecodes.BTN_A not in keys or ecodes.ABS_X not in axes or ecodes.ABS_Y not in axes:
            return False
        if not self.name_fragments:
            return True
        lowered = (dev.name or "").lower()
        return any(fragment in lowered for fragment in self.name_fragments)

    def try_connect(self) -> None:
        if self.device is not None:
            return
        denied: list[str] = []
        for path in self.candidate_paths():
            try:
                dev = InputDevice(path)
            except PermissionError:
                denied.append(path)
                continue
            except OSError:
                continue
            if self.matches(dev):
                self.device = dev
                self.axis_roles = build_axis_roles(dev)
                self.frame_id = f"joy_connected:{dev.name}"
                self.get_logger().info(
                    f"手柄已连接：{dev.name}（{path}，{dev.phys}）；"
                    f"轴布局 = {self.describe_axis_roles()}"
                )
                return
            dev.close()
        if denied:
            # 只有"确实像手柄、但打不开"时才提示权限（见 candidate_paths 的注释）
            self.get_logger().error(
                f"{len(denied)} 个输入设备打不开（没有权限）：{'、'.join(denied)}；{PERMISSION_HINT}",
                once=True,
            )
            return
        self.get_logger().warning(
            "还没找到手柄：插上实体手柄，或先跑仿真手柄（sim_joy/xbox_sim_joy.py）",
            throttle_duration_sec=5.0,
        )

    def disconnect(self, reason: str) -> None:
        if self.device is not None:
            self.get_logger().warning(f"手柄断开（{reason}）")
            try:
                self.device.close()
            except OSError:
                pass
        self.device = None
        self.axes = [0.0] * AXIS_COUNT
        self.buttons = [0] * BUTTON_COUNT
        self.frame_id = "joy_disconnected"

    # ------------------------------------------------------------------ 读与发
    def describe_axis_roles(self) -> str:
        """一行说清"哪个轴去了哪个槽位、什么量纲"（启动日志与真机核对都用它）。"""
        names = {}
        for code in self.axis_roles:
            label = ecodes.ABS.get(code)
            names[code] = label if isinstance(label, str) else f"0x{code:x}"
        parts = [
            f"{names[code]}→axes[{slot}]/{kind}" + (f"({center:.0f}±{half:.0f})" if kind != "hat" else "")
            for code, (slot, kind, center, half) in sorted(self.axis_roles.items(), key=lambda kv: kv[1][0])
        ]
        unknown = []
        caps = self.device.capabilities() if self.device is not None else {}
        for code, _info in caps.get(ecodes.EV_ABS, []):
            if code not in self.axis_roles and code not in HAT_SLOTS:
                unknown.append(ecodes.ABS.get(code, str(code)))
        text = "、".join(parts) if parts else "（没有可用轴）"
        if unknown:
            text += f"；忽略 {unknown}"
        return text

    def apply_event(self, event) -> None:
        if event.type == ecodes.EV_ABS:
            role = self.axis_roles.get(event.code)
            if role is None:
                return
            slot, kind, center, half = role
            if kind == "stick":  # 双极性 + 死区
                value = self.stick(event.value, center, half)
            elif kind == "trigger":  # 单极性：松开 = -1、踩到底 = +1
                value = self.trigger(event.value, half)
            else:  # DPad hat：本来就是 -1 / 0 / 1
                value = float(event.value)
            # evdev 正负号 → 规范化正负号（见 SLOT_SIGN）；+ 0.0 把 −0.0 变回 0.0，echo 出来不带负号
            self.axes[slot] = SLOT_SIGN[slot] * value + 0.0
        elif event.type == ecodes.EV_KEY:
            index = BUTTON_INDEX.get(event.code)
            if index is not None:
                self.buttons[index] = 1 if event.value else 0

    def stick(self, raw: int, center: float, half: float) -> float:
        value = (raw - center) / half if half > 0 else 0.0
        if abs(value) < self.deadband:
            return 0.0
        return max(-1.0, min(1.0, value))

    @staticmethod
    def trigger(raw: int, full_scale: float) -> float:
        # 松开 = -1、踩到底 = +1（与摇杆同一个区间，控制器侧不必区分量纲）
        if full_scale <= 0:
            return 0.0
        return max(-1.0, min(1.0, raw / full_scale * 2.0 - 1.0))

    def poll(self) -> None:
        if self.device is not None:
            try:
                ready, _, _ = select.select([self.device.fd], [], [], 0.0)
                while ready:
                    for event in self.device.read():
                        self.apply_event(event)
                    ready, _, _ = select.select([self.device.fd], [], [], 0.0)
            except OSError:
                self.disconnect("设备读失败，可能被拔掉了")
        self.publisher.publish(
            make_joy(self.frame_id, self.axes, self.buttons, self.get_clock().now().to_msg())
        )

    # ------------------------------------------------------------------ 回程
    def on_status(self, msg) -> None:
        """控制器发来的状态（回程）。模式一变就打印，其余每 2 s 打一行。"""
        now = self.get_clock().now().nanoseconds / 1e9
        changed = msg.mode != self.last_mode
        if not changed and now - self.last_status_log < 2.0:
            return
        self.last_status_log = now
        self.last_mode = msg.mode
        self.get_logger().info(
            f"控制器回程：模式={msg.mode} 斜坡={msg.ramp_progress * 100:.0f}% "
            f"已发指令={msg.command_count} 条"
        )


def main() -> int:
    """跑起来，并在 Ctrl-C 时**干净退出**。

    两处容易踩的（实测）：① `rclpy.spin()` 在 SIGINT 时抛 `ExternalShutdownException`，
      它已经把 context shutdown 过了，我们再调一次 `rclpy.shutdown()` 就会抛
      `RCLError: failed to shutdown: rcl_shutdown already called`，于是 Ctrl-C 打出一段
      traceback、退出码 1（第一次三节点联跑就是这么结束的）；② context 已经关了以后
      `destroy_node()` 也可能报错。所以这两步都要"确认还活着再做"。
    """
    rclpy.init()
    node = JoyNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node.device is not None:
            try:
                node.device.close()
            except OSError:
                pass
        try:
            node.destroy_node()
        except Exception:  # noqa: BLE001  已经 shutdown 过时 destroy 会抛，不值得我们再崩一次
            pass
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
