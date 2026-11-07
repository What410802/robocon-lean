/**
 * @file sim_node.cpp
 * @brief 仿真运行节点（C++）：MuJoCo 里的 black 四足，收 MIT 指令，发电机反馈与 IMU。
 *
 * 分工：本节点 = **电机 + 机器人**那一侧（实机上由驱动板 + 本体承担）。控制器节点发来的是
 * MIT 五参数（/mit_command），本节点用电机模型（include/quadruped_ros2/motor.hpp）把它算成
 * 关节力矩写进 `data.ctrl`，步进物理，再把 {q, dq, ddq, tau, cur} 与 IMU 发回去。
 *
 * 线程（**两条**，2026-10-06 从"官方界面 + 三线程"改成自建窗口的单线程仿真）：
 *   * **主线程 = 仿真 + 渲染**：装现场 → 建窗口（`include/quadruped_ros2/viewer.hpp`）→
 *     一个循环里"按墙钟补齐物理步（取指令 → 算力矩 → `mj_step` → 发反馈）→ 画一帧"。
 *     `mjModel`/`mjData` **只被这一条线程碰**，所以不需要任何锁——比"到处加锁"更安全，
 *     因为根本不存在共享可变状态（换来的是窗口必须在这个线程里建，GLFW 本来也这么要求）；
 *   * **ROS 执行器线程**：只收 `/mit_command`、答 `/sim_reset`。它只碰 `SimNode` 自己的
 *     命令行槽位（一把只护 12 个 double 的小锁）与一个原子复位标志，**不碰 `mjData`**。
 *
 * 为什么不用官方 `Simulate` 界面了：实测鼠标一动它就掉到 43 fps（p95 41.8 ms）、
 * 自建窗口同样负载 56 fps；差别来自"两线程共享递归锁 + 每事件控件命中测试/重绘"这套结构。
 * 数据、复现命令与取舍见 UPSTREAM.md §2。
 *
 * 为什么仿真用 C++（没变）：Python 版（`launch_passive`）开窗口受 GIL 限制，实测只有 0.93x 实时。
 *
 * 时间：无窗口时物理按墙钟节流（deadline pacing，误差不累积、落后就重新对齐不追赶）；
 * 有窗口时"每帧把物理补齐到当前墙钟再画"，于是物理仍是 500 Hz、渲染是 vsync 的帧率。
 * `realtime:=false` 只对无窗口有意义（全速跑，专供回归脚本）。
 * 控制周期 = 物理步长 = 0.002 s（500 Hz，与实机主控同量级）。
 */

#include "quadruped_ros2/attitude.hpp"     // 状态行里的倾角
#include "quadruped_ros2/control.hpp"      // kNu（与控制器、站姿表同源）
#include "quadruped_ros2/depth_camera.hpp" // 深度相机（plum 深度策略的 sim 侧输入）
#include "quadruped_ros2/motor.hpp"

#include "quadruped_ros2/msg/depth_frame.hpp"
#include "quadruped_ros2/msg/mit_command.hpp"
#include "quadruped_ros2/msg/motor_state.hpp"

#include <mujoco/mujoco.h>

#include "quadruped_ros2/viewer.hpp" // 自建窗口（GLFW + mjv/mjr），见文件头

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/empty.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

namespace fs = std::filesystem;

using quadruped_ros2::msg::DepthFrame;
using quadruped_ros2::msg::MitCommand;
using quadruped_ros2::msg::MotorState;
using sensor_msgs::msg::Imu;
using std_srvs::srv::Empty;

constexpr int kNu = quadruped::control::kNu; // 12：自检/控制器的固定宽度；运行时用 model->nu

double WallNow()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/// 只保留最新一条的 QoS（与控制器节点一致）：传感器流要的是"现在"。
/// best_effort 不会被 slow subscriber 的流控顶住；`ros2 topic echo` 的默认 QoS 是
/// sensor_data（best_effort），所以命令行看这几条开箱可用（见 UPSTREAM.md §2）。
rclcpp::QoS LatestQoS()
{
    return rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
}

/// 一次仿真要用的东西：模型 + 数据 + 实际加载的场景路径。
/// 自己持有 m/d、析构里释放（窗口只是拿它渲染，不改所有权）。
/// 否则同一个指针会被释放两次（官方的析构里也会 mj_delete*）。
struct Scene
{
    mjModel* m = nullptr;
    mjData* d = nullptr;
    fs::path path;

    Scene() = default;
    ~Scene()
    {
        if (d != nullptr)
        {
            mj_deleteData(d);
        }
        if (m != nullptr)
        {
            mj_deleteModel(m);
        }
    }
    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;

    void Release()
    {
        m = nullptr;
        d = nullptr;
    }
};

/// 每个执行器作用在哪根关节上（qpos / qvel·qacc 下标），启动时查一次。
/// 不写死关节名：换模型这里不用改（只有"站姿表 12 个数"是按本任务写死的）。
struct ActuatorIndex
{
    std::vector<int> qpos;
    std::vector<int> dof;
};

/// IMU 三个传感器在 sensordata 里的起始下标（模型见 models/black_description.xml 的 <sensor>）。
struct ImuAddress
{
    int quat = -1;
    int gyro = -1;
    int acc = -1;

    bool ok() const { return quat >= 0 && gyro >= 0 && acc >= 0; }
};

/// 视口尺寸（逻辑像素）。本机是 2x 缩放显示，**framebuffer 会是这里的两倍**，
/// 渲染开销按 framebuffer 算——这也是"阴影贴图别开太大"的原因，见 §2.1 的实测。
constexpr int kViewerWidth = 1280;
constexpr int kViewerHeight = 720;

/// 视口里的阴影贴图边长（默认值，可被 `viewer_shadow_size` 覆盖）。取 4096 = 模型自己的默认值：
/// 2026-10-07 用 [`scripts/agent_scripts/shadow_probe.py`] 重测后**推翻**了原先"4096 太贵所以降到
/// 1024"的判断——那次 15.5 ms 是透过真实窗口（含 vsync/blit）量的，两者不可比；离屏同一路径
/// （2× 超采样缓冲 2560×1440、offsamples=0、11 次中位）的净阴影开销是
/// 1024 → 2.67 ms、2048 → 2.34 ms、4096 → 3.12 ms/帧，即 4096 只多 0.2~0.6 ms，换来的却是
/// 阴影边界偏移 0.728 → **0.171 px**（4.3×，见 UPSTREAM.md §2）。
/// **必须在建 GL 上下文之前设**（上下文建好后改无效，`mjr_makeContext` 只读一次）。
constexpr int kViewerShadowSize = 4096;

/// 阴影正交盒的半宽 = `stat.extent × 这个值`（**只对平行光有效**）。默认 1.0 = 与 MuJoCo 一致
/// （本场景覆盖半径 ±1.33 m）。成因见 UPSTREAM.md §2：MuJoCo 的阴影是
/// **逐片元硬比较**（shadow map 用 `GL_NEAREST` + `COMPARE_R_TO_TEXTURE` 采样，
/// `render/classic/render_context.c` 的 makeShadow），于是阴影边界被量化到**纹素格子**上：
/// 纹素 = 2·extent·clip/(shadowsize−2)，画质只取决于 `clip/shadowsize` 的**比值**。
/// 所以收紧这个值能换画质（clip 0.5 → 纹素减半），代价是覆盖范围同比缩小：狗走远一点就
/// 不在阴影盒里、阴影整块消失（实测 clip=1.0、狗 x=1.5 m 时阴影已经是 0 像素）。
/// 本任务用"不动覆盖范围、只把 shadowsize 回到 4096"来换画质，这个参数留作现场旋钮。
/// 与 shadowsize 同理，它在 `mjr_makeContext` 里只读一次，**必须在建 GL 上下文之前设**。
constexpr double kViewerShadowClip = 1.0;

/// 视口速度档位（`-` / `=` 在它上面走）：照官方 `Simulate::percentRealTime` 的 31 档
/// （`simulate/simulate.h:241`，100 → 80 → 63 → … → 0.1 %），下标越大越慢。
/// 只作用于**窗口模式**的墙钟推进（目标时刻 = 锚点 + 经过的时间 × 倍率）；
/// 无窗口时由 `realtime` 参数决定快慢。
constexpr double kSpeedLevels[] = {100, 80,  63,  50,  40,  32,  25,  20,  16,  13, 10,
                                   8,   6.3, 5.0, 4,   3.2, 2.5, 2,   1.6, 1.3, 1,  .8,
                                   .63, .5,  .4,  .32, .25, .2,  .16, .13, .1};
constexpr int kSpeedLevelsCount = static_cast<int>(sizeof(kSpeedLevels) / sizeof(double));

/// 窗口模式下"落后"超过这么多秒就重新对齐，不去追赶（窗口被拖住/系统卡一下之后不该让物理
/// 在后面狂追；与无窗口的 deadline pacing 同一口径：误差不累积）
constexpr double kMaxLagSeconds = 0.2;

/// 一帧最多补多少物理步：防止"窗口被拖住/系统卡了一下"之后一帧里追赶成千上万步
/// （那会让画面看起来"瞬移"）。60 fps × 2 ms 只需 ~8 步，这里给足余量。
constexpr int kMaxCatchupSteps = 200;

/// 节点参数（启动时读一次，之后不变）。
struct Options
{
    std::string scene; ///< 场景 xml；空 = 从可执行文件往上找 scenes/
    /// 起点：raw（= `mj_resetData` 的模型原姿态，**默认**，与上次任务 `@20260927_motor` 的
    /// `--start` 默认一致）/ rest（场景里的趴卧 keyframe，方便"从趴着开始"看起身）。
    std::string start;
    bool viewer = true;    ///< 开不开官方窗口
    bool realtime = true;  ///< 按墙钟节流
    int publish_every = 1; ///< 每多少步发一次反馈（1 = 每步 = 500 Hz）
    /// 每多少步发一次 eval_state（评测 trace 用；0 = 关）。内容为 base/foot/contact 的 JSON。
    int eval_state_every = 0;
    /// 深度相机：auto（模型里有 depth_cam 就开）/ on（必须有，缺失 = 装配失败）/ off。
    std::string depth = "auto";
    /// 每多少步渲一帧深度并发布 /depth_cam（默认 10 步 = 20 ms = 50 Hz，与反馈共用同一 d->time）。
    int depth_every = 10;
    /// 关节力矩上限 [N·m]。默认 -1 = 自动：优先 tau_max_array，其次标量，最后从模型的
    /// 力矩执行器 `ctrlrange` 读取（blackW = 23.7/23.7/59.25/20）。审计 F1：标量 20 与
    /// 训练/judge 的逐关节限值不等价，必须默认对齐全。
    double tau_max = -1.0;
    /// 逐关节力矩上限（16 个；非空且长度=nu 时优先于 tau_max 标量）。
    std::vector<double> tau_max_array;
    double damp_kd = 0.5; ///< 看门狗退回阻尼模式时的 kd [N·m·s/rad]
    int command_timeout_ms = 200; ///< 多少个控制周期没收到指令就退回阻尼（折算成步数，见下）
    /// 视口阴影：`viewer_shadow` 是**运行时可改**的渲染标志（窗口里按 S 也能切）；
    /// `viewer_shadow_size` / `viewer_shadow_clip` 是阴影贴图边长与正交盒半宽系数，
    /// 只在建窗口（GL 上下文）时用一次，改它们要重启（见 kViewerShadowClip 的注释）。
    bool viewer_shadow = true;
    int viewer_shadow_size = kViewerShadowSize;
    double viewer_shadow_clip = kViewerShadowClip;
    /// 视口多重采样（抗锯齿）：官方界面建窗时也用 4；0 = 关（放大观察会有明显锯齿）
    int viewer_msaa = quadruped::viewer::Window::kDefaultMsaa;
    /// 视口超采样倍率：离屏按 倍率×framebuffer 渲染再 GL_LINEAR 缩回窗口（1 = 关）
    int viewer_render_scale = quadruped::viewer::Window::kDefaultRenderScale;
    /// 相机跟随机器人 base（默认开，与评审视频的跟随视角一致）；Ctrl+F 可运行中切换。
    bool follow_camera = true;

    /// 电机非理想项（默认全关 = 理想力矩源，行为与之前逐位一致）。语义与计数口径照搬
    /// 第三次培训的完整版（`@20260927_motor/cpp/src/motor.h` 的 JointMotors），见
    /// UPSTREAM.md §2。
    double motor_deadzone = 0.0; ///< 静摩擦死区 [N·m]：0 < |τ| < 死区 → 推不动，输出 0
    int motor_delay_cycles = 0; ///< 指令延迟 [控制周期]：用 N 个周期之前下发的那条指令
    double motor_noise = 0.0; ///< 力矩噪声标准差 [N·m]（高斯白噪声，固定种子可复现）
    int motor_seed = 12345; ///< 噪声种子（同参数 + 同 seed → 同一条噪声序列）
    double status_period_s = 5.0; ///< 每隔多少仿真秒打一行状态（基座高度/倾角/接触点）；0 = 不打
    /// Optional episode spawn override applied after the start pose is loaded (Part1 start zone /
    /// Part2 level starts). All NaN = keep the scene's start pose.
    double spawn_x = std::numeric_limits<double>::quiet_NaN();
    double spawn_y = std::numeric_limits<double>::quiet_NaN();
    double spawn_z = std::numeric_limits<double>::quiet_NaN();
    double spawn_yaw = std::numeric_limits<double>::quiet_NaN();
};

class SimNode : public rclcpp::Node
{
  public:
    SimNode() : Node("sim_node")
    {
        Options opt;
        opt.scene = declare_parameter<std::string>("scene", "");
        opt.start = declare_parameter<std::string>("start", "raw");
        opt.viewer = declare_parameter<bool>("viewer", true);
        opt.realtime = declare_parameter<bool>("realtime", true);
        opt.publish_every = declare_parameter<int>("publish_every", 1);
        opt.eval_state_every = declare_parameter<int>("eval_state_every", 0);
        opt.depth = declare_parameter<std::string>("depth", opt.depth);
        opt.depth_every = declare_parameter<int>("depth_every", opt.depth_every);
        opt.tau_max = declare_parameter<double>("tau_max", opt.tau_max);
        opt.tau_max_array =
            declare_parameter<std::vector<double>>("tau_max_array", opt.tau_max_array);
        opt.damp_kd = declare_parameter<double>("damp_kd", opt.damp_kd);
        opt.command_timeout_ms =
            declare_parameter<int>("command_timeout_ms", opt.command_timeout_ms);
        opt.viewer_shadow = declare_parameter<bool>("viewer_shadow", opt.viewer_shadow);
        opt.viewer_shadow_size =
            declare_parameter<int>("viewer_shadow_size", opt.viewer_shadow_size);
        opt.viewer_shadow_clip =
            declare_parameter<double>("viewer_shadow_clip", opt.viewer_shadow_clip);
        opt.viewer_msaa = declare_parameter<int>("viewer_msaa", opt.viewer_msaa);
        opt.viewer_render_scale =
            declare_parameter<int>("viewer_render_scale", opt.viewer_render_scale);
        opt.follow_camera = declare_parameter<bool>("follow_camera", opt.follow_camera);
        opt.motor_deadzone = declare_parameter<double>("motor_deadzone", opt.motor_deadzone);
        opt.motor_delay_cycles =
            declare_parameter<int>("motor_delay_cycles", opt.motor_delay_cycles);
        opt.motor_noise = declare_parameter<double>("motor_noise", opt.motor_noise);
        opt.motor_seed = declare_parameter<int>("motor_seed", opt.motor_seed);
        opt.status_period_s = declare_parameter<double>("status_period_s", opt.status_period_s);
    opt.spawn_x = declare_parameter<double>("spawn_x", opt.spawn_x);
    opt.spawn_y = declare_parameter<double>("spawn_y", opt.spawn_y);
    opt.spawn_z = declare_parameter<double>("spawn_z", opt.spawn_z);
    opt.spawn_yaw = declare_parameter<double>("spawn_yaw", opt.spawn_yaw);

        if (opt.start != "rest" && opt.start != "raw")
        {
            throw std::runtime_error("参数 start 只能是 rest 或 raw，收到：" + opt.start);
        }
        if (opt.publish_every < 1)
        {
            throw std::runtime_error("参数 publish_every 必须 ≥ 1");
        }
        if (opt.eval_state_every < 0)
        {
            throw std::runtime_error("参数 eval_state_every 不能为负（0 = 关）");
        }
        if (opt.depth != "auto" && opt.depth != "on" && opt.depth != "off")
        {
            throw std::runtime_error("参数 depth 只能是 auto / on / off，收到：" + opt.depth);
        }
        if (opt.depth_every < 1)
        {
            throw std::runtime_error("参数 depth_every 必须 ≥ 1（0/负数无效；默认 10 步 = 50 Hz）");
        }
        if (opt.motor_deadzone < 0.0 || opt.motor_noise < 0.0)
        {
            throw std::runtime_error("参数 motor_deadzone / motor_noise 不能为负");
        }
        if (opt.motor_delay_cycles < 0)
        {
            throw std::runtime_error("参数 motor_delay_cycles 不能为负");
        }
        if (opt.motor_seed < 0)
        {
            throw std::runtime_error("参数 motor_seed 不能为负");
        }
        if (opt.viewer_render_scale < 1)
        {
            throw std::runtime_error("参数 viewer_render_scale 必须 ≥ 1（1 = 不超采样）");
        }
        if (opt.viewer_msaa < 0)
        {
            throw std::runtime_error("参数 viewer_msaa 不能为负（0 = 关抗锯齿）");
        }
        if (opt.viewer_shadow_size < 0)
        {
            throw std::runtime_error("参数 viewer_shadow_size 不能为负（0 = 用模型里的默认值）");
        }
        if (opt.viewer_shadow_clip <= 0.0)
        {
            throw std::runtime_error("参数 viewer_shadow_clip 必须 > 0（阴影正交盒半宽系数）");
        }
        // 没有显示服务时（纯 ssh / 无窗口环境）不能建窗口，自动降级成无窗口跑，
        // 而不是让 GLFW 在 glfwInit 里把节点整个搞崩。
        if (opt.viewer && std::getenv("DISPLAY") == nullptr &&
            std::getenv("WAYLAND_DISPLAY") == nullptr)
        {
            RCLCPP_WARN(get_logger(),
                        "没有 DISPLAY / WAYLAND_DISPLAY：viewer 自动关掉，按无窗口跑");
            opt.viewer = false;
        }
        opt_ = opt;

        command_sub_ = create_subscription<MitCommand>(
            "mit_command", LatestQoS(), [this](MitCommand::SharedPtr msg) { OnCommand(msg); });
        state_pub_ = create_publisher<MotorState>("motor_state", LatestQoS());
        imu_pub_ = create_publisher<Imu>("imu", LatestQoS());
        eval_pub_ = create_publisher<std_msgs::msg::String>("eval_state", LatestQoS());
        depth_pub_ = create_publisher<DepthFrame>("depth_cam", LatestQoS());
        reset_srv_ =
            create_service<Empty>("sim_reset", [this](const std::shared_ptr<Empty::Request>,
                                                      std::shared_ptr<Empty::Response>)
                                  { reset_requested_.store(true); });

        RCLCPP_INFO(get_logger(),
                    "仿真节点：起点=%s、viewer=%s、realtime=%s、每 %d 步发一次反馈、"
                    "力矩上限 %s；cur 一律报 0（不做电流推定，理由见 Feedback()）",
                    opt.start.c_str(), opt.viewer ? "开" : "关", opt.realtime ? "开" : "关",
                    opt.publish_every,
                    opt.tau_max > 0.0 ? (std::to_string(opt.tau_max) + " N·m（标量）").c_str()
                                      : "逐关节（启动时打印）");
        RCLCPP_INFO(get_logger(),
                    "等待 /mit_command（收到第一条之前按阻尼模式跑，与上电默认一致）");
    }

    const Options& options() const { return opt_; }

    /// 物理线程问"该收工了吗"：Ctrl-C（rclcpp 的 SIGINT）或窗口关了都算。
    bool quit_requested() const { return quit_.load() || !rclcpp::ok(); }

    void RequestQuit() { quit_.store(true); }

    /// 取最近一条 MIT 指令。返回 false = 一条都还没收到（看门狗由物理线程按步数判，见下）。
    bool TakeCommand(MitCommand* out)
    {
        std::lock_guard<std::mutex> lock(cmd_mutex_);
        if (!have_cmd_)
        {
            return false;
        }
        *out = cmd_;
        return true;
    }

    bool TakeReset() { return reset_requested_.exchange(false); }

    /// 窗口里按 R：与 `/sim_reset` 服务同一条路径（都只是置标志，复位在主线程做）
    void RequestReset() { reset_requested_.store(true); }

    void PublishMotorState(const MotorState& msg) { state_pub_->publish(msg); }

    void PublishImu(const Imu& msg) { imu_pub_->publish(msg); }
    void PublishEvalState(const std::string& text)
    {
        std_msgs::msg::String msg;
        msg.data = text;
        eval_pub_->publish(msg);
    }

    /// 深度帧：`depth` 是 DepthCamera::Render 的输出（64x48 row-major 公制深度）。`sim_time`
    /// 用同一步的 d->time（不是取消息时的墙钟），与 /motor_state.sim_time 对齐。
    void PublishDepth(const std::vector<float>& depth, double sim_time)
    {
        DepthFrame msg;
        msg.header.stamp = now();
        msg.sim_time = sim_time;
        msg.width = quadruped::depth::DepthCamera::kWidth;
        msg.height = quadruped::depth::DepthCamera::kHeight;
        msg.depth = depth;
        depth_pub_->publish(msg);
    }

  private:
    void OnCommand(const MitCommand::SharedPtr& msg)
    {
        // Non-finite command guard (P2-06): MuJoCo warns and recovers from NaN at the next
        // mj_step, but a NaN command must not silently drive a control cycle. Validate before it
        // reaches the motor model; on violation drop the command so the watchdog uses damping.
        const auto all_finite = [](const std::vector<double>& v) {
            for (double x : v)
            {
                if (!std::isfinite(x)) return false;
            }
            return true;
        };
        if (!all_finite(msg->q) || !all_finite(msg->w) || !all_finite(msg->kp) ||
            !all_finite(msg->kd) || !all_finite(msg->tau))
        {
            static int rejected = 0;
            if (rejected++ < 3)
            {
                RCLCPP_ERROR(get_logger(), "rejecting non-finite /mit_command; watchdog falls back to damping");
            }
            std::lock_guard<std::mutex> lock(cmd_mutex_);
            have_cmd_ = false;
            return;
        }
        std::lock_guard<std::mutex> lock(cmd_mutex_);
        cmd_ = *msg;
        if (!have_cmd_)
        {
            have_cmd_ = true;
            RCLCPP_INFO(get_logger(), "收到第一条 /mit_command：电机进入 MIT 模式"
                                      "（τ = tau + kp·(q_des − q) + kd·(w_des − q̇)）");
        }
    }

    Options opt_;
    std::atomic_bool quit_{false};
    std::atomic_bool reset_requested_{false};

    std::mutex cmd_mutex_;
    MitCommand cmd_;
    bool have_cmd_ = false;

    rclcpp::Subscription<MitCommand>::SharedPtr command_sub_;
    rclcpp::Publisher<MotorState>::SharedPtr state_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr eval_pub_;
    rclcpp::Publisher<Imu>::SharedPtr imu_pub_;
    rclcpp::Publisher<DepthFrame>::SharedPtr depth_pub_;
    rclcpp::Service<Empty>::SharedPtr reset_srv_;
};

/// 场景路径：参数给了就用它；没给就从**可执行文件所在目录往上找**带 scenes/ 的那一层
/// （比"往上数两层"健壮：build/、install/、或把可执行文件拷到别处都适用）。
bool ResolveScene(fs::path* out, const std::string& given, std::string* err)
{
    if (!given.empty())
    {
        *out = given;
        return true;
    }
    const fs::path exe_dir = fs::read_symlink("/proc/self/exe").parent_path();
    for (fs::path p = exe_dir; p != p.parent_path(); p = p.parent_path())
    {
        if (fs::is_directory(p / "scenes"))
        {
            *out = p / "scenes/flat_scene.xml";
            return true;
        }
    }
    *err = "从可执行文件所在目录往上找不到带 scenes/ 的任务目录（" + exe_dir.string() +
           "）：用 scene:=<路径> 显式指定场景";
    return false;
}

/// Optional spawn override (Part1 start zone / Part2 level starts); NaN keeps the scene pose.
void ApplySpawnOverride(const Options& opt, mjModel* m, mjData* d)
{
    if (!std::isfinite(opt.spawn_x) && !std::isfinite(opt.spawn_y) &&
        !std::isfinite(opt.spawn_z) && !std::isfinite(opt.spawn_yaw))
    {
        return;
    }
    const double x = std::isfinite(opt.spawn_x) ? opt.spawn_x : 0.0;
    const double y = std::isfinite(opt.spawn_y) ? opt.spawn_y : 0.0;
    const double z = std::isfinite(opt.spawn_z) ? opt.spawn_z : 0.58;
    const double yaw = std::isfinite(opt.spawn_yaw) ? opt.spawn_yaw : 0.0;
    d->qpos[0] = x;
    d->qpos[1] = y;
    d->qpos[2] = z;
    d->qpos[3] = std::cos(0.5 * yaw);
    d->qpos[4] = 0.0;
    d->qpos[5] = 0.0;
    d->qpos[6] = std::sin(0.5 * yaw);
    d->qvel[0] = 0.0; d->qvel[1] = 0.0; d->qvel[2] = 0.0;
    std::cout << "spawn override: x=" << x << " y=" << y << " z=" << z
              << " yaw=" << yaw << std::endl;
    mj_forward(m, d);
}

/// 装配现场：加载场景 + 建 data + 摆到起点。
bool LoadScene(const Options& opt, Scene* out, std::string* err)
{
    fs::path path;
    if (!ResolveScene(&path, opt.scene, err))
    {
        return false;
    }
    out->path = path;
    out->m = mj_loadXML(path.c_str(), nullptr, nullptr, 0);
    if (out->m == nullptr)
    {
        *err = "加载场景失败：" + path.string();
        return false;
    }
    out->d = mj_makeData(out->m);
    if (out->d == nullptr)
    {
        *err = "创建 mjData 失败";
        return false;
    }
    // 起点：raw = 模型原姿态（mj_resetData 的 qpos0，**默认**）；rest = 场景里的 <keyframe
    // name="rest">（index 0，趴卧） （直腿站立、脚底刚好触地，零力矩下自然塌成趴卧）。keyframe
    // 不会自动加载， 见 UPSTREAM.md §2。
    if (opt.start == "rest")
    {
        if (out->m->nkey == 0)
        {
            *err = "场景里没有 keyframe，无法用 start:=rest（改用 start:=raw）";
            return false;
        }
        mj_resetDataKeyframe(out->m, out->d, 0);
    }
    ApplySpawnOverride(opt, out->m, out->d);
    mj_forward(out->m, out->d);
    return true;
}

/// 复位（手柄 X → /sim_reset）：回到起点姿态并回阻尼（控制器侧会同时切阻尼）。
/// **保留仿真时间轴**：控制器的斜坡按 sim_time 算，把 time 清零会让"复位后再起身"先卡住
/// （这个坑上一版也踩过，见 @20260927_motor/cpp/essential_core/src/main.cpp 的注释）。
void ResetToStart(const Options& opt, mjModel* m, mjData* d)
{
    const double t_keep = d->time;
    if (opt.start == "rest" && m->nkey > 0)
    {
        mj_resetDataKeyframe(m, d, 0);
    }
    else
    {
        mj_resetData(m, d);
    }
    d->time = t_keep;
    ApplySpawnOverride(opt, m, d);
    mj_forward(m, d);
}

ActuatorIndex BuildIndex(const mjModel* m)
{
    ActuatorIndex idx;
    idx.qpos.resize(static_cast<std::size_t>(m->nu));
    idx.dof.resize(static_cast<std::size_t>(m->nu));
    for (int i = 0; i < m->nu; ++i)
    {
        const int j = m->actuator_trnid[2 * i]; // 该执行器作用的关节 id
        idx.qpos[static_cast<std::size_t>(i)] = m->jnt_qposadr[j];
        idx.dof[static_cast<std::size_t>(i)] = m->jnt_dofadr[j];
    }
    return idx;
}

ImuAddress FindImu(const mjModel* m)
{
    ImuAddress adr;
    const auto find = [m](const char* name, int min_dim, int* out)
    {
        const int id = mj_name2id(m, mjOBJ_SENSOR, name);
        if (id >= 0 && m->sensor_dim[id] >= min_dim)
        {
            *out = m->sensor_adr[id];
        }
    };
    find("imu_quat", 4, &adr.quat);
    find("imu_gyro", 3, &adr.gyro);
    find("imu_acc", 3, &adr.acc);
    return adr;
}

/// 看门狗/未收到指令时的兜底：纯阻尼（kp = 0、kd = damp_kd）。
MitCommand DampingCommand(int nu, double kd)
{
    MitCommand cmd;
    cmd.kp.assign(static_cast<std::size_t>(nu), 0.0);
    cmd.kd.assign(static_cast<std::size_t>(nu), kd);
    cmd.q.assign(static_cast<std::size_t>(nu), 0.0);
    cmd.w.assign(static_cast<std::size_t>(nu), 0.0);
    cmd.tau.assign(static_cast<std::size_t>(nu), 0.0);
    return cmd;
}

void PublishFeedback(SimNode* node, const mjData* d, const ActuatorIndex& idx,
                     const ImuAddress& imu, int nu)
{
    MotorState state;
    state.header.stamp = node->now();
    state.sim_time = d->time;
    state.q.assign(static_cast<std::size_t>(nu), 0.0);
    state.dq.assign(static_cast<std::size_t>(nu), 0.0);
    state.ddq.assign(static_cast<std::size_t>(nu), 0.0);
    state.tau.assign(static_cast<std::size_t>(nu), 0.0);
    state.cur.assign(static_cast<std::size_t>(nu), 0.0);
    for (int i = 0; i < nu; ++i)
    {
        const auto k = static_cast<std::size_t>(i);
        state.q[k] = d->qpos[idx.qpos[k]];
        state.dq[k] = d->qvel[idx.dof[k]];
        state.ddq[k] = d->qacc[idx.dof[k]];
        state.tau[k] = d->actuator_force[i];
        // cur **一律 0**：MuJoCo 的执行器是纯力矩源，模型里没有电流环，也没有传感器；
        // 按 tau/(减速比×Kt) 折算看似"有量纲"，其实 Kt 只能靠猜（讲义与手册都没给 GO-M8010-6 的
        // 值），算出来的数与实机电流没有可验证的对应关系——**没有支撑模型与数据就不编**。
        // 字段本身保留（任务书的接口定义就是 {q, dq, ddq, tau, cur}），含义明确为"未测量，置 0"。
        state.cur[k] = 0.0;
    }
    // 协方差一律留 0：仿真是确定性、无噪声的（要"有噪声的 IMU"就等电机非理想项那一轮再加）。
    node->PublishMotorState(state);

    if (!imu.ok())
    {
        return;
    }
    Imu out;
    out.header.stamp = state.header.stamp;
    out.header.frame_id = "imu"; // 本任务没有 TF 树，只是给这份数据标个来源
    // MuJoCo 的 framequat 是 (w, x, y, z)，ROS 的 Imu 是 (x, y, z, w) —— 顺序不能照抄
    out.orientation.w = d->sensordata[imu.quat + 0];
    out.orientation.x = d->sensordata[imu.quat + 1];
    out.orientation.y = d->sensordata[imu.quat + 2];
    out.orientation.z = d->sensordata[imu.quat + 3];
    // gyro / accelerometer 都是**机体系**：静止时加速度计 z ≈ +9.81 m/s²、自由落体时 ≈ 0
    out.angular_velocity.x = d->sensordata[imu.gyro + 0];
    out.angular_velocity.y = d->sensordata[imu.gyro + 1];
    out.angular_velocity.z = d->sensordata[imu.gyro + 2];
    out.linear_acceleration.x = d->sensordata[imu.acc + 0];
    out.linear_acceleration.y = d->sensordata[imu.acc + 1];
    out.linear_acceleration.z = d->sensordata[imu.acc + 2];
    node->PublishImu(out);
}

/// 物理线程主体：装现场 → 递给界面 → 一步一循环。返回值 = 进程退出码（0 正常 / 1 装配失败）。
/// 电机非理想项自检（`sim_node --self-test`）：
/// 不需要模型、窗口、硬件，跑完就退。检查口径与数字见 UPSTREAM.md §2。
int MotorSelfTest()
{
    using quadruped::control::kNu;
    using quadruped::motor::Bank;
    using quadruped::motor::Cmd;
    using quadruped::motor::NonIdeal;

    int bad = 0;
    const auto check = [&bad](const char* what, bool ok)
    {
        std::printf("  %-54s %s\n", what, ok ? "✓" : "✗");
        if (!ok)
        {
            ++bad;
        }
    };
    std::printf("电机非理想项自检（不需要硬件/窗口）\n");

    // ① 全关 = 理想公式（逐位一致，保证"不引入非理想项"的回归结果一字不差）
    {
        Bank bank(NonIdeal{}, quadruped::motor::kDefaultTauMax);
        bool same = true;
        for (int i = 0; i < kNu && same; ++i)
        {
            for (const double q : {-1.0, -0.3, 0.0, 0.37, 2.0})
            {
                for (const double dq : {-4.0, -0.2, 0.0, 0.9, 6.0})
                {
                    std::array<Cmd, kNu> c{};
                    std::array<double, kNu> jq{};
                    std::array<double, kNu> jdq{};
                    std::array<double, kNu> tau{};
                    c[static_cast<std::size_t>(i)] = Cmd{0.5, 30.0, 0.2, 1.5, -0.1};
                    jq[static_cast<std::size_t>(i)] = q;
                    jdq[static_cast<std::size_t>(i)] = dq;
                    bank.Apply(c.data(), jq.data(), jdq.data(), tau.data(), static_cast<std::size_t>(kNu));
                    if (tau[static_cast<std::size_t>(i)] !=
                        quadruped::motor::Torque(c[static_cast<std::size_t>(i)], q, dq,
                                                 quadruped::motor::kDefaultTauMax))
                    {
                        same = false;
                    }
                }
            }
        }
        check("全关时与理想公式 Torque() 逐位一致", same);
    }

    // ② 死区：0 < |τ| < 死区 → 0；τ 正好 0 不算落死区（完整版踩过的坑）
    {
        Bank bank(NonIdeal{0.5, 0, 0.0, 1}, 100.0);
        std::array<Cmd, kNu> c{};
        std::array<double, kNu> jq{};
        std::array<double, kNu> jdq{};
        std::array<double, kNu> tau{};
        c[0].kp = 100.0;
        c[0].q_des = 0.001; // τ = 0.1 N·m < 0.5
        c[1].kp = 100.0;
        c[1].q_des = 0.02; // τ = 2 N·m
        bank.Apply(c.data(), jq.data(), jdq.data(), tau.data(), static_cast<std::size_t>(kNu));
        check("死区：0.1 N·m（< 0.5）被削成 0", tau[0] == 0.0);
        check("死区：2 N·m 原样通过", tau[1] == 2.0);
        check("死区：只统计被削掉的那 1 个电机·步", bank.stats().dead == 1);

        Bank damp_bank(NonIdeal{0.5, 0, 0.0, 1}, 100.0);
        std::array<Cmd, kNu> damp{};
        for (Cmd& d : damp)
        {
            d.kd = 2.0; // 阻尼模式，dq 全 0 ⇒ τ 正好 0
        }
        for (int s = 0; s < 100; ++s)
        {
            damp_bank.Apply(damp.data(), jq.data(), jdq.data(), tau.data(), static_cast<std::size_t>(kNu));
        }
        check("死区：τ 正好 0 不计入 dead（否则会报出上万次假计数）",
              damp_bank.stats().dead == 0 && damp_bank.stats().dead_events == 0);
    }

    // ③ 指令延迟：N 个周期之前的指令，前 N 步还是"没收到指令"（0）
    {
        Bank bank(NonIdeal{0.0, 3, 0.0, 1}, 1000.0);
        std::array<Cmd, kNu> c{};
        std::array<double, kNu> jq{};
        std::array<double, kNu> jdq{};
        std::array<double, kNu> tau{};
        double first = -1.0;
        double fourth = -1.0;
        for (int s = 0; s <= 3; ++s)
        {
            for (Cmd& cc : c)
            {
                cc.tau_ff = (s == 0) ? 1.0 : 5.0;
            }
            bank.Apply(c.data(), jq.data(), jdq.data(), tau.data(), static_cast<std::size_t>(kNu));
            if (s == 0)
            {
                first = tau[0];
            }
            if (s == 3)
            {
                fourth = tau[0];
            }
        }
        check("延迟 3 周期：第 1 步仍是 0（还没收到指令）", first == 0.0);
        check("延迟 3 周期：第 4 步才用上第 1 步的 1 N·m", fourth == 1.0);
    }

    // ④ 噪声：同 seed 可复现、不同 seed 不同、RMS ≈ 标准差
    {
        Bank a(NonIdeal{0.0, 0, 0.1, 7}, 100.0);
        Bank b(NonIdeal{0.0, 0, 0.1, 7}, 100.0);
        Bank c(NonIdeal{0.0, 0, 0.1, 8}, 100.0);
        std::array<Cmd, kNu> zero{};
        std::array<double, kNu> jq{};
        std::array<double, kNu> jdq{};
        std::array<double, kNu> ta{};
        std::array<double, kNu> tb{};
        std::array<double, kNu> tc{};
        bool same = true;
        bool differ = false;
        double sum_sq = 0.0;
        int n = 0;
        for (int s = 0; s < 200; ++s)
        {
            a.Apply(zero.data(), jq.data(), jdq.data(), ta.data(), static_cast<std::size_t>(kNu));
            b.Apply(zero.data(), jq.data(), jdq.data(), tb.data(), static_cast<std::size_t>(kNu));
            c.Apply(zero.data(), jq.data(), jdq.data(), tc.data(), static_cast<std::size_t>(kNu));
            same = same && (ta[0] == tb[0]);
            differ = differ || (ta[0] != tc[0]);
            sum_sq += ta[0] * ta[0];
            ++n;
        }
        const double rms = std::sqrt(sum_sq / n);
        check("噪声：同 seed 两条序列逐位一致（可复现）", same);
        check("噪声：换 seed 序列就不同", differ);
        check("噪声：RMS ≈ 标准差 0.1 N·m（±30%）", rms > 0.07 && rms < 0.13);
    }

    // ⑤ 限幅 + 统计
    {
        Bank bank(NonIdeal{}, 5.0);
        std::array<Cmd, kNu> c{};
        std::array<double, kNu> jq{};
        std::array<double, kNu> jdq{};
        std::array<double, kNu> tau{};
        c[0].tau_ff = 100.0;
        bank.Apply(c.data(), jq.data(), jdq.data(), tau.data(), static_cast<std::size_t>(kNu));
        check("限幅：100 N·m 压到 tau_max=5 N·m 且计入 sat",
              tau[0] == 5.0 && bank.stats().sat == 1);
    }

    std::printf("电机自检：%s（失败 %d 项）\n", bad == 0 ? "全部通过 ✓" : "有失败项 ✗", bad);
    return bad == 0 ? 0 : 1;
}

/// 深度相机自检（`sim_node --depth-self-test <scene.xml>`）：不需要 ROS/窗口，跑完即退。
/// 用 EGL 离屏渲染场景**默认姿态**的一帧（就是 LoadScene 摆完的样子），把 64x48 row-major
/// float32（单位 m）写到 /tmp/depth_selftest.f32，供 eval 侧 Python 渲染器对同一场景逐位对拍。
int DepthSelfTest(const char* scene_path)
{
    Options opt;
    opt.scene = scene_path;
    Scene scene;
    std::string err;
    if (!LoadScene(opt, &scene, &err))
    {
        std::fprintf(stderr, "深度自检失败：%s\n", err.c_str());
        return 1;
    }
    mjModel* m = scene.m;
    std::printf("深度相机自检（EGL 离屏，不需要 ROS/窗口）\n场景：%s\nMuJoCo %s\n",
                scene.path.c_str(), mj_versionString());

    quadruped::depth::DepthCamera camera;
    if (!camera.Init(m, false))
    {
        std::fprintf(stderr, "深度自检失败：%s\n", camera.error().c_str());
        return 1;
    }
    std::vector<float> depth;
    if (!camera.Render(m, scene.d, &depth))
    {
        std::fprintf(stderr, "深度自检失败：%s\n", camera.error().c_str());
        return 1;
    }

    const char* out_path = "/tmp/depth_selftest.f32";
    std::FILE* file = std::fopen(out_path, "wb");
    if (file == nullptr)
    {
        std::fprintf(stderr, "深度自检失败：无法写 %s\n", out_path);
        return 1;
    }
    const std::size_t written = std::fwrite(depth.data(), sizeof(float), depth.size(), file);
    std::fclose(file);
    if (written != depth.size())
    {
        std::fprintf(stderr, "深度自检失败：%s 只写了 %zu/%zu 个 float32\n", out_path, written,
                     depth.size());
        return 1;
    }

    double min = std::numeric_limits<double>::infinity();
    double max = 0.0;
    double sum = 0.0;
    std::size_t finite = 0;
    for (const float value : depth)
    {
        if (std::isfinite(value))
        {
            ++finite;
            min = std::min(min, static_cast<double>(value));
            max = std::max(max, static_cast<double>(value));
            sum += value;
        }
    }
    const int camera_id = mj_name2id(m, mjOBJ_CAMERA, "depth_cam");
    const double znear = m->vis.map.znear * m->stat.extent;
    const double zfar = m->vis.map.zfar * m->stat.extent;
    std::printf("相机：depth_cam id=%d；输出 %dx%d row-major float32（单位 m）\n", camera_id,
                quadruped::depth::DepthCamera::kWidth, quadruped::depth::DepthCamera::kHeight);
    std::printf("模型 vis：znear=%.6g zfar=%.6g extent=%.6g -> 近端 %.6g m / 远端 %.6g m\n",
                m->vis.map.znear, m->vis.map.zfar, m->stat.extent, znear, zfar);
    std::printf("深度统计：min=%.6f max=%.6f mean=%.6f finite=%zu/%zu（无命中 = 远端 %.6g m）\n",
                finite > 0 ? min : 0.0, max,
                finite > 0 ? sum / static_cast<double>(finite) : 0.0, finite, depth.size(), zfar);
    std::printf("已写 %s（%zu 个 float32 = %zu 字节）\n", out_path, depth.size(),
                depth.size() * sizeof(float));
    return 0;
}

/// 仿真主循环：**物理 + 渲染都在这一条线程**（主线程），`mjModel`/`mjData` 不被别人碰。
///
/// 两种节奏：
///   * 有窗口：每帧把物理"补齐到当前墙钟"（上限 `kMaxCatchupSteps` 步）再画一帧 —— 物理仍是
///     500 Hz，渲染跟 vsync 走；窗口/系统卡一下之后不会攒出一大堆步一次补完；
///   * 无窗口：一步一节流（deadline pacing），与自检脚本、回归脚本的口径完全一致。
int SimLoop(SimNode* node)
{
    const Options& opt = node->options();

    Scene scene;
    std::string err;
    if (!LoadScene(opt, &scene, &err))
    {
        std::fprintf(stderr, "仿真节点装配失败：%s\n", err.c_str());
        return 1;
    }
    mjModel* m = scene.m;
    mjData* d = scene.d;
    const ActuatorIndex idx = BuildIndex(m);
    const ImuAddress imu = FindImu(m);
    const double dt = m->opt.timestep;
    // 深度相机：模型里有没有 depth_cam（blackW_description.xml 自带，所有场景都 include 它）。
    // depth:=on 表示"必须有"：与其它装配失败一样直接退出，而不是悄悄跑成盲走。
    const int depth_camera_id = mj_name2id(m, mjOBJ_CAMERA, "depth_cam");
    if (opt.depth == "on" && depth_camera_id < 0)
    {
        std::fprintf(stderr, "仿真节点装配失败：depth:=on 要求模型里有 depth_cam 相机，"
                             "但场景 %s 里没有\n", scene.path.c_str());
        return 1;
    }

    // 逐关节力矩上限：tau_max_array > tau_max 标量 > 模型力矩执行器 ctrlrange。
    // blackW MJCF 的 <motor ctrlrange> 就是关节力矩限幅（23.7/23.7/59.25/20）；
    // 带偏置的执行器（位置伺服）的 ctrlrange 是角度，不能当力矩用，跳过。
    std::vector<double> tau_limits(static_cast<std::size_t>(m->nu),
                                   quadruped::motor::kDefaultTauMax);
    std::string tau_source = "default";
    if (!opt.tau_max_array.empty())
    {
        if (opt.tau_max_array.size() != static_cast<std::size_t>(m->nu))
        {
            std::fprintf(stderr, "tau_max_array 长度 %zu != nu %ld\n",
                         opt.tau_max_array.size(), static_cast<long>(m->nu));
            return 1;
        }
        tau_limits = opt.tau_max_array;
        tau_source = "tau_max_array";
    }
    else if (opt.tau_max > 0.0)
    {
        std::fill(tau_limits.begin(), tau_limits.end(), opt.tau_max);
        tau_source = "tau_max";
    }
    else
    {
        bool all_torque = true;
        for (int i = 0; i < m->nu; ++i)
        {
            if (m->actuator_biastype[i] != mjBIAS_NONE)
            {
                all_torque = false;
                continue;
            }
            const double lo = m->actuator_ctrlrange[2 * i];
            const double hi = m->actuator_ctrlrange[2 * i + 1];
            const double mag = std::max(std::fabs(lo), std::fabs(hi));
            if (mag > 0.0 && std::isfinite(mag))
            {
                tau_limits[static_cast<std::size_t>(i)] = mag;
            }
        }
        tau_source = all_torque ? "model ctrlrange" : "model ctrlrange (mixed actuators)";
    }

    // 四轮世界坐标（部署逐档 smoke / judge 对拍用；审计 F2）。
    const std::array<const char*, 4> wheel_names = {"FL_foot", "FR_foot", "RL_foot", "RR_foot"};
    std::array<int, 4> wheel_body = {-1, -1, -1, -1};
    for (std::size_t i = 0; i < wheel_names.size(); ++i)
    {
        wheel_body[i] = mj_name2id(m, mjOBJ_BODY, wheel_names[i]);
    }
    const int bar_body = mj_name2id(m, mjOBJ_BODY, "height_bar_beam_body");

    std::printf("MuJoCo %s\n场景：%s\n起点：%s（nq=%ld nv=%ld nu=%ld dt=%.4g s）\n",
                mj_versionString(), scene.path.c_str(), opt.start.c_str(), static_cast<long>(m->nq),
                static_cast<long>(m->nv), static_cast<long>(m->nu), m->opt.timestep);
    if (!imu.ok())
    {
        std::fprintf(stderr, "警告：模型里找不到 imu_quat / imu_gyro / imu_acc 传感器，"
                             "/imu 不会发（见 models/black_description.xml 的 <sensor>）\n");
    }

    // 窗口（可选）：GLFW 要求"谁建窗口谁用它"，本函数就在主线程，所以在这里建。
    std::unique_ptr<quadruped::viewer::Window> win;
    // 深度相机的 mjr 上下文只该跟着**模型默认**的 offsamples 走（eval/Python 渲染器就是默认值）。
    // 窗口下面会把 m->vis.quality.offsamples 改成 viewer 的 MSAA 选择，那只对已建好的窗口
    // 上下文有效；深度 Init 之前再改回来，窗口模式的 /depth_cam 数值才与 headless/评测一致
    // （实测 offsamples 4 vs 0 在近地平线行能差到 ~10 m——不是小噪声）。
    const int model_offsamples = m->vis.quality.offsamples;
    if (opt.viewer)
    {
        if (opt.viewer_shadow_size > 0)
        {
            m->vis.quality.shadowsize = opt.viewer_shadow_size; // 必须在建 GL 上下文之前设
        }
        // 阴影正交盒半宽：只对平行光有效；收紧它是"提高阴影有效分辨率"的免费手段，
        // 代价是覆盖范围变小（盒子以**光源的 xy** 为中心）——见 kViewerShadowClip 的注释。
        m->vis.map.shadowclip = opt.viewer_shadow_clip;
        // **超采样与 MSAA 互斥**（本机实测、可复现：两者同开时窗口只剩 HUD、画面全黑）：
        // 走超采样时，离屏采样数与**窗口的 MSAA hint** 都要关——只关离屏那个仍然黑屏，
        // 说明触发点在"多重采样窗口 + 离屏 blit"这条路上。见 UPSTREAM.md §2。
        const int window_msaa = opt.viewer_render_scale > 1 ? 0 : std::max(0, opt.viewer_msaa);
        // 离屏缓冲的多重采样数在建 GL 上下文时被读走（与 shadowsize 同理，晚了没用）
        m->vis.quality.offsamples = window_msaa;
        if (opt.viewer_render_scale > 1 && opt.viewer_msaa > 0)
        {
            std::printf("提示：viewer_render_scale=%d（超采样）已生效，MSAA 自动关掉"
                        "（两者同开在本机驱动上画面全黑，实测见 UPSTREAM.md §2）\n",
                        opt.viewer_render_scale);
        }
        win = std::make_unique<quadruped::viewer::Window>(
            m, "quadruped_ros2 sim_node", kViewerWidth, kViewerHeight, opt.viewer_shadow,
            window_msaa, opt.viewer_render_scale);
        std::printf("视口：阴影=%s（运行中按 S 切换）、阴影贴图 %d×%d、正交盒半宽 %.3f m"
                    "（纹素 %.2f mm，改这两个要重启；成因与实测见 UPSTREAM.md §2）\n",
                    opt.viewer_shadow ? "开" : "关", m->vis.quality.shadowsize,
                    m->vis.quality.shadowsize, m->stat.extent * m->vis.map.shadowclip,
                    2000.0 * m->stat.extent * m->vis.map.shadowclip /
                        std::max(1, m->vis.quality.shadowsize - 2));
        win->SetFollowBase(opt.follow_camera);
        std::printf("相机：%s（Ctrl+F 切换；鼠标左键转、右键平移、滚轮缩放）\n",
                    opt.follow_camera ? "跟随机器人 base（默认）" : "固定 iso 视角");
        std::printf("节点按键：空格 暂停/继续、→ 暂停时单步、Ctrl+R 复位、Ctrl+F 相机跟随/固定、"
                    "Ctrl+Q 或 Esc 退出、-/= 速度（31 档，当前 %.4g%%）；"
                    "HUD 右上角显示当前开着的显示开关\n",
                    kSpeedLevels[0]);
        if (!opt.realtime)
        {
            std::printf("注意：窗口模式下仍按墙钟节流（要全速跑请 viewer:=false）\n");
        }
    }
    // 深度相机：必须**在窗口建好之后**初始化——有窗口就复用 GLFW 已经 make current 的上下文
    // （同一线程，没必要再开第二个），无窗口才自建 EGL 离屏上下文（见 depth_camera.hpp）。
    // 失败只影响 /depth_cam：打印一条警告后照常跑，盲走策略与物理完全不受影响。
    m->vis.quality.offsamples = model_offsamples; // 见窗口前 model_offsamples 的注释
    quadruped::depth::DepthCamera depth_camera;
    bool depth_on = false;
    if (opt.depth != "off" && depth_camera_id >= 0)
    {
        if (depth_camera.Init(m, win != nullptr))
        {
            depth_on = true;
            std::printf("深度相机：开（%dx%d，每 %d 步渲一帧 = %.0f Hz；"
                        "先于同一步的 /motor_state 发布，同一 d->time）\n",
                        quadruped::depth::DepthCamera::kWidth,
                        quadruped::depth::DepthCamera::kHeight, opt.depth_every,
                        1.0 / (dt * static_cast<double>(opt.depth_every)));
        }
        else
        {
            std::fprintf(stderr, "警告：深度相机初始化失败（%s），/depth_cam 不发；"
                                 "仿真与盲走策略照常\n", depth_camera.error().c_str());
        }
    }
    mj_forward(m, d); // 先算一遍派生量，第一帧才不是空的

    // 电机（含非理想项）：全关时 Bank 内部直接走理想公式，行为与改动前逐位一致
    quadruped::motor::Bank motors(
        quadruped::motor::NonIdeal{opt.motor_deadzone, opt.motor_delay_cycles, opt.motor_noise,
                                   static_cast<unsigned>(opt.motor_seed)},
        tau_limits, static_cast<std::size_t>(m->nu));
    std::printf("电机：逐关节力矩上限 [ ");
    for (std::size_t i = 0; i < tau_limits.size(); ++i)
    {
        std::printf("%.2f%s", tau_limits[i], i + 1 == tau_limits.size() ? "" : ", ");
    }
    std::printf(" ] N·m（来源：%s）\n", tau_source.c_str());
    if (motors.ideal())
    {
        std::printf("电机：理想力矩源（死区/延迟/噪声全关）\n");
    }
    else
    {
        std::printf("电机非理想项：死区 %.4g N·m、指令延迟 %d 周期、力矩噪声 %.4g N·m（seed %d）\n",
                    opt.motor_deadzone, opt.motor_delay_cycles, opt.motor_noise, opt.motor_seed);
    }

    long steps = 0;
    long published = 0;
    long last_cmd_step = 0;
    const long timeout_steps =
        std::max(1L, static_cast<long>(opt.command_timeout_ms / (dt * 1000.0)));
    bool warned_stale = false;
    bool paused = false;
    int speed_level = 0; // 100% 起步（下标越大越慢，见 kSpeedLevels）
    double status_sim = d->time;
    double report_wall = WallNow();
    double report_sim = d->time;
    double realtime_factor = 1.0;
    // 有窗口：墙钟与仿真时间的一对锚点 —— "墙钟走到 wall_ref 时，仿真该到 sim_ref"。
    // 不能只用"循环开始以来的墙钟差"：复位保留仿真时间轴（d->time 不清零），一旦把基准重置成
    // "现在"，d->time 就永远大于"已经过的时间"，物理再也不步进 —— 2026-10-06 实测 bug：
    // 手柄按 X 复位后狗卡在 raw 姿势不动（暂停后恢复也会同样卡住）。
    double wall_ref = report_wall;
    double sim_ref = d->time;
    double deadline = report_wall; // 无窗口：deadline pacing 的下一次期限

    // 一步：取指令（含看门狗）→ 电机力矩 → `mj_step` → 反馈 → 状态行。
    // 看门狗按**控制周期数**算（不是墙上时间）：500 Hz 下 200 ms = 100 步；这样
    // `realtime:=false` 的全速回归与实时跑行为一致（实机上驱动板也是数控制周期的）。
    const int nu = m->nu;
    std::vector<float> depth_buf; // 深度帧输出（64x48），复用以避免每帧分配
    int depth_failures = 0;
    const auto step_once = [&]()
    {
        MitCommand cmd;
        const bool got = node->TakeCommand(&cmd);
        if (got)
        {
            last_cmd_step = steps;
        }
        const bool stale = !got || (steps - last_cmd_step) > timeout_steps;
        if (stale)
        {
            cmd = DampingCommand(nu, opt.damp_kd);
            if (!warned_stale)
            {
                warned_stale = true;
                if (got)
                {
                    std::printf(
                        "看门狗：连续 %ld 个控制周期（%.0f ms @ %.0f Hz）没收到 /mit_command，"
                        "退回阻尼模式\n",
                        steps - last_cmd_step, static_cast<double>(opt.command_timeout_ms),
                        1.0 / dt);
                }
                else
                {
                    std::printf("还没收到 /mit_command，按阻尼模式跑\n");
                }
            }
        }
        else
        {
            warned_stale = false;
        }

        std::vector<quadruped::motor::Cmd> cmds(static_cast<std::size_t>(nu));
        std::vector<double> joint_q(static_cast<std::size_t>(nu));
        std::vector<double> joint_dq(static_cast<std::size_t>(nu));
        std::vector<double> tau(static_cast<std::size_t>(nu));
        for (int i = 0; i < nu; ++i)
        {
            const auto k = static_cast<std::size_t>(i);
            cmds[k] = quadruped::motor::Cmd{cmd.tau[k], cmd.kp[k], cmd.q[k], cmd.kd[k], cmd.w[k]};
            joint_q[k] = d->qpos[idx.qpos[k]];
            joint_dq[k] = d->qvel[idx.dof[k]];
        }
        // 每步只调一次：指令延迟靠它推进历史（按关节调会把 N 周期延迟变成 N/12 周期）
        motors.Apply(cmds.data(), joint_q.data(), joint_dq.data(), tau.data(), static_cast<std::size_t>(nu));
        for (int i = 0; i < nu; ++i)
        {
            d->ctrl[i] = tau[static_cast<std::size_t>(i)];
        }
        mj_step(m, d);
        ++steps;

        // 深度帧：与同一步的 /motor_state 用同一个 d->time，且**先于** PublishFeedback 发出——
        // 深度策略的消费端要在反馈回调驱动下一步之前先拿到这一帧（见 DepthFrame.msg 的注释）。
        // 渲染是 64x48 的离屏矩形（实测 ~0.1 ms），不改变物理/发布节奏。
        if (depth_on && steps % opt.depth_every == 0)
        {
            if (depth_camera.Render(m, d, &depth_buf))
            {
                depth_failures = 0;
                node->PublishDepth(depth_buf, d->time);
            }
            else if (++depth_failures >= 3)
            {
                depth_on = false;
                std::fprintf(stderr, "警告：深度渲染连续失败，停用 /depth_cam（仿真继续）\n");
            }
        }

        if (steps % opt.publish_every == 0)
        {
            PublishFeedback(node, d, idx, imu, nu);
            ++published;
        }

        // Optional eval trace (P2-11/P3-09): JSON frames with base pose, foot positions and raw
        // contact geom-name pairs; the eval recorder turns them into judge surface tags. Off by
        // default; `eval_state_every:=10` gives one frame per 20 ms of sim time.
        if (opt.eval_state_every > 0 && steps % opt.eval_state_every == 0)
        {
            std::ostringstream os;
            os << std::setprecision(9);
            os << "{\"t\":" << d->time << ",\"base\":[" << d->qpos[0] << "," << d->qpos[1]
               << "," << d->qpos[2] << "],\"quat\":[" << d->qpos[3] << "," << d->qpos[4]
               << "," << d->qpos[5] << "," << d->qpos[6] << "],\"feet\":{";
            for (std::size_t i = 0; i < wheel_names.size(); ++i)
            {
                if (i) os << ",";
                os << "\"" << wheel_names[i] << "\":[";
                if (wheel_body[i] >= 0)
                {
                    os << d->xpos[3 * wheel_body[i] + 0] << "," << d->xpos[3 * wheel_body[i] + 1]
                       << "," << d->xpos[3 * wheel_body[i] + 2];
                }
                os << "]";
            }
            os << "},\"bar_z\":";
            if (bar_body >= 0)
            {
                os << d->xpos[3 * bar_body + 2];
            }
            else
            {
                os << "null";
            }
            os << ",\"contacts\":[";
            bool first = true;
            for (int c = 0; c < d->ncon; ++c)
            {
                const mjContact& con = d->contact[c];
                const char* g1 = mj_id2name(m, mjOBJ_GEOM, con.geom[0]);
                const char* g2 = mj_id2name(m, mjOBJ_GEOM, con.geom[1]);
                if (g1 == nullptr || g2 == nullptr)
                {
                    continue;
                }
                if (!first) os << ",";
                first = false;
                os << "[\"" << g1 << "\",\"" << g2 << "\"]";
            }
            os << "]";
            os << ",\"cmd_step\":" << last_cmd_step << ",\"steps\":" << steps << ",\"cmd_stale\":" << (stale ? "true" : "false")
               << ",\"cmd\":[";
            for (int i = 0; i < nu; ++i)
            {
                if (i) os << ",";
                os << "[" << cmd.tau[i] << "," << cmd.kp[i] << "," << cmd.kd[i] << ","
                   << cmd.q[i] << "," << cmd.w[i] << "]";
            }
            os << "]";
            os << ",\"q\":[";
            for (int i = 0; i < nu; ++i)
            {
                if (i) os << ",";
                os << d->qpos[idx.qpos[i]];
            }
            os << "],\"dq\":[";
            for (int i = 0; i < nu; ++i)
            {
                if (i) os << ",";
                os << d->qvel[idx.dof[i]];
            }
            os << "],\"tau\":[";
            for (int i = 0; i < nu; ++i)
            {
                if (i) os << ",";
                os << d->actuator_force[i];
            }
            os << "]";
            os << "}";
            node->PublishEvalState(os.str());
        }

        // 状态行（默认每 5 秒仿真时间一行，回归脚本会 grep 它；status_period_s 可改）：基座高度 / 倾角 / 接触点数
        if (opt.status_period_s > 0.0 && d->time - status_sim >= opt.status_period_s)
        {
            status_sim = d->time;
            double tilt = 0.0;
            if (imu.ok())
            {
                tilt = quadruped::attitude::TiltDeg(
                    d->sensordata[imu.quat + 0], d->sensordata[imu.quat + 1],
                    d->sensordata[imu.quat + 2], d->sensordata[imu.quat + 3]);
            }
            std::printf("t=%.3f x=%.3f y=%.3f z=%.4f tilt=%.1f ncon=%d cmd=%s depth=%s", d->time,
                        d->qpos[0], d->qpos[1], d->qpos[2], tilt, d->ncon,
                        (!stale && got) ? "ok" : "damping", depth_on ? "on" : "off");
            if (wheel_body[0] >= 0 && wheel_body[1] >= 0 && wheel_body[2] >= 0 &&
                wheel_body[3] >= 0)
            {
                std::printf(" wx=%.3f,%.3f,%.3f,%.3f wz=%.3f,%.3f,%.3f,%.3f",
                            d->xpos[3 * wheel_body[0] + 0], d->xpos[3 * wheel_body[1] + 0],
                            d->xpos[3 * wheel_body[2] + 0], d->xpos[3 * wheel_body[3] + 0],
                            d->xpos[3 * wheel_body[0] + 2], d->xpos[3 * wheel_body[1] + 2],
                            d->xpos[3 * wheel_body[2] + 2], d->xpos[3 * wheel_body[3] + 2]);
            }
            std::printf("\n");
        }
    };

    // 每 5 s 墙钟打一行：仿真秒 / 实时率 / 步数 / 发出去的反馈条数
    const auto report_throughput = [&]()
    {
        const double wall = WallNow() - report_wall;
        if (wall < 5.0)
        {
            return;
        }
        const double sim_seconds = d->time - report_sim;
        realtime_factor = sim_seconds / wall;
        std::printf("仿真 %.2f s / 墙钟 %.2f s = %.3fx 实时；%ld 步，发反馈 %ld 条\n", sim_seconds,
                    wall, realtime_factor, steps, published);
        report_wall = WallNow();
        report_sim = d->time;
    };

    const auto draw_frame = [&]()
    {
        if (win == nullptr)
        {
            return;
        }
        double tilt = 0.0;
        if (imu.ok())
        {
            tilt = quadruped::attitude::TiltDeg(
                d->sensordata[imu.quat + 0], d->sensordata[imu.quat + 1],
                d->sensordata[imu.quat + 2], d->sensordata[imu.quat + 3]);
        }
        // HUD 一律 ASCII（MuJoCo 内置字体画不出中文，见 viewer.hpp）
        char left[200];
        std::snprintf(left, sizeof(left), "t=%.2fs  x=%.2f y=%.2f z=%.3fm  tilt=%.1fdeg  ncon=%d  steps=%ld",
                      d->time, d->qpos[0], d->qpos[1], d->qpos[2], tilt, d->ncon, steps);
        char right[400];
        const std::string flags = win->FlagsSummary();
        std::snprintf(right, sizeof(right),
                      "%s  %.2fx realtime  %.4g%%\n"
                      "flags: %s\n"
                      "[space]pause [right]step [ctrl+r]reset [ctrl+q]/[esc]quit "
                      "[-/=]speed [f6/f7]frame/label [home]view",
                      paused ? "PAUSED" : "running", realtime_factor, kSpeedLevels[speed_level],
                      flags.c_str());
        win->Draw(m, d, left, right);
    };

    while (!node->quit_requested())
    {
        if (win != nullptr)
        {
            if (!win->Poll())
            {
                break; // 用户点掉窗口
            }
            quadruped::viewer::KeyEvent event;
            while (win->TakeKey(&event))
            {
                // 顺序：Ctrl 组合（节点动作）→ 窗口的显示开关表 → 剩下的节点动作
                if ((event.mods & GLFW_MOD_CONTROL) != 0)
                {
                    if (event.key == GLFW_KEY_R)
                    {
                        node->RequestReset();
                    }
                    else if (event.key == GLFW_KEY_Q)
                    {
                        node->RequestQuit();
                    }
                    else if (event.key == GLFW_KEY_F)
                    {
                        win->SetFollowBase(!win->FollowBase());
                        std::printf("相机：%s\n",
                                    win->FollowBase() ? "跟随机器人 base" : "固定 iso 视角");
                    }
                }
                else if (win->HandleKey(event))
                {
                    // 显示开关/F6/F7/Home：窗口自己消化了，HUD 里能看到变化
                }
                else if (event.key == GLFW_KEY_SPACE)
                {
                    paused = !paused;
                    std::printf("%s\n", paused ? "暂停（空格继续）" : "继续");
                }
                else if (event.key == GLFW_KEY_RIGHT)
                {
                    if (paused) // 官方语义：暂停时单步前进
                    {
                        step_once();
                        draw_frame();
                    }
                }
                else if (event.key == GLFW_KEY_MINUS)
                {
                    if (speed_level < kSpeedLevelsCount - 1)
                    {
                        std::printf("速度：%.4g%%\n", kSpeedLevels[++speed_level]);
                    }
                }
                else if (event.key == GLFW_KEY_EQUAL)
                {
                    if (speed_level > 0)
                    {
                        std::printf("速度：%.4g%%\n", kSpeedLevels[--speed_level]);
                    }
                }
                else if (event.key == GLFW_KEY_Q || event.key == GLFW_KEY_ESCAPE)
                {
                    node->RequestQuit();
                }
            }
            if (node->quit_requested())
            {
                break;
            }
        }

        if (node->TakeReset())
        {
            ResetToStart(opt, m, d);
            std::printf("复位：回到起点（%s），保留仿真时间轴 t=%.3f s\n", opt.start.c_str(),
                        d->time);
            wall_ref = WallNow(); // 重新锚定：d->time 保留，等价于"接着走"
            sim_ref = d->time;
            deadline = wall_ref;
        }

        if (paused)
        {
            draw_frame(); // 暂停时也画：HUD 上能看到 PAUSED
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            wall_ref = WallNow(); // 暂停的这段时间不算"落后"：锚点一起前移
            sim_ref = d->time;
            deadline = wall_ref;
            continue;
        }

        if (win != nullptr)
        {
            const double target =
                sim_ref + (WallNow() - wall_ref) * kSpeedLevels[speed_level] / 100.0;
            int guard = 0;
            while (d->time < target && guard < kMaxCatchupSteps && !node->quit_requested())
            {
                step_once();
                ++guard;
            }
            if (target - d->time > kMaxLagSeconds) // 落后太多：重新对齐，不追这一段
            {
                wall_ref = WallNow();
                sim_ref = d->time;
            }
            report_throughput();
            draw_frame();
        }
        else
        {
            step_once();
            if (opt.realtime)
            {
                deadline += dt;
                const double delay = deadline - WallNow();
                if (delay > 0.0)
                {
                    std::this_thread::sleep_for(std::chrono::duration<double>(delay));
                }
                else
                {
                    deadline = WallNow();
                }
            }
            report_throughput();
        }
    }
    if (!motors.ideal())
    {
        quadruped::motor::PrintStats(motors);
    }
    std::printf("仿真节点退出：%ld 步、发反馈 %ld 条、仿真时间 %.3f s\n", steps, published,
                d->time);
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0); // 交互式运行：日志要立刻可见，别等缓冲区满
    // 两个离线自检都放在 rclcpp::init 之前（不需要 ROS，也免得把未知参数塞给它）：
    //   --self-test                    电机模型的离线自检
    //   --depth-self-test <scene.xml>  EGL 离屏深度渲染自检（供 eval 侧 Python 逐位对拍）
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--self-test")
        {
            return MotorSelfTest();
        }
        if (std::string(argv[i]) == "--depth-self-test")
        {
            if (i + 1 >= argc)
            {
                std::fprintf(stderr, "用法：%s --depth-self-test <scene.xml>\n", argv[0]);
                return 1;
            }
            return DepthSelfTest(argv[i + 1]);
        }
    }
    rclcpp::init(argc, argv);

    std::shared_ptr<SimNode> node;
    try
    {
        node = std::make_shared<SimNode>();
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "参数有问题：%s\n", e.what());
        rclcpp::shutdown();
        return 1;
    }

    // ROS 执行器自己一条线程：只收 /mit_command、答 /sim_reset，不碰 mjData（见文件头）。
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    std::thread ros_thread([&executor] { executor.spin(); });

    const int rc = SimLoop(node.get()); // 主线程：物理 + 渲染（单线程，无锁）

    node->RequestQuit();
    executor.cancel();
    if (ros_thread.joinable())
    {
        ros_thread.join();
    }
    rclcpp::shutdown();
    return rc;
}
