/**
 * @file controller_node.cpp
 * @brief 控制器节点（C++）：订阅 /motor_state + /imu + /joy，发布 /mit_command + /control_status。
 *
 * 数据流（任务第 2 项与第 3 项的接缝）：
 *
 *     /mit_command  →  仿真节点（电机 + 机器人）  →  /motor_state + /imu
 *          ↑                                              │
 *          └──────────── 控制器节点 ←────────────────────┘
 *                            ↑
 *                          /joy ← 手柄节点（A 站立 / B 阻尼 / X 复位）
 *
 * 控制周期 = 物理步长：**每收到一条 /motor_state 就回一条 /mit_command**，不自己起定时器。
 * 这样"控制周期"就是仿真步长（0.002 s = 500 Hz，与实机主控周期同量级），
 * 也不需要引入第二个时钟；丢一条状态就少发一条指令，仿真侧照旧保持上一条。
 * 回程的 /control_status 只给手柄节点看（模式 / 斜坡进度 / 倾角 / 指令计数）。
 *
 * 时基：起身斜坡用 **MotorState.sim_time**（仿真时间），不是墙上时间——理由见 MotorState.msg。
 */

#include "quadruped_ros2/attitude.hpp"
#include "quadruped_ros2/control.hpp"

#include "quadruped_ros2/msg/control_status.hpp"
#include "quadruped_ros2/msg/mit_command.hpp"
#include "quadruped_ros2/msg/motor_state.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <std_srvs/srv/empty.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

namespace ctrl = quadruped::control;

using quadruped_ros2::msg::ControlStatus;
using quadruped_ros2::msg::MitCommand;
using quadruped_ros2::msg::MotorState;
using sensor_msgs::msg::Imu;
using sensor_msgs::msg::Joy;
using std_srvs::srv::Empty;

/// 控制回路要的是"现在"，不是"全部"：best_effort + 只留最新一条。
/// reliable 的流控会在订阅者变慢时把发布者顶住（仿真侧是物理线程，顶住就会掉实时），
/// 而 best_effort 正好也是传感器流的语义。`ros2 topic echo` 的默认 QoS 就是 sensor_data
/// （best_effort），所以命令行看这几条话题开箱可用；反过来显式加 `--qos-reliability reliable`
/// 会订不上（实测过）。
rclcpp::QoS LatestQoS()
{
    return rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
}

class ControllerNode : public rclcpp::Node
{
  public:
    ControllerNode() : Node("controller_node")
    {
        // ---- 参数：默认值就是第三次培训那套（讲义 §1.4 的实机输出侧一组）----
        ctrl::Param param;
        param.kp = declare_parameter<double>("kp", param.kp);
        param.kd = declare_parameter<double>("kd", param.kd);
        param.kd_damp = declare_parameter<double>("kd_damp", param.kd_damp);
        param.ramp = declare_parameter<double>("ramp", param.ramp);
        button_stand_ = declare_parameter<int>("button_stand", 0);
        button_damp_ = declare_parameter<int>("button_damp", 1);
        button_reset_ = declare_parameter<int>("button_reset", 2);
        tilt_warn_deg_ = declare_parameter<double>("tilt_warn_deg", 60.0);
        const auto imu_log_s = declare_parameter<double>("imu_log_period_s", 0.0);
        const auto status_ms = declare_parameter<int>("status_period_ms", 200);
        if (imu_log_s < 0.0)
        {
            throw std::invalid_argument("参数 imu_log_period_s 不能为负（0 = 不打 IMU 日志）");
        }
        // RCLCPP_*_THROTTLE 的周期单位是 ms；>0 但不足 1 ms 的写法夹到 1 ms，避免静默失效
        imu_log_period_ms_ = imu_log_s > 0.0 ? std::max(1, static_cast<int>(imu_log_s * 1000.0)) : 0;
        param_ = param;
        sm_ = std::make_unique<ctrl::StateMachine>(param);

        command_pub_ = create_publisher<MitCommand>("mit_command", LatestQoS());
        status_pub_ = create_publisher<ControlStatus>("control_status", rclcpp::QoS(1));

        state_sub_ = create_subscription<MotorState>(
            "motor_state", LatestQoS(), [this](MotorState::SharedPtr msg) { OnMotorState(msg); });
        imu_sub_ = create_subscription<Imu>("imu", LatestQoS(),
                                            [this](Imu::SharedPtr msg) { OnImu(msg); });
        // /joy 用 reliable：它是低频的"命令"而不是传感器流，reliable 也让
        // `ros2 topic echo /joy` 开箱可用（手柄节点与它匹配）。
        joy_sub_ = create_subscription<Joy>("joy", rclcpp::QoS(1),
                                            [this](Joy::SharedPtr msg) { OnJoy(msg); });

        reset_client_ = create_client<Empty>("sim_reset");
        status_timer_ =
            create_wall_timer(std::chrono::milliseconds(status_ms), [this] { PublishStatus(); });

        // 参数在线修改：验收时常用"把 kp 调大看起身更硬""把 ramp 调长看慢起身"，
        // 所以注册回调，让这 4 个控制参数改完立刻生效（`ros2 param set /controller_node ramp 8`）；
        // 按键映射也能改，方便换手柄或改键位。
        param_callback_ = add_on_set_parameters_callback(
            [this](const std::vector<rclcpp::Parameter>& params)
            {
                rcl_interfaces::msg::SetParametersResult result;
                result.successful = true;
                for (const auto& p : params)
                {
                    if (p.get_name() == "kp")
                    {
                        param_.kp = p.as_double();
                    }
                    else if (p.get_name() == "kd")
                    {
                        param_.kd = p.as_double();
                    }
                    else if (p.get_name() == "kd_damp")
                    {
                        param_.kd_damp = p.as_double();
                    }
                    else if (p.get_name() == "ramp")
                    {
                        param_.ramp = p.as_double();
                    }
                    else if (p.get_name() == "button_stand")
                    {
                        button_stand_ = static_cast<int>(p.as_int());
                    }
                    else if (p.get_name() == "button_damp")
                    {
                        button_damp_ = static_cast<int>(p.as_int());
                    }
                    else if (p.get_name() == "button_reset")
                    {
                        button_reset_ = static_cast<int>(p.as_int());
                    }
                }
                sm_->SetParam(param_);
                RCLCPP_INFO(get_logger(),
                            "控制参数已更新：kp=%.4g kd=%.4g kd_damp=%.4g 斜坡=%.3g s", param_.kp,
                            param_.kd, param_.kd_damp, param_.ramp);
                return result;
            });

        RCLCPP_INFO(get_logger(),
                    "控制器节点：kp=%.4g kd=%.4g kd_damp=%.4g 斜坡=%.3g s；站姿基座 z=%.4f m"
                    "（12 个关节角内联在 control.hpp）；上电状态 = 阻尼模式",
                    param.kp, param.kd, param.kd_damp, param.ramp, ctrl::kStanceZ);
        RCLCPP_INFO(get_logger(),
                    "手柄按键：A（buttons[%d]）= 站立、B（buttons[%d]）= 阻尼、X（buttons[%d]）="
                    " 复位；每收到一条 /motor_state 回一条 /mit_command",
                    button_stand_, button_damp_, button_reset_);
    }

  private:
    // ------------------------------------------------------------ 闭环
    /// 收到电机反馈 → 状态机生成 MIT 五参数 → 发回去。
    void OnMotorState(const MotorState::SharedPtr& msg)
    {
        if (msg->q.size() != static_cast<std::size_t>(ctrl::kNu))
        {
            RCLCPP_WARN_ONCE(get_logger(),
                             "收到的 /motor_state 有 %zu 个关节，本控制器按 %d 个写死，忽略",
                             msg->q.size(), ctrl::kNu);
            return;
        }
        for (std::size_t i = 0; i < static_cast<std::size_t>(ctrl::kNu); ++i) { last_q_[i] = msg->q[i]; }
        sim_time_ = msg->sim_time;

        const ctrl::Command cmd = sm_->Make(sim_time_);
        MitCommand out;
        out.header.stamp = now();
        out.kp.assign(cmd.kp.begin(), cmd.kp.end());
        out.kd.assign(cmd.kd.begin(), cmd.kd.end());
        out.q.assign(cmd.q.begin(), cmd.q.end());
        out.w.assign(cmd.w.begin(), cmd.w.end());
        out.tau.assign(cmd.tau.begin(), cmd.tau.end());
        command_pub_->publish(out);
        ++command_count_;
    }

    /// `/imu` 是任务书要求的反馈项（仿真节点 → 控制器），控制律**不用**它。这个回调只做两件事，
    /// 都不改变控制输出（见 UPSTREAM.md §2）：
    ///   * 安全告警：倾角 > `tilt_warn_deg` 打 WARN（`tilt_warn_deg <= 0` = 关掉）；
    ///   * 可选低频打印：`imu_log_period_s > 0` 时按该周期打一行，用来在验收现场证明这条链路在传数据。
    void OnImu(const Imu::SharedPtr& msg)
    {
        const double tilt_deg = quadruped::attitude::TiltDeg(
            msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z);
        if (tilt_warn_deg_ > 0.0 && tilt_deg > tilt_warn_deg_)
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                                 "机身倾角 %.1f° 超过 %.1f°：狗可能已经翻倒（纯 PD 不会自己翻身）",
                                 tilt_deg, tilt_warn_deg_);
        }
        if (imu_log_period_ms_ > 0)
        {
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), imu_log_period_ms_,
                                 "IMU 回传：倾角=%.1f°（/imu 500 Hz 在传，控制律不用它）", tilt_deg);
        }
    }

    // ------------------------------------------------------------ 手柄
    void OnJoy(const Joy::SharedPtr& msg)
    {
        // "没连上"的帧不动作：手柄节点没找到设备时发的就是它（frame_id = joy_disconnected），
        // 主办者仓库的 gateway 也有同样的护栏（`joy_require_connection_frame`）。
        // 这样"手柄还没插好"时的杂散帧不会把状态机带走。
        static const std::string kDisconnected = "joy_disconnected";
        if (msg->header.frame_id.rfind(kDisconnected, 0) == 0)
        {
            RCLCPP_WARN_ONCE(get_logger(), "收到 joy_disconnected 的 /joy：手柄还没连上，忽略按键");
            return;
        }

        const bool stand = Button(*msg, button_stand_);
        const bool damp = Button(*msg, button_damp_);
        const bool reset = Button(*msg, button_reset_);

        // 只认"按下沿"：按住不放不会反复触发（与上一版键盘的 S/D/R 一致）
        if (stand && !prev_stand_)
        {
            RequestStand();
        }
        if (damp && !prev_damp_)
        {
            RequestDamp();
        }
        if (reset && !prev_reset_)
        {
            RequestReset();
        }
        prev_stand_ = stand;
        prev_damp_ = damp;
        prev_reset_ = reset;
    }

    static bool Button(const Joy& msg, int index)
    {
        return index >= 0 && static_cast<std::size_t>(index) < msg.buttons.size() &&
               msg.buttons[static_cast<std::size_t>(index)] != 0;
    }

    void RequestStand()
    {
        if (!sm_->Request(ctrl::State::Standing, last_q_, sim_time_))
        {
            return; // 已经在站立模式：幂等，不重开斜坡
        }
        RCLCPP_INFO(get_logger(),
                    "手柄 A → 站立模式：斜坡 %.3g s，从当前姿态推到站姿（基座 z=%.4f m）",
                    param_.ramp, ctrl::kStanceZ);
    }

    void RequestDamp()
    {
        if (!sm_->Request(ctrl::State::Damping, last_q_, sim_time_))
        {
            return;
        }
        RCLCPP_INFO(get_logger(), "手柄 B → 阻尼模式：全部 kp=0、kd=kd_damp，狗在重力下自己塌回去");
    }

    void RequestReset()
    {
        sm_->Request(ctrl::State::Damping, last_q_, sim_time_);
        if (reset_client_->service_is_ready())
        {
            reset_client_->async_send_request(std::make_shared<Empty::Request>());
            RCLCPP_INFO(get_logger(), "手柄 X → 请求仿真复位（/sim_reset），并切回阻尼模式");
        }
        else
        {
            RCLCPP_WARN(get_logger(),
                        "手柄 X → 想复位，但 /sim_reset 服务不可用（仿真节点没在跑？）");
        }
    }

    // ------------------------------------------------------------ 回程状态
    void PublishStatus()
    {
        ControlStatus status;
        status.header.stamp = now();
        status.mode = ctrl::Name(sm_->state());
        status.ramp_progress = sm_->RampProgress(sim_time_);
        status.command_count = command_count_;
        status_pub_->publish(status);
    }

    std::unique_ptr<ctrl::StateMachine> sm_;
    ctrl::Param param_{};
    rclcpp::Publisher<MitCommand>::SharedPtr command_pub_;
    rclcpp::Publisher<ControlStatus>::SharedPtr status_pub_;
    rclcpp::Subscription<MotorState>::SharedPtr state_sub_;
    rclcpp::Subscription<Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<Joy>::SharedPtr joy_sub_;
    rclcpp::Client<Empty>::SharedPtr reset_client_;
    rclcpp::TimerBase::SharedPtr status_timer_;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_;

    std::array<double, ctrl::kNu> last_q_{}; ///< 最近一次的关节角（切站立时的斜坡起点）
    double sim_time_ = 0.0;                  ///< 最近一次反馈的仿真时间 [s]
    double tilt_warn_deg_ = 60.0; ///< 倾角告警阈值 [deg]；<= 0 = 不告警
    int imu_log_period_ms_ = 0;   ///< /imu 的低频打印周期 [ms]；0 = 不打（验收现场可用 -p 打开）
    uint64_t command_count_ = 0;
    int button_stand_ = 0;
    int button_damp_ = 1;
    int button_reset_ = 2;
    bool prev_stand_ = false;
    bool prev_damp_ = false;
    bool prev_reset_ = false;
};

} // namespace

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    // 参数非法时给一句人话再退出（与 sim_node 同一口径），别让异常直接 abort
    std::shared_ptr<ControllerNode> node;
    try
    {
        node = std::make_shared<ControllerNode>();
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "参数有问题：%s\n", e.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
