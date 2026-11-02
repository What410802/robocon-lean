/**
 * @file control.hpp
 * @brief 控制器核心：阻尼 / 站立两状态机 + MIT 五参数指令的生成。
 *
 * 移植来源：@20260927_motor/cpp/essential_core/src/state.h（第三次培训子任务项一的"核心版"）。
 * 控制律本身一个数没改，只做两处为 ROS 2 拆分所必需的改动：
 *
 *   1. **不再持有 mjModel**：原来状态机启动时查一遍"每个执行器作用在哪根关节上"
 *      （actuator_trnid → jnt_qposadr/jnt_dofadr），现在关节角 / 角速度是随 ROS 消息
 *      （MotorState 的 q、dq）一起送来的，而且**已经是执行器顺序**，那张表整段去掉。
 *   2. **输出从"直接写 d->ctrl"改成"填 5 个数组"**（与 msg/MitCommand.msg 一一对应）：
 *      控制器只发 MIT 五参数，力矩由仿真侧的电机模型自己算。这正是实机的分工
 *      （讲义 §1.2/§2.3：驱动板里封好了 FOC 电流环），也顺带把"两道限幅"里的第一道
 *      （控制程序这一层）交给了电机侧——本任务只有电机侧那一道，见 sim 侧的 motor.hpp。
 *
 * 两状态的含义（讲义 §1.3/§1.4）：
 *   * **阻尼模式**（上电默认）：kp = 0、q = w = tau = 0、kd = kd_damp ⇒ τ = −kd·q̇。
 *     像粘稠液体，甩一下就停，永远不会自己站起来。
 *   * **站立模式**（手柄按 A / 键盘按 S）：位置项目标 q_des 从"切换到站立那一刻的关节角"
 *     用 smoothstep 平滑推到站姿 ⇒ 不管起点是原姿态、半站还是正在往下掉，同一段代码都能
 *     把它连续地带回站姿（不是瞬移）。
 *
 * 时基：所有时间都是**仿真时间**（MotorState.sim_time），不是墙上时间——理由见
 * msg/MotorState.msg 的注释（仿真全速跑与实时跑量出来的轨迹才一致）。
 */

#pragma once

#include <array>
#include <cstddef>

namespace quadruped::control
{

/// 本任务的关节数（模型 nu = 12）；站姿表按它写死，换模型要一起改。
inline constexpr int kNu = 12;

/// 站姿（控制目标）：12 个关节角 [rad]，**按执行器顺序**（FL/FR/RR/RL × hip/thigh/calf）。
///
/// 与 @20260927_motor/cpp/essential_core/src/state.h 的 kStanceQ / kStanceZ **逐位相同**
/// （重搜出来的低站姿：膝 1.689 rad、基座 z 0.400 m）。为什么挑这一组、实测数据、
/// 生成办法见 UPSTREAM.md §2——那里有两条关键的：
/// ① 低站姿离"阻尼模式下正在塌落"的狗更近，上电后任意时刻切站立都站得住；
/// ② 质心再前移 1 cm，四足受力均匀。
/// 沿用同一组数，本任务的数字（起身时间、末态高度）才能直接与第三次培训的 sim 对照。
inline constexpr double kStanceZ = 0.4000802544414247;

/// 大腿 / 膝各写一个名字：数值是 0.55×1.7 − 0.011 与 1.7 − 0.011 **按浮点算出来**的结果
/// （大腿那个比字面量 0.946 高 1 ulp——直接写计算结果，才能与搜索工具生成的站姿逐位相同）。
inline constexpr double kStanceThigh = 0.94600000000000006;
inline constexpr double kStanceCalf = 1.6890000000000001;

inline constexpr std::array<double, kNu> kStanceQ = {
    0.0, kStanceThigh,  -kStanceCalf, 0.0, -kStanceThigh, kStanceCalf,
    0.0, -kStanceThigh, kStanceCalf,  0.0, kStanceThigh,  -kStanceCalf,
};

/// 控制参数：默认值就是第三次培训那套（讲义 §1.4 提到的实机输出侧那一组）。
struct Param
{
    double kp = 80.0;     ///< 站立模式：位置刚度 [N·m/rad]
    double kd = 3.0;      ///< 站立模式：速度刚度 [N·m·s/rad]
    double kd_damp = 0.5; ///< 阻尼模式：阻尼 [N·m·s/rad]
    double ramp = 1.0;    ///< q_des 从"切换那一刻的关节角"推到站姿的时长 [s]
};

/// 状态机的两个状态。上电（构造）即阻尼模式——任务书要求。
enum class State
{
    Damping,
    Standing,
};

/// 状态名（给消息字段与日志用；中文说明留给各节点的日志文案）。
inline const char* Name(State s)
{
    return s == State::Damping ? "damping" : "standing";
}

/// 斜坡插值：u = 0→0、u = 1→1，两端导数为 0（平滑起步、平滑到位）；u 先夹到 [0, 1]。
inline double Smoothstep(double u)
{
    const double x = u < 0.0 ? 0.0 : (u > 1.0 ? 1.0 : u);
    return x * x * (3.0 - 2.0 * x);
}

/// 一条 MIT 指令（关节侧），5 个数组与 msg/MitCommand.msg 的字段一一对应。
struct Command
{
    std::array<double, kNu> kp{};
    std::array<double, kNu> kd{};
    std::array<double, kNu> q{};
    std::array<double, kNu> w{};
    std::array<double, kNu> tau{};
};

/// 两状态机：只由"状态 + 时刻"决定输出，不碰仿真、不碰硬件。
class StateMachine
{
  public:
    explicit StateMachine(const Param& param = Param{},
                          const std::array<double, kNu>& q_stand = kStanceQ)
        : param_(param), target_(q_stand), from_(q_stand)
    {
    }

    State state() const { return state_; }

    /// 在线改参数（控制器节点把它接到 ROS 参数回调上：`ros2 param set` 改完立刻生效）。
    /// 只影响之后的指令：正在走的斜坡不会重算起点，只是从下一帧起用新的增益/斜率。
    void SetParam(const Param& param) { param_ = param; }

    /// 切换请求（手柄按键 / 键盘按下时调）。返回 true 表示状态真的变了
    /// （幂等：重复按同一个键不会重开斜坡）。
    ///
    /// @param q_now 当前 12 个关节角（执行器顺序）：切到站立时它就是斜坡起点——
    ///              "从任意初始位置连续地站起来"里的"任意"就落在这里。
    /// @param t_now 当前仿真时间 [s]。
    bool Request(State s, const std::array<double, kNu>& q_now, double t_now)
    {
        if (s == state_)
        {
            return false;
        }
        state_ = s;
        if (s == State::Standing)
        {
            from_ = q_now;
            ramp_t0_ = t_now;
        }
        return true;
    }

    /// 按当前状态与时刻生成一条指令。
    Command Make(double t_now) const
    {
        Command cmd;
        if (state_ == State::Damping)
        {
            // τ = −kd·q̇：kp / q / w / tau 全 0，只剩阻尼
            cmd.kd.fill(param_.kd_damp);
            return cmd;
        }
        const double a = Smoothstep((t_now - ramp_t0_) / param_.ramp);
        for (int i = 0; i < kNu; ++i)
        {
            const auto k = static_cast<std::size_t>(i);
            cmd.kp[k] = param_.kp;
            cmd.kd[k] = param_.kd;
            cmd.q[k] = from_[k] + (target_[k] - from_[k]) * a; // 目标逐帧插值
        }
        return cmd;
    }

    /// 起身斜坡进度 0..1（阻尼模式恒 0）——回给手柄节点看的 ControlStatus.ramp_progress。
    double RampProgress(double t_now) const
    {
        if (state_ == State::Damping)
        {
            return 0.0;
        }
        return Smoothstep((t_now - ramp_t0_) / param_.ramp);
    }

  private:
    Param param_;
    std::array<double, kNu> target_; ///< 站姿（控制目标）
    std::array<double, kNu> from_;   ///< 切到站立那一刻的关节角（斜坡起点）
    State state_ = State::Damping;   ///< 上电 = 阻尼模式
    double ramp_t0_ = 0.0;           ///< 斜坡开始时刻 [s]（仿真时间）
};

} // namespace quadruped::control
