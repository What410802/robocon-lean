/**
 * @file motor.hpp
 * @brief 仿真侧的"关节电机"：把 MIT 五参数变成关节力矩。
 *
 * 分工上这是**实机驱动板**那一侧：控制器节点只发 {kp, kd, q, w, tau}（msg/MitCommand.msg），
 * 电机（这里就是本文件）自己算
 *
 *     τ = τ_ff + kp·(q_des − q) + kd·(w_des − q̇)
 *
 * 再把结果按电机能力限幅，最后写进 MuJoCo 的 `data.ctrl`（本模型的执行器是 `<motor gear="1">`，
 * `ctrl` 就是关节侧力矩，见 models/black_description.xml 的 <actuator> 段）。
 *
 * 与第三次培训完整版（@20260927_motor/cpp/src/motor.h）的关系：**同一套语义**。
 *   * 理想部分：本文件的 `Torque()`，一条 MIT 公式 + 一道限幅（控制器与算法的"思考语言"）；
 *   * 非理想部分：`Bank`（死区 / 指令延迟 / 力矩噪声 / 统计），参数与计数口径照搬完整版，
 *     这样两个任务的数字可以直接对比（详见 UPSTREAM.md §2）。
 *   * 放在仿真侧而不是控制器侧是刻意的：实机上死区、电流限幅、指令延迟都发生在驱动板，
 *     上层控制器看不见（讲义 §1.1）。
 *
 * 量纲：全部是**关节侧**（输出端）。**本文件不做电流推定**：`MotorState.cur` 一律报 0，
 * 理由见 sim_node.cpp 里那一处注释（我们没有支撑它的电机模型与实测数据）。
 * 转子侧的换算（讲义 §2.3 的 ×N / ÷N / ÷N²）等将来真要做电流/力矩标定时再说，
 * 那时需要的是"电机电气模型 + 实测标定数据"，不是这里加一个除法。
 */

#pragma once

#include "quadruped_ros2/control.hpp" // kNu（关节数，与控制器/站姿表同源）

#include <algorithm>
#include <stdexcept>
#include <string>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace quadruped::motor
{

/// 关节侧力矩上限 [N·m] 的兜底默认值（仅当模型/参数都拿不到逐关节限值时使用）。
/// blackW 的 MJCF 是逐关节 `ctrlrange`（23.7/59.25/20），sim_node 默认从模型读取，
/// 不再用这个标量一刀切（2026-10-26 审计 F1：标量 20 与训练/judge 的 59.25 不等价）。
inline constexpr double kDefaultTauMax = 20.0;

/// 减速比：GO-M8010-6 是 6.33（讲义 §2.2：转子转 19 圈、输出端转 3 圈）。
/// 这是讲义给的真值，但**本任务不拿它算任何东西**——留在这里是为了将来做转子侧换算
/// （力矩标定、电流环建模）时有个出处，不要用它去"推"电流（见上面的说明）。
inline constexpr double kDefaultGear = 6.33;

/// 一条 MIT 指令（单个关节）。
struct Cmd
{
    double tau_ff = 0.0; ///< 前馈力矩 [N·m]
    double kp = 0.0;     ///< 位置刚度 [N·m/rad]
    double q_des = 0.0;  ///< 期望关节角 [rad]
    double kd = 0.0;     ///< 速度刚度 [N·m·s/rad]
    double w_des = 0.0;  ///< 期望关节角速度 [rad/s]
};

/// 一条指令 + 当前状态 → 关节力矩（含电机侧限幅）。
inline double Torque(const Cmd& c, double q, double dq, double tau_max)
{
    const double tau = c.tau_ff + c.kp * (c.q_des - q) + c.kd * (c.w_des - dq);
    return std::clamp(tau, -tau_max, tau_max);
}

/// 非理想项参数（默认全 0 = 理想力矩源，与"没有这个类"逐位一致）
struct NonIdeal
{
    double deadzone = 0.0; ///< 静摩擦死区 [N·m]：0 < |τ| < 死区 → 推不动，输出 0
    int delay_cycles = 0; ///< 指令延迟 [控制周期]：用 N 个周期**之前**下发的那条指令
    double noise = 0.0; ///< 力矩噪声标准差 [N·m]（高斯白噪声，固定种子可复现）
    unsigned seed = 12345; ///< 噪声种子（同一组参数 + 同一 seed → 同一条噪声序列）
};

/// 12 个关节的电机组：有状态（指令延迟要记历史、噪声要记种子、统计要累计）。
///
/// **全关时与纯函数 `Torque()` 逐位一致**：`Torque()` 里第一件事就是走理想分支，
/// 保证"不引入非理想项"的回归结果与改动前一字不差。
///
/// 三个非理想项的语义与完整版（@20260927_motor/cpp/src/motor.h 的 `JointMotors`）一致：
///   * 死区：`0 < |τ| < deadzone` 时把力矩当 0（**正好 0 不算落死区**，否则阻尼模式下
///     狗停住时会报出成千上万次假计数 —— 完整版踩过这个坑，见其 §4 踩坑 15）；
///   * 延迟：延迟的是**指令**（驱动板收到得晚），反馈 q/dq 仍是当前值；
///   * 噪声：加在**输出力矩**上（原型里注释写的是"编码器/电流环噪声"），固定种子。
class Bank
{
  public:
    Bank() = default;
    Bank(const NonIdeal& cfg, double tau_max,
         std::size_t joints = static_cast<std::size_t>(quadruped::control::kNu))
        : cfg_(cfg), width_(joints), tau_max_(joints, tau_max)
    {
        delay_ = std::max(0, cfg_.delay_cycles);
        hist_.assign(width_ * static_cast<std::size_t>(delay_ + 1), Cmd{});
        in_dead_.assign(width_, false);
        rng_.seed(cfg_.seed);
    }

    /// 逐关节力矩上限版本（blackW 用）：tau_max.size() 必须等于 joints。
    Bank(const NonIdeal& cfg, const std::vector<double>& tau_max,
         std::size_t joints = static_cast<std::size_t>(quadruped::control::kNu))
        : cfg_(cfg), width_(joints), tau_max_(tau_max)
    {
        if (tau_max_.size() != width_)
        {
            throw std::invalid_argument(
                "motor::Bank: tau_max size " + std::to_string(tau_max_.size()) +
                " != joints " + std::to_string(width_));
        }
        delay_ = std::max(0, cfg_.delay_cycles);
        hist_.assign(width_ * static_cast<std::size_t>(delay_ + 1), Cmd{});
        in_dead_.assign(width_, false);
        rng_.seed(cfg_.seed);
    }

    /// 有没有开任何非理想项（节点用它决定要不要打统计行）
    bool ideal() const { return cfg_.deadzone <= 0.0 && delay_ <= 0 && cfg_.noise <= 0.0; }

    int delayCycles() const { return delay_; }

    /// **每步调用一次**（指令延迟靠它推进历史，所以不能每个关节调一次）：
    /// 算 12 个关节的力矩 = MIT 公式 → 死区 → 限幅 → 噪声，写进 `tau_out`。
    /// 提示：只推进一次历史是**语义要求**——按关节调用会把 N 周期延迟变成 N/12 周期。
    void Apply(const Cmd* cmds, const double* q, const double* dq, double* tau_out,
               std::size_t width)
    {
        const Cmd* eff = cmds;
        if (delay_ > 0)
        {
            // 与完整版 JointMotors::EffectiveCommand() 同一套环形下标：
            // 把这一步的指令写进"现在"这格，再取 delay_ 格之前的那一份
            const std::size_t slots = static_cast<std::size_t>(delay_ + 1);
            const std::size_t now_slot = hist_step_ % slots;
            for (std::size_t k = 0; k < width; ++k)
            {
                hist_[now_slot * width + k] = cmds[k];
            }
            const std::size_t use_slot =
                (hist_step_ + slots - static_cast<std::size_t>(delay_)) % slots;
            eff = hist_.data() + use_slot * width;
            ++hist_step_;
        }

        for (std::size_t k = 0; k < width; ++k)
        {
            // 理想公式**内部已经限幅**（`clamp(τ, ±tau_max)`），所以这里不再重复限幅，
            // 而是"观察"它有没有打满——语义与完整版的"撞限幅"计数一致（恒力矩区的天花板）。
            const double tau_max = tau_max_[k];
            double tau = motor::Torque(eff[k], q[k], dq[k], tau_max);
            if (std::fabs(tau) >= tau_max)
            {
                ++stats_.sat;
            }
            const double mag = std::fabs(tau);
            if (cfg_.deadzone > 0.0)
            {
                // "在死区里"包含 mag == 0（完全没指令），但只有真的被削掉才算数
                const bool inside = mag < cfg_.deadzone;
                if (inside && mag > 0.0)
                {
                    tau = 0.0;
                    ++stats_.dead;
                    if (!in_dead_[k])
                    {
                        ++stats_.dead_events;
                    }
                }
                in_dead_[k] = inside;
            }
            // 噪声加在限幅**之后**（完整版也是这个顺序）：量纲上是力矩纹波，
            // 所以输出可以略微超过 tau_max——这是刻意的，别"顺手修掉"。
            if (cfg_.noise > 0.0)
            {
                tau += cfg_.noise * normal_(rng_);
            }
            tau_out[k] = tau;
            ++stats_.steps;
            const double out = std::fabs(tau);
            stats_.tau_peak = std::max(stats_.tau_peak, out);
            stats_.tau_abs_sum += out;
        }
    }

    /// 统计（写进摘要，用来说明"非理想项到底起没起作用"）
    struct Stats
    {
        long steps = 0; ///< 累计"电机·步"
        long sat = 0;   ///< 撞限幅的电机·步
        long dead = 0;  ///< 落死区被当 0 的电机·步（不含本来就正好是 0 的）
        long dead_events = 0;  ///< 进入死区的次数（每个关节一次上升沿算一次）
        double tau_peak = 0.0; ///< |τ| 峰值
        double tau_abs_sum = 0.0;
        double tau_mean_abs() const
        {
            return steps > 0 ? tau_abs_sum / static_cast<double>(steps) : 0.0;
        }
    };
    const Stats& stats() const { return stats_; }

  private:
    NonIdeal cfg_{};
    std::size_t width_ = 0;
    std::vector<double> tau_max_{}; ///< 逐关节力矩上限 [N·m]
    int delay_ = 0;
    std::size_t hist_step_ = 0;
    std::vector<Cmd> hist_;     ///< [步][关节] 摊平存放，长度 = kNu×(delay+1)
    std::vector<bool> in_dead_; ///< 上一拍是否在死区（数"进入死区"的上升沿）
    std::mt19937 rng_{12345};
    std::normal_distribution<double> normal_{0.0, 1.0};
    Stats stats_;
};

/// 一行统计摘要（措辞与完整版一致，两个任务可以横向对比）
inline void PrintStats(const Bank& motors)
{
    const Bank::Stats& st = motors.stats();
    std::printf("电机：累计 %ld 电机·步，撞限幅 %ld 次（%.3f%%），落死区 %ld 电机·步（%.3f%%，"
                "进出死区 %ld 次），|τ| 峰值 %.2f N·m、均值 %.2f N·m（延迟 %d 周期）\n",
                st.steps, st.sat, 100.0 * static_cast<double>(st.sat) / std::max(1L, st.steps),
                st.dead, 100.0 * static_cast<double>(st.dead) / std::max(1L, st.steps),
                st.dead_events, st.tau_peak, st.tau_mean_abs(), motors.delayCycles());
}

} // namespace quadruped::motor
