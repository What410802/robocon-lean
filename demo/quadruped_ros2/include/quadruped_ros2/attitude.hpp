/**
 * @file attitude.hpp
 * @brief 由四元数算机身姿态角：控制器（判断是不是翻倒）与仿真节点（状态行）共用。
 *
 * 只用到一个数：**倾角** max(|roll|, |pitch|) [deg]。站着时 ≈ 0，趴着时也不大
 * （趴着是俯仰小、横滚小），只有翻倒才会很大，所以它适合当"狗还好吗"的粗判据。
 */

#pragma once

#include <algorithm>
#include <cmath>

namespace quadruped::attitude
{

constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;

/// 四元数 (w, x, y, z)（MuJoCo 的顺序）→ 倾角 [deg] = max(|roll|, |pitch|)。
inline double TiltDeg(double w, double x, double y, double z)
{
    const double roll = std::atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
    const double sin_pitch = 2.0 * (w * y - z * x);
    const double pitch = std::abs(sin_pitch) >= 1.0
                             ? std::copysign(0.5 * 3.14159265358979323846, sin_pitch)
                             : std::asin(sin_pitch);
    return std::max(std::abs(roll), std::abs(pitch)) * kRadToDeg;
}

} // namespace quadruped::attitude
