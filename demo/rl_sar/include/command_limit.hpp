/*
 * New file for @20261007_assignment - no upstream counterpart (package LICENSE covers it).
 *
 * Command-side limiting. This node never sends torque: it sends an MIT command
 * {q_des, dq_des, kp, kd, tau_ff} and the motor (or the sim's motor model) turns that into
 *
 *     tau = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)
 *
 * So "limit the torque" has to be expressed on the command, using the measured q/dq we already have
 * from /motor_state. Four rules, applied per joint, in this order (each one is the cheapest way to
 * bound the quantity the rule before it could not):
 *
 *   1. gains:      kp <= kp_max, kd <= kd_max (and >= 0)
 *   2. dq_des:     clamped to +/- dq_max
 *   3. slew:       |q_des - q_des_prev| <= slew * dt          (bounds how fast the implied torque can grow)
 *   4. torque:     the *implied* torque at the measured state must satisfy |tau| <= tau_limit, so the
 *                  position error is budgeted: kp*|q_des - q| <= tau_limit - |tau_ff| - kd*|dq_des - dq|
 *
 * What this does and does not guarantee (be honest about it):
 *   * guaranteed: the torque implied by the command *at the measured state* is within tau_limit - that
 *     is exactly the quantity a driver would compute one control cycle later;
 *   * not guaranteed: what happens if the joint lags behind and the error grows - that is bounded by
 *     rule 3 (the error can only grow as fast as `slew`) and by kp, i.e. the torque can overshoot the
 *     limit by roughly kp * slew * dt per cycle until the next command arrives. With the defaults
 *     (kp = 40, slew = 20 rad/s, dt = 5 ms) that is 4 N.m per cycle - worth knowing, and the reason
 *     `slew` should not be raised carelessly.
 *
 * See UPSTREAM.md §2 for the team-side history; the pre-migration task docs are not shipped in
 * this repo.
 */

#ifndef COMMAND_LIMIT_HPP
#define COMMAND_LIMIT_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace command_limit
{

/// Which rules may bite, and their thresholds. All values come from policy/black/base.yaml.
struct Limits
{
    double tau = 33.5;    ///< implied torque cap [N.m]; 33.5 = the trained envelope (the model's own 20 N.m stays downstream)
    double kp_max = 80.0; ///< gain caps: 80/10 are the largest gains this port ever sends (get-up ramps)
    double kd_max = 10.0;
    double dq_max = 20.0; ///< |dq_des| cap [rad/s]
    double slew = 60.0;   ///< max |q_des| rate [rad/s] (0 = disable); only catches steps
    /// Optional absolute position window (policy order, radians). Empty = not enforced: the MJCF joint
    /// limits are a *model* parameter and are documented, not enforced here (safety-limits.md section 4).
    std::vector<double> q_lower;
    std::vector<double> q_upper;
};

/// Per-joint measured state + the previous command, both in policy order.
struct State
{
    std::vector<double> q;
    std::vector<double> dq;
    std::vector<double> q_des_prev;
    bool has_prev = false;
};

/// How many joints each rule touched in the last call (for the throttled warning / status line).
struct Counts
{
    int gains = 0;
    int dq_des = 0;
    int slew = 0;
    int torque = 0;
    int window = 0;

    bool any() const { return gains + dq_des + slew + torque + window > 0; }
};

/// Clamp one command in place. Returns which rules bit.
inline Counts Apply(std::vector<double> &q_des, std::vector<double> &dq_des,
                    std::vector<double> &kp, std::vector<double> &kd, std::vector<double> &tau_ff,
                    const State &state, const Limits &lim, double dt)
{
    Counts counts;
    const std::size_t n = q_des.size();

    for (std::size_t i = 0; i < n; ++i)
    {
        // 1. gains
        if (kp[i] > lim.kp_max)
        {
            kp[i] = lim.kp_max;
            ++counts.gains;
        }
        if (kd[i] > lim.kd_max)
        {
            kd[i] = lim.kd_max;
            ++counts.gains;
        }
        if (kp[i] < 0.0) kp[i] = 0.0;
        if (kd[i] < 0.0) kd[i] = 0.0;

        // 2. dq_des
        if (std::abs(dq_des[i]) > lim.dq_max)
        {
            dq_des[i] = std::copysign(lim.dq_max, dq_des[i]);
            ++counts.dq_des;
        }

        // 3. slew: how far may the target move in one control cycle?
        if (lim.slew > 0.0 && state.has_prev && i < state.q_des_prev.size())
        {
            const double step = lim.slew * dt;
            const double lo = state.q_des_prev[i] - step;
            const double hi = state.q_des_prev[i] + step;
            if (q_des[i] < lo || q_des[i] > hi)
            {
                q_des[i] = std::clamp(q_des[i], lo, hi);
                ++counts.slew;
            }
        }

        // 4a. absolute window, if the caller provided one. Deliberately applied *after* the slew rule:
        // pulling a target back into the allowed range faster than `slew` is the safe direction.
        if (i < lim.q_lower.size() && i < lim.q_upper.size() && lim.q_lower[i] < lim.q_upper[i])
        {
            const double clamped = std::clamp(q_des[i], lim.q_lower[i], lim.q_upper[i]);
            if (clamped != q_des[i])
            {
                q_des[i] = clamped;
                ++counts.window;
            }
        }

        // 4b. implied-torque budget: |tau_ff| + kp*|q_des - q| + kd*|dq_des - dq| <= tau_limit
        bool touched = false;
        if (std::abs(tau_ff[i]) > lim.tau)
        {
            tau_ff[i] = std::copysign(lim.tau, tau_ff[i]);
            touched = true;
        }
        const double budget = lim.tau - std::abs(tau_ff[i]);
        if (budget <= 0.0)
        {
            // No room left for the PD part: ask for exactly the measured state (a pure damper).
            if (i < state.q.size() && q_des[i] != state.q[i])
            {
                q_des[i] = state.q[i];
                touched = true;
            }
            if (dq_des[i] != 0.0)
            {
                dq_des[i] = 0.0;
                touched = true;
            }
        }
        else if (i < state.dq.size())
        {
            const double pd_budget = budget - kd[i] * std::abs(dq_des[i] - state.dq[i]);
            if (pd_budget <= 0.0)
            {
                if (dq_des[i] != state.dq[i])
                {
                    dq_des[i] = state.dq[i];
                    touched = true;
                }
                if (i < state.q.size() && q_des[i] != state.q[i])
                {
                    q_des[i] = state.q[i];
                    touched = true;
                }
            }
            else if (kp[i] > 0.0 && i < state.q.size())
            {
                const double allowed = pd_budget / kp[i];
                const double lo = state.q[i] - allowed;
                const double hi = state.q[i] + allowed;
                if (q_des[i] < lo || q_des[i] > hi)
                {
                    q_des[i] = std::clamp(q_des[i], lo, hi);
                    touched = true;
                }
            }
        }
        if (touched)
        {
            ++counts.torque;
        }
    }
    return counts;
}

} // namespace command_limit

#endif // COMMAND_LIMIT_HPP
