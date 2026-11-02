/*
 * New file for @20261007_assignment - no upstream counterpart (package LICENSE covers it).
 *
 * Self-test for include/command_limit.hpp: pure math, no ROS graph, no sim, no hardware.
 * Run it with `ros2 run rl_sar command_limit_selftest` (also registered with CTest).
 */

#include "command_limit.hpp"

#include <cstdio>
#include <vector>

namespace
{

using command_limit::Counts;
using command_limit::Limits;
using command_limit::State;

int g_failures = 0;

void Expect(const char *what, bool ok)
{
    if (!ok)
    {
        ++g_failures;
    }
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
}

void ExpectNear(const char *what, double got, double want, double tol = 1e-9)
{
    Expect(what, std::abs(got - want) <= tol);
    if (std::abs(got - want) > tol)
    {
        std::printf("         got %.6f want %.6f\n", got, want);
    }
}

/// One joint, all the plumbing a single case needs.
struct Rig
{
    std::vector<double> q_des{0.0}, dq_des{0.0}, kp{40.0}, kd{1.2}, tau{0.0};
    State state;
    Limits lim;

    Rig()
    {
        state.q = {0.0};
        state.dq = {0.0};
        state.q_des_prev = {0.0};
        state.has_prev = true;
        // The cases below are written against *these* numbers - do not inherit the header defaults,
        // they are calibrated for the node (33.5 N.m / 60 rad/s) and are allowed to change.
        lim.tau = 20.0;
        lim.slew = 20.0;
    }

    Counts Apply(double dt = 0.005)
    {
        return command_limit::Apply(q_des, dq_des, kp, kd, tau, state, lim, dt);
    }
};

} // namespace

int main()
{
    std::printf("command_limit self-test\n");

    std::printf("1. a sane command passes through untouched\n");
    {
        Rig r;
        // 0.05 rad in one 5 ms cycle = 10 rad/s (under slew 20); error 0.04 rad * kp 40 = 1.6 N.m
        r.q_des = {0.05};
        r.state.q = {0.01};
        r.dq_des = {0.0};
        r.state.dq = {0.0};
        const Counts c = r.Apply();
        ExpectNear("q_des unchanged", r.q_des[0], 0.05);
        Expect("no rule bit", !c.any());
    }

    std::printf("2. gain caps\n");
    {
        Rig r;
        r.kp = {200.0};
        r.kd = {50.0};
        const Counts c = r.Apply();
        ExpectNear("kp clamped to 80", r.kp[0], 80.0);
        ExpectNear("kd clamped to 10", r.kd[0], 10.0);
        Expect("counted as gains", c.gains == 2);
    }

    std::printf("3. dq_des cap\n");
    {
        Rig r;
        r.kd = {0.0};              // kd = 0 so the dq term cannot consume the torque budget
        r.dq_des = {-50.0};
        const Counts c = r.Apply();
        ExpectNear("dq_des clamped to -20", r.dq_des[0], -20.0);
        Expect("counted as dq_des", c.dq_des == 1);
    }
    {
        // The caps are not independent: kd * dq_des is torque. 20 rad/s with kd 1.2 implies 24 N.m,
        // so the torque rule pulls dq_des back to the measured velocity (0 here).
        Rig r;
        r.dq_des = {-20.0};
        const Counts c = r.Apply();
        ExpectNear("torque rule overrides the dq_des cap", r.dq_des[0], 0.0);
        Expect("counted as torque", c.torque >= 1);
    }

    std::printf("4. slew: the target may only move slew*dt per control cycle\n");
    {
        Rig r;                     // prev = 0, slew 20 rad/s, dt 5 ms -> 0.1 rad max
        r.q_des = {1.0};
        r.kp = {0.0};              // isolate the slew rule from the torque rule
        const Counts c = r.Apply();
        ExpectNear("q_des clamped to one step", r.q_des[0], 0.1);
        Expect("counted as slew", c.slew == 1);
    }

    std::printf("5. implied-torque budget: kp*|q_des - q| <= tau_limit\n");
    {
        Rig r;                     // kp 40, tau 20, dq_des == dq -> |q_des - q| <= 0.5 rad
        r.q_des = {10.0};
        const Counts c = r.Apply(1.0);   // dt large: slew (20 rad) must not be the binding rule
        ExpectNear("q_des clamped to q + budget/kp", r.q_des[0], 0.5);
        Expect("counted as torque", c.torque == 1);
    }
    {
        Rig r;                     // 15 N.m of feed-forward leaves 5 N.m for the PD part
        r.tau = {15.0};
        r.q_des = {10.0};
        const Counts c = r.Apply(1.0);
        ExpectNear("budget shrinks with tau_ff", r.q_des[0], 5.0 / 40.0);
        Expect("counted as torque", c.torque == 1);
    }
    {
        Rig r;                     // kd * |dq_des - dq| eats the whole budget -> pure damper
        r.kd = {10.0};
        r.state.dq = {2.0};
        r.dq_des = {-2.0};         // |dq_des - dq| = 4, kd*4 = 40 > 20
        r.q_des = {1.0};
        const Counts c = r.Apply(1.0);
        ExpectNear("q_des falls back to measured q", r.q_des[0], 0.0);
        ExpectNear("dq_des falls back to measured dq", r.dq_des[0], 2.0);
        Expect("counted as torque", c.torque >= 1);
    }
    {
        Rig r;                     // tau_ff itself over the cap: clamp it, then budget the rest
        r.tau = {35.0};
        r.q_des = {1.0};
        const Counts c = r.Apply(1.0);
        ExpectNear("tau_ff clamped to +20", r.tau[0], 20.0);
        ExpectNear("no torque budget left -> q_des = q", r.q_des[0], 0.0);
        Expect("counted as torque", c.torque >= 1);
    }

    std::printf("6. kp = 0 (passive damping) must not divide by zero\n");
    {
        Rig r;
        r.kp = {0.0};
        r.q_des = {5.0};
        const Counts c = r.Apply();
        ExpectNear("q_des only slew-limited", r.q_des[0], 0.1);
        Expect("slew bit", c.slew == 1);
    }

    std::printf("7. optional absolute window\n");
    {
        Rig r;
        r.lim.q_lower = {-0.5};
        r.lim.q_upper = {0.5};
        r.q_des = {3.0};
        r.kp = {0.0};              // isolate: only the window and the slew rule can act
        const Counts c = r.Apply(1.0);
        ExpectNear("q_des clamped into the window", r.q_des[0], 0.5);
        Expect("counted as window", c.window == 1);
    }

    std::printf("8b. repeated control cycles keep the configured slew rate (audit F6)\n");
    {
        Rig r;                     // slew 20 rad/s, dt 5 ms, 4 control cycles per inference
        r.q_des = {10.0};
        r.kp = {0.0};
        for (int i = 0; i < 4; ++i)
        {
            r.q_des = {10.0};      // the node re-copies the policy target every cycle
            r.Apply();
            // ... and keeps the *limited* target as the next cycle's prev
            // (rl_sim SetCommand: limited_q_des_prev = q_des).
            r.state.q_des_prev = r.q_des;
        }
        // 4 cycles * 20 rad/s * 0.005 s = 0.4 rad per 20 ms policy period -> 20 rad/s,
        // i.e. the configured limit is the effective limit (not slew/decimation).
        ExpectNear("allowed change over one policy period", r.q_des[0], 0.4);
    }

    std::printf("8. the whole chain on one joint\n");
    {
        Rig r;                     // worst case: huge target, big tau_ff, big dq_des, big gains
        r.q_des = {100.0};
        r.dq_des = {100.0};
        r.kp = {500.0};
        r.kd = {500.0};
        r.tau = {100.0};
        r.state.dq = {0.0};
        const Counts c = r.Apply();
        const double tau_implied = r.tau[0] + r.kp[0] * (r.q_des[0] - r.state.q[0])
            + r.kd[0] * (r.dq_des[0] - r.state.dq[0]);
        Expect("several rules bit", c.gains >= 1 && c.dq_des == 1 && c.torque >= 1);
        ExpectNear("implied torque within the cap", std::abs(tau_implied), 20.0, 1e-6);
    }

    if (g_failures == 0)
    {
        std::printf("result: all checks passed\n");
        return 0;
    }
    std::printf("result: %d check(s) failed\n", g_failures);
    return 1;
}
