// Real SDK/FSM output tests; no ROS graph, hardware, or physics.
#include "rl_sdk.hpp"
#include "command_limit.hpp"
#include "blackW/fsm.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>

namespace
{
int failures = 0;
void Expect(const char *name, bool ok)
{
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) ++failures;
}
bool Near(double a, double b) { return std::abs(a - b) < 2e-6; }

struct Rig : RL
{
    torch::Tensor Forward() override { return torch::zeros({1, 16}); }
    void GetState(RobotState<double> *state) override { *state = robot_state; }
    void SetCommand(const RobotCommand<double> *) override {}

    Rig()
    {
        robot_name = "blackW";
        config_name_override = "vt_locked12";
        ReadYamlBase(robot_name);
        ReadYamlRL(robot_name + "/vt_locked12");
        auto state = std::make_shared<blackw_fsm::RLFSMStateRL_Locomotion>(this);
        state->fsm_state = &robot_state;
        state->fsm_command = &robot_command;
        fsm.current_state_ = state;
        for (int i : params.wheel_indices)
            robot_state.motor_state.q[i] = 12.125 - i * 1.75;
        InitWheelLock();
        InitObservations();
        InitOutputs();
        rl_init_done = true;
    }

    // Same SDK override + limiter boundary used by RL_Sim::SetCommand.
    void OutputBoundary()
    {
        auto &c = robot_command.motor_command;
        command_limit::Limits lim;
        if (WheelLockActive())
        {
            ApplyWheelLockTargets(c.q, c.dq, c.kp, c.kd, c.tau);
            lim.position_holds = params.wheel_indices;
        }
        command_limit::State measured;
        measured.q = robot_state.motor_state.q;
        measured.dq = robot_state.motor_state.dq;
        measured.has_prev = true;
        measured.q_des_prev.assign(32, 0.0);
        command_limit::Apply(c.q, c.dq, c.kp, c.kd, c.tau, measured, lim, params.dt);
    }
};
}

int main()
{
    torch::autograd::GradMode::set_enabled(false);
    Rig r;
    const auto initial = r.locked_wheel_q;
    bool init_ok = true;
    for (int i : r.params.wheel_indices)
        init_ok &= Near(r.output_dof_pos[0][i].item<double>(), initial[i])
            && r.output_dof_vel[0][i].item<double>() == 0.0
            && r.params.default_dof_pos[0][i].item<double>() == 0.0;
    Expect("startup captures distinct unwrapped angles without changing defaults", init_ok);

    auto &c = r.robot_command.motor_command;
    c.q.assign(32, -50.0);
    c.dq.assign(32, 8.0);
    c.kp.assign(32, 0.0);
    c.kd.assign(32, 1.0);
    c.tau.assign(32, 5.0);
    r.ApplyWheelLockTargets(c.q, c.dq, c.kp, c.kd, c.tau);
    bool boundary_ok = c.q[0] == -50.0 && c.dq[0] == 8.0 && c.tau[0] == 5.0;
    for (int i : r.params.wheel_indices)
        boundary_ok &= c.q[i] == initial[i] && c.dq[i] == 0.0 && c.kp[i] == 80.0
            && c.kd[i] == 3.0 && c.tau[i] == 0.0;
    Expect("first command overrides stale wheel velocity and preserves leg commands", boundary_ok);

    bool held = true;
    for (int tick = 0; tick < 120; ++tick)
    {
        r.obs.dof_pos = torch::full({1, 16}, 20.0 + tick);
        r.obs.dof_vel = torch::full({1, 16}, -30.0);
        for (int i : r.params.wheel_indices)
        {
            r.robot_state.motor_state.q[i] += 0.1;
            r.robot_state.motor_state.dq[i] = -40.0;
        }
        torch::Tensor q, dq, tau;
        r.ComputeOutput(torch::full({1, 16}, 0.2 * tick - 10.0), q, dq, tau);
        r.OutputBoundary();
        for (int i : r.params.wheel_indices)
        {
            held &= Near(q[0][i].item<double>(), initial[i]) && dq[0][i].item<double>() == 0.0;
            held &= c.q[i] == initial[i] && c.dq[i] == 0.0 && r.locked_wheel_q[i] == initial[i];
            const double effort = c.kp[i] * std::abs(c.q[i] - r.robot_state.motor_state.q[i])
                                + c.kd[i] * std::abs(r.robot_state.motor_state.dq[i]);
            held &= effort <= 33.5 + 1e-8;
        }
    }
    Expect("120 inference/output cycles keep one capture and bounded PD torque", held);
    r.output_dof_pos_queue.push(torch::ones({1, 16}));
    r.output_dof_vel_queue.push(torch::ones({1, 16}));
    r.output_dof_tau_queue.push(torch::ones({1, 16}));
    r.InitOutputs();
    Expect("output reset clears old queues without recapturing", r.output_dof_pos_queue.empty()
        && r.output_dof_vel_queue.empty() && r.output_dof_tau_queue.empty() && r.locked_wheel_q == initial);

    r.ReadYamlRL("blackW/row2p_p40w40_v1");
    r.InitWheelLock();
    r.InitObservations();
    torch::Tensor q, dq, tau;
    r.ComputeOutput(torch::full({1, 16}, 0.7), q, dq, tau);
    bool unlocked = !r.params.lock_wheels && !r.locked_wheel_ready && !r.WheelLockActive();
    for (int i : r.params.wheel_indices)
        unlocked &= Near(dq[0][i].item<double>(), 0.7 * r.params.action_scale[0][i].item<double>());
    Expect("switch to 16DoF preserves signed scaled wheel velocity", unlocked);

    r.ReadYamlRL("blackW/vt_locked12");
    r.InitWheelLock();
    const auto next = r.robot_state.motor_state.q;
    r.InitObservations();
    r.OutputBoundary();
    bool recaptured = r.locked_wheel_q == next && next != initial;
    for (int i : r.params.wheel_indices) recaptured &= c.q[i] == next[i] && c.dq[i] == 0.0;
    Expect("switch back captures current angles once", recaptured);

    for (const auto &behavior : {"low_bar_drive", "bridge_drive", "car_drive"})
    {
        auto fixed = std::make_shared<blackw_fsm::RLFSMStateFixedDrive>(&r, behavior, behavior);
        fixed->fsm_state = &r.robot_state;
        fixed->fsm_command = &r.robot_command;
        r.fsm.current_state_ = fixed;
        fixed->Enter();
        r.control.x = 0.4;
        fixed->Run();
        const auto desired = c.dq;
        // Even a stale rl_init_done must not let an RL hold overwrite a fixed action.
        r.rl_init_done = true;
        r.ApplyWheelLockTargets(c.q, c.dq, c.kp, c.kd, c.tau);
        bool fixed_ok = !r.WheelLockActive() && c.dq == desired;
        for (int i : r.params.wheel_indices) fixed_ok &= c.kp[i] == 0.0 && std::abs(c.dq[i]) > 0.0;
        Expect(behavior, fixed_ok);
    }

    // Exercise actual FSM Enter/Exit including InitRL model loading when deployment weights exist.
    const std::string model = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/policy/blackW/vt_locked12/policy.pt";
    if (std::filesystem::exists(model))
    {
        auto locomotion = std::make_shared<blackw_fsm::RLFSMStateRL_Locomotion>(&r);
        r.fsm.current_state_ = locomotion;
        r.robot_state.motor_state.q[3] = 31.125;
        locomotion->Enter();
        const bool entered = r.WheelLockActive() && r.locked_wheel_q[3] == 31.125;
        locomotion->Exit();
        const bool exited = !r.WheelLockActive();
        r.robot_state.motor_state.q[3] = -45.375;
        locomotion->Enter();
        Expect("real FSM same-policy re-entry captures once and exit disables hold", entered && exited
            && r.WheelLockActive() && r.locked_wheel_q[3] == -45.375);
    }
    else std::printf("  [SKIP] FSM model-loading integration (deployment weights absent)\n");

    r.robot_state.motor_state.q[3] = std::numeric_limits<double>::quiet_NaN();
    bool rejected = false;
    try { r.InitWheelLock(); }
    catch (const std::runtime_error &) { rejected = true; }
    Expect("nonfinite capture fails without marking wheel hold ready", rejected && !r.locked_wheel_ready);
    std::printf("result: %d failed\n", failures);
    return failures == 0 ? 0 : 1;
}
