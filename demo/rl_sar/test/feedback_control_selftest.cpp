// Exercise actual RL output queues/FSM consumers without ROS or physics.
#include "feedback_control.hpp"
#include "blackW/fsm.hpp"

#include <cstdio>
#include <memory>
#include <vector>

namespace
{
int failures = 0;
void Expect(const char *name, bool ok)
{
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) ++failures;
}

struct Rig : RL
{
    int controls = 0, inferences = 0, publishes = 0;
    std::vector<double> delivered;
    torch::Tensor Forward() override { return torch::zeros({1, 16}); }
    void GetState(RobotState<double> *) override {}
    void SetCommand(const RobotCommand<double> *) override
    {
        ++publishes;
        delivered.push_back(robot_command.motor_command.q[0]);
    }
    Rig(int course)
    {
        part = course;
        robot_name = "blackW";
        params.num_of_dofs = 16;
        params.rl_kp = torch::full({1, 16}, 50.0);
        params.rl_kd = torch::full({1, 16}, 1.2);
        robot_command.motor_command.q.assign(32, 0.0);
        robot_command.motor_command.dq.assign(32, 0.0);
        robot_command.motor_command.kp.assign(32, 0.0);
        robot_command.motor_command.kd.assign(32, 0.0);
        robot_command.motor_command.tau.assign(32, 0.0);
        auto state = std::make_shared<blackw_fsm::RLFSMStateRL_Locomotion>(this);
        state->fsm_state = &robot_state;
        state->fsm_command = &robot_command;
        fsm.current_state_ = state;
        rl_init_done = true;
    }
    void Tick(bool due)
    {
        feedback_control::Step(*this, due,
            [this](bool deferred)
            {
                ++controls;
                fsm.current_state_->Run();
                if (!deferred) SetCommand(&robot_command);
            },
            [this]()
            {
                ++inferences;
                if (rl_init_done)
                {
                    output_dof_pos_queue.push(torch::full({1, 16}, 0.125 * inferences));
                    output_dof_vel_queue.push(torch::full({1, 16}, -0.25 * inferences));
                }
            },
            [this]() { SetCommand(&robot_command); });
    }
};

struct FixedState : FSMState
{
    int runs = 0;
    FixedState() : FSMState("RLFSMStateFixedDrive") {}
    void Enter() override {}
    void Exit() override {}
    void Run() override { ++runs; }
    std::string CheckChange() override { return GetStateName(); }
};
}

int main()
{
    torch::autograd::GradMode::set_enabled(false);
    for (int part : {1, 2})
    {
        Rig r(part);
        for (int tick = 1; tick <= 9; ++tick) r.Tick(tick % 4 == 0);
        Expect(part == 2 ? "Part2 new action reaches the same inference tick"
                         : "Part1 keeps next-tick action delivery",
               r.delivered[3] == (part == 2 ? 0.125 : 0.0)
               && r.delivered[4] == 0.125
               && r.delivered[7] == (part == 2 ? 0.25 : 0.125));
        Expect("one input/control and one output/limiter boundary per tick",
               r.controls == 9 && r.publishes == 9 && r.inferences == 2);
        Expect("paired real FSM queues are drained and gains/velocity preserved",
               r.output_dof_pos_queue.empty() && r.output_dof_vel_queue.empty()
               && r.robot_command.motor_command.kp[0] == 50.0
               && r.robot_command.motor_command.kd[0] > 1.19
               && r.robot_command.motor_command.dq[0] == -0.5);
    }
    Rig fixed(2);
    auto state = std::make_shared<FixedState>();
    fixed.fsm.current_state_ = state;
    fixed.rl_init_done = false;
    fixed.Tick(true);
    Expect("fixed FSM is never run again by the inference flush",
           state->runs == 1 && fixed.controls == 1 && fixed.publishes == 1);

    Rig transitioning(2);
    auto target = std::make_shared<FixedState>();
    feedback_control::Step(transitioning, true,
        [&](bool deferred)
        {
            transitioning.fsm.current_state_ = target;
            transitioning.rl_init_done = false;
            target->Run();
            if (!deferred) transitioning.SetCommand(&transitioning.robot_command);
        }, []() {},
        [&]() { transitioning.SetCommand(&transitioning.robot_command); });
    Expect("a transition out of RL cannot flush the new fixed behavior twice",
           target->runs == 1 && transitioning.publishes == 1);
    std::printf("result: %d failed\n", failures);
    return failures == 0 ? 0 : 1;
}
