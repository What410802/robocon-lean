/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 *
 * Modified for @20261007_assignment from rl_sar e5c2f41 policy/go2/fsm.hpp.
 * Change list: see UPSTREAM.md §2.
 */

#ifndef BLACK_FSM_HPP
#define BLACK_FSM_HPP

#include "fsm_core.hpp"
#include "rl_sdk.hpp"

#include "black/knobs.hpp"

#include "input_map.hpp"

#include <algorithm>
#include <iostream>
#include <string>

namespace black_fsm
{

// This robot has one policy config; upstream hardcodes the same literal inside the RL state.
inline constexpr const char *kPolicyConfigName = "himloco";

/// Transitions read the intents resolved from the raw inputs (input_map.hpp) instead of the library's
/// single-value latch. An intent is an edge, so it exists for exactly one control cycle - and only for
/// the state that was active when the key was pressed, which is the rule the old latch lacked.
///
/// Some transitions additionally wait for a condition (GetUp waits for its 3 s ramp), and a user may
/// well press the key a moment *before* that condition is met. So each state accumulates the intents it
/// has seen since it was entered: "pressed while in this state" counts even if the gate was still
/// closed, and the accumulator is cleared on every enter/use - the same contract upstream's sticky
/// latch gave by accident, but explicit and bounded to one state.
struct SeenIntents
{
    input_map::IntentMask mask = 0;

    void Reset() { mask = 0; }
    void Tick() { mask |= input_map::Current().intents; }
    bool Has(input_map::Intent intent) const { return (mask & input_map::Mask(intent)) != 0; }
    /// Edge seen while in this state, or the keys are simply still held (covers "was already holding
    /// the combination when the state was entered").
    bool HasOrHeld(input_map::Intent intent) const { return Has(intent) || input_map::Held(intent); }
};

class RLFSMStatePassive : public RLFSMState
{
public:
    SeenIntents seen_;
    RLFSMStatePassive(RL *rl) : RLFSMState(*rl, "RLFSMStatePassive") {}

    void Enter() override
    {
        seen_.Reset();
        rl.running_percent = 0.0f;
        // Startup check: both files are on disk, so a mismatch is visible before the robot moves.
        WarnIfBaseConfigMismatch(rl.robot_name, kPolicyConfigName);
        std::cout << LOGGER::NOTE << "Entered passive mode. Press '0' (Keyboard) or 'A' (Gamepad) to switch to RLFSMStateGetUp." << std::endl;
    }

    void Run() override
    {
        for (int i = 0; i < rl.params.num_of_dofs; ++i)
        {
            // fsm_command->motor_command.q[i] = fsm_state->motor_state.q[i];
            fsm_command->motor_command.dq[i] = 0;
            fsm_command->motor_command.kp[i] = 0;
            fsm_command->motor_command.kd[i] = GetKnobs(rl.robot_name).kd_passive;
            fsm_command->motor_command.tau[i] = 0;
        }
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        seen_.Tick();
        if (seen_.Has(input_map::Intent::StandUp))
        {
            return "RLFSMStateGetUp";
        }
        return state_name_;
    }
};

class RLFSMStateGetUp : public RLFSMState
{
public:
    SeenIntents seen_;
    RLFSMStateGetUp(RL *rl) : RLFSMState(*rl, "RLFSMStateGetUp") {}

    float pre_running_percent = 0.0f;
    // Crouch pose reached before standing up (policy order FL / FR / RL / RR).
    // go2 uses the same sign on all four legs; black's right legs are mirrored, and go2's
    // calf -2.65 is outside black's calf range [-2.5, -0.85]. Values from the organizer's
    // black fork of rl_sar (policy/black/fsm.hpp), all within our model's joint ranges.
    std::vector<float> pre_running_pos = {
        0.00,  1.40, -2.20,
        0.00, -1.40,  2.20,
        0.00,  1.40, -2.20,
        0.00, -1.40,  2.20
    };

    void Enter() override
    {
        seen_.Reset();
        pre_running_percent = 0.0f;
        rl.running_percent = 0.0f;
        rl.now_state = *fsm_state;
        rl.start_state = rl.now_state;
    }

    void Run() override
    {
        if (pre_running_percent < 1.0f)
        {
            pre_running_percent += 1.0f / static_cast<float>(GetKnobs(rl.robot_name).getup_pre_cycles);
            pre_running_percent = std::min(pre_running_percent, 1.0f);

            for (int i = 0; i < rl.params.num_of_dofs; ++i)
            {
                fsm_command->motor_command.q[i] = (1 - pre_running_percent) * rl.now_state.motor_state.q[i] + pre_running_percent * pre_running_pos[i];
                fsm_command->motor_command.dq[i] = 0;
                fsm_command->motor_command.kp[i] = rl.params.fixed_kp[0][i].item<double>();
                fsm_command->motor_command.kd[i] = rl.params.fixed_kd[0][i].item<double>();
                fsm_command->motor_command.tau[i] = 0;
            }
            std::cout << "\r\033[K" << std::flush << LOGGER::INFO << "Pre Getting up " << std::fixed << std::setprecision(2) << pre_running_percent * 100.0f << "%" << std::flush;
        }

        if (pre_running_percent == 1 && rl.running_percent < 1.0f)
        {
            rl.running_percent += 1.0f / static_cast<float>(GetKnobs(rl.robot_name).getup_cycles);
            rl.running_percent = std::min(rl.running_percent, 1.0f);

            for (int i = 0; i < rl.params.num_of_dofs; ++i)
            {
                fsm_command->motor_command.q[i] = (1 - rl.running_percent) * pre_running_pos[i] + rl.running_percent * rl.params.default_dof_pos[0][i].item<double>();
                fsm_command->motor_command.dq[i] = 0;
                fsm_command->motor_command.kp[i] = rl.params.fixed_kp[0][i].item<double>();
                fsm_command->motor_command.kd[i] = rl.params.fixed_kd[0][i].item<double>();
                fsm_command->motor_command.tau[i] = 0;
            }
            std::cout << "\r\033[K" << std::flush << LOGGER::INFO << "Getting up " << std::fixed << std::setprecision(2) << rl.running_percent * 100.0f << "%" << std::flush;
        }
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        seen_.Tick();
        if (seen_.Has(input_map::Intent::Damp))
        {
            return "RLFSMStatePassive";
        }
        if (rl.running_percent == 1.0f)
        {
            // After the ramp, a fresh press or simply holding the keys enters RL / lies down. Upstream's
            // sticky latch made "press early and hold" work by accident; here it is an explicit level query
            // (`Held`), used only behind the running_percent == 1.0f gate so it cannot re-fire.
            if (seen_.HasOrHeld(input_map::Intent::EnterRL))
            {
                return "RLFSMStateRL_Locomotion";
            }
            else if (seen_.HasOrHeld(input_map::Intent::LieDown))
            {
                return "RLFSMStateGetDown";
            }
        }
        return state_name_;
    }
};

class RLFSMStateGetDown : public RLFSMState
{
public:
    SeenIntents seen_;
    RLFSMStateGetDown(RL *rl) : RLFSMState(*rl, "RLFSMStateGetDown") {}

    void Enter() override
    {
        seen_.Reset();
        rl.running_percent = 0.0f;
        rl.now_state = *fsm_state;
    }

    void Run() override
    {
        if (rl.running_percent < 1.0f)
        {
            rl.running_percent += 1.0f / static_cast<float>(GetKnobs(rl.robot_name).getdown_cycles);
            rl.running_percent = std::min(rl.running_percent, 1.0f);

            for (int i = 0; i < rl.params.num_of_dofs; ++i)
            {
                fsm_command->motor_command.q[i] = (1 - rl.running_percent) * rl.now_state.motor_state.q[i] + rl.running_percent * rl.start_state.motor_state.q[i];
                fsm_command->motor_command.dq[i] = 0;
                fsm_command->motor_command.kp[i] = rl.params.fixed_kp[0][i].item<double>();
                fsm_command->motor_command.kd[i] = rl.params.fixed_kd[0][i].item<double>();
                fsm_command->motor_command.tau[i] = 0;
            }
            std::cout << "\r\033[K" << std::flush << LOGGER::INFO << "Getting down "<< std::fixed << std::setprecision(2) << rl.running_percent * 100.0f << "%" << std::flush;
        }
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        seen_.Tick();
        if (seen_.Has(input_map::Intent::Damp) || rl.running_percent == 1.0f)
        {
            return "RLFSMStatePassive";
        }
        else if (seen_.Has(input_map::Intent::StandUp))
        {
            return "RLFSMStateGetUp";
        }
        return state_name_;
    }
};

class RLFSMStateRL_Locomotion : public RLFSMState
{
public:
    SeenIntents seen_;
    RLFSMStateRL_Locomotion(RL *rl) : RLFSMState(*rl, "RLFSMStateRL_Locomotion") {}

    void Enter() override
    {
        seen_.Reset();
        rl.episode_length_buf = 0;

        // One-off cue: S1 moved the old 200 Hz "RL Controller ..." status line into RL_Sim's
        // status_period_ms (default 0 = off), so without this line the terminal keeps showing the
        // previous state's "\r" progress line and RL entry looks like it never happened.
        std::cout << std::endl << LOGGER::INFO << "Entered RL_Locomotion; the per-cycle status line is the "
                  << "rl_sim parameter status_period_ms (0 = off, e.g. 500 = one line per 0.5 s)" << std::endl;

        // read params from yaml
        rl.config_name = kPolicyConfigName;
        std::string robot_path = rl.robot_name + "/" + rl.config_name;
        try
        {
            rl.InitRL(robot_path);
            rl.rl_init_done = true;
            // config.yaml was just (re-)read, so compare it against base.yaml every time we enter RL:
            // editing config.yaml and re-entering RL is the only supported way to reload it (base.yaml
            // is read once in the node constructor, so that one needs a restart).
            WarnIfBaseConfigMismatch(rl.robot_name, rl.config_name);
        }
        catch (const std::exception& e)
        {
            std::cout << LOGGER::ERROR << "InitRL() failed: " << e.what() << std::endl;
            rl.rl_init_done = false;
            rl.control.current_keyboard = Input::Keyboard::Num0;
        }

        // pos init
    }

    void Run() override
    {
        torch::Tensor _output_dof_pos, _output_dof_vel;
        if (rl.output_dof_pos_queue.try_pop(_output_dof_pos) && rl.output_dof_vel_queue.try_pop(_output_dof_vel))
        {
            for (int i = 0; i < rl.params.num_of_dofs; ++i)
            {
                if (_output_dof_pos.defined() && _output_dof_pos.numel() > 0)
                {
                    fsm_command->motor_command.q[i] = _output_dof_pos[0][i].item<double>();
                }
                if (_output_dof_vel.defined() && _output_dof_vel.numel() > 0)
                {
                    fsm_command->motor_command.dq[i] = _output_dof_vel[0][i].item<double>();
                }
                fsm_command->motor_command.kp[i] = rl.params.rl_kp[0][i].item<double>();
                fsm_command->motor_command.kd[i] = rl.params.rl_kd[0][i].item<double>();
                fsm_command->motor_command.tau[i] = 0;
            }
        }
    }

    void Exit() override
    {
        rl.rl_init_done = false;
    }

    std::string CheckChange() override
    {
        seen_.Tick();
        if (seen_.Has(input_map::Intent::Damp))
        {
            return "RLFSMStatePassive";
        }
        else if (seen_.Has(input_map::Intent::LieDown))
        {
            return "RLFSMStateGetDown";
        }
        else if (seen_.Has(input_map::Intent::StandUp))
        {
            return "RLFSMStateGetUp";
        }
        return state_name_;
    }
};

} // namespace black_fsm

class BlackFSMFactory : public FSMFactory
{
public:
    BlackFSMFactory(const std::string& initial) : initial_state_(initial) {}
    std::shared_ptr<FSMState> CreateState(void *context, const std::string &state_name) override
    {
        RL *rl = static_cast<RL *>(context);
        if (state_name == "RLFSMStatePassive")
            return std::make_shared<black_fsm::RLFSMStatePassive>(rl);
        else if (state_name == "RLFSMStateGetUp")
            return std::make_shared<black_fsm::RLFSMStateGetUp>(rl);
        else if (state_name == "RLFSMStateGetDown")
            return std::make_shared<black_fsm::RLFSMStateGetDown>(rl);
        else if (state_name == "RLFSMStateRL_Locomotion")
            return std::make_shared<black_fsm::RLFSMStateRL_Locomotion>(rl);
        return nullptr;
    }
    std::string GetType() const override { return "black"; }
    std::vector<std::string> GetSupportedStates() const override
    {
        return {
            "RLFSMStatePassive",
            "RLFSMStateGetUp",
            "RLFSMStateGetDown",
            "RLFSMStateRL_Locomotion"
        };
    }
    std::string GetInitialState() const override { return initial_state_; }
private:
    std::string initial_state_;
};

REGISTER_FSM_FACTORY(BlackFSMFactory, "RLFSMStatePassive")

#endif // BLACK_FSM_HPP
