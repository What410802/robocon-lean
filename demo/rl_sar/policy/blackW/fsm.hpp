/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 *
 * Modified for @20261007_assignment from rl_sar_blackw e5c2f41 policy/go2/fsm.hpp.
 * Change list: see UPSTREAM.md §2.
 */

#ifndef BLACKW_FSM_HPP
#define BLACKW_FSM_HPP

#include "fsm_core.hpp"
#include "rl_sdk.hpp"

#include "../black/knobs.hpp"

#include "input_map.hpp"

#include <algorithm>
#include <iostream>
#include <string>

namespace blackw_fsm
{

// This robot has one policy config; upstream hardcodes the same literal inside the RL state.
inline constexpr const char *kPolicyConfigName = "flat";

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
        black_fsm::WarnIfBaseConfigMismatch(rl.robot_name, kPolicyConfigName);
        std::cout << LOGGER::NOTE << "Entered passive mode. Press '0' (Keyboard) or 'A' (Gamepad) to switch to RLFSMStateGetUp." << std::endl;
    }

    void Run() override
    {
        for (int i = 0; i < rl.params.num_of_dofs; ++i)
        {
            // fsm_command->motor_command.q[i] = fsm_state->motor_state.q[i];
            fsm_command->motor_command.dq[i] = 0;
            fsm_command->motor_command.kp[i] = 0;
            fsm_command->motor_command.kd[i] = black_fsm::GetKnobs(rl.robot_name).kd_passive;
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
    // black fork of rl_sar_blackw (policy/black/fsm.hpp), all within our model's joint ranges.
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
        if (rl.running_percent < 1.0f)
        {
            rl.running_percent += 1.0f / static_cast<float>(black_fsm::GetKnobs(rl.robot_name).getup_cycles);
            rl.running_percent = std::min(rl.running_percent, 1.0f);
            for (int i = 0; i < rl.params.num_of_dofs; ++i)
            {
                fsm_command->motor_command.q[i] =
                    (1 - rl.running_percent) * rl.now_state.motor_state.q[i] +
                    rl.running_percent * rl.params.default_dof_pos[0][i].item<double>();
                fsm_command->motor_command.dq[i] = 0;
                fsm_command->motor_command.kp[i] = rl.params.fixed_kp[0][i].item<double>();
                fsm_command->motor_command.kd[i] = rl.params.fixed_kd[0][i].item<double>();
                fsm_command->motor_command.tau[i] = 0;
            }
            std::cout << "\r\033[K" << std::flush << LOGGER::INFO << "blackW Getting up "
                      << std::fixed << std::setprecision(2) << rl.running_percent * 100.0f << "%"
                      << std::flush;
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
            rl.running_percent += 1.0f / static_cast<float>(black_fsm::GetKnobs(rl.robot_name).getdown_cycles);
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
        rl.config_name = rl.config_name_override.empty() ? kPolicyConfigName : rl.config_name_override;
        std::string robot_path = rl.robot_name + "/" + rl.config_name;
        try
        {
            rl.InitRL(robot_path);
            rl.rl_init_done = true;
            // config.yaml was just (re-)read, so compare it against base.yaml every time we enter RL:
            // editing config.yaml and re-entering RL is the only supported way to reload it (base.yaml
            // is read once in the node constructor, so that one needs a restart).
            black_fsm::WarnIfBaseConfigMismatch(rl.robot_name, rl.config_name);
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
        // Part 2 is RL-only: fixed-action terrain behaviors are Part 1 manual aids only.
        if (rl.part == 1 && seen_.Has(input_map::Intent::LowBarDrive)) return "RLFSMStateLowBarDrive";
        if (rl.part == 1 && seen_.Has(input_map::Intent::BridgeDrive)) return "RLFSMStateBridgeDrive";
        if (rl.part == 1 && seen_.Has(input_map::Intent::CarDrive)) return "RLFSMStateCarDrive";
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


// --------------------------------------------------------------------------- fixed-pose wheel drives
// Part1-only helpers ported from the organizer's blackW low_bar_drive / bridge_drive / car_drive
// (ReadOnly.d/quadruped_control/configs/behaviors/blackW/*.yaml). They are fixed actions, which the
// Part1 rules allow; Part2 is RL-only and must not use them.
struct FixedDriveConfig
{
    std::vector<double> target_positions;
    std::vector<double> kp;
    std::vector<double> kd;
    std::vector<double> wheel_velocity_sign;
    std::vector<int> wheel_sides;  // 0 = left, 1 = right
    int prepare_cycles = 150;
    double max_x = 2.0;
    double max_yaw = 3.0;
    double wheel_velocity_scale = 10.0;
    double yaw_to_wheel_velocity = 5.0;
};

inline FixedDriveConfig LoadFixedDriveConfig(const std::string& robot_name, const std::string& behavior)
{
    FixedDriveConfig cfg;
    const std::string path = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/policy/" + robot_name +
                             "/behaviors/" + behavior + ".yaml";
    const YAML::Node node = YAML::LoadFile(path);
    auto read_vec = [&node](const char* key) {
        std::vector<double> out;
        for (const auto& v : node[key]) out.push_back(v.as<double>());
        return out;
    };
    cfg.target_positions = read_vec("target_joint_positions");
    cfg.kp = read_vec("kp");
    cfg.kd = read_vec("kd");
    cfg.wheel_velocity_sign = read_vec("wheel_velocity_sign");
    cfg.prepare_cycles = node["prepare_cycles"] ? node["prepare_cycles"].as<int>() : 150;
    cfg.max_x = node["max_x"] ? node["max_x"].as<double>() : 2.0;
    cfg.max_yaw = node["max_yaw"] ? node["max_yaw"].as<double>() : 3.0;
    cfg.wheel_velocity_scale = node["wheel_velocity_scale"] ? node["wheel_velocity_scale"].as<double>() : 10.0;
    cfg.yaw_to_wheel_velocity = node["yaw_to_wheel_velocity"] ? node["yaw_to_wheel_velocity"].as<double>() : 5.0;
    for (const auto& v : node["wheel_sides"]) cfg.wheel_sides.push_back(v.as<std::string>() == "left" ? 0 : 1);
    return cfg;
}

inline bool IsWheelIndex(const std::vector<int>& wheels, int i)
{
    return std::find(wheels.begin(), wheels.end(), i) != wheels.end();
}

inline int WheelOrdinal(const std::vector<int>& wheels, int i)
{
    const auto it = std::find(wheels.begin(), wheels.end(), i);
    return it == wheels.end() ? -1 : static_cast<int>(it - wheels.begin());
}

class RLFSMStateFixedDrive : public RLFSMState
{
public:
    SeenIntents seen_;
    RLFSMStateFixedDrive(RL *rl, const std::string& state_name, const std::string& behavior)
        : RLFSMState(*rl, state_name), behavior_(behavior)
    {
        try
        {
            config_ = LoadFixedDriveConfig(rl->robot_name, behavior_);
            loaded_ = config_.target_positions.size() == static_cast<std::size_t>(rl->params.num_of_dofs);
        }
        catch (const std::exception& e)
        {
            std::cout << LOGGER::ERROR << "fixed drive config load failed: " << e.what() << std::endl;
        }
        if (!loaded_)
        {
            std::cout << LOGGER::ERROR << "fixed drive " << behavior_ << " is not available" << std::endl;
        }
    }

    void Enter() override
    {
        seen_.Reset();
        rl.running_percent = 0.0f;
        rl.rl_init_done = false;
        interp_cycles_ = 0;
        prepared_ = false;
        start_q_.assign(rl.params.num_of_dofs, 0.0);
        for (int i = 0; i < rl.params.num_of_dofs; ++i) start_q_[i] = fsm_state->motor_state.q[i];
        std::cout << std::endl << LOGGER::INFO << "Entered " << state_name_ << " (fixed pose wheel drive)" << std::endl;
    }

    void Run() override
    {
        if (!loaded_) return;
        const int n = rl.params.num_of_dofs;
        const std::array<int, 4> fallback_wheels = {3, 7, 11, 15};
        const std::vector<int> wheels = rl.params.wheel_indices.size() == 4 ? rl.params.wheel_indices
                                                                           : std::vector<int>(fallback_wheels.begin(), fallback_wheels.end());
        if (!prepared_)
        {
            interp_cycles_ += 1;
            if (interp_cycles_ >= config_.prepare_cycles) prepared_ = true;
        }
        const double p = prepared_ ? 1.0 : std::min(1.0, static_cast<double>(interp_cycles_) /
                                                            static_cast<double>(std::max(1, config_.prepare_cycles)));

        const double x = std::clamp(rl.control.x, -config_.max_x, config_.max_x);
        const double yaw = std::clamp(rl.control.yaw, -config_.max_yaw, config_.max_yaw);
        const double left = config_.wheel_velocity_scale * x - config_.yaw_to_wheel_velocity * yaw;
        const double right = config_.wheel_velocity_scale * x + config_.yaw_to_wheel_velocity * yaw;

        for (int i = 0; i < n; ++i)
        {
            const int ordinal = WheelOrdinal(wheels, i);
            fsm_command->motor_command.tau[i] = 0.0;
            if (ordinal >= 0)
            {
                const double side = config_.wheel_sides.size() > static_cast<std::size_t>(ordinal) &&
                                            config_.wheel_sides[ordinal] == 1
                                        ? right
                                        : left;
                const double sign = config_.wheel_velocity_sign.size() > static_cast<std::size_t>(ordinal)
                                        ? config_.wheel_velocity_sign[ordinal]
                                        : 1.0;
                fsm_command->motor_command.q[i] = fsm_state->motor_state.q[i];
                fsm_command->motor_command.dq[i] = sign * side;
                fsm_command->motor_command.kp[i] = 0.0;
            }
            else
            {
                const double target = config_.target_positions[i];
                fsm_command->motor_command.q[i] = (1.0 - p) * start_q_[i] + p * target;
                fsm_command->motor_command.dq[i] = 0.0;
                fsm_command->motor_command.kp[i] = config_.kp[i];
            }
            fsm_command->motor_command.kd[i] = config_.kd[i];
        }
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        seen_.Tick();
        if (seen_.Has(input_map::Intent::Damp)) return "RLFSMStatePassive";
        if (seen_.Has(input_map::Intent::EnterRL)) return "RLFSMStateRL_Locomotion";
        if (seen_.Has(input_map::Intent::StandUp)) return "RLFSMStateGetUp";
        if (seen_.Has(input_map::Intent::LieDown)) return "RLFSMStateGetDown";
        return state_name_;
    }

private:
    std::string behavior_;
    FixedDriveConfig config_;
    bool loaded_ = false;
    bool prepared_ = false;
    int interp_cycles_ = 0;
    std::vector<double> start_q_;
};

} // namespace blackw_fsm

class BlackWFSMFactory : public FSMFactory
{
public:
    BlackWFSMFactory(const std::string& initial) : initial_state_(initial) {}
    std::shared_ptr<FSMState> CreateState(void *context, const std::string &state_name) override
    {
        RL *rl = static_cast<RL *>(context);
        if (state_name == "RLFSMStatePassive")
            return std::make_shared<blackw_fsm::RLFSMStatePassive>(rl);
        else if (state_name == "RLFSMStateGetUp")
            return std::make_shared<blackw_fsm::RLFSMStateGetUp>(rl);
        else if (state_name == "RLFSMStateGetDown")
            return std::make_shared<blackw_fsm::RLFSMStateGetDown>(rl);
        else if (state_name == "RLFSMStateRL_Locomotion")
            return std::make_shared<blackw_fsm::RLFSMStateRL_Locomotion>(rl);
        else if (state_name == "RLFSMStateLowBarDrive")
            return std::make_shared<blackw_fsm::RLFSMStateFixedDrive>(rl, state_name, "low_bar_drive");
        else if (state_name == "RLFSMStateBridgeDrive")
            return std::make_shared<blackw_fsm::RLFSMStateFixedDrive>(rl, state_name, "bridge_drive");
        else if (state_name == "RLFSMStateCarDrive")
            return std::make_shared<blackw_fsm::RLFSMStateFixedDrive>(rl, state_name, "car_drive");
        return nullptr;
    }
    std::string GetType() const override { return "blackW"; }
    std::vector<std::string> GetSupportedStates() const override
    {
        return {
            "RLFSMStatePassive",
            "RLFSMStateGetUp",
            "RLFSMStateGetDown",
            "RLFSMStateRL_Locomotion",
            "RLFSMStateLowBarDrive",
            "RLFSMStateBridgeDrive",
            "RLFSMStateCarDrive"
        };
    }
    std::string GetInitialState() const override { return initial_state_; }
private:
    std::string initial_state_;
};

REGISTER_FSM_FACTORY(BlackWFSMFactory, "RLFSMStatePassive")

#endif // BLACKW_FSM_HPP
