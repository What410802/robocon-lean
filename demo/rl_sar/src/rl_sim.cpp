/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 *
 * Modified for @20261007_assignment from rl_sar e5c2f41 src/rl_sim.cpp.
 * Change list: see UPSTREAM.md §2.
 */

#include "rl_sim.hpp"
#include <fstream>
#include <iostream>
#include <exception>
#include <cstdlib>
#include <iomanip>

#include "command_limit.hpp"
#include "feedback_control.hpp"
#include "input_map.hpp"

#include <torch/utils.h> // torch::set_num_threads (libtorch >= 2.x no longer pulls it in via torch/script.h)

#include <chrono>
#include <cmath>

namespace
{
// Same QoS as the sim node uses for its 500 Hz topics: keep only the latest sample, best effort.
// A reliable subscriber would never match the sim's best_effort publishers.
rclcpp::QoS LatestQoS()
{
    return rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
}
} // namespace

RL_Sim::RL_Sim()
    : rclcpp::Node("rl_sim_node")
{
    this->ang_vel_type = "ang_vel_body";
    this->robot_name = this->declare_parameter<std::string>("robot_name", "black");
    this->config_name_override = this->declare_parameter<std::string>("policy_name", "flat");
    this->part = this->declare_parameter<int>("part", 1);
    this->joy_command_scale = this->declare_parameter<double>("joy_command_scale", this->joy_command_scale);
    // Status line period in ms; 0 = off. The line used to be printed by the RL state on every control
    // cycle (200 Hz, '\r'-overwritten); it is throttled here instead so the rate is a node parameter.
    this->status_period_ms = this->declare_parameter<int>("status_period_ms", this->status_period_ms);
    std::cout << LOGGER::INFO << "robot_name: " << this->robot_name
              << " part: " << this->part << std::endl;

    // library/core's Control has no initializer for its input enums, so the keyboard latch starts out
    // indeterminate; the keyboard adapter reads it, so make "nothing pressed" explicit. (The gamepad
    // latch is no longer read anywhere: input_map.hpp derives the pad state from the raw /joy frame.)
    this->control.current_keyboard = Input::Keyboard::None;
    this->control.last_keyboard = Input::Keyboard::None;

    // read params from yaml
    this->ReadYamlBase(this->robot_name);
    this->ValidateRobotManifest(this->robot_name, false);

    // auto load FSM by robot_name
    if (FSMManager::GetInstance().IsTypeSupported(this->robot_name))
    {
        auto fsm_ptr = FSMManager::GetInstance().CreateFSM(this->robot_name, this);
        if (fsm_ptr)
        {
            this->fsm = *fsm_ptr;
        }
    }
    else
    {
        std::cout << LOGGER::ERROR << "No FSM registered for robot: " << this->robot_name << std::endl;
    }

    // init torch
    torch::autograd::GradMode::set_enabled(false);
    torch::set_num_threads(4);

    // init robot
    this->InitOutputs();
    this->InitControl();
    // unbounded ROS arrays (12-joint black and 16-joint blackW share the message types)
    const std::size_t n = static_cast<std::size_t>(this->params.num_of_dofs);
    this->mit_command_msg.q.assign(n, 0.0);
    this->mit_command_msg.w.assign(n, 0.0);
    this->mit_command_msg.kp.assign(n, 0.0);
    this->mit_command_msg.kd.assign(n, 0.0);
    this->mit_command_msg.tau.assign(n, 0.0);
    this->last_sent_command.q.assign(n, 0.0);
    this->last_sent_command.w.assign(n, 0.0);
    this->last_sent_command.kp.assign(n, 0.0);
    this->last_sent_command.kd.assign(n, 0.0);
    this->last_sent_command.tau.assign(n, 0.0);

    // publisher
    this->mit_command_publisher = this->create_publisher<quadruped_ros2::msg::MitCommand>("mit_command", LatestQoS());

    // subscriber
    this->cmd_vel_subscriber = this->create_subscription<geometry_msgs::msg::Twist>(
        "cmd_vel", rclcpp::SystemDefaultsQoS(),
        [this] (const geometry_msgs::msg::Twist::SharedPtr msg) {this->CmdvelCallback(msg);}
    );
    // Relative topic names (P2-a): with absolute names a --namespace could move only half of the stack.
    // Without a namespace these still resolve to /joy, /cmd_vel, ... - see UPSTREAM.md §2
    this->joy_subscriber = this->create_subscription<sensor_msgs::msg::Joy>(
        "joy", rclcpp::SystemDefaultsQoS(),
        [this] (const sensor_msgs::msg::Joy::SharedPtr msg) {this->JoyCallback(msg);}
    );
    this->imu_subscriber = this->create_subscription<sensor_msgs::msg::Imu>(
        "imu", LatestQoS(), [this] (const sensor_msgs::msg::Imu::SharedPtr msg) {this->ImuCallback(msg);}
    );
    this->motor_state_subscriber = this->create_subscription<quadruped_ros2::msg::MotorState>(
        "motor_state", LatestQoS(),
        [this] (const quadruped_ros2::msg::MotorState::SharedPtr msg) {this->MotorStateCallback(msg);}
    );
    // Depth camera for depth-conditioned policies (sim_node publishes it; blind
    // policies never read it) and the automation policy-switch hook: the Part2
    // route asks for the plum depth policy here because all easy gamepad combos
    // are already taken.
    this->depth_subscriber = this->create_subscription<quadruped_ros2::msg::DepthFrame>(
        "depth_cam", LatestQoS(),
        [this] (const quadruped_ros2::msg::DepthFrame::SharedPtr msg) {this->DepthCallback(msg);}
    );
    this->policy_switch_subscriber = this->create_subscription<std_msgs::msg::String>(
        "policy_switch", LatestQoS(),
        [this] (const std_msgs::msg::String::SharedPtr msg) {this->PolicySwitchCallback(msg);}
    );

    // service
    this->sim_reset_client = this->create_client<std_srvs::srv::Empty>("sim_reset");
    auto empty_request = std::make_shared<std_srvs::srv::Empty::Request>();
    auto result = this->sim_reset_client->async_send_request(empty_request);

    // S4: no wall-clock loop for control or inference any more. /motor_state drives both through
    // ControlStep(): control every `dt` = 5 ms of SIM time, inference every `decimation` control cycles
    // (20 ms). Pause / slow-motion / fast-forward therefore behave, and everything runs on the ROS
    // executor thread (races gone) - UPSTREAM.md §2. Only the keyboard keeps a 50 ms wall loop.
    std::cout << LOGGER::INFO << "Feedback-driven clock (sim time): control every " << this->params.dt * 1000.0
              << " ms, inference every " << this->params.dt * this->params.decimation * 1000.0
              << " ms; nothing runs while no feedback arrives" << std::endl;

    // keyboard
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Sim::KeyboardInterface, this));
    this->loop_keyboard->start();

#ifdef CSV_LOGGER
    this->CSVInit(this->robot_name);
#endif

    std::cout << LOGGER::INFO << "RL_Sim start" << std::endl;
}

RL_Sim::~RL_Sim()
{
    this->loop_keyboard->shutdown();
    std::cout << LOGGER::INFO << "RL_Sim exit" << std::endl;
}

void RL_Sim::GetState(RobotState<double> *state)
{
    const auto &orientation = this->imu_msg.orientation;
    const auto &angular_velocity = this->imu_msg.angular_velocity;

    state->imu.quaternion[0] = orientation.w;
    state->imu.quaternion[1] = orientation.x;
    state->imu.quaternion[2] = orientation.y;
    state->imu.quaternion[3] = orientation.z;

    state->imu.gyroscope[0] = angular_velocity.x;
    state->imu.gyroscope[1] = angular_velocity.y;
    state->imu.gyroscope[2] = angular_velocity.z;

    // joint_mapping[i]: index in the sim's actuator order of policy joint i (policy/black/base.yaml)
    for (int i = 0; i < this->params.num_of_dofs; ++i)
    {
        state->motor_state.q[i] = this->motor_state_msg.q[this->params.joint_mapping[i]];
        state->motor_state.dq[i] = this->motor_state_msg.dq[this->params.joint_mapping[i]];
        state->motor_state.tau_est[i] = this->motor_state_msg.tau[this->params.joint_mapping[i]];
    }
}

void RL_Sim::SetCommand(const RobotCommand<double> *command)
{
    // S5: limit the command before it leaves this node. We do not send torque, so the rules in
    // command_limit.hpp work on what we do send (q_des / dq_des / kp / kd / tau_ff) and budget the
    // implied torque at the measured state. command_limit.hpp has the design and the
    // guarantees; the counters are reported once a second at most so a biting limit is visible.
    const black_fsm::Knobs &knobs = black_fsm::GetKnobs(this->robot_name);
    command_limit::Limits lim;
    lim.tau = knobs.command_tau_limit;
    lim.kp_max = knobs.command_kp_max;
    lim.kd_max = knobs.command_kd_max;
    lim.dq_max = knobs.command_dq_max;
    lim.slew = knobs.command_slew;
    command_limit::State lim_state;
    lim_state.q = this->robot_state.motor_state.q;
    lim_state.dq = this->robot_state.motor_state.dq;
    lim_state.q_des_prev = this->limited_q_des_prev;
    lim_state.has_prev = this->command_limited_once;
    std::vector<double> q_des(command->motor_command.q);
    std::vector<double> dq_des(command->motor_command.dq);
    std::vector<double> kp(command->motor_command.kp);
    std::vector<double> kd(command->motor_command.kd);
    std::vector<double> tau_ff(command->motor_command.tau);
    if (this->WheelLockActive())
    {
        this->ApplyWheelLockTargets(q_des, dq_des, kp, kd, tau_ff);
        lim.position_holds = this->params.wheel_indices;
    }
    const command_limit::Counts limited = command_limit::Apply(q_des, dq_des, kp, kd, tau_ff,
                                                              lim_state, lim, this->params.dt);
    this->limited_q_des_prev = q_des;
    this->command_limited_once = true;
    if (limited.any())
    {
        this->limit_events++;
        const auto now = std::chrono::steady_clock::now();
        if (now - this->last_limit_warn >= std::chrono::seconds(1))
        {
            this->last_limit_warn = now;
            std::cout << std::endl << LOGGER::WARNING << "command limited (events " << this->limit_events
                      << "): slew=" << limited.slew << " torque=" << limited.torque
                      << " gains=" << limited.gains << " dq_des=" << limited.dq_des
                      << " window=" << limited.window << std::endl;
        }
    }

    for (int i = 0; i < this->params.num_of_dofs; ++i)
    {
        this->mit_command_msg.q[this->params.joint_mapping[i]] = q_des[i];
        this->mit_command_msg.w[this->params.joint_mapping[i]] = dq_des[i];
        this->mit_command_msg.kp[this->params.joint_mapping[i]] = kp[i];
        this->mit_command_msg.kd[this->params.joint_mapping[i]] = kd[i];
        this->mit_command_msg.tau[this->params.joint_mapping[i]] = tau_ff[i];
    }

    // S4: publish only when the target changes - the policy output is constant between inferences, so at
    // full speed this drops /mit_command from one per control cycle to one per 20 ms of sim time. The
    // heartbeat keeps the sim's watchdog fed (100 steps = 200 ms of sim time; we send every 100 ms).
    constexpr int kCommandHeartbeatCycles = 20;
    const bool changed = !this->command_sent_once || this->mit_command_msg.q != this->last_sent_command.q
        || this->mit_command_msg.kp != this->last_sent_command.kp
        || this->mit_command_msg.kd != this->last_sent_command.kd
        || this->mit_command_msg.w != this->last_sent_command.w
        || this->mit_command_msg.tau != this->last_sent_command.tau;
    if (!changed && ++this->cycles_since_command_publish < kCommandHeartbeatCycles)
    {
        return;
    }
    this->cycles_since_command_publish = 0;
    this->command_sent_once = true;
    this->last_sent_command = this->mit_command_msg;

    this->mit_command_msg.header.stamp = this->now();
    this->mit_command_publisher->publish(this->mit_command_msg);
}

void RL_Sim::SwitchPolicy(const std::string &policy_name)
{
    if (policy_name.empty() || policy_name == this->config_name_override)
    {
        return;
    }
    this->config_name_override = policy_name;
    if (!this->rl_init_done)
    {
        std::cout << std::endl << LOGGER::INFO << "[policy] next RL entry will load " << policy_name << std::endl;
        return;
    }
    this->rl_init_done = false;
    try
    {
        // A /policy_switch callback can occur between control ticks. Capture
        // the newest feedback, not the previous inference's observation.
        this->GetState(&this->robot_state);
        const auto switch_start = std::chrono::steady_clock::now();
        this->InitRL(this->robot_name + "/" + policy_name);
        const auto switch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - switch_start).count();
        this->rl_init_done = true;
        this->control.x = 0.0;
        this->control.y = 0.0;
        this->control.yaw = 0.0;
        torch::Tensor drop;
        while (this->output_dof_pos_queue.try_pop(drop)) {}
        while (this->output_dof_vel_queue.try_pop(drop)) {}
        while (this->output_dof_tau_queue.try_pop(drop)) {}
        this->episode_length_buf = 0;
        std::cout << std::endl << LOGGER::INFO << "[policy] switched to " << policy_name
                  << " (" << switch_ms << " ms)" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cout << std::endl << LOGGER::ERROR << "[policy] switch to " << policy_name << " failed: " << e.what() << std::endl;
        this->rl_init_done = false;
    }
}

std::uint32_t RL_Sim::KeyboardMask() const
{
    // library/core's KeyboardInterface() only ever stores the LAST key (and ignores a repeat of the same
    // key), so this adapter maps that single value to one bit; input_map then derives edges by comparing
    // it with the previous cycle. See input_map.hpp for why the keyboard cannot report two keys at once.
    switch (this->control.current_keyboard)
    {
    case Input::Keyboard::Num0: return input_map::Bit(input_map::KbKey::Num0);
    case Input::Keyboard::Num1: return input_map::Bit(input_map::KbKey::Num1);
    case Input::Keyboard::Num2: return input_map::Bit(input_map::KbKey::Num2);
    case Input::Keyboard::Num3: return input_map::Bit(input_map::KbKey::Num3);
    case Input::Keyboard::Num4: return input_map::Bit(input_map::KbKey::Num4);
    case Input::Keyboard::Num5: return input_map::Bit(input_map::KbKey::Num5);
    case Input::Keyboard::Num6: return input_map::Bit(input_map::KbKey::Num6);
    case Input::Keyboard::Num7: return input_map::Bit(input_map::KbKey::Num7);
    case Input::Keyboard::Num8: return input_map::Bit(input_map::KbKey::Num8);
    case Input::Keyboard::Num9: return input_map::Bit(input_map::KbKey::Num9);
    case Input::Keyboard::W: return input_map::Bit(input_map::KbKey::W);
    case Input::Keyboard::A: return input_map::Bit(input_map::KbKey::A);
    case Input::Keyboard::S: return input_map::Bit(input_map::KbKey::S);
    case Input::Keyboard::D: return input_map::Bit(input_map::KbKey::D);
    case Input::Keyboard::Q: return input_map::Bit(input_map::KbKey::Q);
    case Input::Keyboard::E: return input_map::Bit(input_map::KbKey::E);
    case Input::Keyboard::F: return input_map::Bit(input_map::KbKey::F);
    case Input::Keyboard::T: return input_map::Bit(input_map::KbKey::T);
    case Input::Keyboard::B: return input_map::Bit(input_map::KbKey::B);
    case Input::Keyboard::P: return input_map::Bit(input_map::KbKey::P);
    case Input::Keyboard::R: return input_map::Bit(input_map::KbKey::R);
    case Input::Keyboard::N: return input_map::Bit(input_map::KbKey::N);
    case Input::Keyboard::Space: return input_map::Bit(input_map::KbKey::Space);
    default: return 0;
    }
}

void RL_Sim::RobotControl(bool defer_publish)
{
    // Resolve the input edges once per cycle (input_map.hpp): an edge is a pure function of this cycle
    // and the previous one, so it fires exactly once and never has to be "consumed" — that replaces the
    // library's single-value latch and the four bugs it caused (see UPSTREAM.md §2).
    input_map::Frame &in = input_map::Current();
    const std::uint32_t pad = input_map::PadMaskFromJoy(this->joy_msg.buttons, this->joy_msg.axes);
    const std::uint32_t kb = this->KeyboardMask();
    std::array<float, 8> axes{};
    if (this->joy_msg.axes.size() >= 8)
    {
        std::copy(this->joy_msg.axes.begin(), this->joy_msg.axes.begin() + 8, axes.begin());
    }
    input_map::BeginCycleWithJoy(pad, kb, axes, this->have_joy_frame);
    const bool keyboard_edge = in.kb.any_pressed();

    if (in.has(input_map::Intent::ResetSim))
    {
        auto empty_request = std::make_shared<std_srvs::srv::Empty::Request>();
        auto result = this->sim_reset_client->async_send_request(empty_request);
    }
    // The library's keyboard latch ignores a repeated key, so keep upstream's "act, then clear" for the
    // keyboard only: without it, pressing the same key twice in a row would look like one press.
    if (keyboard_edge)
    {
        this->control.current_keyboard = this->control.last_keyboard;
    }

    if (simulation_running)
    {
        this->motiontime++;

        if (in.has(input_map::Intent::JogForward)) this->control.x += 0.1;
        if (in.has(input_map::Intent::JogBackward)) this->control.x -= 0.1;
        if (in.has(input_map::Intent::JogLeft)) this->control.y += 0.1;
        if (in.has(input_map::Intent::JogRight)) this->control.y -= 0.1;
        if (in.has(input_map::Intent::JogYawLeft)) this->control.yaw += 0.1;
        if (in.has(input_map::Intent::JogYawRight)) this->control.yaw -= 0.1;
        if (in.has(input_map::Intent::ZeroCommand))
        {
            this->control.x = 0;
            this->control.y = 0;
            this->control.yaw = 0;
        }
        if (in.has(input_map::Intent::PolicyFlat)) this->SwitchPolicy("flat");
        if (in.has(input_map::Intent::PolicyStair)) this->SwitchPolicy("stair");
        if (in.has(input_map::Intent::PolicyObstacle)) this->SwitchPolicy("obstacle");
        // Frontier roster (docs/joystick.md §3): one generalist + one Part1 specialist. Policy names are
        // the deployment tree directory names under policy/blackW/, so a renamed/re-exported round only
        // needs this line + one binding row.
        if (in.has(input_map::Intent::PolicyMain)) this->SwitchPolicy("row2p_p40w40_v1");
        if (in.has(input_map::Intent::PolicyTStairs)) this->SwitchPolicy("st30_ppo_st30_s100_600");
        if (in.has(input_map::Intent::PolicyTrench30)) this->SwitchPolicy("s1d_trench_reward_v75_m100");
        // Part2 stairs_30cm specialist (only known 30 cm stair passer on the field); RB+left / B.
        if (in.has(input_map::Intent::PolicyBand)) this->SwitchPolicy("vt_band_v1");
        if (in.has(input_map::Intent::PolicyPose)) this->SwitchPolicy("vt_pose_v1");
        if (in.has(input_map::Intent::PolicyEdge)) this->SwitchPolicy("vt_edge_v1");
        if (in.has(input_map::Intent::PolicyTrack)) this->SwitchPolicy("vt_track_v1");
        if (in.has(input_map::Intent::PolicyTrackV2)) this->SwitchPolicy("vt_track_v2");
        if (in.has(input_map::Intent::PolicyNwwolf))
        {
            // Default keeps the N-W-wolf teacher; the Part1 route can point the
            // same button at a locally trained locked-wheel connector without a
            // second gamepad binding (e.g. PART1_LOCKED_POLICY=vt_locked12).
            const char* locked = std::getenv("PART1_LOCKED_POLICY");
            this->SwitchPolicy(locked != nullptr && *locked != '\0' ? locked : "vt_locked12");
        }
        if (in.has(input_map::Intent::ToggleNavigation))
        {
            this->control.navigation_mode = !this->control.navigation_mode;
            std::cout << std::endl << LOGGER::INFO << "Navigation mode: " << (this->control.navigation_mode ? "ON" : "OFF") << std::endl;
        }

        this->GetState(&this->robot_state);
        this->StateController(&this->robot_state, &this->robot_command);
        if (!defer_publish) this->SetCommand(&this->robot_command);

        if (this->status_period_ms > 0)
        {
            const auto now = std::chrono::steady_clock::now();
            if (now - this->last_status_print >= std::chrono::milliseconds(this->status_period_ms))
            {
                this->last_status_print = now;
                std::cout << "\r\033[K" << std::flush << LOGGER::INFO << "RL Controller x:" << this->control.x
                          << " y:" << this->control.y << " yaw:" << this->control.yaw
                          << " cmd_limit_events:" << this->limit_events << std::flush;
            }
        }
    }
}

void RL_Sim::ImuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    this->imu_msg = *msg;
}

void RL_Sim::CmdvelCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
    this->cmd_vel = *msg;
}

void RL_Sim::JoyCallback(const sensor_msgs::msg::Joy::SharedPtr msg)
{
    // Our joy_node (quadruped_ros2/scripts/joy_node.py) already publishes the layout and signs
    // used by input_map.hpp (docs/joystick.md §1, UPSTREAM.md §2), so here we only guard the length: indexing
    // a shorter message would read past the end. The key map itself lives in input_map.hpp and is
    // resolved once per control cycle in RobotControl() - see UPSTREAM.md §2.
    if (msg->axes.size() < 8 || msg->buttons.size() < 11)
    {
        return;
    }
    this->joy_msg = *msg;
    this->have_joy_frame = true;

    // Continuous commands stay LEVELS applied once per /joy frame (same place and timing as upstream):
    // input_map.hpp above only resolves edges. Keyboard jogging still happens in RobotControl() at
    // 200 Hz, which is why the keyboard path is documented with `joy:=false` (no /joy frames, so these
    // three lines cannot zero the jog).
    this->control.x = this->joy_msg.axes[1] * this->joy_command_scale;   // LY
    this->control.y = this->joy_msg.axes[0] * this->joy_command_scale;   // LX
    this->control.yaw = this->joy_msg.axes[3] * this->joy_command_scale; // RX
}

void RL_Sim::MotorStateCallback(const quadruped_ros2::msg::MotorState::SharedPtr msg)
{
    this->motor_state_msg = *msg;

    // S4: the feedback is the clock - each /motor_state drives the control step below.
    this->ControlStep(msg->sim_time);
}

void RL_Sim::DepthCallback(const quadruped_ros2::msg::DepthFrame::SharedPtr msg)
{
    // Keep the newest frame only (LatestQoS).  Size / staleness checks are done in
    // DepthReady() when a depth policy actually asks for an inference.
    if (msg->depth.empty())
    {
        return;
    }
    this->depth_frame_msg = *msg;
    this->depth_sim_time = msg->sim_time;
    this->depth_frames++;
}

void RL_Sim::PolicySwitchCallback(const std_msgs::msg::String::SharedPtr msg)
{
    if (!msg->data.empty())
    {
        this->SwitchPolicy(msg->data);
    }
}

bool RL_Sim::DepthReady(double sim_time) const
{
    if (!this->params.depth_enabled)
    {
        return true;
    }
    if (this->depth_sim_time < 0.0)
    {
        return false;
    }
    const std::size_t expected = static_cast<std::size_t>(this->params.depth_width)
                               * static_cast<std::size_t>(this->params.depth_height);
    if (this->depth_frame_msg.depth.size() != expected)
    {
        return false;
    }
    // sim_node renders one frame per 20 ms of sim time; accept the newest frame
    // if it is at most 50 ms older than the current control stamp.
    return std::abs(sim_time - this->depth_sim_time) <= 0.05;
}

torch::Tensor RL_Sim::DepthInput() const
{
    // Same normalization as eval/robocon_eval/policy_runner.py::render_depth and
    // the training camera: non-positive (no hit) -> cutoff, clamp to
    // [min_depth, cutoff], divide by cutoff.
    const std::vector<float> &raw = this->depth_frame_msg.depth;
    const int width = this->params.depth_width;
    const int height = this->params.depth_height;
    const float cutoff = static_cast<float>(this->params.depth_cutoff);
    const float near = static_cast<float>(this->params.depth_min_depth);
    std::vector<float> normalized(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i)
    {
        const float d = raw[i] > 0.0f ? raw[i] : cutoff;
        normalized[i] = std::min(std::max(d, near), cutoff) / cutoff;
    }
    return torch::tensor(normalized, torch::kFloat32).view({1, 1, height, width});
}

void RL_Sim::ControlStep(double sim_time)
{
    // Both control and inference are paced by SIM time - not by "one step per feedback message" and not
    // by the wall clock: control every `params.dt` (5 ms), inference every `decimation` control cycles
    // (20 ms). That keeps the work per simulated second constant, and nothing runs while no feedback
    // arrives. Control-per-message instead pushed the feedback to 4-5 kHz at full speed (and /mit_command
    // with it) until the two ends starved each other - see UPSTREAM.md §2 item 18.
    const double control_period = this->params.dt;

    if (sim_time < this->last_control_sim_time && this->WheelLockActive())
    {
        // A reset starts a new episode. Capture the reset feedback once; a
        // dropped-frame gap or an ordinary control tick must never recapture.
        this->GetState(&this->robot_state);
        this->InitWheelLock();
        this->InitOutputs();
    }
    if (this->last_control_sim_time < 0.0 || sim_time < this->last_control_sim_time
        || sim_time - this->last_control_sim_time > 10.0 * control_period)
    {
        // First frame / sim time went backwards / feedback gap over 10 periods: re-anchor, do not catch up.
        this->last_control_sim_time = sim_time;
        this->cycles_since_inference = 0;
        return;
    }
    if (sim_time - this->last_control_sim_time < control_period)
    {
        return;   // next control period not reached yet
    }

    // Catch up along sim time: normally one step per feedback frame, at most 10 after dropped ones.
    while (sim_time - this->last_control_sim_time >= control_period)
    {
        this->last_control_sim_time += control_period;
        const bool inference_due = ++this->cycles_since_inference >= this->params.decimation;
        if (inference_due) this->cycles_since_inference = 0;
        feedback_control::Step(*this, inference_due,
            [this](bool deferred) { this->RobotControl(deferred); },
            [this]() { this->RunModel(); },
            [this]() { this->SetCommand(&this->robot_command); });
    }
}

void RL_Sim::RunModel()
{
    if (this->rl_init_done && simulation_running)
    {
        if (!this->DepthReady(this->last_control_sim_time))
        {
            // The first inference after switching to a depth policy can beat the
            // next /depth_cam frame by one control period; hold the previous
            // command instead of feeding the policy an empty image.
            static int waiting = 0;
            if (waiting++ < 3)
            {
                std::cout << std::endl << LOGGER::WARNING
                          << "waiting for /depth_cam before the first depth inference" << std::endl;
            }
            return;
        }
        this->episode_length_buf += 1;
        this->obs.ang_vel = torch::tensor(this->robot_state.imu.gyroscope).unsqueeze(0);
        if (this->control.navigation_mode)
        {
            this->obs.commands = torch::tensor({{this->cmd_vel.linear.x, this->cmd_vel.linear.y, this->cmd_vel.angular.z}});
        }
        else
        {
            this->obs.commands = torch::tensor({{this->control.x, this->control.y, this->control.yaw}});
        }
        this->obs.base_quat = torch::tensor(this->robot_state.imu.quaternion).unsqueeze(0);
        this->obs.dof_pos = torch::tensor(this->robot_state.motor_state.q).narrow(0, 0, this->params.num_of_dofs).unsqueeze(0);
        this->obs.dof_vel = torch::tensor(this->robot_state.motor_state.dq).narrow(0, 0, this->params.num_of_dofs).unsqueeze(0);

        this->obs.actions = this->Forward();
        this->ComputeOutput(this->obs.actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);

        if (this->output_dof_pos.defined() && this->output_dof_pos.numel() > 0)
        {
            output_dof_pos_queue.push(this->output_dof_pos);
        }
        if (this->output_dof_vel.defined() && this->output_dof_vel.numel() > 0)
        {
            output_dof_vel_queue.push(this->output_dof_vel);
        }
        // output_dof_tau is deliberately *not* pushed: upstream pushes it from here and from the five
        // rl_real_*.cpp, but no FSM anywhere pops it (grep try_pop), and tbb::concurrent_queue is
        // unbounded, so it is a pure leak that grows with every inference. This port never uses the
        // feed-forward tau (the FSM sets tau = 0), so we drop the push. See UPSTREAM.md §2 and
        // UPSTREAM.md §2.

        // this->TorqueProtect(this->output_dof_tau);

#ifdef CSV_LOGGER
        torch::Tensor tau_est = torch::tensor(this->robot_state.motor_state.tau_est).narrow(0, 0, this->params.num_of_dofs).unsqueeze(0);
        this->CSVLogger(this->output_dof_tau, tau_est, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
#endif
    }
}

torch::Tensor RL_Sim::Forward()
{
    torch::autograd::GradMode::set_enabled(false);

    torch::Tensor clamped_obs = this->ComputeObservation();
    if (!torch::isfinite(clamped_obs).all().item<bool>())
    {
        static int guard_obs = 0;
        if (guard_obs++ < 5) std::cerr << "[guard] non-finite observation -> sanitized" << std::endl;
        clamped_obs = torch::nan_to_num(clamped_obs, 0.0, 100.0, -100.0);
    }

    // Depth policies are exported as policy(obs_1d, [1x1xHxW depth]); blind ones
    // keep the historical single-input call.  The image is normalized exactly like
    // the eval runner / training camera (see DepthInput()).
    const auto run_policy = [this](const torch::Tensor &input)
    {
        if (!this->params.depth_enabled)
        {
            return this->model.forward({input}).toTensor();
        }
        return this->model.forward({input, std::vector<torch::Tensor>{this->DepthInput()}}).toTensor();
    };

    torch::Tensor actions;
    if (this->params.observations_history.size() != 0)
    {
        if (!this->history_prefilled)
        {
            // Backfill all history slots with the first RL observation (both oldest
            // and newest slots), matching the MjLab/rsl_rl observation history
            // semantics the trained policies expect on episode reset.
            for (std::size_t i = 0; i < this->params.observations_history.size(); ++i)
            {
                this->history_obs_buf.insert(clamped_obs);
            }
            this->history_prefilled = true;
        }
        this->history_obs_buf.insert(clamped_obs);
        this->history_obs = this->history_obs_buf.get_obs_vec(this->params.observations_history);
        actions = run_policy(this->history_obs);
    }
    else
    {
        actions = run_policy(clamped_obs);
    }

    if (!torch::isfinite(actions).all().item<bool>())
    {
        static int guard_act = 0;
        if (guard_act++ < 5) std::cerr << "[guard] non-finite action -> zeroed" << std::endl;
        actions = torch::zeros_like(actions);
    }

    // Debug hook: dump the first RL observations/actions for contract comparison
    // against the MuJoCo evaluator (enabled only with RL_DEBUG_OBS=1).
    static int debug_count = 0;
    static const int debug_max = []() {
        const char *m = std::getenv("RL_DEBUG_OBS_MAX");
        return (m != nullptr) ? std::max(1, std::atoi(m)) : 60;
    }();
    if (std::getenv("RL_DEBUG_OBS") != nullptr && debug_count < debug_max)
    {
        static std::ofstream debug_csv("/tmp/rl_obs_debug.csv", std::ios::app);
        if (debug_count == 0) debug_csv << "# obs57,act16\n";
        for (int i = 0; i < int(clamped_obs.numel()); ++i)
            debug_csv << clamped_obs[0][i].item<double>() << (i + 1 == int(clamped_obs.numel()) ? ";" : ",");
        for (int i = 0; i < int(actions.numel()); ++i)
            debug_csv << actions[0][i].item<double>() << (i + 1 == int(actions.numel()) ? "\n" : ",");
        debug_count++;
    }
    // Depth-policy deployment dump (RL_DEPTH_DEBUG=1): the exact policy input (obs
    // history + the raw metric depth frame that DepthInput() normalizes) and the raw
    // model output, so this C++ inference can be compared bit-for-bit with the same
    // policy run from Python on identical inputs (demo/tests/check_depth_inference.py).
    if (this->params.depth_enabled && std::getenv("RL_DEPTH_DEBUG") != nullptr)
    {
        static int depth_debug_count = 0;
        static const int depth_debug_max = []() {
            const char *m = std::getenv("RL_DEPTH_DEBUG_MAX");
            return (m != nullptr) ? std::max(1, std::atoi(m)) : 40;
        }();
        static std::ofstream depth_debug_csv("/tmp/rl_depth_debug.csv", std::ios::trunc);
        if (depth_debug_count < depth_debug_max)
        {
            const torch::Tensor &policy_input =
                (this->params.observations_history.size() != 0) ? this->history_obs : clamped_obs;
            depth_debug_csv << std::setprecision(17)
                            << this->last_control_sim_time << ";" << this->depth_frame_msg.sim_time << ";";
            for (int i = 0; i < int(policy_input.numel()); ++i)
                depth_debug_csv << policy_input[0][i].item<double>()
                                << (i + 1 == int(policy_input.numel()) ? ";" : ",");
            for (int i = 0; i < int(actions.numel()); ++i)
                depth_debug_csv << actions[0][i].item<double>()
                                << (i + 1 == int(actions.numel()) ? ";" : ",");
            const auto &raw_depth = this->depth_frame_msg.depth;
            for (int i = 0; i < int(raw_depth.size()); ++i)
                depth_debug_csv << raw_depth[i] << (i + 1 == int(raw_depth.size()) ? "\n" : ",");
            depth_debug_count++;
        }
    }
    if (this->params.clip_actions_upper.numel() != 0 && this->params.clip_actions_lower.numel() != 0)
    {
        return torch::clamp(actions, this->params.clip_actions_lower, this->params.clip_actions_upper);
    }
    else
    {
        return actions;
    }
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    int rc = 0;
    try
    {
        rclcpp::spin(std::make_shared<RL_Sim>());
    }
    catch (const std::exception &e)
    {
        // F15: a policy/config error must not abort the whole process; log it once
        // and exit non-zero so a supervisor can restart with a corrected config.
        std::cerr << "[rl_sim] fatal exception: " << e.what() << std::endl;
        rc = 1;
    }
    catch (...)
    {
        std::cerr << "[rl_sim] fatal unknown exception" << std::endl;
        rc = 1;
    }
    rclcpp::shutdown();
    return rc;
}
