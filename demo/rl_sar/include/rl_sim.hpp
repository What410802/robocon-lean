/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 *
 * Modified for @20261007_assignment from rl_sar e5c2f41 include/rl_sim.hpp.
 * Change list: see UPSTREAM.md §2.
 */

#ifndef RL_SIM_HPP
#define RL_SIM_HPP

// #define CSV_LOGGER

#include "rl_sdk.hpp"
#include "observation_buffer.hpp"
#include "loop.hpp"
#include "fsm.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <vector>
#include <string>
#include <fstream>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_srvs/srv/empty.hpp>
#include "quadruped_ros2/msg/mit_command.hpp"
#include "quadruped_ros2/msg/motor_state.hpp"

class RL_Sim : public RL, public rclcpp::Node
{
public:
    RL_Sim();
    ~RL_Sim();

private:
    // rl functions
    torch::Tensor Forward() override;
    void GetState(RobotState<double> *state) override;
    void SetCommand(const RobotCommand<double> *command) override;
    void RunModel();
    void RobotControl();
    void SwitchPolicy(const std::string &policy_name);

    // input: the raw frame is stored here, resolved to intents once per cycle (input_map.hpp, S3.5)
    bool have_joy_frame = false;

    // S5 command-side limiting: the previous command (to slew-limit against) and the counters that
    // decide whether the throttled warning is worth printing (include/command_limit.hpp).
    std::vector<double> limited_q_des_prev;
    bool command_limited_once = false;
    std::chrono::steady_clock::time_point last_limit_warn{};
    long limit_events = 0;

    // S4 feedback clock: control and inference are paced by SIM time, anchored on /motor_state.
    // -1 = not initialised yet (the first feedback frame only aligns the beat).
    double last_control_sim_time = -1.0;
    int cycles_since_inference = 0;
    int cycles_since_command_publish = 0;
    bool command_sent_once = false;
    quadruped_ros2::msg::MitCommand last_sent_command;

    // status line (node parameter; 0 = off)
    int status_period_ms = 0;
    std::chrono::steady_clock::time_point last_status_print{};

    // loop
    std::shared_ptr<LoopFunc> loop_keyboard;
    // S4: control/inference no longer use wall-clock LoopFuncs; only the keyboard poll remains.

    // ros interface
    sensor_msgs::msg::Imu imu_msg;
    geometry_msgs::msg::Twist cmd_vel;
    sensor_msgs::msg::Joy joy_msg;
    quadruped_ros2::msg::MitCommand mit_command_msg;
    quadruped_ros2::msg::MotorState motor_state_msg;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscriber;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_subscriber;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_subscriber;
    rclcpp::Subscription<quadruped_ros2::msg::MotorState>::SharedPtr motor_state_subscriber;
    rclcpp::Publisher<quadruped_ros2::msg::MitCommand>::SharedPtr mit_command_publisher;
    rclcpp::Client<std_srvs::srv::Empty>::SharedPtr sim_reset_client;
    void ImuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
    void CmdvelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
    void MotorStateCallback(const quadruped_ros2::msg::MotorState::SharedPtr msg);
    void JoyCallback(const sensor_msgs::msg::Joy::SharedPtr msg);
    std::uint32_t KeyboardMask() const;
    void ControlStep(double sim_time);

    // others
    int motiontime = 0;
    double joy_command_scale = 1.5;
};

#endif // RL_SIM_HPP
