/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_sdk.hpp"

#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>

void RL::StateController(const RobotState<double>* state, RobotCommand<double>* command)
{
    auto updateState = [&](std::shared_ptr<FSMState> statePtr)
    {
        if (auto rl_fsm_state = std::dynamic_pointer_cast<RLFSMState>(statePtr))
        {
            rl_fsm_state->fsm_state = state;
            rl_fsm_state->fsm_command = command;
        }
    };
    for (auto& pair : fsm.states_)
    {
        updateState(pair.second);
    }

    fsm.Run();
}

torch::Tensor RL::ComputeObservation()
{
    std::vector<torch::Tensor> obs_list;

    for (const std::string &observation : this->params.observations)
    {
        if (observation == "lin_vel")
        {
            obs_list.push_back(this->obs.lin_vel * this->params.lin_vel_scale);
        }
        else if (observation == "ang_vel_body")
        {
            obs_list.push_back(this->obs.ang_vel * this->params.ang_vel_scale);
        }
        else if (observation == "ang_vel_world")
        {
            obs_list.push_back(this->QuatRotateInverse(this->obs.base_quat, this->obs.ang_vel) * this->params.ang_vel_scale);
        }
        else if (observation == "gravity_vec")
        {
            obs_list.push_back(this->QuatRotateInverse(this->obs.base_quat, this->obs.gravity_vec));
        }
        else if (observation == "commands")
        {
            obs_list.push_back(this->obs.commands * this->params.commands_scale);
        }
        else if (observation == "dof_pos")
        {
            torch::Tensor dof_pos_rel = this->obs.dof_pos - this->params.default_dof_pos;
            for (int i : this->params.wheel_indices)
            {
                dof_pos_rel[0][i] = 0.0;
            }
            obs_list.push_back(dof_pos_rel * this->params.dof_pos_scale);
        }
        else if (observation == "dof_vel")
        {
            obs_list.push_back(this->obs.dof_vel * this->params.dof_vel_scale);
        }
        else if (observation == "actions")
        {
            obs_list.push_back(this->obs.actions);
        }
        else if (observation == "phase")
        {
            float motion_time = this->episode_length_buf * this->params.dt * this->params.decimation;
            torch::Tensor phase = torch::tensor({{3.1415926 * motion_time / 2}});
            torch::Tensor phase_tensor = torch::cat({
                torch::sin(phase),
                torch::cos(phase),
                torch::sin(phase / 2),
                torch::cos(phase / 2),
                torch::sin(phase / 4),
                torch::cos(phase / 4),
            }, -1);
            obs_list.push_back(phase_tensor);
        }
        else if (observation == "g1_phase")
        {
            float motion_time = this->episode_length_buf * this->params.dt * this->params.decimation;
            torch::Tensor period = torch::tensor({{0.8f}});
            torch::Tensor count = torch::tensor({{motion_time}});
            torch::Tensor phase = torch::fmod(count, period) / period;
            torch::Tensor phase_tensor = torch::cat({
                torch::sin(2 * 3.1415926f * phase),
                torch::cos(2 * 3.1415926f * phase),
            }, -1);
            obs_list.push_back(phase_tensor);
        }
        else if (observation == "g1_mimic_phase")
        {
            float motion_time = this->episode_length_buf * this->params.dt * this->params.decimation;
            torch::Tensor count = torch::tensor({{motion_time}});
            torch::Tensor phase = count / this->motion_length;
            obs_list.push_back(phase);
        }
    }

    torch::Tensor obs = torch::cat(obs_list, 1);
    torch::Tensor clamped_obs = torch::clamp(obs, -this->params.clip_obs, this->params.clip_obs);
    return clamped_obs;
}

void RL::InitObservations()
{
    this->obs.lin_vel = torch::tensor({{0.0, 0.0, 0.0}});
    this->obs.ang_vel = torch::tensor({{0.0, 0.0, 0.0}});
    this->obs.gravity_vec = torch::tensor({{0.0, 0.0, -1.0}});
    this->obs.commands = torch::tensor({{0.0, 0.0, 0.0}});
    this->obs.base_quat = torch::tensor({{0.0, 0.0, 0.0, 1.0}});
    this->obs.dof_pos = this->params.default_dof_pos;
    this->obs.dof_vel = torch::zeros({1, this->params.num_of_dofs});
    this->obs.actions = torch::zeros({1, this->params.num_of_dofs});
}

void RL::InitOutputs()
{
    this->output_dof_tau = torch::zeros({1, this->params.num_of_dofs});
    this->output_dof_pos = this->params.default_dof_pos;
    this->output_dof_vel = torch::zeros({1, this->params.num_of_dofs});
}

void RL::InitControl()
{
    this->control.x = 0.0;
    this->control.y = 0.0;
    this->control.yaw = 0.0;
}

void RL::InitRL(std::string robot_path)
{
    this->ReadYamlRL(robot_path);
    this->ValidateRobotManifest(this->robot_name, true);
    // Per-policy frequency must match the loaded base.yaml dt/decimation (audit
    // F7); otherwise the policy silently runs at a different period than trained.
    if (this->params.control_hz > 0.0 || this->params.policy_hz > 0.0)
    {
        const double want_control = 1.0 / this->params.dt;
        const double want_policy = want_control / static_cast<double>(this->params.decimation);
        const bool bad_control = this->params.control_hz > 0.0
            && std::abs(this->params.control_hz - want_control) > 0.02 * want_control;
        const bool bad_policy = this->params.policy_hz > 0.0
            && std::abs(this->params.policy_hz - want_policy) > 0.02 * want_policy;
        if (bad_control || bad_policy)
        {
            throw std::runtime_error(
                robot_path + ": config.yaml frequency (control_hz="
                + std::to_string(this->params.control_hz) + " policy_hz="
                + std::to_string(this->params.policy_hz) + ") != base.yaml dt="
                + std::to_string(this->params.dt) + " decimation="
                + std::to_string(this->params.decimation));
        }
    }
    for (std::string &observation : this->params.observations)
    {
        if (observation == "ang_vel")
        {
            // In ROS1 Gazebo, the coordinate system for angular velocity is in the world coordinate system.
            // In ROS2 Gazebo and real robot, the coordinate system for angular velocity is in the body coordinate system.
            observation = this->ang_vel_type;
        }
    }
    // init rl
    if (!this->params.observations_history.empty())
    {
        this->history_obs_buf = ObservationBuffer(1, this->params.num_observations, this->params.observations_history.size());
        this->history_prefilled = false;
    }
    // model
    std::string model_path = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/policy/" + robot_path + "/" + this->params.model_name;
    this->model = torch::jit::load(model_path);

    this->InitObservations();
    this->InitOutputs();
    this->InitControl();
}

void RL::ComputeOutput(const torch::Tensor &actions, torch::Tensor &output_dof_pos, torch::Tensor &output_dof_vel, torch::Tensor &output_dof_tau)
{
    torch::Tensor actions_scaled = actions * this->params.action_scale;
    torch::Tensor pos_actions_scaled = actions_scaled.clone();
    torch::Tensor vel_actions_scaled = torch::zeros_like(actions);
    for (int i : this->params.wheel_indices)
    {
        pos_actions_scaled[0][i] = 0.0;
        vel_actions_scaled[0][i] = actions_scaled[0][i];
    }
    torch::Tensor all_actions_scaled = pos_actions_scaled + vel_actions_scaled;
    output_dof_pos = pos_actions_scaled + this->params.default_dof_pos;
    output_dof_vel = vel_actions_scaled;
    output_dof_tau = this->params.rl_kp * (all_actions_scaled + this->params.default_dof_pos - this->obs.dof_pos) - this->params.rl_kd * this->obs.dof_vel;
    output_dof_tau = torch::clamp(output_dof_tau, -(this->params.torque_limits), this->params.torque_limits);
}

torch::Tensor RL::QuatRotateInverse(torch::Tensor q, torch::Tensor v)
{
    torch::Tensor q_w;
    torch::Tensor q_vec;

    // wxyz
    q_w = q.index({torch::indexing::Slice(), 0});
    q_vec = q.index({torch::indexing::Slice(), torch::indexing::Slice(1, 4)});

    c10::IntArrayRef shape = q.sizes();

    torch::Tensor a = v * (2.0 * torch::pow(q_w, 2) - 1.0).unsqueeze(-1);
    torch::Tensor b = torch::cross(q_vec, v, -1) * q_w.unsqueeze(-1) * 2.0;
    torch::Tensor c = q_vec * torch::bmm(q_vec.view({shape[0], 1, 3}), v.view({shape[0], 3, 1})).squeeze(-1) * 2.0;
    return a - b + c;
}

void RL::TorqueProtect(torch::Tensor origin_output_dof_tau)
{
    std::vector<int> out_of_range_indices;
    std::vector<double> out_of_range_values;
    for (int i = 0; i < origin_output_dof_tau.size(1); ++i)
    {
        double torque_value = origin_output_dof_tau[0][i].item<double>();
        double limit_lower = -this->params.torque_limits[0][i].item<double>();
        double limit_upper = this->params.torque_limits[0][i].item<double>();

        if (torque_value < limit_lower || torque_value > limit_upper)
        {
            out_of_range_indices.push_back(i);
            out_of_range_values.push_back(torque_value);
        }
    }
    if (!out_of_range_indices.empty())
    {
        for (int i = 0; i < out_of_range_indices.size(); ++i)
        {
            int index = out_of_range_indices[i];
            double value = out_of_range_values[i];
            double limit_lower = -this->params.torque_limits[0][index].item<double>();
            double limit_upper = this->params.torque_limits[0][index].item<double>();

            std::cout << LOGGER::WARNING << "Torque(" << index + 1 << ")=" << value << " out of range(" << limit_lower << ", " << limit_upper << ")" << std::endl;
        }
        // Just a reminder, no protection
        // this->control.SetKeyboard(Input::Keyboard::P);
        // std::cout << LOGGER::INFO << "Switching to STATE_POS_GETDOWN"<< std::endl;
    }
}

void RL::AttitudeProtect(const std::vector<double> &quaternion, float pitch_threshold, float roll_threshold)
{
    float rad2deg = 57.2958;
    float w, x, y, z;

    w = quaternion[0];
    x = quaternion[1];
    y = quaternion[2];
    z = quaternion[3];

    // Calculate roll (rotation around the X-axis)
    float sinr_cosp = 2 * (w * x + y * z);
    float cosr_cosp = 1 - 2 * (x * x + y * y);
    float roll = std::atan2(sinr_cosp, cosr_cosp) * rad2deg;

    // Calculate pitch (rotation around the Y-axis)
    float sinp = 2 * (w * y - z * x);
    float pitch;
    if (std::fabs(sinp) >= 1)
    {
        pitch = std::copysign(90.0, sinp); // Clamp to avoid out-of-range values
    }
    else
    {
        pitch = std::asin(sinp) * rad2deg;
    }

    if (std::fabs(roll) > roll_threshold)
    {
        // this->control.SetKeyboard(Input::Keyboard::P);
        std::cout << LOGGER::WARNING << "Roll exceeds " << roll_threshold << " degrees. Current: " << roll << " degrees." << std::endl;
    }
    if (std::fabs(pitch) > pitch_threshold)
    {
        // this->control.SetKeyboard(Input::Keyboard::P);
        std::cout << LOGGER::WARNING << "Pitch exceeds " << pitch_threshold << " degrees. Current: " << pitch << " degrees." << std::endl;
    }
}

#include <termios.h>
#include <sys/ioctl.h>
static bool kbhit()
{
    termios term;
    tcgetattr(0, &term);

    termios term2 = term;
    term2.c_lflag &= ~ICANON;
    tcsetattr(0, TCSANOW, &term2);

    int byteswaiting;
    ioctl(0, FIONREAD, &byteswaiting);

    tcsetattr(0, TCSANOW, &term);

    return byteswaiting > 0;
}

void RL::KeyboardInterface()
{
    if (kbhit())
    {
        int c = fgetc(stdin);
        switch (c)
        {
        case '0': this->control.SetKeyboard(Input::Keyboard::Num0); break;
        case '1': this->control.SetKeyboard(Input::Keyboard::Num1); break;
        case '2': this->control.SetKeyboard(Input::Keyboard::Num2); break;
        case '3': this->control.SetKeyboard(Input::Keyboard::Num3); break;
        case '4': this->control.SetKeyboard(Input::Keyboard::Num4); break;
        case '5': this->control.SetKeyboard(Input::Keyboard::Num5); break;
        case '6': this->control.SetKeyboard(Input::Keyboard::Num6); break;
        case '7': this->control.SetKeyboard(Input::Keyboard::Num7); break;
        case '8': this->control.SetKeyboard(Input::Keyboard::Num8); break;
        case '9': this->control.SetKeyboard(Input::Keyboard::Num9); break;
        case 'a': case 'A': this->control.SetKeyboard(Input::Keyboard::A); break;
        case 'b': case 'B': this->control.SetKeyboard(Input::Keyboard::B); break;
        case 'c': case 'C': this->control.SetKeyboard(Input::Keyboard::C); break;
        case 'd': case 'D': this->control.SetKeyboard(Input::Keyboard::D); break;
        case 'e': case 'E': this->control.SetKeyboard(Input::Keyboard::E); break;
        case 'f': case 'F': this->control.SetKeyboard(Input::Keyboard::F); break;
        case 'g': case 'G': this->control.SetKeyboard(Input::Keyboard::G); break;
        case 'h': case 'H': this->control.SetKeyboard(Input::Keyboard::H); break;
        case 'i': case 'I': this->control.SetKeyboard(Input::Keyboard::I); break;
        case 'j': case 'J': this->control.SetKeyboard(Input::Keyboard::J); break;
        case 'k': case 'K': this->control.SetKeyboard(Input::Keyboard::K); break;
        case 'l': case 'L': this->control.SetKeyboard(Input::Keyboard::L); break;
        case 'm': case 'M': this->control.SetKeyboard(Input::Keyboard::M); break;
        case 'n': case 'N': this->control.SetKeyboard(Input::Keyboard::N); break;
        case 'o': case 'O': this->control.SetKeyboard(Input::Keyboard::O); break;
        case 'p': case 'P': this->control.SetKeyboard(Input::Keyboard::P); break;
        case 'q': case 'Q': this->control.SetKeyboard(Input::Keyboard::Q); break;
        case 'r': case 'R': this->control.SetKeyboard(Input::Keyboard::R); break;
        case 's': case 'S': this->control.SetKeyboard(Input::Keyboard::S); break;
        case 't': case 'T': this->control.SetKeyboard(Input::Keyboard::T); break;
        case 'u': case 'U': this->control.SetKeyboard(Input::Keyboard::U); break;
        case 'v': case 'V': this->control.SetKeyboard(Input::Keyboard::V); break;
        case 'w': case 'W': this->control.SetKeyboard(Input::Keyboard::W); break;
        case 'x': case 'X': this->control.SetKeyboard(Input::Keyboard::X); break;
        case 'y': case 'Y': this->control.SetKeyboard(Input::Keyboard::Y); break;
        case 'z': case 'Z': this->control.SetKeyboard(Input::Keyboard::Z); break;
        case ' ': this->control.SetKeyboard(Input::Keyboard::Space); break;
        case '\n': case '\r': this->control.SetKeyboard(Input::Keyboard::Enter); break;
        case 27: this->control.SetKeyboard(Input::Keyboard::Escape); break;
        case 0xE0:
        {
            int ext = fgetc(stdin);
            switch (ext)
            {
            case 72: this->control.SetKeyboard(Input::Keyboard::Up); break;
            case 80: this->control.SetKeyboard(Input::Keyboard::Down); break;
            case 75: this->control.SetKeyboard(Input::Keyboard::Left); break;
            case 77: this->control.SetKeyboard(Input::Keyboard::Right); break;
            default:  break;
            }
        } break;
        default:  break;
        }
    }
}

template <typename T>
std::vector<T> ReadVectorFromYaml(const YAML::Node &node)
{
    std::vector<T> values;
    for (const auto &val : node)
    {
        values.push_back(val.as<T>());
    }
    return values;
}

void RL::ValidateRobotManifest(std::string robot_name, bool check_loaded_params)
{
    namespace fs = std::filesystem;
    const fs::path repo_root = fs::path(CMAKE_CURRENT_SOURCE_DIR).parent_path().parent_path().parent_path();
    const fs::path manifest_path = repo_root / "robots" / robot_name / "robot.yaml";
    if (!fs::is_regular_file(manifest_path))
    {
        throw std::runtime_error("robot manifest missing: " + manifest_path.string());
    }
    const YAML::Node man = YAML::LoadFile(manifest_path.string());
    const int num_dofs = man["num_dofs"].as<int>();
    const YAML::Node obs = man["obs_contract"];
    const int num_obs = obs["num_observations"].as<int>();
    const int history = obs["history"].as<int>();
    const std::vector<int> mapping = ReadVectorFromYaml<int>(man["joint_mapping"]);
    const std::vector<double> action_scale = ReadVectorFromYaml<double>(man["action_contract"]["scale"]);
    if (static_cast<int>(mapping.size()) != num_dofs
        || static_cast<int>(action_scale.size()) != num_dofs)
    {
        throw std::runtime_error("robot manifest inconsistent: " + manifest_path.string());
    }
    if (!check_loaded_params)
    {
        return;
    }
    if (this->params.num_of_dofs != num_dofs)
    {
        throw std::runtime_error("policy/robot mismatch: num_of_dofs=" + std::to_string(this->params.num_of_dofs)
                                 + " manifest=" + std::to_string(num_dofs));
    }
    if (this->params.num_observations != num_obs)
    {
        throw std::runtime_error("policy/robot mismatch: num_observations="
                                 + std::to_string(this->params.num_observations)
                                 + " manifest=" + std::to_string(num_obs));
    }
    if (static_cast<int>(this->params.observations_history.size()) != history)
    {
        throw std::runtime_error("policy/robot mismatch: history="
                                 + std::to_string(this->params.observations_history.size())
                                 + " manifest=" + std::to_string(history));
    }
    if (this->params.joint_mapping != mapping)
    {
        throw std::runtime_error("policy/robot mismatch: joint_mapping differs from robot.yaml");
    }
}

void RL::ReadYamlBase(std::string robot_path)
{
    // The config file is located at "rl_sar/src/rl_sar/policy/<robot_path>/base.yaml"
    std::string config_path = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/policy/" + robot_path + "/base.yaml";
    YAML::Node config;
    try
    {
        config = YAML::LoadFile(config_path)[robot_path];
    }
    catch (YAML::BadFile &e)
    {
        std::cout << LOGGER::ERROR << "The file '" << config_path << "' does not exist" << std::endl;
        return;
    }

    this->params.dt = config["dt"].as<double>();
    this->params.decimation = config["decimation"].as<int>();
    // Frequency contract (audit F7): if base.yaml declares control_hz/policy_hz,
    // they must agree with dt/decimation.  A silent mismatch changes the policy
    // period without any operator-visible signal.
    if (config["control_hz"] && config["policy_hz"] && this->params.dt > 0.0
        && this->params.decimation > 0)
    {
        const double control_hz = config["control_hz"].as<double>();
        const double policy_hz = config["policy_hz"].as<double>();
        const double want_control = 1.0 / this->params.dt;
        const double want_policy = want_control / static_cast<double>(this->params.decimation);
        if (std::abs(control_hz - want_control) > 0.02 * want_control
            || std::abs(policy_hz - want_policy) > 0.02 * want_policy)
        {
            throw std::runtime_error(
                "base.yaml frequency mismatch: control_hz=" + std::to_string(control_hz)
                + " policy_hz=" + std::to_string(policy_hz) + " but dt="
                + std::to_string(this->params.dt) + " decimation="
                + std::to_string(this->params.decimation));
        }
    }
    this->params.wheel_indices = ReadVectorFromYaml<int>(config["wheel_indices"]);
    this->params.num_of_dofs = config["num_of_dofs"].as<int>();
    this->params.fixed_kp = torch::tensor(ReadVectorFromYaml<double>(config["fixed_kp"])).view({1, -1});
    this->params.fixed_kd = torch::tensor(ReadVectorFromYaml<double>(config["fixed_kd"])).view({1, -1});
    this->params.torque_limits = torch::tensor(ReadVectorFromYaml<double>(config["torque_limits"])).view({1, -1});
    this->params.default_dof_pos = torch::tensor(ReadVectorFromYaml<double>(config["default_dof_pos"])).view({1, -1});
    this->params.joint_names = ReadVectorFromYaml<std::string>(config["joint_names"]);
    this->params.joint_controller_names = ReadVectorFromYaml<std::string>(config["joint_controller_names"]);
    this->params.joint_mapping = ReadVectorFromYaml<int>(config["joint_mapping"]);
}

void RL::ReadYamlRL(std::string robot_path)
{
    // The config file is located at "rl_sar/src/rl_sar/policy/<robot_path>/config.yaml"
    std::string config_path = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/policy/" + robot_path + "/config.yaml";
    YAML::Node document;
    try
    {
        document = YAML::LoadFile(config_path);
    }
    catch (YAML::BadFile &e)
    {
        std::cout << LOGGER::ERROR << "The file '" << config_path << "' does not exist" << std::endl;
        return;
    }
    // Convention: one top-level key naming the policy dir ("blackW/<policy>"). The legacy weight
    // archive copied many configs from the comp_bc_v5 template and kept the template's wrapper key,
    // while every other tool in the repo (launch preflight, manifest builders, eval) already unwraps
    // the single-entry map generically. Accept that single-entry form here too, so a policy is
    // loadable by its directory name (switchable roster: docs/joystick.md §3/§4).
    YAML::Node config = document[robot_path];
    if (!config.IsDefined() || config.IsNull())
    {
        if (document.size() != 1)
        {
            throw std::runtime_error("policy config has no node '" + robot_path + "' and is not a "
                                     "single-entry map: " + config_path);
        }
        YAML::Node only = document.begin()->second;
        if (!only.IsMap() || !only["num_observations"])
        {
            throw std::runtime_error("policy config has no node '" + robot_path + "' and its single "
                                     "entry is not a policy payload: " + config_path);
        }
        std::cout << LOGGER::WARNING << "policy config wrapper key mismatch in '" << config_path
                  << "' (no node '" << robot_path << "'); using the single top-level entry" << std::endl;
        config = only;
    }

    this->params.model_name = config["model_name"].as<std::string>();
    // Optional per-policy frequency declaration (written by the exporter/config
    // template).  Validated against base.yaml in InitRL.
    this->params.control_hz = config["control_hz"] ? config["control_hz"].as<double>() : -1.0;
    this->params.policy_hz = config["policy_hz"] ? config["policy_hz"].as<double>() : -1.0;
    this->params.num_observations = config["num_observations"].as<int>();
    this->params.observations = ReadVectorFromYaml<std::string>(config["observations"]);
    if (config["observations_history"].IsNull())
    {
        this->params.observations_history = {};
    }
    else
    {
        this->params.observations_history = ReadVectorFromYaml<int>(config["observations_history"]);
    }
    this->params.clip_obs = config["clip_obs"].as<double>();
    if (config["clip_actions_lower"].IsNull() && config["clip_actions_upper"].IsNull())
    {
        this->params.clip_actions_upper = torch::tensor({}).view({1, -1});
        this->params.clip_actions_lower = torch::tensor({}).view({1, -1});
    }
    else
    {
        this->params.clip_actions_upper = torch::tensor(ReadVectorFromYaml<double>(config["clip_actions_upper"])).view({1, -1});
        this->params.clip_actions_lower = torch::tensor(ReadVectorFromYaml<double>(config["clip_actions_lower"])).view({1, -1});
    }
    this->params.action_scale = torch::tensor(ReadVectorFromYaml<double>(config["action_scale"])).view({1, -1});
    this->params.wheel_indices = ReadVectorFromYaml<int>(config["wheel_indices"]);
    this->params.num_of_dofs = config["num_of_dofs"].as<int>();
    this->params.lin_vel_scale = config["lin_vel_scale"].as<double>();
    this->params.ang_vel_scale = config["ang_vel_scale"].as<double>();
    this->params.dof_pos_scale = config["dof_pos_scale"].as<double>();
    this->params.dof_vel_scale = config["dof_vel_scale"].as<double>();
    this->params.commands_scale = torch::tensor(ReadVectorFromYaml<double>(config["commands_scale"])).view({1, -1});
    // this->params.commands_scale = torch::tensor({this->params.lin_vel_scale, this->params.lin_vel_scale, this->params.ang_vel_scale});
    this->params.rl_kp = torch::tensor(ReadVectorFromYaml<double>(config["rl_kp"])).view({1, -1});
    this->params.rl_kd = torch::tensor(ReadVectorFromYaml<double>(config["rl_kd"])).view({1, -1});
    this->params.fixed_kp = torch::tensor(ReadVectorFromYaml<double>(config["fixed_kp"])).view({1, -1});
    this->params.fixed_kd = torch::tensor(ReadVectorFromYaml<double>(config["fixed_kd"])).view({1, -1});
    this->params.torque_limits = torch::tensor(ReadVectorFromYaml<double>(config["torque_limits"])).view({1, -1});
    this->params.default_dof_pos = torch::tensor(ReadVectorFromYaml<double>(config["default_dof_pos"])).view({1, -1});
    this->params.joint_mapping = ReadVectorFromYaml<int>(config["joint_mapping"]);
}

void RL::CSVInit(std::string robot_path)
{
    csv_filename = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/policy/" + robot_path + "/motor";

    // Uncomment these lines if need timestamp for file name
    // auto now = std::chrono::system_clock::now();
    // std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    // std::stringstream ss;
    // ss << std::put_time(std::localtime(&now_c), "%Y%m%d%H%M%S");
    // std::string timestamp = ss.str();
    // csv_filename += "_" + timestamp;

    csv_filename += ".csv";
    std::ofstream file(csv_filename.c_str());

    for(int i = 0; i < this->params.num_of_dofs; ++i) { file << "tau_cal_" << i << ","; }
    for(int i = 0; i < this->params.num_of_dofs; ++i) { file << "tau_est_" << i << ","; }
    for(int i = 0; i < this->params.num_of_dofs; ++i) { file << "joint_pos_" << i << ","; }
    for(int i = 0; i < this->params.num_of_dofs; ++i) { file << "joint_pos_target_" << i << ","; }
    for(int i = 0; i < this->params.num_of_dofs; ++i) { file << "joint_vel_" << i << ","; }

    file << std::endl;

    file.close();
}

void RL::CSVLogger(torch::Tensor torque, torch::Tensor tau_est, torch::Tensor joint_pos, torch::Tensor joint_pos_target, torch::Tensor joint_vel)
{
    std::ofstream file(csv_filename.c_str(), std::ios_base::app);

    for(int i = 0; i < this->params.num_of_dofs; ++i) { file << torque[0][i].item<double>() << ","; }
    for(int i = 0; i < this->params.num_of_dofs; ++i) { file << tau_est[0][i].item<double>() << ","; }
    for(int i = 0; i < this->params.num_of_dofs; ++i) { file << joint_pos[0][i].item<double>() << ","; }
    for(int i = 0; i < this->params.num_of_dofs; ++i) { file << joint_pos_target[0][i].item<double>() << ","; }
    for(int i = 0; i < this->params.num_of_dofs; ++i) { file << joint_vel[0][i].item<double>() << ","; }

    file << std::endl;

    file.close();
}
