/*
 * New file for @20261007_assignment - no upstream counterpart, so it carries no upstream notice
 * (the package LICENSE covers it; see UPSTREAM.md §2).
 *
 * What it holds: the deployment knobs this port reads from policy/<robot>/base.yaml instead of
 * hardcoding them in the state machine, plus the base.yaml vs <policy>/config.yaml consistency check.
 * Kept out of policy/black/fsm.hpp so that the state machine stays close to upstream
 * policy/go2/fsm.hpp. See UPSTREAM.md §2.
 */

#ifndef BLACK_KNOBS_HPP
#define BLACK_KNOBS_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace black_fsm
{

/// Deployment knobs that upstream hardcodes in this file. We read them from policy/<robot>/base.yaml
/// so the stand-up duration and the passive damping can be tuned without a rebuild. Defaults are the
/// values this port used to hardcode. Missing file or keys -> defaults (and one warning).
struct Knobs
{
    double kd_passive = 8.0;      ///< kd in passive mode (upstream README: "Motor passive mode (kp=0, kd=8)")
    int getup_pre_cycles = 200;   ///< control cycles of the pre-stand crouch ramp
    int getup_cycles = 400;       ///< control cycles of the stand-up ramp
    int getdown_cycles = 500;     ///< control cycles of the lie-down ramp

    // Command-side limiting (command_limit.hpp): the node never sends torque, so the
    // limits are expressed on the MIT command we do send - see include/command_limit.hpp.
    double command_tau_limit = 33.5;  ///< implied |tau| cap [N.m] at the measured state (= the trained envelope)
    double command_kp_max = 80.0;     ///< gain caps (80/10 = the largest this port ever sends)
    double command_kd_max = 10.0;
    double command_dq_max = 20.0;     ///< |dq_des| cap [rad/s]
    double command_slew = 60.0;       ///< max |q_des| rate [rad/s]; 0 disables the slew rule
};

inline std::string PolicyDir(const std::string &robot_name)
{
    return std::string(CMAKE_CURRENT_SOURCE_DIR) + "/policy/" + robot_name;
}

inline YAML::Node LoadYamlOrWarn(const std::string &path, const char *what)
{
    try
    {
        return YAML::LoadFile(path);
    }
    catch (const std::exception &e)
    {
        std::cout << LOGGER::WARNING << "[yaml] cannot read " << what << " " << path << ": " << e.what() << std::endl;
        return YAML::Node(YAML::NodeType::Undefined);
    }
}

/// Read the deployment knobs once (lazily). Explicit values win; anything missing keeps the default.
inline Knobs LoadKnobs(const std::string &robot_name)
{
    Knobs k;
    const std::string path = PolicyDir(robot_name) + "/base.yaml";
    YAML::Node base = LoadYamlOrWarn(path, "base.yaml");
    const YAML::Node node = base[robot_name] ? base[robot_name] : base;
    if (node["kd_passive"])
    {
        k.kd_passive = node["kd_passive"].as<double>();
    }
    if (node["getup_pre_cycles"])
    {
        k.getup_pre_cycles = node["getup_pre_cycles"].as<int>();
    }
    if (node["getup_cycles"])
    {
        k.getup_cycles = node["getup_cycles"].as<int>();
    }
    if (node["getdown_cycles"])
    {
        k.getdown_cycles = node["getdown_cycles"].as<int>();
    }
    if (node["command_tau_limit"]) k.command_tau_limit = node["command_tau_limit"].as<double>();
    if (node["command_kp_max"]) k.command_kp_max = node["command_kp_max"].as<double>();
    if (node["command_kd_max"]) k.command_kd_max = node["command_kd_max"].as<double>();
    if (node["command_dq_max"]) k.command_dq_max = node["command_dq_max"].as<double>();
    if (node["command_slew"]) k.command_slew = node["command_slew"].as<double>();
    k.getup_pre_cycles = std::max(1, k.getup_pre_cycles);
    k.getup_cycles = std::max(1, k.getup_cycles);
    k.getdown_cycles = std::max(1, k.getdown_cycles);
    std::cout << LOGGER::INFO << "[yaml] knobs from " << path << ": kd_passive=" << k.kd_passive
              << " getup_pre_cycles=" << k.getup_pre_cycles << " getup_cycles=" << k.getup_cycles
              << " getdown_cycles=" << k.getdown_cycles << " command_tau_limit=" << k.command_tau_limit
              << " command_kp_max=" << k.command_kp_max << " command_kd_max=" << k.command_kd_max
              << " command_dq_max=" << k.command_dq_max << " command_slew=" << k.command_slew
              << std::endl;
    return k;
}

inline const Knobs &GetKnobs(const std::string &robot_name)
{
    static const Knobs knobs = LoadKnobs(robot_name); // one robot per process
    return knobs;
}

/// Compact one-line rendering of a yaml sequence for the mismatch warning.
inline std::string FormatSeq(const YAML::Node &node, size_t max_items = 4)
{
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < node.size(); ++i)
    {
        if (i > 0)
        {
            out << ", ";
        }
        if (i == max_items)
        {
            out << "... (" << node.size() << ")";
            break;
        }
        out << node[i].as<double>();
    }
    out << "]";
    return out.str();
}

/// The 7 keys that appear in both base.yaml and <policy>/config.yaml; config.yaml overwrites base.yaml
/// when entering RL, so equal values are required. Warn (do not fail) on mismatch - see
/// UPSTREAM.md §2.
inline void WarnIfBaseConfigMismatch(const std::string &robot_name, const std::string &config_name)
{
    const std::string base_path = PolicyDir(robot_name) + "/base.yaml";
    const std::string conf_path = PolicyDir(robot_name) + "/" + config_name + "/config.yaml";
    YAML::Node base = LoadYamlOrWarn(base_path, "base.yaml");
    YAML::Node conf = LoadYamlOrWarn(conf_path, "config.yaml");
    const YAML::Node b = base[robot_name] ? base[robot_name] : base;
    const YAML::Node c = conf[robot_name + "/" + config_name] ? conf[robot_name + "/" + config_name] : conf;

    const char *keys[] = {"default_dof_pos", "fixed_kp", "fixed_kd", "joint_mapping",
                          "num_of_dofs", "torque_limits", "wheel_indices"};
    std::vector<std::string> bad;
    for (const char *key : keys)
    {
        if (!b[key] || !c[key])
        {
            continue; // one side missing: nothing to compare (the loader has its own required-field errors)
        }
        const YAML::Node bn = b[key];
        const YAML::Node cn = c[key];
        bool same = bn.size() == cn.size();
        for (size_t i = 0; same && i < bn.size(); ++i)
        {
            same = std::fabs(bn[i].as<double>() - cn[i].as<double>()) <= 1e-9;
        }
        if (!same)
        {
            bad.emplace_back(std::string(key) + " (base " + FormatSeq(bn) + " vs config " + FormatSeq(cn) + ")");
        }
    }
    if (!bad.empty())
    {
        std::cout << LOGGER::WARNING << "[yaml] base.yaml and " << config_name
                  << "/config.yaml disagree, so behaviour changes when entering RL:" << std::endl;
        for (const std::string &item : bad)
        {
            std::cout << LOGGER::WARNING << "[yaml]   " << item << std::endl;
        }
        std::cout << LOGGER::WARNING << "[yaml] make the " << bad.size()
                  << " key(s) above identical in both files (UPSTREAM.md §2)" << std::endl;
    }
}

} // namespace black_fsm

#endif // BLACK_KNOBS_HPP
