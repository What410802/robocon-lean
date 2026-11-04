/*
 * New file for @20261007_assignment - no upstream counterpart, so it carries no upstream notice
 * (the package LICENSE covers it; see UPSTREAM.md §2).
 *
 * What it is: the input layer this port uses instead of the verbatim library's `Input` latch.
 * library/core's `Input` keeps ONE enum per device ("last write wins"), so "was just pressed",
 * "already consumed" and "which key wins" all have to be approximated by mutating it at call sites -
 * that produced several bugs (including the RB_A/RB_B combo aliasing; see UPSTREAM.md §2).
 *
 * Here instead:
 *   * every key is a bit: `down` (level now) + `prev` (level last cycle) -> `pressed()` is a pure
 *     function of two consecutive cycles, so an edge happens exactly once and needs no clearing;
 *   * the pad's hat directions are just more bits, so a combination like RB + D-pad up is one mask,
 *     not a hand-written enum value that some consumers know and others do not;
 *   * `kBindings` is the key map as DATA - the operator-facing table in docs/joystick.md §2 and
 *     this table are the same list, so adding a key means adding one row (the switchable policy
 *     roster and the Part1 task playbook live in docs/joystick.md §3/§5);
 *   * presses made while in a previous FSM state are simply gone (the edge was consumed there),
 *     which is exactly the rule the old latch had to approximate by hand.
 *
 * Long-press: not needed today. `KeyBits` is the place to add it (`held_cycles[]` or a timestamp per
 * key) - rl_sim sees every /joy frame, so it can measure duration itself without any message change.
 *
 * Visibility: the FSM states only receive `RL&`, not the node, so the resolved intents live in a
 * process-global (`Current()`); this port assumes one rl_sim per process, same as `GetKnobs()`.
 * See UPSTREAM.md §2 for the porting history (the pre-migration task docs are not shipped here).
 */

#ifndef INPUT_MAP_HPP
#define INPUT_MAP_HPP

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace input_map
{

// ---------------------------------------------------------------- keys as bits

/// Gamepad keys, including the four hat directions (derived from axes[6]/[7], not from buttons[]).
enum class PadKey : std::uint8_t
{
    A, B, X, Y, LB, RB, Back, Start, Guide, LStick, RStick,
    DPadUp, DPadDown, DPadLeft, DPadRight,
    COUNT
};

/// Keyboard keys we use. The library's KeyboardInterface() is a single-value latch (a terminal
/// reports key presses only, never releases), so this source can only ever have one key "down".
enum class KbKey : std::uint8_t
{
    Num0, Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9,
    W, A, S, D, Q, E, F, T, P, R, N, Space, B,
    COUNT
};

inline constexpr std::size_t kPadKeyCount = static_cast<std::size_t>(PadKey::COUNT);
inline constexpr std::size_t kKbKeyCount = static_cast<std::size_t>(KbKey::COUNT);

inline constexpr std::uint32_t Bit(PadKey k) { return 1u << static_cast<std::uint32_t>(k); }
inline constexpr std::uint32_t Bit(KbKey k) { return 1u << static_cast<std::uint32_t>(k); }

/// Level + previous level for one input source. `pressed()` is one-shot by construction.
struct KeyBits
{
    std::uint32_t down = 0;
    std::uint32_t prev = 0;

    /// All bits of the mask are down right now.
    bool held(std::uint32_t mask) const { return (down & mask) == mask && mask != 0; }
    /// The mask is complete now but was not complete last cycle: fires once, on the cycle the *last*
    /// key of a combination arrives (for a single key this is the plain rising edge, hence no separate
    /// "single key" path is needed).
    bool pressed(std::uint32_t mask) const { return held(mask) && (prev & mask) != mask; }
    /// The mask was complete last cycle and is not complete now.
    bool released(std::uint32_t mask) const { return !held(mask) && (prev & mask) == mask && mask != 0; }
    /// Any bit rose this cycle.
    bool any_pressed() const { return (down & ~prev) != 0; }
    void advance(std::uint32_t now) { prev = down; down = now; }
};

// ---------------------------------------------------------------- intents

/// What the app is asked to do this cycle (edges only; levels are handled as commands below).
enum class Intent : std::uint8_t
{
    StandUp,            // A / 0
    LieDown,            // B / 9
    EnterRL,            // RB + D-pad up / 1
    Damp,               // LB + X / P
    ResetSim,           // RB + Y / R
    ToggleNavigation,   // X / N
    JogForward,         // W   (keyboard-only command jog, one step per press)
    JogBackward,        // S
    JogLeft,            // A
    JogRight,           // D
    JogYawLeft,         // Q
    JogYawRight,        // E
    ZeroCommand,        // Space
    PolicyFlat,         // Start+D-pad left / 2
    PolicyStair,        // Start+D-pad up / 3
    PolicyObstacle,     // Start+D-pad right / 4
    PolicyMain,         // Start+D-pad down / 8   (row2p_p40w40_v1: Part2 main, Part1 wall/bridge/slope)
    PolicyTStairs,      // RB+D-pad down / F      (st30_ppo_st30_s100_600: Part1 T-stairs specialist)
    PolicyTrench30,     // RB+D-pad right / T     (s1d_trench_reward_v75_m100: Part2 trench_30cm specialist)
    PolicyBand,         // RB+D-pad left / B      (vt_band_v1: Part2 stairs_30cm specialist)
    PolicyPose,         // LB+D-pad left           (vt_pose_v1: low-speed closed-loop pose/alignment)
    PolicyEdge,         // Back+D-pad up           (vt_edge_v1: wheel-off-edge recovery specialist)
    PolicyTrack,        // LB+D-pad right          (vt_track_v1: velocity-tracking Part1 handover primitive)
    PolicyTrackV2,      // LB+D-pad up             (vt_track_v2: contact-plane v2 recipe)
    PolicyNwwolf,       // LB+D-pad down           (vt_locked12: 12-DoF locked-wheel connector)
    LowBarDrive,        // Back+D-pad down / 5
    BridgeDrive,        // Back+D-pad right / 6
    CarDrive,           // Back+D-pad left / 7
    COUNT
};

inline constexpr std::size_t kIntentCount = static_cast<std::size_t>(Intent::COUNT);
using IntentMask = std::uint32_t;
inline constexpr IntentMask Mask(Intent i) { return 1u << static_cast<std::uint32_t>(i); }

/// One row of the key map: this pad mask OR this keyboard key produces this intent.
/// A pad row fires on the rising edge of "all bits of the mask are down", so RB+D-pad up is simply
/// two bits and an unbound combination (RB + A) produces nothing at all.
struct Binding
{
    std::uint32_t pad;
    std::uint32_t kb;
    Intent intent;
};

inline constexpr Binding kBindings[] =
{
    // pad (mask)                    keyboard (single key)      intent
    { Bit(PadKey::A),                0,                         Intent::StandUp },
    { Bit(PadKey::B),                0,                         Intent::LieDown },
    { Bit(PadKey::RB) | Bit(PadKey::DPadUp), 0,                Intent::EnterRL },
    { Bit(PadKey::LB) | Bit(PadKey::X), 0,                     Intent::Damp },
    { Bit(PadKey::RB) | Bit(PadKey::Y), 0,                     Intent::ResetSim },
    { Bit(PadKey::X),                0,                         Intent::ToggleNavigation },
    { 0,                             Bit(KbKey::Num0),          Intent::StandUp },
    { 0,                             Bit(KbKey::Num9),          Intent::LieDown },
    { 0,                             Bit(KbKey::Num1),          Intent::EnterRL },
    { 0,                             Bit(KbKey::P),             Intent::Damp },
    { 0,                             Bit(KbKey::R),             Intent::ResetSim },
    { 0,                             Bit(KbKey::N),             Intent::ToggleNavigation },
    { 0,                             Bit(KbKey::W),             Intent::JogForward },
    { 0,                             Bit(KbKey::S),             Intent::JogBackward },
    { 0,                             Bit(KbKey::A),             Intent::JogLeft },
    { 0,                             Bit(KbKey::D),             Intent::JogRight },
    { 0,                             Bit(KbKey::Q),             Intent::JogYawLeft },
    { 0,                             Bit(KbKey::E),             Intent::JogYawRight },
    { 0,                             Bit(KbKey::Space),         Intent::ZeroCommand },
    // Policy and fixed-drive combos are modifier + D-pad only (Start / Back / RB): those modifiers have
    // no single-key binding, so a combo cannot also fire the single-key action it is built on.
    // Frontier-model roster (2026-10-31): Start+down = PolicyMain (row2p_p40w40_v1, Part2 main and
    // Part1 high wall / bridge blocks / slope), RB+down = PolicyTStairs (st30_ppo_st30_s100_600,
    // Part1 T-stairs), RB+right = PolicyTrench30 (s1d_trench_reward_v75_m100, Part2 trench_30cm).
    // See docs/joystick.md §3 for the roster and §5/§6 for the task playbook.
    { Bit(PadKey::Start) | Bit(PadKey::DPadLeft), 0,          Intent::PolicyFlat },
    { Bit(PadKey::Start) | Bit(PadKey::DPadUp), 0,            Intent::PolicyStair },
    { Bit(PadKey::Start) | Bit(PadKey::DPadRight), 0,         Intent::PolicyObstacle },
    { Bit(PadKey::Start) | Bit(PadKey::DPadDown), 0,          Intent::PolicyMain },
    { Bit(PadKey::RB) | Bit(PadKey::DPadDown), 0,           Intent::PolicyTStairs },
    { Bit(PadKey::RB) | Bit(PadKey::DPadRight), 0,          Intent::PolicyTrench30 },
    { Bit(PadKey::RB) | Bit(PadKey::DPadLeft), 0,           Intent::PolicyBand },
    { Bit(PadKey::LB) | Bit(PadKey::DPadLeft), 0,           Intent::PolicyPose },
    { Bit(PadKey::LB) | Bit(PadKey::DPadRight), 0,          Intent::PolicyTrack },
    { Bit(PadKey::LB) | Bit(PadKey::DPadUp), 0,             Intent::PolicyTrackV2 },
    { Bit(PadKey::LB) | Bit(PadKey::DPadDown), 0,           Intent::PolicyNwwolf },
    { Bit(PadKey::Back) | Bit(PadKey::DPadUp), 0,           Intent::PolicyEdge },
    { 0,                             Bit(KbKey::Num2),          Intent::PolicyFlat },
    { 0,                             Bit(KbKey::Num3),          Intent::PolicyStair },
    { 0,                             Bit(KbKey::Num4),          Intent::PolicyObstacle },
    { 0,                             Bit(KbKey::Num8),          Intent::PolicyMain },
    { 0,                             Bit(KbKey::F),             Intent::PolicyTStairs },
    { 0,                             Bit(KbKey::T),             Intent::PolicyTrench30 },
    { 0,                             Bit(KbKey::B),             Intent::PolicyBand },
    { Bit(PadKey::Back) | Bit(PadKey::DPadDown), 0,           Intent::LowBarDrive },
    { Bit(PadKey::Back) | Bit(PadKey::DPadRight), 0,          Intent::BridgeDrive },
    { Bit(PadKey::Back) | Bit(PadKey::DPadLeft), 0,           Intent::CarDrive },
    { 0,                             Bit(KbKey::Num5),          Intent::LowBarDrive },
    { 0,                             Bit(KbKey::Num6),          Intent::BridgeDrive },
    { 0,                             Bit(KbKey::Num7),          Intent::CarDrive },
};

/// Everything the resolver needs: the two key sources' levels, plus the axes as levels.
struct Frame
{
    KeyBits pad;
    KeyBits kb;
    std::array<float, 8> axes{};
    IntentMask intents = 0;    ///< edges resolved for THIS cycle
    bool had_joy_frame = false; ///< false until the first /joy message arrives

    bool has(Intent i) const { return (intents & Mask(i)) != 0; }
    std::vector<Intent> list() const
    {
        std::vector<Intent> out;
        for (std::size_t i = 0; i < kIntentCount; ++i)
        {
            if (intents & (1u << i))
            {
                out.push_back(static_cast<Intent>(i));
            }
        }
        return out;
    }
};

/// Process-global frame: rl_sim writes it once per control cycle, the FSM states read the intents.
inline Frame &Current()
{
    static Frame frame;
    return frame;
}

/// Build the pad key mask from a /joy message's raw arrays (indices follow docs/joystick.md §1).
inline std::uint32_t PadMaskFromJoy(const std::vector<int32_t> &buttons, const std::vector<float> &axes)
{
    std::uint32_t mask = 0;
    if (buttons.size() >= 11)
    {
        if (buttons[0]) mask |= Bit(PadKey::A);
        if (buttons[1]) mask |= Bit(PadKey::B);
        if (buttons[2]) mask |= Bit(PadKey::X);
        if (buttons[3]) mask |= Bit(PadKey::Y);
        if (buttons[4]) mask |= Bit(PadKey::LB);
        if (buttons[5]) mask |= Bit(PadKey::RB);
        if (buttons[6]) mask |= Bit(PadKey::Back);
        if (buttons[7]) mask |= Bit(PadKey::Start);
        if (buttons[8]) mask |= Bit(PadKey::Guide);
        if (buttons[9]) mask |= Bit(PadKey::LStick);
        if (buttons[10]) mask |= Bit(PadKey::RStick);
    }
    if (axes.size() >= 8)
    {
        // Hat as four keys: up / right are +, per the same convention.
        if (axes[7] > 0.5f) mask |= Bit(PadKey::DPadUp);
        if (axes[7] < -0.5f) mask |= Bit(PadKey::DPadDown);
        if (axes[6] > 0.5f) mask |= Bit(PadKey::DPadRight);
        if (axes[6] < -0.5f) mask |= Bit(PadKey::DPadLeft);
    }
    return mask;
}

/// Resolve the edges of the current cycle into intents. Call once per control cycle, before the FSM.
inline void BeginCycle(std::uint32_t pad_down, std::uint32_t kb_down, const std::array<float, 8> &axes)
{
    Frame &frame = Current();
    frame.pad.advance(pad_down);
    frame.kb.advance(kb_down);
    frame.axes = axes;
    frame.intents = 0;
    for (const Binding &b : kBindings)
    {
        if ((b.pad && frame.pad.pressed(b.pad)) || (b.kb && frame.kb.pressed(b.kb)))
        {
            frame.intents |= Mask(b.intent);
        }
    }
}

/// Same, but also remember that a /joy frame has been seen (used to decide whether to apply axes).
inline void BeginCycleWithJoy(std::uint32_t pad_down, std::uint32_t kb_down,
                              const std::array<float, 8> &axes, bool had_joy_frame)
{
    BeginCycle(pad_down, kb_down, axes);
    Current().had_joy_frame = Current().had_joy_frame || had_joy_frame;
}

/// Level query: are the keys bound to this intent all down right now?
/// Intents are edges by design; this exists for the one place where holding through a ramp is the
/// natural gesture (GetUp: press RB+D-pad up a moment before the robot finishes standing and keep
/// holding). Callers must gate it themselves so it cannot re-fire in a loop.
inline bool Held(Intent intent)
{
    const Frame &frame = Current();
    for (const Binding &b : kBindings)
    {
        if (b.intent != intent)
        {
            continue;
        }
        if ((b.pad && frame.pad.held(b.pad)) || (b.kb && frame.kb.held(b.kb)))
        {
            return true;
        }
    }
    return false;
}

/// Names, for logs and for the self-test.
inline const char *Name(Intent i)
{
    switch (i)
    {
    case Intent::StandUp: return "StandUp";
    case Intent::LieDown: return "LieDown";
    case Intent::EnterRL: return "EnterRL";
    case Intent::Damp: return "Damp";
    case Intent::ResetSim: return "ResetSim";
    case Intent::ToggleNavigation: return "ToggleNavigation";
    case Intent::JogForward: return "JogForward";
    case Intent::JogBackward: return "JogBackward";
    case Intent::JogLeft: return "JogLeft";
    case Intent::JogRight: return "JogRight";
    case Intent::JogYawLeft: return "JogYawLeft";
    case Intent::JogYawRight: return "JogYawRight";
    case Intent::ZeroCommand: return "ZeroCommand";
    case Intent::PolicyFlat: return "PolicyFlat";
    case Intent::PolicyStair: return "PolicyStair";
    case Intent::PolicyObstacle: return "PolicyObstacle";
    case Intent::PolicyMain: return "PolicyMain";
    case Intent::PolicyTStairs: return "PolicyTStairs";
    case Intent::PolicyTrench30: return "PolicyTrench30";
    case Intent::PolicyBand: return "PolicyBand";
    case Intent::PolicyPose: return "PolicyPose";
    case Intent::PolicyNwwolf: return "PolicyNwwolf";
    case Intent::LowBarDrive: return "LowBarDrive";
    case Intent::BridgeDrive: return "BridgeDrive";
    case Intent::CarDrive: return "CarDrive";
    default: return "?";
    }
}

} // namespace input_map

#endif // INPUT_MAP_HPP
