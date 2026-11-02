/*
 * New file for @20261007_assignment - no upstream counterpart (package LICENSE covers it).
 *
 * Self-test for include/input_map.hpp: feeds synthetic frames and asserts the resolved intents. It
 * needs no ROS graph, no sim and no hardware - the point of moving the key map into a pure function
 * (UPSTREAM.md §2). Case names are kept short and English to match the rest of the package; the
 * Chinese explanation of each group is in UPSTREAM.md §2. Run it with:
 *
 *     ros2 run rl_sar input_map_selftest          # direct, most convenient
 *     colcon test --packages-select rl_sar        # via CTest (from the demo env)
 *
 * Exit code 0 = all cases pass, 1 = at least one failed (printed with the observed intents).
 */

#include "input_map.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace
{

using input_map::Bit;
using input_map::Intent;
using input_map::PadKey;
using input_map::KbKey;

int g_failures = 0;

std::string Names(input_map::IntentMask mask)
{
    std::string out;
    for (std::size_t i = 0; i < input_map::kIntentCount; ++i)
    {
        if (mask & (1u << i))
        {
            if (!out.empty())
            {
                out += "+";
            }
            out += input_map::Name(static_cast<Intent>(i));
        }
    }
    return out.empty() ? "(none)" : out;
}

/// Feed one cycle and return the intents it produced.
input_map::IntentMask Step(std::uint32_t pad, std::uint32_t kb = 0)
{
    input_map::BeginCycle(pad, kb, {});
    return input_map::Current().intents;
}

void Check(const std::string &what, input_map::IntentMask got, input_map::IntentMask want)
{
    const bool ok = got == want;
    if (!ok)
    {
        ++g_failures;
    }
    std::printf("  [%s] %-52s got %-22s want %s\n", ok ? "PASS" : "FAIL", what.c_str(),
                Names(got).c_str(), Names(want).c_str());
}

void ExpectTrue(const std::string &what, bool ok)
{
    if (!ok)
    {
        ++g_failures;
    }
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
}

constexpr std::uint32_t A = Bit(PadKey::A);
constexpr std::uint32_t B = Bit(PadKey::B);
constexpr std::uint32_t X = Bit(PadKey::X);
constexpr std::uint32_t Y = Bit(PadKey::Y);
constexpr std::uint32_t LB = Bit(PadKey::LB);
constexpr std::uint32_t RB = Bit(PadKey::RB);
constexpr std::uint32_t START = Bit(PadKey::Start);
constexpr std::uint32_t UP = Bit(PadKey::DPadUp);
constexpr std::uint32_t DOWN = Bit(PadKey::DPadDown);
constexpr std::uint32_t LEFT = Bit(PadKey::DPadLeft);
constexpr std::uint32_t RIGHT = Bit(PadKey::DPadRight);

constexpr input_map::IntentMask I(Intent i) { return input_map::Mask(i); }

} // namespace

int main()
{
    std::printf("input_map self-test\n");

    std::printf("1. edges are one-shot by construction\n");
    Check("idle", Step(0), 0);
    Check("press A", Step(A), I(Intent::StandUp));
    Check("A still held (no repeat)", Step(A), 0);
    Check("release A", Step(0), 0);
    Check("press A again", Step(A), I(Intent::StandUp));

    std::printf("2. a latched D-pad must not swallow buttons\n");
    Check("D-pad down latched", Step(DOWN), 0);
    Check("latched down + press A", Step(DOWN | A), I(Intent::StandUp));
    Check("release A, D-pad still latched", Step(DOWN), 0);
    Check("latched down + press B", Step(DOWN | B), I(Intent::LieDown));

    std::printf("3. combinations (the hat is just another bit)\n");
    Check("hold RB", Step(RB), 0);
    Check("then D-pad up -> RB+up = EnterRL", Step(RB | UP), I(Intent::EnterRL));
    Check("both held (no repeat)", Step(RB | UP), 0);
    Check("release D-pad", Step(RB), 0);
    Check("D-pad up again -> EnterRL again", Step(RB | UP), I(Intent::EnterRL));
    Check("D-pad up alone (unbound)", Step(UP), 0);
    Check("LB + X -> Damp and ToggleNavigation (X is bound too)",
          Step(LB | X), I(Intent::Damp) | I(Intent::ToggleNavigation));

    std::printf("3b. level query Held() for press-early-and-hold gestures\n");
    Step(RB | UP);
    ExpectTrue("Held(EnterRL) while RB+up is down", input_map::Held(Intent::EnterRL));
    Step(RB);
    ExpectTrue("Held(EnterRL) false after releasing D-pad", !input_map::Held(Intent::EnterRL));
    ExpectTrue("Held(StandUp) false (A not held)", !input_map::Held(Intent::StandUp));

    std::printf("4. holding RB while pressing A still fires A\n");
    Check("hold RB", Step(RB), 0);
    Check("press A while RB is held", Step(RB | A), I(Intent::StandUp));
    Check("release A (RB still held)", Step(RB), 0);

    std::printf("5. reset combo and single-key bindings\n");
    Check("RB+Y -> ResetSim", Step(RB | Y), I(Intent::ResetSim));
    Check("X -> ToggleNavigation", Step(X), I(Intent::ToggleNavigation));

    std::printf("6. keyboard source: single-value latch, a change is a press\n");
    Check("press 0 -> StandUp", Step(0, Bit(KbKey::Num0)), I(Intent::StandUp));
    Check("0 still the last key (no repeat)", Step(0, Bit(KbKey::Num0)), 0);
    Check("press W -> JogForward", Step(0, Bit(KbKey::W)), I(Intent::JogForward));
    Check("press 1 -> EnterRL", Step(0, Bit(KbKey::Num1)), I(Intent::EnterRL));
    Check("press space -> ZeroCommand", Step(0, Bit(KbKey::Space)), I(Intent::ZeroCommand));

    std::printf("7. pad and keyboard can fire in the same cycle\n");
    Check("pad A + keyboard 1", Step(A, Bit(KbKey::Num1)), I(Intent::StandUp) | I(Intent::EnterRL));

    std::printf("8. axes mapping: hat up/right positive, usable as bits\n");
    ExpectTrue("axes[7]=+1 -> DPadUp",
               input_map::PadMaskFromJoy({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 1}) == UP);
    ExpectTrue("axes[7]=-1 -> DPadDown",
               input_map::PadMaskFromJoy({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, -1}) == DOWN);
    ExpectTrue("axes[6]=+1 -> DPadRight",
               input_map::PadMaskFromJoy({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 1, 0}) ==
               Bit(PadKey::DPadRight));
    ExpectTrue("buttons[5] -> RB",
               input_map::PadMaskFromJoy({0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0}, {}) == RB);
    ExpectTrue("too-short message is not treated as any key",
               input_map::PadMaskFromJoy({1, 0, 0}, {0, 0, 0}) == 0);

    std::printf("9. frontier policy roster bindings (docs/joystick.md §3)\n");
    Check("Start alone (unbound)", Step(START), 0);
    Check("D-pad down alone (unbound)", Step(DOWN), 0);
    Check("hold Start", Step(START), 0);
    Check("then D-pad left -> PolicyFlat", Step(START | LEFT), I(Intent::PolicyFlat));
    Check("both held (no repeat)", Step(START | LEFT), 0);
    Check("release D-pad, press up -> PolicyStair", Step(START | UP), I(Intent::PolicyStair));
    Check("press right -> PolicyObstacle", Step(START | RIGHT), I(Intent::PolicyObstacle));
    Check("press down -> PolicyMain (row2p_p40w40_v1)", Step(START | DOWN), I(Intent::PolicyMain));
    Check("Start+down held (no repeat)", Step(START | DOWN), 0);
    Check("RB alone (unbound)", Step(RB), 0);
    Check("RB+D-pad down -> PolicyTStairs (st30_ppo_st30_s100_600)",
          Step(RB | DOWN), I(Intent::PolicyTStairs));
    Check("RB+D-pad right -> PolicyTrench30 (s1d_trench_reward_v75_m100)",
          Step(RB | RIGHT), I(Intent::PolicyTrench30));
    Check("RB+up still EnterRL", Step(RB | UP), I(Intent::EnterRL));

    std::printf("10. keyboard twins for the frontier policies: 8 = main, F = T-stairs\n");
    Check("press 8 -> PolicyMain", Step(0, Bit(KbKey::Num8)), I(Intent::PolicyMain));
    Check("8 still last key (no repeat)", Step(0, Bit(KbKey::Num8)), 0);
    Check("press F -> PolicyTStairs", Step(0, Bit(KbKey::F)), I(Intent::PolicyTStairs));
    Check("press 2 -> PolicyFlat", Step(0, Bit(KbKey::Num2)), I(Intent::PolicyFlat));
    Check("press T -> PolicyTrench30", Step(0, Bit(KbKey::T)), I(Intent::PolicyTrench30));

    if (g_failures == 0)
    {
        std::printf("result: all checks passed\n");
        return 0;
    }
    std::printf("result: %d check(s) failed\n", g_failures);
    return 1;
}
