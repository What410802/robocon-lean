#pragma once

#include "rl_sdk.hpp"

namespace feedback_control
{
inline bool ActiveRL(const RL &rl)
{
    return rl.simulation_running && rl.rl_init_done
        && rl.fsm.mode_ == FSM::Mode::NORMAL && rl.fsm.current_state_
        && rl.fsm.current_state_->GetStateName() == "RLFSMStateRL_Locomotion";
}

// Keep input/FSM processing and command limiting once per control tick. Part 2
// inference ticks consume the newly queued action before that one publication;
// the recorded Part 1 route retains its original next-tick action delivery.
template <class Control, class Infer, class Publish>
void Step(RL &rl, bool inference_due, Control control, Infer infer, Publish publish)
{
    const bool deferred = inference_due && rl.part == 2 && ActiveRL(rl);
    control(deferred);
    if (!inference_due) return;
    infer();
    if (deferred && rl.simulation_running)
    {
        // Only the RL state's queue consumer may run twice. Running the whole
        // FSM again would repeat transitions, input edges and fixed behaviors.
        if (ActiveRL(rl)) rl.fsm.current_state_->Run();
        publish();
    }
}
} // namespace feedback_control
