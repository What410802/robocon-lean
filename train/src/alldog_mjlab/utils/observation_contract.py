"""blackW observation layout contract (single source of truth).

Native training layout: term-major, oldest->newest, each term flattened.
Deployment layout: frame-major, newest->oldest, 57-D per frame.
"""

from __future__ import annotations

BLACKW_TERMS = ("command", "base_ang_vel", "projected_gravity",
                "joint_pos", "joint_vel", "actions")
BLACKW_TERM_DIMS = (3, 3, 3, 16, 16, 16)
BLACKW_FRAME_DIM = sum(BLACKW_TERM_DIMS)  # 57
BLACKW_HISTORY = 6
BLACKW_ACTOR_DIM = BLACKW_FRAME_DIM * BLACKW_HISTORY
BLACKW_DEPLOY_LAYOUT = ((3, 0), (3, 3), (3, 6), (16, 9), (16, 25), (16, 41))
BLACKW_ACTION_TERM_OFFSET = 3 + 3 + 3 + 16 + 16  # 41
BLACKW_WHEEL_ACTION_INDICES = (3, 7, 11, 15)


def validate_terms(terms) -> None:
    if tuple(terms) != BLACKW_TERMS:
        raise ValueError(f"blackW observation terms {tuple(terms)} != {BLACKW_TERMS}")


def validate_history(history) -> None:
    if history is None or list(history) != list(range(BLACKW_HISTORY)):
        raise ValueError(
            f"blackW observations_history must be [0..{BLACKW_HISTORY - 1}], got {history}"
        )


def native_to_deployment_frame(native_row, deployment_layout=BLACKW_DEPLOY_LAYOUT,
                               frame_dim=BLACKW_FRAME_DIM, history=BLACKW_HISTORY):
    """Return the permuted deployment columns for one native flat row."""
    out = []
    for term_dim, offset in deployment_layout:
        for frame in range(history):
            for d in range(term_dim):
                out.append(frame * frame_dim + offset + d)
    return out


def build_native_history(obs, episode):
    """(N,57) single frames + episode ids -> (N,342) native term-major oldest->newest."""
    import numpy as np

    obs = np.asarray(obs, dtype=np.float32)
    episode = np.asarray(episode)
    n = len(obs)
    out = np.zeros((n, BLACKW_ACTOR_DIM), dtype=np.float32)
    offsets = [0]
    for d in BLACKW_TERM_DIMS:
        offsets.append(offsets[-1] + d)
    for ep in np.unique(episode):
        idx = np.where(episode == ep)[0]
        if len(idx) == 0:
            continue
        frames = np.zeros((len(idx), BLACKW_HISTORY, BLACKW_FRAME_DIM), dtype=np.float32)
        frames[:] = obs[idx[0]][None, None, :]
        for t, row in enumerate(idx):
            if t > 0:
                frames[t, :-1] = frames[t - 1, 1:]
            frames[t, -1] = obs[row]
        cols = []
        for k, dim in enumerate(BLACKW_TERM_DIMS):
            lo, hi = offsets[k], offsets[k + 1]
            cols.append(frames[:, :, lo:hi].reshape(len(idx), dim * BLACKW_HISTORY))
        out[idx] = np.concatenate(cols, axis=1)
    return out
