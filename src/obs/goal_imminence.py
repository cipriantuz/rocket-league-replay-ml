"""Goal-imminence label derivation for the SHORT-HORIZON prediction task.

WHY this reformulation matters
--------------------------------
The original "next goal" target (which team scores the NEXT goal?) suffers from
unbounded horizon: a window at the start of a 5-minute match has no physics
signal about a goal that may arrive 4 minutes later.  This collapses all models
to near-random (~0.33 acc on a 3-class task) because the label is dominated by
match-level priors, not by the dynamics visible in the 3-second window.

Short-horizon imminence flips the question to: **"Will any goal happen within
H seconds after this window ends?"**  Goals in [0, H] seconds are preceded by
identifiable physical precursors — aerial trajectories, breakaway sprints, ball
racing toward goal — that are encoded in the raw frame physics.  A transformer
attending to fine spatiotemporal dynamics *should* outperform tabular aggregates
that flatten these dynamics into scalar statistics.

PR-AUC vs accuracy
--------------------
Windows with an imminent goal are rare (positive base rate ≈ 5–25% depending on
H).  Accuracy is dominated by the majority class (non-imminent) and gives no
information about discriminative power.  PR-AUC (average_precision_score)
integrates precision-recall across all thresholds and directly measures how well
the model ranks positives above negatives — the correct metric for rare-event
binary classification.

API
----
    compute_imminence_labels(frames_df, *, window=45, stride=15, horizon_s=10.0, hz=15)
        -> (binary_labels [n_windows,], threeclass_labels [n_windows,])

    binary_labels[i]  = 1  if a goal occurs within horizon_s of window-end, else 0
    threeclass[i]     = 0  if BLUE  scores within horizon_s
                      = 1  if ORANGE scores within horizon_s
                      = 2  if NO goal within horizon_s (no imminence)

Window/stride semantics are identical to WindowSpec(length=45, stride=15) used
by ReplayWindowDataset so that label[i] aligns with raw-frame window[i] and
aggregate feature window[i] produced by compute_match_feature_sequence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _find_goal_frames(frames_df: pd.DataFrame) -> list[tuple[int, int]]:
    """Return list of (frame_index, scoring_team) for every goal event.

    A goal is detected by a score increment in blue_score or orange_score
    (more reliable than goal_scored column which is inconsistently populated).
    scoring_team: 0 = blue, 1 = orange.

    Multiple increments in the same frame are treated as one goal (blue wins
    tie by convention — this is an edge case that should not occur in practice).
    """
    blue = frames_df["blue_score"].to_numpy(dtype=np.int32)
    orange = frames_df["orange_score"].to_numpy(dtype=np.int32)

    goals: list[tuple[int, int]] = []
    for i in range(1, len(blue)):
        blue_inc = blue[i] - blue[i - 1]
        orange_inc = orange[i] - orange[i - 1]
        if blue_inc > 0:
            goals.append((i, 0))
        elif orange_inc > 0:
            goals.append((i, 1))
    return goals


def compute_imminence_labels(
    frames_df: pd.DataFrame,
    *,
    window: int = 45,
    stride: int = 15,
    horizon_s: float = 10.0,
    hz: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-window binary and 3-class goal-imminence labels.

    For each sliding window (identical spec to WindowSpec(length=45, stride=15)):
      - end frame index e = start + window - 1
      - Scan frames after e; find the next goal frame g > e
      - time_to_goal_s = (g - e) / hz
      - binary label   = 1 if time_to_goal_s <= horizon_s else 0
      - 3-class label  = scoring_team (0=blue, 1=orange) if imminent else 2

    Edge conditions handled deterministically (no NaN):
      - Window end is past the last goal → label 0 / class 2
      - Match has no goals at all → all zeros / all class 2

    Args:
        frames_df:  Full frames.parquet DataFrame for one replay (pandas).
                    Must have columns: blue_score, orange_score (int).
        window:     Number of frames per window (default 45 = 3 s at 15 Hz).
        stride:     Frame stride between window starts (default 15 = 1 s).
        horizon_s:  Imminence horizon in seconds.  Goals further away than this
                    from the window end are treated as "not imminent".
        hz:         Frame rate (default 15.0 Hz, consistent with ingestion pipeline).

    Returns:
        binary_labels:     int8 ndarray of shape (n_windows,), values in {0, 1}.
        threeclass_labels: int8 ndarray of shape (n_windows,), values in {0, 1, 2}.
                           Aligned index-for-index with binary_labels.

    Both arrays have the same length as the number of valid windows.  Window i
    corresponds to frames[start_i : start_i + window] where start_i = i * stride.
    """
    n = len(frames_df)
    horizon_frames = int(horizon_s * hz)  # e.g. 10 s * 15 Hz = 150 frames

    # Pre-compute all goal events once (O(n)) rather than inside the window loop
    goals = _find_goal_frames(frames_df)  # sorted ascending by frame_index

    # Convert to arrays for binary search
    if goals:
        goal_indices = np.array([g[0] for g in goals], dtype=np.int32)
        goal_teams = np.array([g[1] for g in goals], dtype=np.int8)
    else:
        goal_indices = np.array([], dtype=np.int32)
        goal_teams = np.array([], dtype=np.int8)

    # Generate window starts (same formula as ReplayWindowDataset._build_index)
    starts = list(range(0, n - window + 1, stride))
    n_windows = len(starts)

    binary_labels = np.zeros(n_windows, dtype=np.int8)
    threeclass_labels = np.full(n_windows, 2, dtype=np.int8)  # default = class 2 (no goal)

    for i, start in enumerate(starts):
        e = start + window - 1  # last frame index of this window (0-based)

        if len(goal_indices) == 0:
            continue  # no goals in match — stay at defaults

        # Find first goal strictly AFTER frame e using binary search (O(log G))
        pos = np.searchsorted(goal_indices, e + 1, side="left")
        if pos >= len(goal_indices):
            continue  # no future goal — stay at defaults

        g = goal_indices[pos]
        time_to_goal_s = (g - e) / hz

        if time_to_goal_s <= horizon_s:
            binary_labels[i] = 1
            threeclass_labels[i] = goal_teams[pos]

    return binary_labels, threeclass_labels
