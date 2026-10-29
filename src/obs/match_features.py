"""Match-level / window-level aggregate feature extractor for win prediction.

WHY this is a separate layer from frame_features.py
-----------------------------------------------------
``frame_features.py`` defines the per-frame physics parity contract: the same
code drives both offline training (replay-derived GameState) and live inference
(RLBot GameTickPacket). Modifying that contract requires changing the rlgym
extractor and re-generating every parquet.

This module operates *post-hoc* on ``frames.parquet`` DataFrames that have
already been written by the ingestion pipeline.  It aggregates over sliding
windows to produce match-context features for the HMLT (Hierarchical
Match-Level Transformer) win-prediction head.  No GameState objects are
touched — only pandas/numpy, so this module is importable anywhere without
rlgym or PyTorch.

Feature columns are derived post-hoc from frames.parquet.
MMR comes from external replay metadata (BallChasing API / Kaggle dataset),
defaulting to a neutral value (1000 / 2000 = 0.5) when unknown.

Car columns are discovered dynamically via regex on the DataFrame's column
list, so the same code handles 1v1 (car_blue0/car_orange0), 2v2, and 3v3
without any hardcoded car count.

Ball Y-axis convention (inherited from frame_features.py):
    Positive Y  →  orange half  →  blue attacking
    Negative Y  →  blue half    →  orange attacking
    Blue's goal is at Y < 0; orange's goal is at Y > 0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Fixed output dimension of compute_window_match_features.
MATCH_FEATURE_SIZE = 23

#: MMR ceiling used as normalization denominator.
#: ~2000 corresponds to Grand Champion / SSL boundary in competitive Rocket League.
_MMR_NORM = 2000.0

#: Default episode length in seconds (standard Rocket League match = 5 min).
_MATCH_DURATION_S = 300.0

#: Default goal normalization: 5 goals difference = saturated advantage.
_GOAL_DIFF_NORM = 5.0

#: Default total goals normalization: 10 goals = saturated match tempo.
_TOTAL_GOALS_NORM = 10.0

#: Max goals per window for momentum normalization.
_WINDOW_GOALS_NORM = 3.0

# Regex patterns for dynamic car column discovery.
_RE_BLUE_BOOST = re.compile(r"^car_blue\d+_boost$")
_RE_ORANGE_BOOST = re.compile(r"^car_orange\d+_boost$")
_RE_BLUE_SUPERSONIC = re.compile(r"^car_blue\d+_is_supersonic$")
_RE_ORANGE_SUPERSONIC = re.compile(r"^car_orange\d+_is_supersonic$")


# ---------------------------------------------------------------------------
# MatchMeta dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MatchMeta:
    """External metadata for a single match, sourced from BallChasing / Kaggle.

    These are NOT present in frames.parquet — they come from the replay index.

    Attributes:
        blue_mmr:   Estimated MMR of the blue team (average of team members).
        orange_mmr: Estimated MMR of the orange team.
    """
    blue_mmr: float = 1000.0    # neutral default → 0.5 when normalized by _MMR_NORM
    orange_mmr: float = 1000.0


# ---------------------------------------------------------------------------
# Feature name registry
# ---------------------------------------------------------------------------

def match_feature_names() -> list[str]:
    """Return the 23 feature names in exact output order.

    Groups:
        score_state  [0:4]   — goal differential + time interaction
        time         [4:6]   — time remaining + overtime flag
        boost        [6:11]  — per-team boost economy + supersonic fractions
        possession   [11:15] — ball field-position indicator
        momentum     [15:18] — in-window scoring delta + ball speed
        xg_estimate  [18:20] — lightweight expected-goal heuristic
        mmr_prior    [20:23] — external skill prior (BallChasing / Kaggle)
    """
    names = [
        # score_state (4)
        "goal_diff",
        "total_goals",
        "is_blue_leading",
        "score_x_time",
        # time (2)
        "time_remaining_norm",
        "is_overtime",
        # boost (5)
        "blue_boost_mean",
        "orange_boost_mean",
        "boost_diff",
        "blue_supersonic_frac",
        "orange_supersonic_frac",
        # possession (4)
        "frac_ball_orange_half",
        "frac_ball_blue_half",
        "mean_ball_y",
        "ball_y_trend",
        # momentum (3)
        "goals_this_window",
        "blue_goal_momentum",
        "mean_ball_speed",
        # xg_estimate (2)
        "blue_threat",
        "orange_threat",
        # mmr_prior (3)
        "blue_mmr_norm",
        "orange_mmr_norm",
        "mmr_diff",
    ]
    assert len(names) == MATCH_FEATURE_SIZE, (
        f"match_feature_names(): expected {MATCH_FEATURE_SIZE} names, got {len(names)}"
    )
    return names


# ---------------------------------------------------------------------------
# Column discovery helpers
# ---------------------------------------------------------------------------

def _find_cols(df: pd.DataFrame, pattern: re.Pattern[str]) -> list[str]:
    """Return column names matching *pattern*, sorted (stable order)."""
    return sorted(c for c in df.columns if pattern.match(c))


# ---------------------------------------------------------------------------
# Core window feature extractor
# ---------------------------------------------------------------------------

def compute_window_match_features(
    window_df: pd.DataFrame,
    *,
    meta: MatchMeta | None = None,
) -> np.ndarray:
    """Compute a 23-dim match-context vector from a window of consecutive frames.

    Args:
        window_df: Slice of a frames.parquet DataFrame (arbitrary number of rows,
                   but typically ``window=45`` rows at 15 Hz = 3 seconds).
                   Must contain the META_COLS listed in dataset.py plus the
                   physics feature columns written by the ingestion pipeline.
        meta:      Optional external metadata for MMR-based features.
                   When ``None``, MMR features default to neutral (0.5 / 0.0).

    Returns:
        float32 ndarray of shape (MATCH_FEATURE_SIZE,) = (23,).
        Guaranteed NaN-free and within documented ranges.
    """
    if window_df.empty:
        out = np.zeros(MATCH_FEATURE_SIZE, dtype=np.float32)
        # MMR prior — neutral
        if meta is not None:
            out[20] = float(np.clip(meta.blue_mmr / _MMR_NORM, 0.0, 1.0))
            out[21] = float(np.clip(meta.orange_mmr / _MMR_NORM, 0.0, 1.0))
            out[22] = float(np.clip(out[20] - out[21], -1.0, 1.0))
        else:
            out[20] = 0.5
            out[21] = 0.5
            out[22] = 0.0
        return out

    last = window_df.iloc[-1]
    first = window_df.iloc[0]

    # --- Score state (4) ---
    blue_score_last = float(last["blue_score"])
    orange_score_last = float(last["orange_score"])
    goal_diff_raw = blue_score_last - orange_score_last
    goal_diff_norm = float(np.clip(goal_diff_raw / _GOAL_DIFF_NORM, -1.0, 1.0))
    total_goals_norm = float(np.clip(
        (blue_score_last + orange_score_last) / _TOTAL_GOALS_NORM, 0.0, 1.0
    ))
    if goal_diff_raw > 0:
        is_blue_leading = 1.0
    elif goal_diff_raw < 0:
        is_blue_leading = 0.0
    else:
        is_blue_leading = 0.5

    time_remaining_raw = float(last["episode_seconds_remaining"])
    time_remaining_norm = float(np.clip(time_remaining_raw / _MATCH_DURATION_S, 0.0, 1.0))
    score_x_time = float(np.clip(goal_diff_norm * time_remaining_norm, -1.0, 1.0))

    # --- Time (2) ---
    is_overtime = 1.0 if time_remaining_raw <= 0.0 else 0.0

    # --- Boost economy (5) — dynamic car column discovery ---
    blue_boost_cols = _find_cols(window_df, _RE_BLUE_BOOST)
    orange_boost_cols = _find_cols(window_df, _RE_ORANGE_BOOST)
    blue_super_cols = _find_cols(window_df, _RE_BLUE_SUPERSONIC)
    orange_super_cols = _find_cols(window_df, _RE_ORANGE_SUPERSONIC)

    def _team_mean(cols: list[str]) -> float:
        if not cols:
            return 0.5  # neutral if no cars found
        return float(window_df[cols].values.mean())

    blue_boost_mean = float(np.clip(_team_mean(blue_boost_cols), 0.0, 1.0))
    orange_boost_mean = float(np.clip(_team_mean(orange_boost_cols), 0.0, 1.0))
    boost_diff = float(np.clip(blue_boost_mean - orange_boost_mean, -1.0, 1.0))
    blue_supersonic_frac = float(np.clip(_team_mean(blue_super_cols), 0.0, 1.0))
    orange_supersonic_frac = float(np.clip(_team_mean(orange_super_cols), 0.0, 1.0))

    # --- Possession indicators (4) — ball_pos_y is normalized by BACK_WALL_Y in frame_features ---
    ball_y = window_df["ball_pos_y"].values.astype(np.float32)
    frac_ball_orange_half = float(np.mean(ball_y > 0.0))  # positive Y = orange half
    frac_ball_blue_half = 1.0 - frac_ball_orange_half
    mean_ball_y = float(np.clip(np.mean(ball_y), -1.0, 1.0))
    ball_y_trend = float(np.clip(float(ball_y[-1]) - float(ball_y[0]), -1.0, 1.0))

    # --- Momentum (3) ---
    blue_score_first = float(first["blue_score"])
    orange_score_first = float(first["orange_score"])
    total_first = blue_score_first + orange_score_first
    total_last = blue_score_last + orange_score_last
    goals_delta = total_last - total_first
    goals_this_window = float(np.clip(goals_delta / _WINDOW_GOALS_NORM, 0.0, 1.0))

    blue_delta = blue_score_last - blue_score_first
    orange_delta = orange_score_last - orange_score_first
    blue_goal_momentum = float(np.clip(blue_delta - orange_delta, -1.0, 1.0))

    ball_vx = window_df["ball_linvel_x"].values.astype(np.float32)
    ball_vy = window_df["ball_linvel_y"].values.astype(np.float32)
    ball_vz = window_df["ball_linvel_z"].values.astype(np.float32)
    ball_speed = np.sqrt(ball_vx ** 2 + ball_vy ** 2 + ball_vz ** 2)
    mean_ball_speed = float(np.clip(np.mean(ball_speed), 0.0, 1.0))

    # --- xG estimate (2) ---
    # blue_threat: ball in orange half, moving toward orange goal (linvel_y > 0), weighted by speed
    # Velocities are already normalized by BALL_VEL_NORM=6000 in frame_features, so linvel_y ∈ [-1,1].
    blue_threat = float(np.clip(
        np.mean((ball_y > 0).astype(np.float32) * np.maximum(ball_vy, 0.0)),
        0.0, 1.0,
    ))
    # orange_threat: symmetric — ball in blue half, moving toward blue goal (linvel_y < 0)
    orange_threat = float(np.clip(
        np.mean((ball_y < 0).astype(np.float32) * np.maximum(-ball_vy, 0.0)),
        0.0, 1.0,
    ))

    # --- MMR prior (3) ---
    if meta is not None:
        blue_mmr_norm = float(np.clip(meta.blue_mmr / _MMR_NORM, 0.0, 1.0))
        orange_mmr_norm = float(np.clip(meta.orange_mmr / _MMR_NORM, 0.0, 1.0))
    else:
        blue_mmr_norm = 0.5
        orange_mmr_norm = 0.5
    mmr_diff = float(np.clip(blue_mmr_norm - orange_mmr_norm, -1.0, 1.0))

    # --- Assemble vector ---
    out = np.array(
        [
            # score_state (4)
            goal_diff_norm,
            total_goals_norm,
            is_blue_leading,
            score_x_time,
            # time (2)
            time_remaining_norm,
            is_overtime,
            # boost (5)
            blue_boost_mean,
            orange_boost_mean,
            boost_diff,
            blue_supersonic_frac,
            orange_supersonic_frac,
            # possession (4)
            frac_ball_orange_half,
            frac_ball_blue_half,
            mean_ball_y,
            ball_y_trend,
            # momentum (3)
            goals_this_window,
            blue_goal_momentum,
            mean_ball_speed,
            # xg_estimate (2)
            blue_threat,
            orange_threat,
            # mmr_prior (3)
            blue_mmr_norm,
            orange_mmr_norm,
            mmr_diff,
        ],
        dtype=np.float32,
    )

    # Final safety pass: replace any residual NaN/Inf with 0.0.
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
    assert out.shape == (MATCH_FEATURE_SIZE,)
    return out


# ---------------------------------------------------------------------------
# Sequence extractor — sliding window over full match
# ---------------------------------------------------------------------------

def compute_match_feature_sequence(
    frames_df: pd.DataFrame,
    *,
    window: int = 45,
    stride: int = 15,
    meta: MatchMeta | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide over an entire match and produce one match-feature vector per window.

    Windowing convention matches ``WindowSpec(length=45, stride=15)`` used by
    the existing StreamingReplayDataset — one window = 3 s at 15 Hz, slide 1 s.

    Args:
        frames_df: Full frames.parquet DataFrame for one replay.
        window:    Number of frames per window (default 45 = 3 s at 15 Hz).
        stride:    Number of frames between window starts (default 15 = 1 s).
        meta:      Optional MMR metadata (passed unchanged to each window call).

    Returns:
        seq:          float32 array of shape (n_windows, MATCH_FEATURE_SIZE).
        end_indices:  int32 array of shape (n_windows,) — the integer position
                      (iloc index) of the *last* frame in each window.
                      Use this to align seq with per-frame targets.

    If ``frames_df`` has fewer than ``window`` rows, returns empty arrays:
        seq.shape  = (0, MATCH_FEATURE_SIZE)
        end_indices.shape = (0,)
    """
    n = len(frames_df)
    empty_seq = np.empty((0, MATCH_FEATURE_SIZE), dtype=np.float32)
    empty_idx = np.empty((0,), dtype=np.int32)

    if n < window:
        return empty_seq, empty_idx

    starts = range(0, n - window + 1, stride)
    n_windows = len(starts)

    seq = np.empty((n_windows, MATCH_FEATURE_SIZE), dtype=np.float32)
    end_indices = np.empty(n_windows, dtype=np.int32)

    for i, start in enumerate(starts):
        end = start + window  # exclusive slice bound
        w_df = frames_df.iloc[start:end]
        seq[i] = compute_window_match_features(w_df, meta=meta)
        end_indices[i] = np.int32(end - 1)  # last frame position (0-based)

    return seq, end_indices
