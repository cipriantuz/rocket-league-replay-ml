"""Tests for src/obs/match_features.py — match-level aggregate feature extractor.

Synthetic DataFrames are built entirely in-code (no file I/O) so the test
suite runs without any processed replay data on disk.

Column names follow the exact convention written by the ingestion pipeline:
    META_COLS + physics features (ball_pos_x/y/z, ball_linvel_x/y/z,
    ball_angvel_x/y/z, car_<team><idx>_<field>, boost_pad_*).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.obs.match_features import (
    MATCH_FEATURE_SIZE,
    MatchMeta,
    compute_match_feature_sequence,
    compute_window_match_features,
    match_feature_names,
)

# ---------------------------------------------------------------------------
# Helpers — synthetic DataFrame builders
# ---------------------------------------------------------------------------

_META_COLS = [
    "frame_idx", "src_frame_idx", "time_s", "episode_seconds_remaining",
    "next_scoring_team", "winning_team", "blue_score", "orange_score", "goal_scored",
]

_ACTION_COLS = [
    "act_throttle", "act_steer", "act_pitch", "act_yaw",
    "act_roll", "act_jump", "act_boost", "act_handbrake",
]


def _make_frames(
    n: int,
    *,
    n_blue: int = 2,
    n_orange: int = 2,
    blue_score: int = 0,
    orange_score: int = 0,
    time_remaining: float = 150.0,
    ball_pos_y: float = 0.0,
    ball_linvel_y: float = 0.0,
    ball_speed: float = 0.3,
) -> pd.DataFrame:
    """Build a synthetic frames DataFrame with ``n`` identical rows.

    Physics values are uniform across all rows (sufficient for aggregate tests).
    Boost is set to 0.6 for blue, 0.4 for orange (non-trivial for boost_diff).
    """
    rng = np.random.default_rng(42)
    rows = {}

    # Meta columns
    rows["frame_idx"] = np.arange(n, dtype=np.int64)
    rows["src_frame_idx"] = np.arange(n, dtype=np.int64)
    rows["time_s"] = np.linspace(0.0, n / 15.0, n, dtype=np.float32)
    rows["episode_seconds_remaining"] = np.full(n, time_remaining, dtype=np.float32)
    rows["next_scoring_team"] = np.full(n, -1, dtype=np.int64)
    rows["winning_team"] = np.full(n, 0, dtype=np.int64)
    rows["blue_score"] = np.full(n, blue_score, dtype=np.int64)
    rows["orange_score"] = np.full(n, orange_score, dtype=np.int64)
    rows["goal_scored"] = np.zeros(n, dtype=np.int64)

    # Action columns
    for col in _ACTION_COLS:
        rows[col] = np.zeros(n, dtype=np.float32)

    # Ball physics — normalized (same convention as frame_features.py)
    rows["ball_pos_x"] = np.zeros(n, dtype=np.float32)
    rows["ball_pos_y"] = np.full(n, ball_pos_y, dtype=np.float32)
    rows["ball_pos_z"] = np.full(n, 0.05, dtype=np.float32)
    # linvel components that produce the requested speed:
    # if ball_linvel_y specified, set it; distribute remaining speed into x
    linvel_y = ball_linvel_y
    linvel_x_sq = max(0.0, ball_speed ** 2 - linvel_y ** 2)
    linvel_x = float(np.sqrt(linvel_x_sq))
    rows["ball_linvel_x"] = np.full(n, linvel_x, dtype=np.float32)
    rows["ball_linvel_y"] = np.full(n, linvel_y, dtype=np.float32)
    rows["ball_linvel_z"] = np.zeros(n, dtype=np.float32)
    rows["ball_angvel_x"] = np.zeros(n, dtype=np.float32)
    rows["ball_angvel_y"] = np.zeros(n, dtype=np.float32)
    rows["ball_angvel_z"] = np.zeros(n, dtype=np.float32)

    # Car columns — dynamic per n_blue / n_orange
    for i in range(n_blue):
        prefix = f"car_blue{i}"
        rows[f"{prefix}_pos_x"] = rng.uniform(-0.5, 0.5, n).astype(np.float32)
        rows[f"{prefix}_pos_y"] = np.full(n, -0.5, dtype=np.float32)
        rows[f"{prefix}_pos_z"] = np.full(n, 0.008, dtype=np.float32)
        for ax in ("x", "y", "z"):
            rows[f"{prefix}_linvel_{ax}"] = np.zeros(n, dtype=np.float32)
            rows[f"{prefix}_angvel_{ax}"] = np.zeros(n, dtype=np.float32)
        rows[f"{prefix}_boost"] = np.full(n, 0.6, dtype=np.float32)
        rows[f"{prefix}_on_ground"] = np.ones(n, dtype=np.float32)
        rows[f"{prefix}_has_flip"] = np.ones(n, dtype=np.float32)
        rows[f"{prefix}_has_jumped"] = np.zeros(n, dtype=np.float32)
        rows[f"{prefix}_is_supersonic"] = np.zeros(n, dtype=np.float32)
        rows[f"{prefix}_air_time"] = np.zeros(n, dtype=np.float32)
        rows[f"{prefix}_flip_time"] = np.zeros(n, dtype=np.float32)
        rows[f"{prefix}_demo_timer"] = np.zeros(n, dtype=np.float32)
        rows[f"{prefix}_is_orange"] = np.zeros(n, dtype=np.float32)

    for i in range(n_orange):
        prefix = f"car_orange{i}"
        rows[f"{prefix}_pos_x"] = rng.uniform(-0.5, 0.5, n).astype(np.float32)
        rows[f"{prefix}_pos_y"] = np.full(n, 0.5, dtype=np.float32)
        rows[f"{prefix}_pos_z"] = np.full(n, 0.008, dtype=np.float32)
        for ax in ("x", "y", "z"):
            rows[f"{prefix}_linvel_{ax}"] = np.zeros(n, dtype=np.float32)
            rows[f"{prefix}_angvel_{ax}"] = np.zeros(n, dtype=np.float32)
        rows[f"{prefix}_boost"] = np.full(n, 0.4, dtype=np.float32)
        rows[f"{prefix}_on_ground"] = np.ones(n, dtype=np.float32)
        rows[f"{prefix}_has_flip"] = np.ones(n, dtype=np.float32)
        rows[f"{prefix}_has_jumped"] = np.zeros(n, dtype=np.float32)
        rows[f"{prefix}_is_supersonic"] = np.zeros(n, dtype=np.float32)
        rows[f"{prefix}_air_time"] = np.zeros(n, dtype=np.float32)
        rows[f"{prefix}_flip_time"] = np.zeros(n, dtype=np.float32)
        rows[f"{prefix}_demo_timer"] = np.zeros(n, dtype=np.float32)
        rows[f"{prefix}_is_orange"] = np.ones(n, dtype=np.float32)

    # Boost pad columns
    for i in range(6):
        rows[f"boost_pad_large_{i}_timer"] = np.zeros(n, dtype=np.float32)
    rows["boost_pad_small_ready_frac"] = np.ones(n, dtype=np.float32)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1 — shape, dtype, NaN/Inf
# ---------------------------------------------------------------------------

def test_output_shape_dtype_no_nan():
    """compute_window_match_features returns (23,) float32 with no NaN/Inf."""
    df = _make_frames(45)
    out = compute_window_match_features(df)
    assert out.shape == (MATCH_FEATURE_SIZE,), f"Expected shape ({MATCH_FEATURE_SIZE},), got {out.shape}"
    assert out.dtype == np.float32, f"Expected float32, got {out.dtype}"
    assert not np.isnan(out).any(), "Output contains NaN"
    assert not np.isinf(out).any(), "Output contains Inf"


# ---------------------------------------------------------------------------
# Test 2 — all values within documented ranges
# ---------------------------------------------------------------------------

def test_value_ranges():
    """All 23 features within their documented ranges."""
    df = _make_frames(45, blue_score=2, orange_score=1, time_remaining=120.0)
    out = compute_window_match_features(df)
    names = match_feature_names()

    # boost, time, frac-type features ∈ [0, 1]
    nonneg_unit_indices = {
        names.index("total_goals"),
        names.index("time_remaining_norm"),
        names.index("blue_boost_mean"),
        names.index("orange_boost_mean"),
        names.index("blue_supersonic_frac"),
        names.index("orange_supersonic_frac"),
        names.index("frac_ball_orange_half"),
        names.index("frac_ball_blue_half"),
        names.index("goals_this_window"),
        names.index("mean_ball_speed"),
        names.index("blue_threat"),
        names.index("orange_threat"),
        names.index("blue_mmr_norm"),
        names.index("orange_mmr_norm"),
        names.index("is_overtime"),
    }
    for idx in nonneg_unit_indices:
        val = float(out[idx])
        assert 0.0 <= val <= 1.0, (
            f"Feature '{names[idx]}' = {val} is outside [0, 1]"
        )

    # signed features ∈ [-1, 1]
    signed_unit_indices = {
        names.index("goal_diff"),
        names.index("score_x_time"),
        names.index("boost_diff"),
        names.index("mean_ball_y"),
        names.index("ball_y_trend"),
        names.index("blue_goal_momentum"),
        names.index("mmr_diff"),
    }
    for idx in signed_unit_indices:
        val = float(out[idx])
        assert -1.0 <= val <= 1.0, (
            f"Feature '{names[idx]}' = {val} is outside [-1, 1]"
        )

    # is_blue_leading ∈ {0.0, 0.5, 1.0}
    il = float(out[names.index("is_blue_leading")])
    assert il in (0.0, 0.5, 1.0), f"is_blue_leading = {il} not in {{0.0, 0.5, 1.0}}"


# ---------------------------------------------------------------------------
# Test 3 — match_feature_names() length == MATCH_FEATURE_SIZE == 23
# ---------------------------------------------------------------------------

def test_feature_names_length():
    names = match_feature_names()
    assert len(names) == MATCH_FEATURE_SIZE == 23, (
        f"match_feature_names() returned {len(names)} names, expected {MATCH_FEATURE_SIZE}"
    )
    # No duplicates
    assert len(set(names)) == len(names), "Duplicate feature names detected"


# ---------------------------------------------------------------------------
# Test 4 — blue_threat > orange_threat when ball is in orange half + moving toward orange goal
# ---------------------------------------------------------------------------

def test_blue_threat_dominant_when_ball_in_orange_half():
    """Ball at y=+0.7 (orange half) moving toward orange goal (linvel_y=+0.5).
    blue_threat should exceed orange_threat, frac_ball_orange_half > 0.5.
    """
    df = _make_frames(45, ball_pos_y=0.7, ball_linvel_y=0.5, ball_speed=0.5)
    out = compute_window_match_features(df)
    names = match_feature_names()

    blue_threat = float(out[names.index("blue_threat")])
    orange_threat = float(out[names.index("orange_threat")])
    frac_orange = float(out[names.index("frac_ball_orange_half")])

    assert blue_threat > orange_threat, (
        f"Expected blue_threat ({blue_threat:.4f}) > orange_threat ({orange_threat:.4f})"
    )
    assert frac_orange > 0.5, (
        f"Expected frac_ball_orange_half > 0.5, got {frac_orange:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 5 — goal_diff sign correct when blue leads
# ---------------------------------------------------------------------------

def test_goal_diff_sign_blue_leads():
    """goal_diff > 0 when blue_score > orange_score."""
    df = _make_frames(45, blue_score=3, orange_score=1)
    out = compute_window_match_features(df)
    names = match_feature_names()

    goal_diff = float(out[names.index("goal_diff")])
    assert goal_diff > 0.0, (
        f"Expected goal_diff > 0 when blue leads 3-1, got {goal_diff}"
    )
    assert float(out[names.index("is_blue_leading")]) == 1.0


def test_goal_diff_negative_orange_leads():
    """goal_diff < 0 when orange_score > blue_score."""
    df = _make_frames(45, blue_score=0, orange_score=2)
    out = compute_window_match_features(df)
    names = match_feature_names()

    goal_diff = float(out[names.index("goal_diff")])
    assert goal_diff < 0.0, (
        f"Expected goal_diff < 0 when orange leads 2-0, got {goal_diff}"
    )
    assert float(out[names.index("is_blue_leading")]) == 0.0


def test_goal_diff_tie():
    """is_blue_leading == 0.5 on a draw."""
    df = _make_frames(45, blue_score=2, orange_score=2)
    out = compute_window_match_features(df)
    names = match_feature_names()
    assert float(out[names.index("is_blue_leading")]) == 0.5


# ---------------------------------------------------------------------------
# Test 6 — 1v1 DataFrame (only car_blue0 / car_orange0) produces valid output
# ---------------------------------------------------------------------------

def test_1v1_no_keyerror():
    """Dynamic column discovery handles 1v1 (2 cars) without KeyError."""
    df = _make_frames(45, n_blue=1, n_orange=1)
    # Verify no car_blue1 / car_orange1 columns exist
    assert not any("car_blue1" in c for c in df.columns)
    assert not any("car_orange1" in c for c in df.columns)

    out = compute_window_match_features(df)
    assert out.shape == (MATCH_FEATURE_SIZE,)
    assert out.dtype == np.float32
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()


# ---------------------------------------------------------------------------
# Test 7 — compute_match_feature_sequence windowing
# ---------------------------------------------------------------------------

def test_sequence_window_count_and_end_indices():
    """100-row df with window=45, stride=15 → 4 windows, correct end_indices."""
    n = 100
    df = _make_frames(n)
    seq, end_idx = compute_match_feature_sequence(df, window=45, stride=15)

    expected_n_windows = (n - 45) // 15 + 1  # = (55 // 15) + 1 = 3 + 1 = 4
    assert seq.shape == (expected_n_windows, MATCH_FEATURE_SIZE), (
        f"Expected shape ({expected_n_windows}, {MATCH_FEATURE_SIZE}), got {seq.shape}"
    )
    assert seq.dtype == np.float32
    assert end_idx.shape == (expected_n_windows,)
    assert end_idx.dtype == np.int32

    # Verify end indices: window i ends at start_i + 45 - 1 = i*15 + 44
    for i in range(expected_n_windows):
        expected_end = i * 15 + 44
        assert int(end_idx[i]) == expected_end, (
            f"Window {i}: expected end_idx={expected_end}, got {int(end_idx[i])}"
        )


def test_sequence_no_nan_inf():
    """Sequence output is NaN/Inf-free."""
    df = _make_frames(100, blue_score=1, orange_score=0, time_remaining=200.0)
    seq, _ = compute_match_feature_sequence(df, window=45, stride=15)
    assert not np.isnan(seq).any(), "NaN found in sequence output"
    assert not np.isinf(seq).any(), "Inf found in sequence output"


# ---------------------------------------------------------------------------
# Test 8 — frames_df shorter than window → empty arrays with correct shape
# ---------------------------------------------------------------------------

def test_short_df_returns_empty():
    """Fewer frames than window → (0, 23) seq and (0,) end_indices."""
    df = _make_frames(30)  # 30 < 45
    seq, end_idx = compute_match_feature_sequence(df, window=45, stride=15)

    assert seq.shape == (0, MATCH_FEATURE_SIZE), (
        f"Expected shape (0, {MATCH_FEATURE_SIZE}), got {seq.shape}"
    )
    assert end_idx.shape == (0,), f"Expected shape (0,), got {end_idx.shape}"


# ---------------------------------------------------------------------------
# Bonus — MMR prior features
# ---------------------------------------------------------------------------

def test_mmr_prior_neutral_when_meta_none():
    """Without meta, MMR features default to neutral: blue/orange=0.5, diff=0.0."""
    df = _make_frames(45)
    out = compute_window_match_features(df, meta=None)
    names = match_feature_names()

    assert float(out[names.index("blue_mmr_norm")]) == pytest.approx(0.5, abs=1e-6)
    assert float(out[names.index("orange_mmr_norm")]) == pytest.approx(0.5, abs=1e-6)
    assert float(out[names.index("mmr_diff")]) == pytest.approx(0.0, abs=1e-6)


def test_mmr_prior_with_meta():
    """MatchMeta(blue_mmr=2000, orange_mmr=1000) → blue=1.0, orange=0.5, diff=0.5."""
    df = _make_frames(45)
    meta = MatchMeta(blue_mmr=2000.0, orange_mmr=1000.0)
    out = compute_window_match_features(df, meta=meta)
    names = match_feature_names()

    assert float(out[names.index("blue_mmr_norm")]) == pytest.approx(1.0, abs=1e-6)
    assert float(out[names.index("orange_mmr_norm")]) == pytest.approx(0.5, abs=1e-6)
    assert float(out[names.index("mmr_diff")]) == pytest.approx(0.5, abs=1e-5)


# ---------------------------------------------------------------------------
# Bonus — boost economy differential
# ---------------------------------------------------------------------------

def test_boost_diff_blue_advantage():
    """Blue at 0.6 boost, orange at 0.4 → boost_diff > 0."""
    df = _make_frames(45)  # default: blue=0.6, orange=0.4
    out = compute_window_match_features(df)
    names = match_feature_names()

    boost_diff = float(out[names.index("boost_diff")])
    assert boost_diff > 0.0, f"Expected boost_diff > 0, got {boost_diff}"
    assert float(out[names.index("blue_boost_mean")]) == pytest.approx(0.6, abs=1e-5)
    assert float(out[names.index("orange_boost_mean")]) == pytest.approx(0.4, abs=1e-5)
