"""DataFrame → Universal 133-dim feature mapper.

Converts a stored ``frames.parquet`` (any game mode: 1v1, 2v2, or 3v3) into the
same 133-dim universal representation produced by
``extract_universal_frame_features`` (src/obs/frame_features.py) at replay-parse
time.

Why this module exists
-----------------------
The running re-ingest stores per-car columns with MODE-SPECIFIC names:
  - 1v1:  car_blue0_*, car_orange0_*              (52 car dims)
  - 2v2:  car_blue0_*, car_blue1_*, car_orange0_*, car_orange1_*    (88 car dims)
  - 3v3:  car_blue0..2_*, car_orange0..2_*         (124 car dims)

The universal model needs a FIXED 133-dim input regardless of game mode.
This module re-slots the mode-specific columns into the 6-slot universal layout
(blue slots 0-2, orange slots 0-2, zero-padded when absent) and appends the same
presence masks + n_active counts that ``extract_universal_frame_features`` emits.

The output is numerically identical (within float32 tolerance) to what you would
get by calling ``extract_universal_frame_features`` on the original GameState,
making offline-trained models fully compatible with live inference.

Public API
----------
    frames_df_to_universal(df)  ->  np.ndarray  shape (T, 133)
    universal_feature_names()   ->  list[str]   length 133  (re-exported)

Layout (matches frame_features.extract_universal_frame_features exactly):
    [0:9]      ball physics  (pos_x/y/z, linvel_x/y/z, angvel_x/y/z)
    [9:117]    6 car slots × 18 features each:
                 slots 0-2 → blue  cars (car_blue0, car_blue1, car_blue2)
                 slots 3-5 → orange cars (car_orange0, car_orange1, car_orange2)
               Zero-padded for absent slots.
    [117:124]  boost pad summary: 6 large timers + small_ready_frac
    [124:130]  car presence mask (1.0 if slot occupied, else 0.0)
    [130:133]  n_active normalised: [n_blue/3, n_orange/3, n_total/6]
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

from src.obs.frame_features import (
    _UNIVERSAL_SIZE,
    universal_feature_names,
)

# Re-export so callers can do `from src.obs.universal_frames import universal_feature_names`
__all__ = ["frames_df_to_universal", "universal_feature_names"]

# Constants — must stay in sync with frame_features.py
_N_SLOTS = 6          # 3 blue + 3 orange
_N_SLOTS_PER_TEAM = 3
_CAR_FEATURES = 18    # features per car slot
_BALL_FEATURES = 9
_BOOST_SUMMARY_FEATURES = 7

# Subfeature names WITHIN a car block (same order as car_features() in frame_features.py)
_CAR_SUBFEATURES = [
    "pos_x", "pos_y", "pos_z",
    "linvel_x", "linvel_y", "linvel_z",
    "angvel_x", "angvel_y", "angvel_z",
    "boost", "on_ground", "has_flip", "has_jumped",
    "is_supersonic", "air_time", "flip_time", "demo_timer", "is_orange",
]
assert len(_CAR_SUBFEATURES) == _CAR_FEATURES

# Regex for discovering which car columns are present in the DataFrame.
# Matches column names like "car_blue0_pos_x", "car_orange2_boost", etc.
_RE_CAR_PREFIX = re.compile(r"^(car_(?:blue|orange)\d+)_pos_x$")

# Boost pad column names as written by the replay pipeline.
# Large timer columns: boost_pad_large_{i}_timer for i in 0..5
# Small fraction column: boost_pad_small_ready_frac
_LARGE_TIMER_COLS = [f"boost_pad_large_{i}_timer" for i in range(6)]
_SMALL_FRAC_COL = "boost_pad_small_ready_frac"

# Ball columns (9 features, same order as ball_features() output).
_BALL_COLS = [
    "ball_pos_x", "ball_pos_y", "ball_pos_z",
    "ball_linvel_x", "ball_linvel_y", "ball_linvel_z",
    "ball_angvel_x", "ball_angvel_y", "ball_angvel_z",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discover_car_prefixes(df: pd.DataFrame) -> list[str]:
    """Return sorted list of car column prefixes found in ``df``.

    For example: ['car_blue0', 'car_blue1', 'car_orange0', 'car_orange1']

    Discovery is based on the presence of the ``<prefix>_pos_x`` column, which
    is always emitted for every car (same logic as shot_outcome._discover_car_prefixes).
    """
    prefixes: list[str] = []
    for col in df.columns:
        m = _RE_CAR_PREFIX.match(col)
        if m:
            prefix = m.group(1)
            if prefix not in prefixes:
                prefixes.append(prefix)
    return sorted(prefixes)


def _slot_index(prefix: str) -> int:
    """Map a car prefix to its universal slot index 0-5.

    Blue cars:   car_blue0 → 0,  car_blue1 → 1,  car_blue2 → 2
    Orange cars: car_orange0 → 3, car_orange1 → 4, car_orange2 → 5

    Raises ValueError for unrecognised prefixes (defensive programming).
    """
    m = re.match(r"^car_(blue|orange)(\d+)$", prefix)
    if m is None:
        raise ValueError(f"Unrecognised car prefix: {prefix!r}")
    team = m.group(1)
    idx_in_team = int(m.group(2))
    if team == "blue":
        return idx_in_team          # 0, 1, 2
    else:
        return _N_SLOTS_PER_TEAM + idx_in_team   # 3, 4, 5


def _extract_car_block(
    df: pd.DataFrame,
    prefix: str,
    n_rows: int,
) -> Optional[np.ndarray]:
    """Extract the (n_rows, 18) float32 block for one car prefix.

    Returns None if any of the 18 expected columns are missing.

    Column mapping follows the exact order of car_features() output:
      [0:9]  physics (pos_x/y/z, linvel_x/y/z, angvel_x/y/z)
      [9:18] state   (boost, on_ground, has_flip, has_jumped, is_supersonic,
                      air_time, flip_time, demo_timer, is_orange)
    """
    cols = [f"{prefix}_{sf}" for sf in _CAR_SUBFEATURES]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return None
    return df[cols].to_numpy(dtype=np.float32)  # (n_rows, 18)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def frames_df_to_universal(df: pd.DataFrame) -> np.ndarray:
    """Convert a ``frames.parquet`` DataFrame to the 133-dim universal array.

    Works for any game mode (1v1, 2v2, 3v3) by discovering the car columns
    present in ``df`` and re-slotting them into the fixed 6-slot universal layout.
    Absent slots are zero-padded.

    The output layout is IDENTICAL to ``extract_universal_frame_features`` from
    ``src/obs/frame_features.py``, guaranteeing that a model trained offline on
    this representation will produce identical activations at live inference time.

    Args:
        df:  A ``frames.parquet`` DataFrame (full or partial; all rows processed).
             Must contain ball columns, car columns (at least one car), and
             optionally boost pad columns.  Non-feature (meta) columns are ignored.

    Returns:
        np.ndarray of shape ``(T, 133)`` and dtype ``float32``.
        T = number of rows in ``df``.

    Raises:
        ValueError: If no car columns are found in ``df`` (not a valid frame table).
    """
    T = len(df)
    if T == 0:
        return np.empty((0, _UNIVERSAL_SIZE), dtype=np.float32)

    # --- Allocate output buffer (zero-filled → absent car slots stay zero) ---
    out = np.zeros((T, _UNIVERSAL_SIZE), dtype=np.float32)

    # --- Ball (dims 0:9) ---
    ball_missing = [c for c in _BALL_COLS if c not in df.columns]
    if not ball_missing:
        out[:, :_BALL_FEATURES] = df[_BALL_COLS].to_numpy(dtype=np.float32)
    # If ball columns are absent (should not happen for valid parquets), leave zeros.

    # --- Car slots (dims 9:117) ---
    prefixes = _discover_car_prefixes(df)
    if not prefixes:
        raise ValueError(
            "frames_df_to_universal: no car columns found in DataFrame. "
            "Expected columns like 'car_blue0_pos_x'. "
            f"Available columns: {list(df.columns)[:20]!r}"
        )

    blue_present = 0
    orange_present = 0

    for prefix in prefixes:
        try:
            slot = _slot_index(prefix)
        except ValueError:
            continue  # skip unknown prefixes defensively

        car_block = _extract_car_block(df, prefix, T)
        if car_block is None:
            continue  # skip if any subfeature column is absent

        # Slot offset in the output vector
        slot_start = _BALL_FEATURES + slot * _CAR_FEATURES      # 9 + slot * 18
        slot_end = slot_start + _CAR_FEATURES
        out[:, slot_start:slot_end] = car_block

        if slot < _N_SLOTS_PER_TEAM:
            blue_present = max(blue_present, slot + 1)
        else:
            orange_present = max(orange_present, slot - _N_SLOTS_PER_TEAM + 1)

    # --- Boost pad summary (dims 117:124) ---
    boost_start = _BALL_FEATURES + _N_SLOTS * _CAR_FEATURES   # 9 + 108 = 117

    large_cols_present = [c for c in _LARGE_TIMER_COLS if c in df.columns]
    for i, col in enumerate(large_cols_present):
        if i < 6:
            out[:, boost_start + i] = df[col].to_numpy(dtype=np.float32)
    # dims 117+6 = 123: small_ready_frac
    if _SMALL_FRAC_COL in df.columns:
        out[:, boost_start + 6] = df[_SMALL_FRAC_COL].to_numpy(dtype=np.float32)

    # --- Slot presence mask (dims 124:130) ---
    mask_start = boost_start + _BOOST_SUMMARY_FEATURES   # 117 + 7 = 124
    for prefix in prefixes:
        try:
            slot = _slot_index(prefix)
        except ValueError:
            continue
        car_block = _extract_car_block(df, prefix, T)
        if car_block is not None:
            out[:, mask_start + slot] = 1.0

    # --- n_active scalars (dims 130:133) ---
    counts_start = mask_start + _N_SLOTS   # 124 + 6 = 130
    out[:, counts_start + 0] = blue_present / _N_SLOTS_PER_TEAM   # n_blue/3
    out[:, counts_start + 1] = orange_present / _N_SLOTS_PER_TEAM  # n_orange/3
    out[:, counts_start + 2] = (blue_present + orange_present) / _N_SLOTS   # n_total/6

    assert out.shape == (T, _UNIVERSAL_SIZE), (
        f"frames_df_to_universal: shape mismatch — expected ({T}, {_UNIVERSAL_SIZE}), "
        f"got {out.shape}"
    )
    return out
