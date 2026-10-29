"""Tests for src/obs/universal_frames.py — DataFrame → 133-dim universal mapper.

All tests use SYNTHETIC data only — no real frames.parquet files are read.
The test suite verifies:

1. Output shape (T, 133) for any game mode (1v1 / 2v2 / 3v3).
2. Masks and n_active counts are correct.
3. Car data lands in the correct universal slot (blue0→slot0, orange1→slot4, etc.).
4. Ball and boost pad columns are mapped correctly.
5. Absent car slots are zero-padded.
6. Consistency: a 2v2 GameState through extract_universal_frame_features produces
   the SAME 133-dim vector as the equivalent DataFrame row through
   frames_df_to_universal (offline == live representation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.obs.frame_features import (
    _UNIVERSAL_SIZE,
    extract_universal_frame_features,
    universal_feature_size,
)
from src.obs.universal_frames import frames_df_to_universal, universal_feature_names

# ---------------------------------------------------------------------------
# Layout constants (must stay in sync with frame_features.py)
# ---------------------------------------------------------------------------
_BALL_FEATURES = 9
_CAR_FEATURES = 18
_N_SLOTS = 6
_BOOST_SUMMARY_FEATURES = 7

_BALL_START = 0
_BALL_END = 9          # [0:9]
_CARS_START = 9        # [9:117]
_CARS_END = 117
_BOOST_START = 117     # [117:124]
_BOOST_END = 124
_MASK_START = 124      # [124:130]
_MASK_END = 130
_NACT_START = 130      # [130:133]
_NACT_END = 133

# ---------------------------------------------------------------------------
# Helpers — synthetic DataFrame builders
# ---------------------------------------------------------------------------

_CAR_SUBFEATURES = [
    "pos_x", "pos_y", "pos_z",
    "linvel_x", "linvel_y", "linvel_z",
    "angvel_x", "angvel_y", "angvel_z",
    "boost", "on_ground", "has_flip", "has_jumped",
    "is_supersonic", "air_time", "flip_time", "demo_timer", "is_orange",
]
assert len(_CAR_SUBFEATURES) == _CAR_FEATURES

_BALL_COLS = [
    "ball_pos_x", "ball_pos_y", "ball_pos_z",
    "ball_linvel_x", "ball_linvel_y", "ball_linvel_z",
    "ball_angvel_x", "ball_angvel_y", "ball_angvel_z",
]

_META_COLS = [
    "frame_idx", "time_s", "episode_seconds_remaining",
    "next_scoring_team", "winning_team", "blue_score", "orange_score",
    "goal_scored",
]

_LARGE_TIMER_COLS = [f"boost_pad_large_{i}_timer" for i in range(6)]
_SMALL_FRAC_COL = "boost_pad_small_ready_frac"


def _build_car_row(
    prefix: str,
    *,
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    boost: float = 0.5,
    is_orange: float = 0.0,
    seed_offset: float = 0.0,
) -> dict[str, float]:
    """Build a dict of car column values for one row."""
    row: dict[str, float] = {}
    for sf in _CAR_SUBFEATURES:
        if sf == "pos_x":
            row[f"{prefix}_{sf}"] = float(pos[0])
        elif sf == "pos_y":
            row[f"{prefix}_{sf}"] = float(pos[1])
        elif sf == "pos_z":
            row[f"{prefix}_{sf}"] = float(pos[2])
        elif sf == "boost":
            row[f"{prefix}_{sf}"] = boost
        elif sf == "is_orange":
            row[f"{prefix}_{sf}"] = is_orange
        else:
            row[f"{prefix}_{sf}"] = seed_offset + float(_CAR_SUBFEATURES.index(sf)) * 0.01
    return row


def _build_synthetic_df(
    n_blue: int,
    n_orange: int,
    T: int = 3,
    *,
    rng: np.random.Generator | None = None,
    with_boost_pads: bool = True,
    with_meta: bool = True,
) -> pd.DataFrame:
    """Build a synthetic frames.parquet-like DataFrame for testing.

    Args:
        n_blue:   Number of blue cars (0-3).
        n_orange: Number of orange cars (0-3).
        T:        Number of rows (frames).
        rng:      Optional numpy rng for deterministic random values.
        with_boost_pads: Whether to include boost pad columns.
        with_meta: Whether to include meta columns.

    Returns:
        pd.DataFrame with the expected column structure.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    rows: list[dict] = []
    for t in range(T):
        row: dict[str, float | int] = {}

        # Meta columns
        if with_meta:
            row["frame_idx"] = t
            row["time_s"] = t / 15.0
            row["episode_seconds_remaining"] = (300 - t) / 15.0
            row["next_scoring_team"] = 0
            row["winning_team"] = 0
            row["blue_score"] = 0
            row["orange_score"] = 0
            row["goal_scored"] = 0

        # Ball columns — distinct random values
        for col in _BALL_COLS:
            row[col] = float(rng.uniform(-1.0, 1.0))

        # Car columns
        for i in range(n_blue):
            prefix = f"car_blue{i}"
            # Assign distinctive values so we can verify slot mapping
            base = float(i * 10 + t * 0.1)
            row.update(_build_car_row(
                prefix,
                pos=(base, base + 0.1, base + 0.2),
                boost=float(i + 1) * 0.1,
                is_orange=0.0,
                seed_offset=base,
            ))

        for i in range(n_orange):
            prefix = f"car_orange{i}"
            base = float(i * 10 + 100 + t * 0.1)
            row.update(_build_car_row(
                prefix,
                pos=(base, base + 0.1, base + 0.2),
                boost=float(i + 1) * 0.1,
                is_orange=1.0,
                seed_offset=base,
            ))

        # Boost pad columns
        if with_boost_pads:
            for j in range(6):
                row[f"boost_pad_large_{j}_timer"] = float(j) * 0.1
            row["boost_pad_small_ready_frac"] = 0.75

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers for GameState-based testing (consistency check)
# ---------------------------------------------------------------------------

def _build_gamestate_2v2_from_df(df: pd.DataFrame):
    """Build a rlgym GameState from a single-row DataFrame for consistency testing.

    Reconstructs Car and PhysicsObject instances matching the column values.
    Returns the GameState for a 2v2 match (blue0, blue1, orange0, orange1).
    """
    from rlgym.rocket_league.api import Car, GameState, PhysicsObject
    from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM
    from src.obs.frame_features import POS_NORM, VEL_NORM_ANGULAR, BALL_VEL_NORM

    assert len(df) == 1, "Pass a single-row DataFrame"
    row = df.iloc[0]

    def _make_car_from_row(prefix: str, team: int) -> Car:
        car = Car()
        car.team_num = team
        car.hitbox_type = 0
        car.ball_touches = 0
        car.bump_victim_id = None
        car.wheels_with_contact = (True, True, True, True)
        car.supersonic_time = 0.0
        car.boost_active_time = 0.0
        car.handbrake = 0.0
        car.is_jumping = False
        car.is_holding_jump = False
        car.jump_time = 0.0
        car.has_flipped = False
        car.has_double_jumped = False
        car.is_autoflipping = False
        car.autoflip_timer = 0.0
        car.autoflip_direction = 1.0

        # Car state from parquet columns (parquet stores NORMALISED values,
        # but frame_features.car_features() also NORMALISES the raw GameState,
        # so we store normalized-range values in Car physics to mirror the
        # normalization cancellation: norm(raw * scale) = stored_norm.
        # For the consistency test we instead directly embed the parquet-stored
        # normalised values into Car and verify the pipeline is pass-through.)
        phys = PhysicsObject()
        # positions: parquet stores pos/POS_NORM; GameState has raw; car_features normalises
        # So Car.physics.position = stored_value * POS_NORM to undo normalisation
        px = float(row[f"{prefix}_pos_x"]) * POS_NORM[0]
        py = float(row[f"{prefix}_pos_y"]) * POS_NORM[1]
        pz = float(row[f"{prefix}_pos_z"]) * POS_NORM[2]
        phys.position = np.array([px, py, pz], dtype=np.float32)

        vx = float(row[f"{prefix}_linvel_x"]) * BALL_VEL_NORM
        vy = float(row[f"{prefix}_linvel_y"]) * BALL_VEL_NORM
        vz = float(row[f"{prefix}_linvel_z"]) * BALL_VEL_NORM
        phys.linear_velocity = np.array([vx, vy, vz], dtype=np.float32)

        avx = float(row[f"{prefix}_angvel_x"]) * VEL_NORM_ANGULAR
        avy = float(row[f"{prefix}_angvel_y"]) * VEL_NORM_ANGULAR
        avz = float(row[f"{prefix}_angvel_z"]) * VEL_NORM_ANGULAR
        phys.angular_velocity = np.array([avx, avy, avz], dtype=np.float32)
        phys.euler_angles = np.zeros(3, dtype=np.float32)
        car.physics = phys
        car._inverted_physics = None

        car.boost_amount = float(row[f"{prefix}_boost"]) * 100.0
        car.on_ground = bool(row[f"{prefix}_on_ground"] > 0.5)
        car.has_flip = bool(row[f"{prefix}_has_flip"] > 0.5)
        car.has_jumped = bool(row[f"{prefix}_has_jumped"] > 0.5)
        car.is_supersonic = bool(row[f"{prefix}_is_supersonic"] > 0.5)
        car.air_time_since_jump = float(row[f"{prefix}_air_time"]) * 1.5
        car.flip_time = float(row[f"{prefix}_flip_time"])
        car.demo_respawn_timer = float(row[f"{prefix}_demo_timer"]) * 3.0
        return car

    state = GameState()
    state.tick_count = 0
    state.goal_scored = False
    state.config = None
    state.cars = {
        "b0": _make_car_from_row("car_blue0", BLUE_TEAM),
        "b1": _make_car_from_row("car_blue1", BLUE_TEAM),
        "o0": _make_car_from_row("car_orange0", ORANGE_TEAM),
        "o1": _make_car_from_row("car_orange1", ORANGE_TEAM),
    }

    from src.obs.frame_features import BALL_VEL_NORM, POS_NORM, VEL_NORM_ANGULAR
    ball = PhysicsObject()
    ball.position = np.array([
        float(row["ball_pos_x"]) * POS_NORM[0],
        float(row["ball_pos_y"]) * POS_NORM[1],
        float(row["ball_pos_z"]) * POS_NORM[2],
    ], dtype=np.float32)
    ball.linear_velocity = np.array([
        float(row["ball_linvel_x"]) * BALL_VEL_NORM,
        float(row["ball_linvel_y"]) * BALL_VEL_NORM,
        float(row["ball_linvel_z"]) * BALL_VEL_NORM,
    ], dtype=np.float32)
    ball.angular_velocity = np.array([
        float(row["ball_angvel_x"]) * VEL_NORM_ANGULAR,
        float(row["ball_angvel_y"]) * VEL_NORM_ANGULAR,
        float(row["ball_angvel_z"]) * VEL_NORM_ANGULAR,
    ], dtype=np.float32)
    ball.euler_angles = np.zeros(3, dtype=np.float32)
    state.ball = ball
    state._inverted_ball = None
    # Boost pads — reconstruct from summary columns
    pad_timers = np.zeros(34, dtype=np.float32)
    large_indices = [3, 4, 15, 18, 29, 30]
    for i, idx in enumerate(large_indices):
        col = f"boost_pad_large_{i}_timer"
        if col in row.index:
            pad_timers[idx] = float(row[col]) * 10.0  # stored normalised /10
    state.boost_pad_timers = pad_timers
    state._inverted_boost_pad_timers = None

    return state


# ===========================================================================
# Test cases
# ===========================================================================

class TestOutputShape:
    """frames_df_to_universal always returns (T, 133)."""

    def test_1v1_shape(self):
        df = _build_synthetic_df(n_blue=1, n_orange=1, T=5)
        out = frames_df_to_universal(df)
        assert out.shape == (5, 133), f"Expected (5,133), got {out.shape}"
        assert out.dtype == np.float32

    def test_2v2_shape(self):
        df = _build_synthetic_df(n_blue=2, n_orange=2, T=4)
        out = frames_df_to_universal(df)
        assert out.shape == (4, 133)

    def test_3v3_shape(self):
        df = _build_synthetic_df(n_blue=3, n_orange=3, T=6)
        out = frames_df_to_universal(df)
        assert out.shape == (6, 133)

    def test_single_row(self):
        df = _build_synthetic_df(n_blue=2, n_orange=2, T=1)
        out = frames_df_to_universal(df)
        assert out.shape == (1, 133)

    def test_empty_df_returns_empty(self):
        df = _build_synthetic_df(n_blue=2, n_orange=2, T=1)
        df_empty = df.iloc[0:0]  # 0 rows, same columns
        out = frames_df_to_universal(df_empty)
        assert out.shape == (0, 133)

    def test_no_car_cols_raises(self):
        """A DataFrame with only meta/ball columns but NO car columns raises ValueError."""
        df = _build_synthetic_df(n_blue=2, n_orange=2, T=2)
        # Remove all car columns
        car_cols = [c for c in df.columns if c.startswith("car_")]
        df_nocar = df.drop(columns=car_cols)
        with pytest.raises(ValueError, match="no car columns"):
            frames_df_to_universal(df_nocar)


class TestMasksAndCounts:
    """Presence masks and n_active scalars are correct per game mode."""

    def _mask(self, out: np.ndarray) -> np.ndarray:
        return out[:, _MASK_START:_MASK_END]

    def _nact(self, out: np.ndarray) -> np.ndarray:
        return out[:, _NACT_START:_NACT_END]

    def test_1v1_mask(self):
        """1v1: blue0 (slot0) + orange0 (slot3) present; slots 1,2,4,5 absent."""
        df = _build_synthetic_df(n_blue=1, n_orange=1, T=2)
        out = frames_df_to_universal(df)
        mask = self._mask(out)
        expected = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(mask[0], expected)
        np.testing.assert_array_equal(mask[1], expected)  # same across all rows

    def test_2v2_mask(self):
        """2v2: slots 0,1 blue + slots 3,4 orange present; 2 and 5 absent."""
        df = _build_synthetic_df(n_blue=2, n_orange=2, T=2)
        out = frames_df_to_universal(df)
        mask = self._mask(out)
        expected = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(mask[0], expected)

    def test_3v3_mask(self):
        """3v3: all 6 slots present."""
        df = _build_synthetic_df(n_blue=3, n_orange=3, T=2)
        out = frames_df_to_universal(df)
        mask = self._mask(out)
        np.testing.assert_array_equal(mask[0], np.ones(6, dtype=np.float32))

    def test_1v1_n_active(self):
        """1v1: n_blue=1 → n_blue/3=1/3, n_orange=1 → 1/3, n_total=2/6=1/3."""
        df = _build_synthetic_df(n_blue=1, n_orange=1, T=1)
        out = frames_df_to_universal(df)
        nact = self._nact(out)[0]
        assert nact[0] == pytest.approx(1.0 / 3, abs=1e-5), "n_blue_norm 1v1"
        assert nact[1] == pytest.approx(1.0 / 3, abs=1e-5), "n_orange_norm 1v1"
        assert nact[2] == pytest.approx(2.0 / 6, abs=1e-5), "n_total_norm 1v1"

    def test_2v2_n_active(self):
        """2v2: n_blue=2 → 2/3, n_orange=2 → 2/3, n_total=4/6=2/3."""
        df = _build_synthetic_df(n_blue=2, n_orange=2, T=1)
        out = frames_df_to_universal(df)
        nact = self._nact(out)[0]
        assert nact[0] == pytest.approx(2.0 / 3, abs=1e-5)
        assert nact[1] == pytest.approx(2.0 / 3, abs=1e-5)
        assert nact[2] == pytest.approx(4.0 / 6, abs=1e-5)

    def test_3v3_n_active(self):
        """3v3: all three n_active scalars are 1.0."""
        df = _build_synthetic_df(n_blue=3, n_orange=3, T=1)
        out = frames_df_to_universal(df)
        nact = self._nact(out)[0]
        np.testing.assert_allclose(nact, [1.0, 1.0, 1.0], atol=1e-5)


class TestSlotMapping:
    """Car data lands in the correct universal slot."""

    def _slot(self, out: np.ndarray, slot: int, row: int = 0) -> np.ndarray:
        """Extract the 18-dim car block for a given slot."""
        start = _CARS_START + slot * _CAR_FEATURES
        return out[row, start : start + _CAR_FEATURES]

    def test_blue_car_positions(self):
        """car_blue0 goes to slot 0, car_blue1 goes to slot 1."""
        df = _build_synthetic_df(n_blue=2, n_orange=1, T=1)
        out = frames_df_to_universal(df)
        row = df.iloc[0]

        # slot 0 = car_blue0: check pos_x
        assert out[0, _CARS_START + 0] == pytest.approx(
            row["car_blue0_pos_x"], abs=1e-5
        )
        # slot 1 = car_blue1: check pos_x
        assert out[0, _CARS_START + _CAR_FEATURES] == pytest.approx(
            row["car_blue1_pos_x"], abs=1e-5
        )

    def test_orange_car_positions(self):
        """car_orange0 goes to slot 3, car_orange1 goes to slot 4, car_orange2 to slot 5."""
        df = _build_synthetic_df(n_blue=1, n_orange=3, T=1)
        out = frames_df_to_universal(df)
        row = df.iloc[0]

        slot3_start = _CARS_START + 3 * _CAR_FEATURES
        slot4_start = _CARS_START + 4 * _CAR_FEATURES
        slot5_start = _CARS_START + 5 * _CAR_FEATURES

        assert out[0, slot3_start] == pytest.approx(row["car_orange0_pos_x"], abs=1e-5)
        assert out[0, slot4_start] == pytest.approx(row["car_orange1_pos_x"], abs=1e-5)
        assert out[0, slot5_start] == pytest.approx(row["car_orange2_pos_x"], abs=1e-5)

    def test_absent_slots_are_zero(self):
        """Absent car slots (1v1: slots 1,2,4,5) must be all-zero."""
        df = _build_synthetic_df(n_blue=1, n_orange=1, T=2)
        out = frames_df_to_universal(df)
        for slot in (1, 2, 4, 5):
            block = self._slot(out, slot, row=0)
            assert (block == 0).all(), (
                f"Slot {slot} should be zero-padded for 1v1, got {block}"
            )

    def test_full_car_block_values(self):
        """All 18 car sub-features map correctly (not just pos_x)."""
        df = _build_synthetic_df(n_blue=1, n_orange=1, T=1)
        out = frames_df_to_universal(df)
        row = df.iloc[0]

        # Verify all 18 sub-features of car_blue0 (slot 0)
        for sf_idx, sf in enumerate(_CAR_SUBFEATURES):
            col = f"car_blue0_{sf}"
            expected = float(row[col])
            actual = float(out[0, _CARS_START + sf_idx])
            assert actual == pytest.approx(expected, abs=1e-5), (
                f"car_blue0 sub-feature '{sf}' (dim {_CARS_START + sf_idx}): "
                f"expected {expected:.4f}, got {actual:.4f}"
            )


class TestBallAndBoost:
    """Ball and boost pad columns are mapped to the right positions."""

    def test_ball_mapped(self):
        """Ball columns land in dims [0:9]."""
        df = _build_synthetic_df(n_blue=1, n_orange=1, T=1)
        out = frames_df_to_universal(df)
        row = df.iloc[0]
        for i, col in enumerate(_BALL_COLS):
            assert out[0, i] == pytest.approx(float(row[col]), abs=1e-5), (
                f"Ball column '{col}' at dim {i}"
            )

    def test_boost_pad_large_timers(self):
        """Large boost pad timers land in dims [117:123]."""
        df = _build_synthetic_df(n_blue=1, n_orange=1, T=1, with_boost_pads=True)
        out = frames_df_to_universal(df)
        row = df.iloc[0]
        for i in range(6):
            col = f"boost_pad_large_{i}_timer"
            assert out[0, _BOOST_START + i] == pytest.approx(float(row[col]), abs=1e-5), (
                f"Large pad timer {i} at dim {_BOOST_START + i}"
            )

    def test_boost_pad_small_frac(self):
        """Small pad fraction lands in dim 123."""
        df = _build_synthetic_df(n_blue=1, n_orange=1, T=1, with_boost_pads=True)
        out = frames_df_to_universal(df)
        row = df.iloc[0]
        assert out[0, _BOOST_END - 1] == pytest.approx(
            float(row["boost_pad_small_ready_frac"]), abs=1e-5
        )

    def test_no_boost_cols_gives_zeros(self):
        """When boost pad columns are absent, boost dims are zero."""
        df = _build_synthetic_df(n_blue=1, n_orange=1, T=1, with_boost_pads=False)
        out = frames_df_to_universal(df)
        np.testing.assert_array_equal(
            out[0, _BOOST_START:_BOOST_END],
            np.zeros(_BOOST_SUMMARY_FEATURES, dtype=np.float32),
        )


class TestNoNaNOrInf:
    """No NaN or Inf in output for any game mode."""

    @pytest.mark.parametrize("n_blue,n_orange", [(1, 1), (2, 2), (3, 3), (1, 3), (3, 1)])
    def test_no_nan_inf(self, n_blue: int, n_orange: int):
        df = _build_synthetic_df(n_blue=n_blue, n_orange=n_orange, T=3)
        out = frames_df_to_universal(df)
        assert not np.isnan(out).any(), f"NaN for {n_blue}v{n_orange}"
        assert not np.isinf(out).any(), f"Inf for {n_blue}v{n_orange}"


class TestFeatureNames:
    """universal_feature_names() is compatible with the mapper output."""

    def test_names_length(self):
        assert len(universal_feature_names()) == 133

    def test_names_consistent_with_size(self):
        assert len(universal_feature_names()) == universal_feature_size()

    def test_names_no_duplicates(self):
        names = universal_feature_names()
        assert len(names) == len(set(names)), "Duplicate feature names found"


class TestConsistencyOfflineVsLive:
    """Key test: frames_df_to_universal == extract_universal_frame_features for same state.

    Strategy: construct a GameState using the same helper already used in
    test_universal_features.py (_make_state), then construct the EQUIVALENT
    DataFrame row by manually reading back the normalised values that
    extract_universal_frame_features would produce — and verify both paths
    output an identical 133-dim vector.

    This avoids setting Car properties that have no setter (e.g. Car.has_flip),
    using instead the public GameState builder from the existing test fixtures.
    The consistency property we verify is:
        extract_universal_frame_features(state)
            == frames_df_to_universal(state_to_df(state))
    where state_to_df extracts the same normalised columns that the replay
    pipeline writes to frames.parquet.
    """

    def _make_state_2v2(self):
        """Build a 2v2 GameState using the existing helper (no-setter-safe)."""
        from rlgym.rocket_league.api import Car, GameState, PhysicsObject
        from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM

        def _make_car(team: int, x: float, y: float) -> Car:
            car = Car()
            car.team_num = team
            car.hitbox_type = 0
            car.ball_touches = 0
            car.bump_victim_id = None
            car.demo_respawn_timer = 0.5
            car.wheels_with_contact = (True, True, True, True)
            car.supersonic_time = 0.0
            car.boost_amount = 50.0
            car.boost_active_time = 0.0
            car.handbrake = 0.0
            car.is_jumping = False
            car.has_jumped = False
            car.is_holding_jump = False
            car.jump_time = 0.0
            car.has_flipped = False
            car.has_double_jumped = False
            car.air_time_since_jump = 0.3
            car.flip_time = 0.2
            car.flip_torque = np.zeros(3, dtype=np.float32)
            car.is_autoflipping = False
            car.autoflip_timer = 0.0
            car.autoflip_direction = 1.0
            phys = PhysicsObject()
            phys.position = np.array([x, y, 17.0], dtype=np.float32)
            phys.linear_velocity = np.array([300.0, -100.0, 0.0], dtype=np.float32)
            phys.angular_velocity = np.array([0.1, 0.2, -0.1], dtype=np.float32)
            phys.euler_angles = np.zeros(3, dtype=np.float32)
            car.physics = phys
            car._inverted_physics = None
            return car

        state = GameState()
        state.tick_count = 0
        state.goal_scored = False
        state.config = None
        state.cars = {
            "b0": _make_car(BLUE_TEAM, -1000.0, -2000.0),
            "b1": _make_car(BLUE_TEAM, -2000.0, -1000.0),
            "o0": _make_car(ORANGE_TEAM, 1000.0, 2000.0),
            "o1": _make_car(ORANGE_TEAM, 2000.0, 1000.0),
        }

        from rlgym.rocket_league.api import PhysicsObject as PhysObj
        ball = PhysObj()
        ball.position = np.array([100.0, 200.0, 93.0], dtype=np.float32)
        ball.linear_velocity = np.array([500.0, -300.0, 100.0], dtype=np.float32)
        ball.angular_velocity = np.array([1.0, -0.5, 0.3], dtype=np.float32)
        ball.euler_angles = np.zeros(3, dtype=np.float32)
        state.ball = ball
        state._inverted_ball = None
        state.boost_pad_timers = np.linspace(0.0, 5.0, 34, dtype=np.float32)
        state._inverted_boost_pad_timers = None
        return state

    def _state_to_df(self, state) -> pd.DataFrame:
        """Convert a GameState to the parquet-like DataFrame format.

        The replay pipeline writes NORMALISED values to parquet — this mirrors
        what extract_frame_features / extract_universal_frame_features computes
        per dimension:
          - pos / POS_NORM
          - linvel / BALL_VEL_NORM
          - angvel / VEL_NORM_ANGULAR
          - boost / 100
          - boolean states as 0.0 or 1.0
          - air_time clipped / 1.5
          - flip_time clipped at 1.0
          - demo_timer clipped / 3.0
          - is_orange as 0/1
        """
        from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM
        from src.obs.frame_features import (
            BALL_VEL_NORM, POS_NORM, VEL_NORM_ANGULAR,
        )

        row: dict[str, float] = {}

        # Ball
        b = state.ball
        row["ball_pos_x"] = float(b.position[0] / POS_NORM[0])
        row["ball_pos_y"] = float(b.position[1] / POS_NORM[1])
        row["ball_pos_z"] = float(b.position[2] / POS_NORM[2])
        row["ball_linvel_x"] = float(b.linear_velocity[0] / BALL_VEL_NORM)
        row["ball_linvel_y"] = float(b.linear_velocity[1] / BALL_VEL_NORM)
        row["ball_linvel_z"] = float(b.linear_velocity[2] / BALL_VEL_NORM)
        row["ball_angvel_x"] = float(b.angular_velocity[0] / VEL_NORM_ANGULAR)
        row["ball_angvel_y"] = float(b.angular_velocity[1] / VEL_NORM_ANGULAR)
        row["ball_angvel_z"] = float(b.angular_velocity[2] / VEL_NORM_ANGULAR)

        # Cars — sorted same way as extract_universal_frame_features
        cars_blue = [
            c for _, c in sorted(
                ((aid, c) for aid, c in state.cars.items() if c.team_num == BLUE_TEAM),
                key=lambda kv: str(kv[0]),
            )
        ]
        cars_orange = [
            c for _, c in sorted(
                ((aid, c) for aid, c in state.cars.items() if c.team_num == ORANGE_TEAM),
                key=lambda kv: str(kv[0]),
            )
        ]

        def _write_car(prefix: str, car, is_orange_val: float) -> None:
            p = car.physics
            row[f"{prefix}_pos_x"] = float(p.position[0] / POS_NORM[0])
            row[f"{prefix}_pos_y"] = float(p.position[1] / POS_NORM[1])
            row[f"{prefix}_pos_z"] = float(p.position[2] / POS_NORM[2])
            row[f"{prefix}_linvel_x"] = float(p.linear_velocity[0] / BALL_VEL_NORM)
            row[f"{prefix}_linvel_y"] = float(p.linear_velocity[1] / BALL_VEL_NORM)
            row[f"{prefix}_linvel_z"] = float(p.linear_velocity[2] / BALL_VEL_NORM)
            row[f"{prefix}_angvel_x"] = float(p.angular_velocity[0] / VEL_NORM_ANGULAR)
            row[f"{prefix}_angvel_y"] = float(p.angular_velocity[1] / VEL_NORM_ANGULAR)
            row[f"{prefix}_angvel_z"] = float(p.angular_velocity[2] / VEL_NORM_ANGULAR)
            row[f"{prefix}_boost"] = float(car.boost_amount / 100.0)
            row[f"{prefix}_on_ground"] = float(car.on_ground)
            row[f"{prefix}_has_flip"] = float(car.has_flip)
            row[f"{prefix}_has_jumped"] = float(car.has_jumped)
            row[f"{prefix}_is_supersonic"] = float(car.is_supersonic)
            row[f"{prefix}_air_time"] = float(min(car.air_time_since_jump, 1.5) / 1.5)
            row[f"{prefix}_flip_time"] = float(min(car.flip_time, 1.0))
            row[f"{prefix}_demo_timer"] = float(min(car.demo_respawn_timer, 3.0) / 3.0)
            row[f"{prefix}_is_orange"] = is_orange_val

        for i, car in enumerate(cars_blue):
            _write_car(f"car_blue{i}", car, 0.0)
        for i, car in enumerate(cars_orange):
            _write_car(f"car_orange{i}", car, 1.0)

        # Boost pads (summary only — same as boost_pad_summary())
        timers = state.boost_pad_timers
        large_indices = [3, 4, 15, 18, 29, 30]
        for j, idx in enumerate(large_indices):
            row[f"boost_pad_large_{j}_timer"] = float(timers[idx]) / 10.0
        small_mask = np.ones(len(timers), dtype=bool)
        for idx in large_indices:
            small_mask[idx] = False
        small_ready_frac = float((timers[small_mask] <= 1e-3).sum()) / max(small_mask.sum(), 1)
        row["boost_pad_small_ready_frac"] = small_ready_frac

        return pd.DataFrame([row])

    def test_2v2_offline_equals_live(self):
        """frames_df_to_universal == extract_universal_frame_features for 2v2 (within float32 tolerance)."""
        state = self._make_state_2v2()

        # --- Live path: GameState → 133-dim ---
        live_vec = extract_universal_frame_features(state)  # (133,) float32

        # --- Build equivalent DataFrame row (using same normalisation as replay pipeline) ---
        df = self._state_to_df(state)

        # --- Offline path: DataFrame → 133-dim ---
        offline_vec = frames_df_to_universal(df)[0]  # (133,) float32

        # Full vector must match within float32 arithmetic tolerance (1e-5).
        np.testing.assert_allclose(
            offline_vec, live_vec, atol=1e-5,
            err_msg=(
                "frames_df_to_universal does NOT match extract_universal_frame_features.\n"
                "Offline/live representation inconsistency — check mapper column order."
            ),
        )

        # Spot-check structural invariants
        # Mask: 2v2 → [1,1,0,1,1,0]
        np.testing.assert_array_equal(
            offline_vec[_MASK_START:_MASK_END],
            np.array([1., 1., 0., 1., 1., 0.], dtype=np.float32),
        )
        # n_active: [2/3, 2/3, 4/6]
        np.testing.assert_allclose(
            offline_vec[_NACT_START:_NACT_END],
            [2.0/3, 2.0/3, 4.0/6],
            atol=1e-5,
        )
        # Absent slots 2 and 5 are zero
        for slot in (2, 5):
            start = _CARS_START + slot * _CAR_FEATURES
            np.testing.assert_array_equal(
                offline_vec[start : start + _CAR_FEATURES],
                np.zeros(_CAR_FEATURES),
                err_msg=f"Absent slot {slot} should be zero",
            )
