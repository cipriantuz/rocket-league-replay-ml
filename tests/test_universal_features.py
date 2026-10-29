"""Tests for extract_universal_frame_features (133-dim multi-mode extractor)."""

from __future__ import annotations

import numpy as np
import pytest

from rlgym.rocket_league.api import Car, GameState, PhysicsObject
from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM

from src.obs.frame_features import (
    extract_universal_frame_features,
    universal_feature_names,
    universal_feature_size,
)


# ---------------------------------------------------------------------------
# Helpers — reuse pattern from test_frame_features.py
# ---------------------------------------------------------------------------

def _make_car(team: int, x: float, y: float) -> Car:
    car = Car()
    car.team_num = team
    car.hitbox_type = 0
    car.ball_touches = 0
    car.bump_victim_id = None
    car.demo_respawn_timer = 0.0
    car.wheels_with_contact = (True, True, True, True)
    car.supersonic_time = 0.0
    car.boost_amount = 33.0
    car.boost_active_time = 0.0
    car.handbrake = 0.0
    car.is_jumping = False
    car.has_jumped = False
    car.is_holding_jump = False
    car.jump_time = 0.0
    car.has_flipped = False
    car.has_double_jumped = False
    car.air_time_since_jump = 0.0
    car.flip_time = 0.0
    car.flip_torque = np.zeros(3, dtype=np.float32)
    car.is_autoflipping = False
    car.autoflip_timer = 0.0
    car.autoflip_direction = 1.0
    phys = PhysicsObject()
    phys.position = np.array([x, y, 17.0], dtype=np.float32)
    phys.linear_velocity = np.zeros(3, dtype=np.float32)
    phys.angular_velocity = np.zeros(3, dtype=np.float32)
    phys.euler_angles = np.zeros(3, dtype=np.float32)
    car.physics = phys
    car._inverted_physics = None
    return car


def _make_state(n_blue: int = 2, n_orange: int = 2) -> GameState:
    state = GameState()
    state.tick_count = 0
    state.goal_scored = False
    state.config = None
    state.cars = {}
    for i in range(n_blue):
        state.cars[f"blue-{i}"] = _make_car(BLUE_TEAM, -1000.0 * (i + 1), -2000.0)
    for i in range(n_orange):
        state.cars[f"orange-{i}"] = _make_car(ORANGE_TEAM, 1000.0 * (i + 1), 2000.0)
    ball = PhysicsObject()
    ball.position = np.array([0.0, 0.0, 100.0], dtype=np.float32)
    ball.linear_velocity = np.array([100.0, 200.0, 50.0], dtype=np.float32)
    ball.angular_velocity = np.zeros(3, dtype=np.float32)
    ball.euler_angles = np.zeros(3, dtype=np.float32)
    state.ball = ball
    state._inverted_ball = None
    state.boost_pad_timers = np.zeros(34, dtype=np.float32)
    state._inverted_boost_pad_timers = None
    return state


# ---------------------------------------------------------------------------
# Helpers to extract the mask and n_active sections from a 133-dim vector.
# ---------------------------------------------------------------------------
# Layout: [0:9] ball, [9:117] 6 car slots × 18, [117:124] boost, [124:130] masks, [130:133] n_active

_MASK_START = 124   # 9 + 6*18 + 7
_MASK_END = 130
_NACTIVE_START = 130
_NACTIVE_END = 133


def _mask(feat: np.ndarray) -> np.ndarray:
    return feat[_MASK_START:_MASK_END]


def _n_active(feat: np.ndarray) -> np.ndarray:
    return feat[_NACTIVE_START:_NACTIVE_END]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_size_133():
    """universal_feature_size() == 133 is the contract; verify via extractor too."""
    assert universal_feature_size() == 133

    # verify across all game modes
    for nb, no in [(1, 1), (2, 2), (3, 3), (1, 2), (2, 3)]:
        state = _make_state(nb, no)
        feat = extract_universal_frame_features(state)
        assert feat.shape == (133,), f"Expected (133,) for {nb}v{no}, got {feat.shape}"
        assert feat.dtype == np.float32


def test_feature_names_length_matches_size():
    names = universal_feature_names()
    assert len(names) == universal_feature_size()


def test_1v1_state_padded():
    """1v1: 1 blue + 1 orange → mask = [1,0,0,1,0,0]."""
    state = _make_state(n_blue=1, n_orange=1)
    feat = extract_universal_frame_features(state)
    mask = _mask(feat)
    expected = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_array_equal(mask, expected)


def test_2v2_state_padded():
    """2v2: 2 blue + 2 orange → mask = [1,1,0,1,1,0]."""
    state = _make_state(n_blue=2, n_orange=2)
    feat = extract_universal_frame_features(state)
    mask = _mask(feat)
    expected = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    np.testing.assert_array_equal(mask, expected)


def test_3v3_state_full():
    """3v3: 3 blue + 3 orange → mask = [1,1,1,1,1,1] (no padding)."""
    state = _make_state(n_blue=3, n_orange=3)
    feat = extract_universal_frame_features(state)
    mask = _mask(feat)
    expected = np.ones(6, dtype=np.float32)
    np.testing.assert_array_equal(mask, expected)


def test_n_active_correct():
    """n_active scalars reflect actual car counts, normalized to [0,1]."""
    # 1v1: n_blue=1, n_orange=1, n_total=2 → [1/3, 1/3, 2/6]
    state = _make_state(n_blue=1, n_orange=1)
    feat = extract_universal_frame_features(state)
    n_act = _n_active(feat)
    assert n_act[0] == pytest.approx(1 / 3, abs=1e-5), "n_blue_norm 1v1"
    assert n_act[1] == pytest.approx(1 / 3, abs=1e-5), "n_orange_norm 1v1"
    assert n_act[2] == pytest.approx(2 / 6, abs=1e-5), "n_total_norm 1v1"

    # 2v2: [2/3, 2/3, 4/6]
    state = _make_state(n_blue=2, n_orange=2)
    feat = extract_universal_frame_features(state)
    n_act = _n_active(feat)
    assert n_act[0] == pytest.approx(2 / 3, abs=1e-5), "n_blue_norm 2v2"
    assert n_act[1] == pytest.approx(2 / 3, abs=1e-5), "n_orange_norm 2v2"
    assert n_act[2] == pytest.approx(4 / 6, abs=1e-5), "n_total_norm 2v2"

    # 3v3: [1.0, 1.0, 1.0]
    state = _make_state(n_blue=3, n_orange=3)
    feat = extract_universal_frame_features(state)
    n_act = _n_active(feat)
    np.testing.assert_allclose(n_act, [1.0, 1.0, 1.0], atol=1e-5)


def test_padded_slots_are_zero():
    """Zero-padded car slots must be all-zero (no garbage from adjacent memory)."""
    # 1v1: slots 1, 2, 4, 5 are padded
    state = _make_state(n_blue=1, n_orange=1)
    feat = extract_universal_frame_features(state)
    # Slot offsets: 9 + slot*18
    for slot in (1, 2, 4, 5):
        start = 9 + slot * 18
        assert (feat[start : start + 18] == 0).all(), f"Slot {slot} should be zero for 1v1"


def test_no_nan_inf():
    """No NaN or Inf in output for any game mode."""
    for nb, no in [(1, 1), (2, 2), (3, 3)]:
        state = _make_state(nb, no)
        feat = extract_universal_frame_features(state)
        assert not np.isnan(feat).any(), f"NaN found for {nb}v{no}"
        assert not np.isinf(feat).any(), f"Inf found for {nb}v{no}"
