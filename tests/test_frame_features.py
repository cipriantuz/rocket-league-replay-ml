"""Smoke tests for frame_features extractor."""

import numpy as np
import pytest

from rlgym.rocket_league.api import Car, GameState, PhysicsObject
from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM

from src.obs.frame_features import (
    ExtractorConfig,
    extract_frame_features,
    feature_names,
    feature_size,
)


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


def test_feature_size_matches_extracted_vector():
    state = _make_state()
    feat = extract_frame_features(state)
    assert feat.shape == (feature_size(),)
    assert feat.dtype == np.float32


def test_no_nan_inf_on_default_state():
    state = _make_state()
    feat = extract_frame_features(state)
    assert not np.isnan(feat).any()
    assert not np.isinf(feat).any()


def test_feature_names_length_matches_size():
    cfg = ExtractorConfig()
    assert len(feature_names(cfg)) == feature_size(cfg)


def test_1v1_size():
    state = _make_state(n_blue=1, n_orange=1)
    cfg = ExtractorConfig(n_cars=2)
    feat = extract_frame_features(state, config=cfg)
    assert feat.shape == (feature_size(cfg),)
    assert feature_size(cfg) == 52  # 9 ball + 2*18 cars + 7 boost


def test_3v3_size():
    state = _make_state(n_blue=3, n_orange=3)
    cfg = ExtractorConfig(n_cars=6)
    feat = extract_frame_features(state, config=cfg)
    assert feat.shape == (feature_size(cfg),)
    assert feature_size(cfg) == 124  # 9 + 6*18 + 7


def test_zero_padding_for_missing_cars():
    state = _make_state(n_blue=2, n_orange=1)  # 1 orange missing
    feat = extract_frame_features(state)
    # The last orange car block should be zero-padded
    # Block index: ball(9) + 2 blue(2*18=36) + 1 orange(18) ... starts at offset 9+36+18 = 63
    assert (feat[63 : 63 + 18] == 0).all()


def test_inverted_view_flips_ball_x_y_sign():
    state = _make_state()
    cfg_norm = ExtractorConfig(inverted=False)
    cfg_inv = ExtractorConfig(inverted=True)
    feat_norm = extract_frame_features(state, config=cfg_norm)
    feat_inv = extract_frame_features(state, config=cfg_inv)
    # ball_pos_x, ball_pos_y should be negated under inverted view
    assert feat_inv[0] == pytest.approx(-feat_norm[0], abs=1e-6)
    assert feat_inv[1] == pytest.approx(-feat_norm[1], abs=1e-6)
    # ball_pos_z unchanged (Z is up)
    assert feat_inv[2] == pytest.approx(feat_norm[2], abs=1e-6)
