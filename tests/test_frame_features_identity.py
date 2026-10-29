"""Regression test: per-car identity stability in extract_frame_features.

Guards against the id()-sort bug where cars were sorted by memory address
(id(c)) — an unstable key that reshuffled which player landed in car_blue0 /
car_blue1 every frame, producing physically-impossible teleporting trajectories
(~4090 uu/frame per-car jumps).

The fix sorts by str(AgentID) (the stable dict key from rlgym-tools), which is
constant across all frames of a match.

Tests:
  1. extract_frame_features: same AgentIDs in different dict insertion order
     → identical feature vector (same slot assignment).
  2. extract_universal_frame_features: same test for the 133-dim universal extractor.
  3. Slot positions are physically meaningful (player A always in slot 0,
     player B always in slot 1, regardless of dict order).
"""

from __future__ import annotations

import numpy as np
import pytest

from rlgym.rocket_league.api import Car, GameState, PhysicsObject
from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM

from src.obs.frame_features import (
    ExtractorConfig,
    extract_frame_features,
    extract_universal_frame_features,
)


# ---------------------------------------------------------------------------
# Helpers — copied from test_frame_features.py for self-containment
# ---------------------------------------------------------------------------

def _make_car(team: int, x: float, y: float, boost: float = 33.0) -> Car:
    """Build a minimal Car with distinguishable physics (x,y) and boost."""
    car = Car()
    car.team_num = team
    car.hitbox_type = 0
    car.ball_touches = 0
    car.bump_victim_id = None
    car.demo_respawn_timer = 0.0
    car.wheels_with_contact = (True, True, True, True)
    car.supersonic_time = 0.0
    car.boost_amount = boost
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


def _make_ball() -> PhysicsObject:
    ball = PhysicsObject()
    ball.position = np.array([0.0, 0.0, 100.0], dtype=np.float32)
    ball.linear_velocity = np.array([100.0, 200.0, 50.0], dtype=np.float32)
    ball.angular_velocity = np.zeros(3, dtype=np.float32)
    ball.euler_angles = np.zeros(3, dtype=np.float32)
    return ball


def _make_state_with_cars(cars_dict: dict) -> GameState:
    """Build a minimal GameState given a pre-built cars dict."""
    state = GameState()
    state.tick_count = 0
    state.goal_scored = False
    state.config = None
    state.cars = cars_dict
    state.ball = _make_ball()
    state._inverted_ball = None
    state.boost_pad_timers = np.zeros(34, dtype=np.float32)
    state._inverted_boost_pad_timers = None
    return state


# ---------------------------------------------------------------------------
# Core: two distinct blue players with distinguishable positions
# AgentID "player_alpha" sorts before "player_beta" lexicographically
# so alpha → slot 0 (car_blue0), beta → slot 1 (car_blue1)
# ---------------------------------------------------------------------------

# Player alpha: x=-1000, boost=50  (str key "player_alpha" < "player_beta")
# Player beta:  x=-2000, boost=75
_ALPHA_ID = "player_alpha"
_BETA_ID = "player_beta"

_ALPHA_CAR_X = -1000.0
_BETA_CAR_X = -2000.0
_ALPHA_BOOST = 50.0
_BETA_BOOST = 75.0

# 2 orange players (need them to avoid zero-padding changing the vector)
_ORG_A_ID = "org_a"
_ORG_B_ID = "org_b"


def _build_cars_order1() -> dict:
    """alpha inserted first, beta second — one dict ordering."""
    return {
        _ALPHA_ID: _make_car(BLUE_TEAM, _ALPHA_CAR_X, -500.0, boost=_ALPHA_BOOST),
        _BETA_ID:  _make_car(BLUE_TEAM, _BETA_CAR_X, -500.0, boost=_BETA_BOOST),
        _ORG_A_ID: _make_car(ORANGE_TEAM, 1000.0, 500.0),
        _ORG_B_ID: _make_car(ORANGE_TEAM, 2000.0, 500.0),
    }


def _build_cars_order2() -> dict:
    """beta inserted first, alpha second — reversed dict ordering."""
    return {
        _BETA_ID:  _make_car(BLUE_TEAM, _BETA_CAR_X, -500.0, boost=_BETA_BOOST),
        _ALPHA_ID: _make_car(BLUE_TEAM, _ALPHA_CAR_X, -500.0, boost=_ALPHA_BOOST),
        _ORG_B_ID: _make_car(ORANGE_TEAM, 2000.0, 500.0),
        _ORG_A_ID: _make_car(ORANGE_TEAM, 1000.0, 500.0),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExtractFrameFeaturesIdentity:
    """extract_frame_features must be stable regardless of dict insertion order."""

    def test_same_vector_regardless_of_dict_order(self):
        """Core regression: inserting cars in different dict order must produce
        the identical feature vector."""
        cfg = ExtractorConfig(n_cars=4)
        state1 = _make_state_with_cars(_build_cars_order1())
        state2 = _make_state_with_cars(_build_cars_order2())

        feat1 = extract_frame_features(state1, config=cfg)
        feat2 = extract_frame_features(state2, config=cfg)

        np.testing.assert_array_equal(
            feat1, feat2,
            err_msg=(
                "extract_frame_features produced different vectors for the same "
                "players inserted in different dict order — AgentID sort is broken."
            ),
        )

    def test_alpha_in_slot0_beta_in_slot1(self):
        """Verify WHICH player lands in which slot (alpha < beta lexicographically).

        car_blue0 slot starts at offset 9 (after 9 ball features).
        car_blue1 slot starts at offset 27 (9 + 18).
        Position x is at feature index 0 within each car block.
        Boost is at feature index 9 within each car block (after 9 physics floats).
        """
        cfg = ExtractorConfig(n_cars=4)
        state = _make_state_with_cars(_build_cars_order1())
        feat = extract_frame_features(state, config=cfg)

        from src.obs.frame_features import POS_NORM

        # Slot 0 (car_blue0): should be player_alpha (x=-1000, boost=50)
        slot0_start = 9
        slot0_pos_x_norm = feat[slot0_start + 0]
        slot0_boost = feat[slot0_start + 9]
        expected_alpha_x_norm = _ALPHA_CAR_X / POS_NORM[0]
        expected_alpha_boost = _ALPHA_BOOST / 100.0

        assert slot0_pos_x_norm == pytest.approx(expected_alpha_x_norm, abs=1e-5), (
            f"car_blue0 pos_x={slot0_pos_x_norm:.4f} expected {expected_alpha_x_norm:.4f} "
            f"(player_alpha). Wrong player in slot 0."
        )
        assert slot0_boost == pytest.approx(expected_alpha_boost, abs=1e-5), (
            f"car_blue0 boost={slot0_boost:.4f} expected {expected_alpha_boost:.4f}."
        )

        # Slot 1 (car_blue1): should be player_beta (x=-2000, boost=75)
        slot1_start = 9 + 18
        slot1_pos_x_norm = feat[slot1_start + 0]
        slot1_boost = feat[slot1_start + 9]
        expected_beta_x_norm = _BETA_CAR_X / POS_NORM[0]
        expected_beta_boost = _BETA_BOOST / 100.0

        assert slot1_pos_x_norm == pytest.approx(expected_beta_x_norm, abs=1e-5), (
            f"car_blue1 pos_x={slot1_pos_x_norm:.4f} expected {expected_beta_x_norm:.4f} "
            f"(player_beta). Wrong player in slot 1."
        )
        assert slot1_boost == pytest.approx(expected_beta_boost, abs=1e-5), (
            f"car_blue1 boost={slot1_boost:.4f} expected {expected_beta_boost:.4f}."
        )

    def test_slot_assignment_stable_across_reversed_order(self):
        """With reversed insertion order, alpha must STILL be in slot 0."""
        cfg = ExtractorConfig(n_cars=4)
        state_reversed = _make_state_with_cars(_build_cars_order2())
        feat = extract_frame_features(state_reversed, config=cfg)

        from src.obs.frame_features import POS_NORM

        slot0_pos_x_norm = feat[9 + 0]
        expected_alpha_x_norm = _ALPHA_CAR_X / POS_NORM[0]

        assert slot0_pos_x_norm == pytest.approx(expected_alpha_x_norm, abs=1e-5), (
            "After reversing dict insertion order, car_blue0 no longer holds player_alpha. "
            "AgentID sort is not stable — id()-sort regression may have returned."
        )


class TestExtractUniversalFrameFeaturesIdentity:
    """Same stability guarantee for the 133-dim universal extractor."""

    def test_same_vector_regardless_of_dict_order(self):
        state1 = _make_state_with_cars(_build_cars_order1())
        state2 = _make_state_with_cars(_build_cars_order2())

        feat1 = extract_universal_frame_features(state1)
        feat2 = extract_universal_frame_features(state2)

        np.testing.assert_array_equal(
            feat1, feat2,
            err_msg=(
                "extract_universal_frame_features produced different vectors for "
                "the same players inserted in different dict order."
            ),
        )

    def test_alpha_in_slot0_universal(self):
        """Universal extractor: player_alpha (str-sorted first) must land in slot 0.

        Universal layout: [0:9] ball, [9:27] car_blue0, [27:45] car_blue1, ...
        """
        state = _make_state_with_cars(_build_cars_order1())
        feat = extract_universal_frame_features(state)

        from src.obs.frame_features import POS_NORM

        slot0_pos_x_norm = feat[9 + 0]
        expected_alpha_x_norm = _ALPHA_CAR_X / POS_NORM[0]

        assert slot0_pos_x_norm == pytest.approx(expected_alpha_x_norm, abs=1e-5), (
            f"Universal extractor: car_blue0 pos_x={slot0_pos_x_norm:.4f}, "
            f"expected {expected_alpha_x_norm:.4f} (player_alpha)."
        )

    def test_output_shape_133(self):
        from src.obs.frame_features import universal_feature_size
        state = _make_state_with_cars(_build_cars_order1())
        feat = extract_universal_frame_features(state)
        assert feat.shape == (universal_feature_size(),), (
            f"Expected shape ({universal_feature_size()},), got {feat.shape}"
        )


class TestIntegerAgentIDs:
    """Verify the fix also works with integer AgentIDs (as rlgym-tools produces).

    rlgym-tools sets state.cars keys to player unique_ids (integers).
    The str() cast must produce consistent ordering across frames.
    """

    def test_integer_agent_ids_stable(self):
        """Integer keys: str(1) < str(2) < str(10) is NOT numerically sorted
        (str comparison: "1" < "10" < "2"). The fix must be consistent — the
        SAME key→slot mapping every frame, regardless of insertion order."""
        cfg = ExtractorConfig(n_cars=4)

        # Player with id=1 → str "1", player with id=2 → str "2"
        # "1" < "2" lexicographically → id=1 in slot 0, id=2 in slot 1
        car_id1 = _make_car(BLUE_TEAM, -1111.0, -500.0, boost=11.0)
        car_id2 = _make_car(BLUE_TEAM, -2222.0, -500.0, boost=22.0)

        # Order A: {1: ..., 2: ...}
        state_a = _make_state_with_cars({
            1: car_id1,
            2: car_id2,
            101: _make_car(ORANGE_TEAM, 1000.0, 500.0),
            102: _make_car(ORANGE_TEAM, 2000.0, 500.0),
        })
        # Order B: {2: ..., 1: ...}
        state_b = _make_state_with_cars({
            2: car_id2,
            1: car_id1,
            102: _make_car(ORANGE_TEAM, 2000.0, 500.0),
            101: _make_car(ORANGE_TEAM, 1000.0, 500.0),
        })

        feat_a = extract_frame_features(state_a, config=cfg)
        feat_b = extract_frame_features(state_b, config=cfg)

        np.testing.assert_array_equal(
            feat_a, feat_b,
            err_msg="Integer AgentID keys: dict insertion order changed slot assignment.",
        )
