"""Tests for mechanical pattern detectors: flip_reset, double_tap, aerial_dribble, ceiling_shot."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.patterns.aerial_dribble import detect_aerial_dribbles
from src.patterns.ceiling_shot import detect_ceiling_shots
from src.patterns.double_tap import detect_double_taps
from src.patterns.flip_reset import detect_flip_resets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_physics(pos=(0.0, 0.0, 100.0), vel=(0.0, 0.0, 0.0)) -> MagicMock:
    phys = MagicMock()
    phys.position = np.array(pos, dtype=np.float32)
    phys.linear_velocity = np.array(vel, dtype=np.float32)
    phys.angular_velocity = np.zeros(3, dtype=np.float32)
    # rotation_mtx: identity (car right-side-up: up-Z = +1.0)
    phys.rotation_mtx = np.eye(3, dtype=np.float32)
    return phys


def _make_car(
    ball_touches: int = 0,
    on_ground: bool = True,
    air_time_since_jump: float = 0.0,
    has_flip: bool = True,
    flip_time: float = 0.0,
    boost_active_time: float = 0.0,
    is_supersonic: bool = False,
    pos=(0.0, 0.0, 17.0),
) -> MagicMock:
    car = MagicMock()
    car.ball_touches = ball_touches
    car.on_ground = on_ground
    car.air_time_since_jump = air_time_since_jump
    car.has_flip = has_flip
    car.flip_time = flip_time
    car.boost_active_time = boost_active_time
    car.is_supersonic = is_supersonic
    car.physics = _make_physics(pos=pos)
    return car


def _make_ball(pos=(0.0, 0.0, 100.0)) -> MagicMock:
    ball = MagicMock()
    ball.position = np.array(pos, dtype=np.float32)
    return ball


def _make_state(cars: dict | None = None, ball_pos=(0.0, 0.0, 100.0)) -> MagicMock:
    state = MagicMock()
    state.cars = cars if cars is not None else {}
    state.ball = _make_ball(ball_pos)
    return state


def _make_frame(state: Any) -> MagicMock:
    frame = MagicMock()
    frame.state = state
    return frame


def _frames(states) -> list:
    return [_make_frame(s) for s in states]


# ---------------------------------------------------------------------------
# flip_reset tests
# ---------------------------------------------------------------------------

class TestFlipReset:

    def test_flip_reset_empty_sequence_no_crash(self):
        """Empty frame list must return nothing and not raise."""
        result = list(detect_flip_resets([]))
        assert result == []

    def test_flip_reset_single_frame_no_crash(self):
        """Single frame must return nothing and not raise."""
        state = _make_state(cars={0: _make_car()})
        result = list(detect_flip_resets(_frames([state])))
        assert result == []

    def test_flip_reset_touch_delta_not_cumulative(self):
        """BUG #1 regression: ball_touches=5 from frame 0 (already cumulative).

        All frames have ball_touches=5, so delta=0 every frame.
        Car is upside-down, in air, close to ball, has_flip=False, air_time>0.5.
        The pre-1.41 bug would fire because ball_touches > 0.
        The fixed code must NOT yield any event because delta=0.
        """
        # Upside-down car: rotation_mtx[:, 2][2] < -0.7
        # We patch _car_up_z via rotation_mtx[2, 2] = -0.9
        car = _make_car(
            ball_touches=5,       # cumulative, never increments
            on_ground=False,
            air_time_since_jump=1.0,
            has_flip=False,
            flip_time=0.0,
            pos=(0.0, 0.0, 500.0),  # in air
        )
        # Make car appear upside-down: rotation_mtx[2, 2] = -0.9
        car.physics.rotation_mtx = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, -0.9]], dtype=np.float32
        )

        # Ball very close to car
        ball_pos = (0.0, 0.0, 510.0)  # dist < 200 uu
        car.physics.position = np.array([0.0, 0.0, 500.0], dtype=np.float32)

        # Build 10 identical frames — ball_touches never increments
        states = [
            _make_state(cars={1: car}, ball_pos=ball_pos)
            for _ in range(10)
        ]
        events = list(detect_flip_resets(_frames(states)))
        assert events == [], (
            "No flip_reset should fire when ball_touches delta=0 (cumulative never changes)"
        )

    def test_flip_reset_zero_cars_no_crash(self):
        """State with no cars must not raise."""
        states = [_make_state(cars={}) for _ in range(5)]
        result = list(detect_flip_resets(_frames(states)))
        assert result == []


# ---------------------------------------------------------------------------
# double_tap tests
# ---------------------------------------------------------------------------

class TestDoubleTap:

    def test_double_tap_short_sequence_returns_empty(self):
        """Sequences shorter than 10 frames must return no events."""
        state = _make_state(cars={0: _make_car(ball_touches=0)})
        for length in range(0, 10):
            states = [_make_state(cars={0: _make_car()}) for _ in range(length)]
            result = list(detect_double_taps(_frames(states)))
            assert result == [], f"Expected empty for length={length}"

    def test_double_tap_zero_cars_no_crash(self):
        states = [_make_state(cars={}) for _ in range(15)]
        result = list(detect_double_taps(_frames(states)))
        assert result == []


# ---------------------------------------------------------------------------
# aerial_dribble tests
# ---------------------------------------------------------------------------

class TestAerialDribble:

    def test_aerial_dribble_less_than_min_touches_no_event(self):
        """Only 2 aerial touches must NOT emit an event (requires >= 3)."""
        # Frame 0: baseline (ball_touches=0, no increment)
        # Frame 1: first touch (ball_touches goes 0→1, in air, ball_z>400)
        # Frame 2: second touch (ball_touches goes 1→2, in air, ball_z>400)
        # Frame 3..9: no further touches
        def _aerial_car(touches: int) -> MagicMock:
            return _make_car(
                ball_touches=touches,
                on_ground=False,
                air_time_since_jump=1.0,
                pos=(0.0, 0.0, 600.0),
            )

        touch_seq = [0, 1, 2, 2, 2, 2, 2, 2, 2, 2]
        states = [
            _make_state(cars={0: _aerial_car(t)}, ball_pos=(0.0, 0.0, 500.0))
            for t in touch_seq
        ]
        result = list(detect_aerial_dribbles(_frames(states)))
        assert result == [], "2 touches should not meet MIN_TOUCHES=3 threshold"

    def test_aerial_dribble_zero_cars_no_crash(self):
        states = [_make_state(cars={}) for _ in range(10)]
        result = list(detect_aerial_dribbles(_frames(states)))
        assert result == []


# ---------------------------------------------------------------------------
# ceiling_shot tests
# ---------------------------------------------------------------------------

class TestCeilingShot:

    def test_ceiling_shot_no_ceiling_contact_no_event(self):
        """Car that never reaches Z>1900 must not emit any ceiling_shot event."""
        # Car always at normal height, touches ball in air
        def _low_car(touches: int) -> MagicMock:
            return _make_car(
                ball_touches=touches,
                on_ground=False,
                pos=(0.0, 0.0, 800.0),  # well below CEILING_Z_THRESHOLD=1900
            )

        touch_seq = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        states = [
            _make_state(cars={0: _low_car(t)}, ball_pos=(0.0, 0.0, 800.0))
            for t in touch_seq
        ]
        result = list(detect_ceiling_shots(_frames(states)))
        assert result == [], "No ceiling contact means no ceiling_shot event"

    def test_ceiling_shot_zero_cars_no_crash(self):
        states = [_make_state(cars={}) for _ in range(10)]
        result = list(detect_ceiling_shots(_frames(states)))
        assert result == []


# ---------------------------------------------------------------------------
# Cross-detector: all detectors handle state.cars={} gracefully
# ---------------------------------------------------------------------------

def test_all_detectors_zero_cars_no_crash():
    """All detectors must handle state.cars={} without raising."""
    states = [_make_state(cars={}) for _ in range(20)]
    frames = _frames(states)
    assert list(detect_flip_resets(frames)) == []
    assert list(detect_double_taps(frames)) == []
    assert list(detect_aerial_dribbles(frames)) == []
    assert list(detect_ceiling_shots(frames)) == []
