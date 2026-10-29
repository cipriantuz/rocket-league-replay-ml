"""Fast double flip detector (a.k.a. double wave dash).

A fast double flip is TWO separate dodges in quick succession (within ~0.5 s)
for the same airborne car, used to build speed toward supersonic. This is the
mechanic that the v1 wave-dash detector mislabeled: a normal wave dash is one
dodge, while this is two.

Detection consumes the shared dodge-activation list (src/patterns/dodge_utils),
so it is mutually exclusive with wave_dash_v2 by construction: wave_dash_v2
fires only when exactly one activation sits near a landing, while this fires
when two activations fall within the window.

This detector is orthogonal to double_tap: double_tap is defined on ball
touches, this on dodge inputs. They never double-count.

Frames are native 30 Hz; positions/velocities are RAW RocketSim units (uu, uu/s).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from rlgym_tools.rocket_league.replays.replay_frame import ReplayFrame

from src.patterns.base import PatternEvent
from src.patterns.dodge_utils import dodge_activations

DODGE_WINDOW_FRAMES = 15      # 0.5 s @ 30 Hz — max gap between the two dodges
MIN_GAP_FRAMES = 2            # distinct events, not the same dodge re-detected
MIN_SECOND_DODGE_SPEED = 1100.0  # uu/s — builds toward supersonic (2200)
SPEED_WINDOW = 6
COOLDOWN_FRAMES = 30
SUPERSONIC = 2200.0


def _horizontal_speed(vel) -> float:
    return float(np.hypot(float(vel[0]), float(vel[1])))


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def detect_fast_double_flips(frames: Sequence[ReplayFrame]) -> Iterable[PatternEvent]:
    n = len(frames)
    if n < SPEED_WINDOW + 2:
        return

    acts = dodge_activations(frames)

    for agent_id, dframes in acts.items():
        last_emit = -COOLDOWN_FRAMES - 1
        for i in range(len(dframes) - 1):
            d1 = dframes[i]
            d2 = dframes[i + 1]
            gap = d2 - d1
            if gap < MIN_GAP_FRAMES or gap > DODGE_WINDOW_FRAMES:
                continue
            if (d1 - last_emit) <= COOLDOWN_FRAMES:
                continue

            # Both activations are airborne by construction (dodge_activations
            # requires not on_ground); confirm post-second-dodge speed.
            post_idx = min(n - 1, d2 + SPEED_WINDOW)
            pre_idx = max(0, d1 - SPEED_WINDOW)
            post_car = frames[post_idx].state.cars.get(agent_id)
            pre_car = frames[pre_idx].state.cars.get(agent_id)
            if post_car is None or pre_car is None:
                continue
            post2_speed = _horizontal_speed(post_car.physics.linear_velocity)
            if post2_speed < MIN_SECOND_DODGE_SPEED:
                continue
            pre1_speed = _horizontal_speed(pre_car.physics.linear_velocity)
            reached_ss = bool(getattr(post_car, "is_supersonic", False)) or post2_speed >= SUPERSONIC

            last_emit = d1
            gap_term = _clip01((DODGE_WINDOW_FRAMES - gap) / DODGE_WINDOW_FRAMES)
            speed_term = _clip01((post2_speed - MIN_SECOND_DODGE_SPEED) / MIN_SECOND_DODGE_SPEED)
            conf = 0.45 + 0.30 * gap_term + 0.25 * speed_term + (0.10 if reached_ss else 0.0)
            conf = round(min(conf, 0.95), 3)

            yield PatternEvent(
                pattern="fast_double_flip",
                agent_id=agent_id,
                frame_start=d1,
                frame_end=post_idx,
                confidence=conf,
                metadata={
                    "dodge1_frame": d1,
                    "dodge2_frame": d2,
                    "gap_frames": gap,
                    "post2_speed_uu_s": round(post2_speed, 1),
                    "speed_gain_uu_s": round(post2_speed - pre1_speed, 1),
                    "reached_supersonic": reached_ss,
                    "detector_version": "fast_double_flip_v1",
                },
            )


class FastDoubleFlipDetector:
    name = "fast_double_flip"

    def detect(self, frames: Sequence[ReplayFrame]) -> Iterable[PatternEvent]:
        return detect_fast_double_flips(frames)
