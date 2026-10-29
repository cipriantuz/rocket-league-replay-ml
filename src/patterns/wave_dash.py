"""Wave dash detector — heuristic (imperfect, needs visual tuning).

A wave dash: the car lands from a short airborne phase and dodges into the
ground at the moment of contact, converting the dodge impulse into horizontal
ground speed instead of bouncing. Observable signature on decoded replay state:

    1. Landing edge: ``on_ground`` False -> True.
    2. A dodge fired just before/at contact, very low to the ground: ``flip_time``
       becomes active (>0) while airborne and ``car_z`` is small.
    3. Horizontal ground speed is preserved or gained across the maneuver
       (a normal landing scrubs speed; a wave dash does not).

HONEST LIMITATION: this is a heuristic over reconstructed physics, not a ground
truth. `flip_time` semantics and the speed-gain threshold were tuned by eye on a
handful of replays; expect false positives on ordinary low dodges and false
negatives on subtle dashes. Confidence encodes the horizontal speed gain.

Frames are 30 Hz native replay rate. Positions/velocities here are RAW RocketSim
units (uu, uu/s) — NOT the normalized values in frames.parquet.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from rlgym_tools.rocket_league.replays.replay_frame import ReplayFrame

from src.patterns.base import PatternEvent

LOW_Z = 120.0                 # uu — "near ground" altitude for the dodge
LOOKBACK_FRAMES = 12          # ~0.4s before landing to find the dodge
SPEED_WINDOW = 6              # frames before/after to measure horizontal speed
MIN_GROUND_SPEED = 700.0      # uu/s — must end up actually moving along the ground
MIN_SPEED_GAIN = -50.0        # uu/s — allow a small loss; a real landing loses much more
COOLDOWN_FRAMES = 20


def _horizontal_speed(vel) -> float:
    return float(np.hypot(float(vel[0]), float(vel[1])))


def detect_wave_dashes(frames: Sequence[ReplayFrame]) -> Iterable[PatternEvent]:
    n = len(frames)
    if n < SPEED_WINDOW + 2:
        return

    prev_on_ground: dict[int, bool] = {}
    cooldown: dict[int, int] = {}

    for f_idx, frame in enumerate(frames):
        for agent_id, car in frame.state.cars.items():
            on_ground = bool(car.on_ground)
            was_on_ground = prev_on_ground.get(agent_id, on_ground)
            prev_on_ground[agent_id] = on_ground

            # Landing edge only
            if not (on_ground and not was_on_ground):
                continue
            if (f_idx - cooldown.get(agent_id, -COOLDOWN_FRAMES - 1)) <= COOLDOWN_FRAMES:
                continue

            # Find a low-altitude dodge in the lookback window
            dodge_frame = None
            lo = max(0, f_idx - LOOKBACK_FRAMES)
            for j in range(f_idx, lo - 1, -1):
                cj = frames[j].state.cars.get(agent_id)
                if cj is None:
                    continue
                cz = float(cj.physics.position[2])
                flipping = float(getattr(cj, "flip_time", 0.0)) > 0.0
                if flipping and cz < LOW_Z and not cj.on_ground:
                    dodge_frame = j
                    break
            if dodge_frame is None:
                continue

            # Horizontal speed gain across the dash
            pre_idx = max(0, dodge_frame - SPEED_WINDOW)
            post_idx = min(n - 1, f_idx + SPEED_WINDOW)
            pre_car = frames[pre_idx].state.cars.get(agent_id)
            post_car = frames[post_idx].state.cars.get(agent_id)
            if pre_car is None or post_car is None:
                continue
            pre_speed = _horizontal_speed(pre_car.physics.linear_velocity)
            post_speed = _horizontal_speed(post_car.physics.linear_velocity)
            gain = post_speed - pre_speed

            if post_speed < MIN_GROUND_SPEED or gain < MIN_SPEED_GAIN:
                continue

            cooldown[agent_id] = f_idx
            conf = float(np.clip(0.45 + gain / 2000.0, 0.3, 0.9))
            yield PatternEvent(
                pattern="wave_dash",
                agent_id=agent_id,
                frame_start=dodge_frame,
                frame_end=f_idx,
                confidence=round(conf, 3),
                metadata={
                    "horizontal_speed_gain_uu_s": round(gain, 1),
                    "post_speed_uu_s": round(post_speed, 1),
                    "dodge_to_land_frames": f_idx - dodge_frame,
                },
            )


class WaveDashDetector:
    name = "wave_dash"

    def detect(self, frames: Sequence[ReplayFrame]) -> Iterable[PatternEvent]:
        return detect_wave_dashes(frames)
