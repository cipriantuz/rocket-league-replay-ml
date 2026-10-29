"""Musty flick detector — heuristic (imperfect, needs visual tuning).

A musty flick: the ball sits on the car's hood/roof (a carry), then the player
backflips, whipping the ball forward and steeply upward. Observable signature on
decoded replay state:

    1. Carry: the ball is directly ABOVE the car and horizontally close
       (``ball_z > car_z`` and horizontal distance small) for a sustained run.
    2. Flick: the car performs a backflip — a spike in angular velocity — and
       touches the ball.
    3. Launch: immediately after, the ball gains a large UPWARD velocity
       (``ball_linvel_z`` jumps) on top of forward motion.

HONEST LIMITATION: heuristic over reconstructed physics. The backflip is detected
from world-frame angular-velocity magnitude (we don't reconstruct the car-local
pitch axis), so hard turns/other flips can trigger it; the carry + upward-launch
requirement suppresses most. Confidence encodes the upward launch speed. Expect
both false positives and misses — flagged unreliable, like flip_reset.

Frames are 30 Hz native replay rate. Positions/velocities are RAW RocketSim units.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from rlgym_tools.rocket_league.replays.replay_frame import ReplayFrame

from src.patterns.base import PatternEvent

CARRY_HORIZ_DIST = 170.0      # uu — ball horizontally over the hood
CARRY_Z_MARGIN = 40.0         # uu — ball center above car center by at least this
MIN_CARRY_FRAMES = 8          # ~0.27s of carry before the flick
FLICK_ANGVEL = 3.0            # rad/s — backflip angular-velocity magnitude
LAUNCH_Z_VEL = 600.0          # uu/s — upward ball speed gain marking the flick
POST_WINDOW = 8               # frames after touch to measure the launch
COOLDOWN_FRAMES = 30


def _horiz_dist(a, b) -> float:
    return float(np.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def detect_musty_flicks(frames: Sequence[ReplayFrame]) -> Iterable[PatternEvent]:
    n = len(frames)
    if n < MIN_CARRY_FRAMES + POST_WINDOW:
        return

    carry_len: dict[int, int] = {}        # consecutive carry frames per agent
    carry_start: dict[int, int] = {}
    prev_touches: dict[int, int] = {}
    cooldown: dict[int, int] = {}

    for f_idx, frame in enumerate(frames):
        state = frame.state
        ball_pos = state.ball.position
        ball_z = float(ball_pos[2])

        for agent_id, car in state.cars.items():
            car_pos = car.physics.position
            car_z = float(car_pos[2])

            # --- maintain carry state ---
            on_hood = (
                _horiz_dist(car_pos, ball_pos) < CARRY_HORIZ_DIST
                and ball_z > car_z + CARRY_Z_MARGIN
            )
            if on_hood:
                if carry_len.get(agent_id, 0) == 0:
                    carry_start[agent_id] = f_idx
                carry_len[agent_id] = carry_len.get(agent_id, 0) + 1
            else:
                # decay (not hard reset) so brief contact gaps don't break the carry;
                # tolerated gap ≈ half the carry accumulated so far
                carry_len[agent_id] = max(0, carry_len.get(agent_id, 0) - 2)
                if carry_len[agent_id] == 0:
                    carry_start.pop(agent_id, None)

            # --- touch detection ---
            cur = car.ball_touches
            prev = prev_touches.get(agent_id, cur)
            prev_touches[agent_id] = cur
            touched = cur > prev
            if not touched:
                continue
            if carry_len.get(agent_id, 0) < MIN_CARRY_FRAMES:
                continue
            if (f_idx - cooldown.get(agent_id, -COOLDOWN_FRAMES - 1)) <= COOLDOWN_FRAMES:
                continue

            # backflip: angular-velocity magnitude spike at/near the touch
            angvel_mag = float(np.linalg.norm(car.physics.angular_velocity))
            if angvel_mag < FLICK_ANGVEL:
                continue

            # upward launch in the window after the touch — baseline is the ball's
            # vertical velocity JUST BEFORE the touch, so `launch` is a true gain
            # (measuring at the touch frame makes launch>=0 trivially and fires on
            # balls that were already rising).
            pre_vz = float(frames[max(0, f_idx - 1)].state.ball.linear_velocity[2])
            post_idx = min(n - 1, f_idx + POST_WINDOW)
            peak_vz = max(
                float(frames[j].state.ball.linear_velocity[2])
                for j in range(f_idx, post_idx + 1)
            )
            launch = peak_vz - pre_vz
            if launch < LAUNCH_Z_VEL:
                continue

            cooldown[agent_id] = f_idx
            start = carry_start.get(agent_id, f_idx)
            conf = float(np.clip(0.4 + launch / 4000.0, 0.3, 0.85))
            yield PatternEvent(
                pattern="musty_flick",
                agent_id=agent_id,
                frame_start=start,
                frame_end=post_idx,
                confidence=round(conf, 3),
                metadata={
                    "carry_frames": carry_len.get(agent_id, 0),
                    "angvel_mag_rad_s": round(angvel_mag, 2),
                    "upward_launch_uu_s": round(launch, 1),
                },
            )
            carry_len[agent_id] = 0
            carry_start.pop(agent_id, None)


class MustyFlickDetector:
    name = "musty_flick"

    def detect(self, frames: Sequence[ReplayFrame]) -> Iterable[PatternEvent]:
        return detect_musty_flicks(frames)
