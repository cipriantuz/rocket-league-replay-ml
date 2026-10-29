"""Flip reset detector — dodge-regrant definition (v4, validated visually).

Rule (the precise mechanism): a flip reset is when the player
is AIRBORNE, the car's XYZ is CLOSE to the ball's XYZ (contact), and AT that
contact the player RECEIVES the dodge back. The only way to regain the dodge
while airborne is a flip reset (wheels touch the ball), so the observable signal
is `has_flip` (dodge availability) transitioning False -> True WHILE NOT on_ground,
with the car near the ball at that moment.

VALIDATED LIMITATION (visual ground truth, checked in-game): even this
physically-correct rule produces FALSE POSITIVES on replay data — candidates that
look like resets are actually redirects (car rises into the ball, ~4 wheels near
it, but no real dodge-reset). Root cause: the replay->RocketSim reconstruction does
NOT faithfully capture wheel-ball contact (`wheels_with_contact` is world-geometry
only — 0 ball contacts) nor align the dodge-regrant with the contact frame. We tried
4 approaches (orientation 46%; wheels_with_contact=dead; dodge-regrant frame-aligned=0;
+timing window=12/60 all false). CONCLUSION: flip_reset is NOT reliably detectable
from decoded replay data — it requires live-sim `RocketSim CarState.HasFlipReset()`
oracle access. The other 3 detectors (double_tap, aerial_dribble, ceiling_shot)
remain reliable. This detector is kept as the principled rule but flagged unreliable.

Frames are 30 Hz native replay rate.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from rlgym_tools.rocket_league.replays.replay_frame import ReplayFrame

from src.patterns.base import PatternEvent

BALL_PROXIMITY = 200.0    # uu — car XYZ within this of ball XYZ = contact (car~120 + ball~93 radii)
MIN_AIR_TIME = 0.3        # seconds airborne — genuinely in the air (not a ground jump artifact)


def _car_ball_distance(car, ball_pos: np.ndarray) -> float:
    return float(np.linalg.norm(car.physics.position - ball_pos))


def detect_flip_resets(frames: Sequence[ReplayFrame]) -> Iterable[PatternEvent]:
    """Yield one PatternEvent per flip reset: dodge regained (has_flip False->True)
    while airborne and near the ball."""
    if len(frames) < 2:
        return

    CONTACT_WINDOW = 8  # frames (~0.27s) — bridge RocketSim's reconstruction lag between
    #                     the airborne ball contact and the dodge being re-granted.

    prev_has_flip: dict[int, bool] = {}
    last_air_contact: dict[int, tuple[int, float]] = {}  # agent -> (frame, dist) of recent airborne near-ball contact
    cooldown: dict[int, int] = {}

    for f_idx, frame in enumerate(frames):
        state = frame.state
        ball_pos = state.ball.position

        for agent_id, car in state.cars.items():
            has_flip = bool(getattr(car, "has_flip", False))
            prev = prev_has_flip.get(agent_id, has_flip)
            prev_has_flip[agent_id] = has_flip

            airborne = not car.on_ground
            dist = _car_ball_distance(car, ball_pos)
            air_ok = car.air_time_since_jump > MIN_AIR_TIME

            # Record airborne near-ball contact moments (the reset happens at/around these).
            if airborne and dist < BALL_PROXIMITY and air_ok:
                last_air_contact[agent_id] = (f_idx, dist)

            # Reset = dodge regained (False->True) while airborne, within a short window AFTER
            # an airborne near-ball contact (handles reconstruction timing lag).
            dodge_regained_airborne = has_flip and not prev and airborne
            recently_emitted = (f_idx - cooldown.get(agent_id, -100)) < 15  # ~0.5s

            if dodge_regained_airborne and not recently_emitted:
                contact = last_air_contact.get(agent_id)
                if contact is not None and (f_idx - contact[0]) <= CONTACT_WINDOW:
                    cooldown[agent_id] = f_idx
                    cdist = contact[1]
                    conf = min(1.0, 0.6 + 0.4 * max(0.0, 1.0 - cdist / BALL_PROXIMITY))
                    yield PatternEvent(
                        pattern="flip_reset",
                        agent_id=agent_id,
                        frame_start=contact[0],
                        frame_end=f_idx,
                        confidence=round(conf, 3),
                        metadata={
                            "contact_ball_distance_uu": round(cdist, 1),
                            "regrant_lag_frames": f_idx - contact[0],
                            "air_time_s": round(float(car.air_time_since_jump), 2),
                            "car_z": round(float(car.physics.position[2]), 1),
                        },
                    )


class FlipResetDetector:
    name = "flip_reset"

    def detect(self, frames: Sequence[ReplayFrame]) -> Iterable[PatternEvent]:
        return detect_flip_resets(frames)
