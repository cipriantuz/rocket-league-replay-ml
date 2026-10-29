"""Ceiling shot detector.

Definition:
    Car touches ceiling (car.physics.position[2] > 1900 + on_ground from wheels-with-contact)
    -> detaches with boost (car.boost_active_time > 0 within ~30 frames of detach)
    -> hits the ball mid-air (ball_z > 800 at ball touch, not on ground)

We don't have direct "on_ceiling" state, so use heuristic on Z + on_ground.
Ceiling Z ≈ 2044 uu. We treat car_z > 1900 + on_ground=True as ceiling contact.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from rlgym_tools.rocket_league.replays.replay_frame import ReplayFrame

from src.patterns.base import PatternEvent

CEILING_Z_THRESHOLD = 1900.0
DETACH_TO_HIT_MAX_FRAMES = 90    # 3s @ 30Hz
BALL_AIR_MIN_Z = 600.0


def detect_ceiling_shots(frames: Sequence[ReplayFrame]) -> Iterable[PatternEvent]:
    if len(frames) < 5:
        return

    # Per agent state
    on_ceiling: dict[int, int] = {}      # agent_id -> frame_idx where ceiling-contact ended (detach)
    prev_on_ceiling: dict[int, bool] = {}
    prev_touches: dict[int, int] = {}

    for f_idx, frame in enumerate(frames):
        state = frame.state
        ball_z = float(state.ball.position[2])

        for agent_id, car in state.cars.items():
            car_z = float(car.physics.position[2])
            currently_on_ceiling = car.on_ground and car_z > CEILING_Z_THRESHOLD
            was_on_ceiling = prev_on_ceiling.get(agent_id, False)
            prev_on_ceiling[agent_id] = currently_on_ceiling

            if was_on_ceiling and not currently_on_ceiling:
                on_ceiling[agent_id] = f_idx  # detach frame

            cur_touches = car.ball_touches
            prev = prev_touches.get(agent_id, cur_touches)
            prev_touches[agent_id] = cur_touches
            touched_this_frame = cur_touches > prev

            if not touched_this_frame:
                continue

            detach_frame = on_ceiling.get(agent_id)
            if detach_frame is None:
                continue
            if f_idx - detach_frame > DETACH_TO_HIT_MAX_FRAMES:
                del on_ceiling[agent_id]
                continue
            if car.on_ground or ball_z < BALL_AIR_MIN_Z:
                continue
            # Bonus: was the car boosting in window?
            boosted = car.boost_active_time > 0 or car.is_supersonic
            conf = 0.9 if boosted else 0.7
            yield PatternEvent(
                pattern="ceiling_shot",
                agent_id=agent_id,
                frame_start=detach_frame,
                frame_end=f_idx,
                confidence=conf,
                metadata={
                    "detach_to_hit_frames": f_idx - detach_frame,
                    "ball_z_at_hit": ball_z,
                    "boosted": boosted,
                },
            )
            del on_ceiling[agent_id]


class CeilingShotDetector:
    name = "ceiling_shot"

    def detect(self, frames: Sequence[ReplayFrame]) -> Iterable[PatternEvent]:
        return detect_ceiling_shots(frames)
