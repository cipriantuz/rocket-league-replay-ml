"""Aerial dribble detector — iterated heuristic (v2 after QA).

Definition (v2, tighter than v1):
    Single sustained carry by ONE car:
    - At least 3 ball touches by the same car
    - All touches happen in the air (car not on ground)
    - All touches happen with ball Z > 400 uu
    - Touches span at most 3s (90 frames @ 30Hz)
    - Ball does NOT touch the ground between first and last touch
    - Emit ONE event per sustained sequence (not overlapping events)

v1 emitted overlapping 2-touch events → 19 per match (way too many).
v2 tightens to ≥3 touches and merges overlapping sequences → expected ~3-6/meci.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from rlgym_tools.rocket_league.replays.replay_frame import ReplayFrame

from src.patterns.base import PatternEvent

WINDOW_FRAMES = 90        # 3.0s @ 30Hz
MIN_BALL_Z = 400.0
GROUND_TOUCH_Z = 120.0
MIN_TOUCHES = 3
SEQUENCE_GAP_FRAMES = 30  # if gap > 1s between consecutive touches, treat as new sequence


def detect_aerial_dribbles(frames: Sequence[ReplayFrame]) -> Iterable[PatternEvent]:
    if len(frames) < 5:
        return

    # Ground-bounce tracker
    ground_bounce_frames: set[int] = set()
    prev_touches: dict[int, int] = {}
    # In-progress sequences per agent: list of (frame_idx, ball_z) touches
    seqs: dict[int, list[tuple[int, float]]] = {}

    def _emit(agent_id: int, touches: list[tuple[int, float]]) -> PatternEvent | None:
        if len(touches) < MIN_TOUCHES:
            return None
        first, _ = touches[0]
        last, _ = touches[-1]
        # No ground bounce between first and last
        if any(first <= b <= last for b in ground_bounce_frames):
            return None
        if last - first > WINDOW_FRAMES:
            return None
        max_z = max(z for _, z in touches)
        conf = min(1.0, 0.6 + 0.1 * (len(touches) - MIN_TOUCHES))
        return PatternEvent(
            pattern="aerial_dribble",
            agent_id=agent_id,
            frame_start=first,
            frame_end=last,
            confidence=conf,
            metadata={"n_touches": len(touches), "max_ball_z": max_z},
        )

    for f_idx, frame in enumerate(frames):
        state = frame.state
        ball_z = float(state.ball.position[2])
        if ball_z < GROUND_TOUCH_Z:
            ground_bounce_frames.add(f_idx)

        for agent_id, car in state.cars.items():
            cur_touches = car.ball_touches
            prev = prev_touches.get(agent_id, cur_touches)
            prev_touches[agent_id] = cur_touches
            touched_this_frame = cur_touches > prev

            # Close ongoing sequence if last touch too old
            ongoing = seqs.get(agent_id)
            if ongoing and f_idx - ongoing[-1][0] > SEQUENCE_GAP_FRAMES:
                ev = _emit(agent_id, ongoing)
                if ev is not None:
                    yield ev
                seqs.pop(agent_id, None)
                ongoing = None

            if not touched_this_frame:
                continue
            if car.on_ground:
                # Touch on ground breaks any aerial sequence
                if ongoing:
                    ev = _emit(agent_id, ongoing)
                    if ev is not None:
                        yield ev
                    seqs.pop(agent_id, None)
                continue
            if ball_z < MIN_BALL_Z:
                if ongoing:
                    ev = _emit(agent_id, ongoing)
                    if ev is not None:
                        yield ev
                    seqs.pop(agent_id, None)
                continue

            # Append to (possibly new) sequence
            seqs.setdefault(agent_id, []).append((f_idx, ball_z))

    # Flush any remaining sequences at end of replay
    for agent_id, ongoing in seqs.items():
        ev = _emit(agent_id, ongoing)
        if ev is not None:
            yield ev


class AerialDribbleDetector:
    name = "aerial_dribble"

    def detect(self, frames: Sequence[ReplayFrame]) -> Iterable[PatternEvent]:
        return detect_aerial_dribbles(frames)
