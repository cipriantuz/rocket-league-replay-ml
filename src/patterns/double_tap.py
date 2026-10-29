"""Double tap detector — iterated heuristic (v2 after QA).

v1: required explicit wall contact (|ball.Y| ≥ 5100). Only 2 events / 499 matches
    — wall y often doesn't reach exactly 5100 due to rocketsim interpolation and
    wall absorption physics.

v2 definition (relaxed):
    Touch #1 by car A in the air (ball_z > 600)
    -> within 90 frames (3s), ball Y reaches |Y| > 4500 (heading to backboard)
    -> ball Y reverses direction (peak Y reached, then decreasing |Y|)
    -> Touch #2 by SAME car A within 60 frames (2s) of the peak, ball in air (z > 300)
    -> no other car touches the ball between T1 and T2
"""

from __future__ import annotations

from typing import Iterable, Sequence

from rlgym_tools.rocket_league.replays.replay_frame import ReplayFrame

from src.patterns.base import PatternEvent

T1_BALL_Z_MIN = 600.0
T1_TO_PEAK_MAX_FRAMES = 90       # 3s @ 30Hz
PEAK_Y_THRESHOLD = 4500.0        # ball must approach back wall (≈5120 max)
PEAK_TO_T2_MAX_FRAMES = 60       # 2s @ 30Hz
T2_BALL_Z_MIN = 300.0


def detect_double_taps(frames: Sequence[ReplayFrame]) -> Iterable[PatternEvent]:
    if len(frames) < 10:
        return

    prev_touches: dict[int, int] = {}
    # Pending: agent_id -> {'t1_frame', 'peak_frame'(optional), 'peak_y'(optional)}
    pending: dict[int, dict] = {}
    # Track total ball touches across all agents to detect intervening hits
    total_touches_at_t1: dict[int, int] = {}

    for f_idx, frame in enumerate(frames):
        state = frame.state
        ball_y = float(state.ball.position[1])
        ball_z = float(state.ball.position[2])
        global_touches = sum(c.ball_touches for c in state.cars.values())

        # Track per-agent touch deltas
        agent_touched: dict[int, bool] = {}
        for aid, car in state.cars.items():
            cur = car.ball_touches
            prev = prev_touches.get(aid, cur)
            prev_touches[aid] = cur
            agent_touched[aid] = cur > prev

        for agent_id, car in state.cars.items():
            touched_this_frame = agent_touched[agent_id]

            # Expire stale entries
            if agent_id in pending:
                p = pending[agent_id]
                if "peak_frame" not in p:
                    # Phase 1: waiting for ball to reach backboard
                    if f_idx - p["t1_frame"] > T1_TO_PEAK_MAX_FRAMES:
                        del pending[agent_id]
                        total_touches_at_t1.pop(agent_id, None)
                        continue
                    # Check for intervening touch by ANY other agent
                    if global_touches > total_touches_at_t1.get(agent_id, global_touches) and not touched_this_frame:
                        # Someone else touched the ball before us — abort
                        del pending[agent_id]
                        total_touches_at_t1.pop(agent_id, None)
                        continue
                    # Detect peak: ball Y reaches threshold AND starts reversing
                    if abs(ball_y) >= PEAK_Y_THRESHOLD:
                        # Mark peak frame; we'll confirm reversal next iteration
                        if "peak_y" not in p:
                            p["peak_y"] = ball_y
                            p["peak_frame"] = f_idx
                        elif abs(ball_y) < abs(p["peak_y"]):
                            # Ball is now coming back inward; peak confirmed
                            pass
                        else:
                            # Still pushing outward; update peak
                            p["peak_y"] = ball_y
                            p["peak_frame"] = f_idx
                else:
                    # Phase 2: waiting for second tap
                    if f_idx - p["peak_frame"] > PEAK_TO_T2_MAX_FRAMES:
                        del pending[agent_id]
                        total_touches_at_t1.pop(agent_id, None)
                        continue
                    if touched_this_frame and not car.on_ground and ball_z >= T2_BALL_Z_MIN:
                        yield PatternEvent(
                            pattern="double_tap",
                            agent_id=agent_id,
                            frame_start=p["t1_frame"],
                            frame_end=f_idx,
                            confidence=0.85,
                            metadata={
                                "peak_y": p["peak_y"],
                                "peak_frame": p["peak_frame"],
                                "tap2_ball_z": ball_z,
                                "frames_t1_to_t2": f_idx - p["t1_frame"],
                            },
                        )
                        del pending[agent_id]
                        total_touches_at_t1.pop(agent_id, None)
                        continue

            # Try to open a new pending sequence on first tap (in air, high enough)
            if touched_this_frame and not car.on_ground and ball_z >= T1_BALL_Z_MIN:
                pending[agent_id] = {"t1_frame": f_idx}
                total_touches_at_t1[agent_id] = global_touches


class DoubleTapDetector:
    name = "double_tap"

    def detect(self, frames: Sequence[ReplayFrame]) -> Iterable[PatternEvent]:
        return detect_double_taps(frames)
