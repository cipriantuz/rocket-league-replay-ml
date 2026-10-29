"""Simple shot detector — a directed strike toward the opponent goal.

Definition:
    A car touches the ball and the ball leaves the contact with a high speed
    directed toward the OPPONENT goal. We measure the peak goal-ward ball speed
    in a short window after the touch (the contact frame's velocity can lag one
    step in the reconstruction). A geometric on-target test (project the ball's
    post-touch trajectory onto the goal line) raises confidence but is not
    required — many real shots are saved/deflected before crossing.

Goal geometry (RLGym/RocketSim):
    Orange goal at +Y (y = +5120), blue goal at -Y. Blue team (team_num 0)
    attacks +Y; orange (team_num 1) attacks -Y. Goal mouth: |x| < 893,
    z < 642 (crossbar). We use a small margin on the on-target test.

This is intentionally permissive (a "shot attempt" detector) — it is NOT a
goal detector. Honest caveat: aerial passes that happen to go goal-ward will
register; the on-target flag and goal-ward speed in metadata let downstream
filtering tighten it.

Frames are 30 Hz native replay rate.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from rlgym_tools.rocket_league.replays.replay_frame import ReplayFrame

from src.patterns.base import PatternEvent

GOAL_Y = 5120.0
GOAL_HALF_WIDTH = 893.0
GOAL_HEIGHT = 642.0
ON_TARGET_X_MARGIN = 250.0     # allow slight off-post trajectories
ON_TARGET_Z_MARGIN = 150.0

SHOT_SPEED = 1600.0            # uu/s of goal-ward ball speed to count as a shot
POST_TOUCH_WINDOW = 5         # frames to look for peak post-touch speed (~0.17s)
COOLDOWN_FRAMES = 15          # avoid double-emitting the same strike (~0.5s)


def _on_target(ball_pos: np.ndarray, ball_vel: np.ndarray, attack_dir: float) -> bool:
    """Project ball trajectory (constant velocity) to the goal line; is it inside?"""
    vy = float(ball_vel[1])
    # Must be travelling toward the goal line at a meaningful rate
    if attack_dir * vy <= 1.0:
        return False
    goal_line = attack_dir * GOAL_Y
    t = (goal_line - float(ball_pos[1])) / vy
    if t <= 0:
        return False
    x_at = float(ball_pos[0]) + float(ball_vel[0]) * t
    z_at = float(ball_pos[2]) + float(ball_vel[2]) * t
    return (abs(x_at) < GOAL_HALF_WIDTH + ON_TARGET_X_MARGIN
            and 0.0 <= z_at < GOAL_HEIGHT + ON_TARGET_Z_MARGIN)


def detect_simple_shots(frames: Sequence[ReplayFrame]) -> Iterable[PatternEvent]:
    if len(frames) < 3:
        return

    prev_touches: dict[int, int] = {}
    cooldown: dict[int, int] = {}

    n = len(frames)
    for f_idx, frame in enumerate(frames):
        state = frame.state

        for agent_id, car in state.cars.items():
            cur = car.ball_touches
            prev = prev_touches.get(agent_id, cur)
            prev_touches[agent_id] = cur
            if cur <= prev:
                continue
            if (f_idx - cooldown.get(agent_id, -COOLDOWN_FRAMES - 1)) <= COOLDOWN_FRAMES:
                continue

            # Blue (team 0) attacks +Y, orange (team 1) attacks -Y
            attack_dir = 1.0 if car.team_num == 0 else -1.0

            # Peak goal-ward speed in the short window after contact
            best_speed = -1.0
            best_pos = None
            best_vel = None
            for j in range(f_idx, min(f_idx + POST_TOUCH_WINDOW + 1, n)):
                bvel = frames[j].state.ball.linear_velocity
                goalward = attack_dir * float(bvel[1])
                if goalward > best_speed:
                    best_speed = goalward
                    best_pos = frames[j].state.ball.position
                    best_vel = bvel

            if best_speed < SHOT_SPEED or best_pos is None:
                continue

            on_target = _on_target(best_pos, best_vel, attack_dir)
            cooldown[agent_id] = f_idx
            conf = 0.6 + (0.25 if on_target else 0.0) + min(0.15, (best_speed - SHOT_SPEED) / 8000.0)
            yield PatternEvent(
                pattern="simple_shot",
                agent_id=agent_id,
                frame_start=f_idx,
                frame_end=min(f_idx + POST_TOUCH_WINDOW, n - 1),
                confidence=round(min(1.0, conf), 3),
                metadata={
                    "goalward_speed_uu_s": round(best_speed, 1),
                    "on_target": on_target,
                    "ball_z_at_touch": round(float(state.ball.position[2]), 1),
                },
            )


class SimpleShotDetector:
    name = "simple_shot"

    def detect(self, frames: Sequence[ReplayFrame]) -> Iterable[PatternEvent]:
        return detect_simple_shots(frames)
