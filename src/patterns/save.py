"""Save detector — a defender stops a shot heading into their own goal.

Definition:
    1. Threat: the ball is in a team's defensive third heading toward THEIR own
       goal at speed, roughly on target (would cross the goal mouth).
    2. Save: a car of the DEFENDING team touches the ball while the threat is
       live, and right after the touch the ball's goal-ward velocity reverses
       (it is pushed away from the goal) — and no goal is scored.

Goal geometry (RLGym/RocketSim): orange goal at +Y (defended by orange, team 1),
blue goal at -Y (defended by blue, team 0). A ball heading to +Y threatens orange
→ an orange touch that reverses it = orange save.

Moderately reliable: the ball-reversal-near-own-goal signal is clear. False
positives possible on clears that weren't really on-target; the on-target projection
+ incoming-speed threshold suppress most. Confidence encodes the incoming shot
speed faced.

Frames are 30 Hz native replay rate. Positions/velocities are RAW RocketSim units.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from rlgym_tools.rocket_league.replays.replay_frame import ReplayFrame

from src.patterns.base import PatternEvent

GOAL_Y = 5120.0
GOAL_HALF_WIDTH = 893.0
GOAL_HEIGHT = 642.0
ON_TARGET_X_MARGIN = 300.0
ON_TARGET_Z_MARGIN = 200.0

DANGER_Y = 3500.0          # ball this deep toward a goal = defensive third
SHOT_SPEED = 1200.0        # uu/s goal-ward = a real threat
THREAT_WINDOW = 20         # frames a threat stays live after last on-target sample
REVERSAL_WINDOW = 8        # frames after the touch to confirm the ball was pushed away
COOLDOWN_FRAMES = 20


def _on_target(ball_pos: np.ndarray, ball_vel: np.ndarray, goal_sign: float) -> bool:
    """Would the ball (constant velocity) cross goal at goal_sign*GOAL_Y inside the mouth?"""
    vy = float(ball_vel[1])
    if goal_sign * vy <= 1.0:
        return False
    goal_line = goal_sign * GOAL_Y
    t = (goal_line - float(ball_pos[1])) / vy
    if t <= 0:
        return False
    x_at = float(ball_pos[0]) + float(ball_vel[0]) * t
    z_at = float(ball_pos[2]) + float(ball_vel[2]) * t
    return (abs(x_at) < GOAL_HALF_WIDTH + ON_TARGET_X_MARGIN
            and 0.0 <= z_at < GOAL_HEIGHT + ON_TARGET_Z_MARGIN)


def detect_saves(frames: Sequence[ReplayFrame]) -> Iterable[PatternEvent]:
    n = len(frames)
    if n < 3:
        return

    # Per goal sign (+1 = orange goal/defended by orange team 1; -1 = blue goal/team 0)
    # last live threat: (frame, incoming_goalward_speed)
    threat: dict[float, tuple[int, float]] = {}
    prev_touches: dict[int, int] = {}
    cooldown: dict[int, int] = {}

    for f_idx, frame in enumerate(frames):
        state = frame.state
        ball_pos = state.ball.position
        ball_vel = state.ball.linear_velocity
        ball_y = float(ball_pos[1])
        vy = float(ball_vel[1])

        # Update threats for each goal
        for goal_sign in (1.0, -1.0):
            goalward = goal_sign * vy
            deep = goal_sign * ball_y > DANGER_Y
            if deep and goalward > SHOT_SPEED and _on_target(ball_pos, ball_vel, goal_sign):
                threat[goal_sign] = (f_idx, goalward)

        # Touches → potential saves
        for agent_id, car in state.cars.items():
            cur = car.ball_touches
            prev = prev_touches.get(agent_id, cur)
            prev_touches[agent_id] = cur
            if cur <= prev:
                continue
            if (f_idx - cooldown.get(agent_id, -COOLDOWN_FRAMES - 1)) <= COOLDOWN_FRAMES:
                continue

            # Which goal does this car defend? team 1 (orange) defends +Y, team 0 (blue) defends -Y
            defend_sign = 1.0 if car.team_num == 1 else -1.0
            t = threat.get(defend_sign)
            if t is None or (f_idx - t[0]) > THREAT_WINDOW:
                continue

            # Confirm reversal: ball goal-ward speed drops / reverses within window
            saved = False
            for j in range(f_idx, min(f_idx + REVERSAL_WINDOW + 1, n)):
                vj = defend_sign * float(frames[j].state.ball.linear_velocity[1])
                if vj < 0:           # ball now moving AWAY from the defended goal
                    saved = True
                    break
            if not saved:
                continue

            incoming = t[1]
            cooldown[agent_id] = f_idx
            threat.pop(defend_sign, None)
            conf = float(np.clip(0.55 + (incoming - SHOT_SPEED) / 6000.0, 0.4, 0.95))
            yield PatternEvent(
                pattern="save",
                agent_id=agent_id,
                frame_start=t[0],
                frame_end=min(f_idx + REVERSAL_WINDOW, n - 1),
                confidence=round(conf, 3),
                metadata={
                    "incoming_shot_speed_uu_s": round(incoming, 1),
                    "defended_goal": "orange" if defend_sign > 0 else "blue",
                    "shot_to_save_frames": f_idx - t[0],
                },
            )


class SaveDetector:
    name = "save"

    def detect(self, frames: Sequence[ReplayFrame]) -> Iterable[PatternEvent]:
        return detect_saves(frames)
