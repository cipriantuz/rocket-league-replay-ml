"""Demolition detector — trivial, reliable.

Definition:
    A car is demolished when its ``demo_respawn_timer`` transitions from 0 to >0
    (the victim explodes and starts the respawn countdown). This signal is exact
    in the decoded replay state — unlike the geometric heuristics, no tuning is
    needed.

Attribution (who got the demo):
    The replay state gives us only the victim (the car whose timer fired). We
    attribute the demolition to the nearest OPPONENT car that is supersonic at
    the demo frame (supersonic contact is the only way to demolish in normal
    play). If no supersonic opponent is close, we fall back to the nearest
    opponent. The emitted ``agent_id`` is the attacker; the victim is recorded
    in metadata. ``attribution_confident`` flags whether a supersonic opponent
    was actually found.

Frames are 30 Hz native replay rate.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from rlgym_tools.rocket_league.replays.replay_frame import ReplayFrame

from src.patterns.base import PatternEvent

DEMO_TIMER_EPS = 0.05          # timer considered "active" above this (seconds)
ATTACKER_MAX_DIST = 350.0      # uu — supersonic contact happens at car-radius range


def _distance(a, b) -> float:
    return float(np.linalg.norm(a - b))


def detect_demolitions(frames: Sequence[ReplayFrame]) -> Iterable[PatternEvent]:
    if len(frames) < 2:
        return

    prev_timer: dict[int, float] = {}

    for f_idx, frame in enumerate(frames):
        state = frame.state

        for victim_id, victim in state.cars.items():
            timer = float(victim.demo_respawn_timer)
            prev = prev_timer.get(victim_id, timer)
            prev_timer[victim_id] = timer

            # Rising edge: 0 -> active = a fresh demolition this frame
            if not (prev <= DEMO_TIMER_EPS < timer):
                continue

            victim_team = victim.team_num
            victim_pos = victim.physics.position

            # Attribute to nearest supersonic opponent (fallback: nearest opponent)
            attacker_id = victim_id
            best_super = None       # (dist, agent_id) among supersonic opponents
            best_any = None         # (dist, agent_id) among all opponents
            for aid, car in state.cars.items():
                if aid == victim_id or car.team_num == victim_team:
                    continue
                d = _distance(car.physics.position, victim_pos)
                if best_any is None or d < best_any[0]:
                    best_any = (d, aid)
                if car.is_supersonic and (best_super is None or d < best_super[0]):
                    best_super = (d, aid)

            if best_any is None:
                # No opponent present to attribute the demo to (degenerate frame).
                # Skip rather than credit the victim with demolishing itself.
                continue

            confident = best_super is not None and best_super[0] <= ATTACKER_MAX_DIST
            if confident:
                attacker_id = best_super[1]
                dist = best_super[0]
            else:
                attacker_id = best_any[1]
                dist = best_any[0]

            yield PatternEvent(
                pattern="demolition",
                agent_id=attacker_id,
                frame_start=f_idx,
                frame_end=f_idx,
                confidence=0.95 if confident else 0.6,
                metadata={
                    "victim_agent_id": int(victim_id),
                    "attacker_distance_uu": round(dist, 1),
                    "attribution_confident": confident,
                },
            )


class DemolitionDetector:
    name = "demolition"

    def detect(self, frames: Sequence[ReplayFrame]) -> Iterable[PatternEvent]:
        return detect_demolitions(frames)
