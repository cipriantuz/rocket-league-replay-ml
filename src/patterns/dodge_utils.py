"""Shared dodge-activation primitive for movement-tech detectors.

A *dodge* (flip) is the directional second jump. In the reconstructed replay
state it is marked by ``flip_time`` becoming positive: rlgym-tools sets
``flip_time`` to ``1/tps`` on the dodge tick and counts it up while the flip
torque is active, then resets it to 0 on landing. A plain *double jump* leaves
``flip_time == 0`` (it sets ``has_double_jumped`` instead), so the
``flip_time`` 0 -> >0 transition uniquely marks a real dodge and excludes
double jumps.

Both the wave-dash and the fast-double-flip detectors consume the SAME
activation list produced here, which is what makes them mutually exclusive by
construction (a wave dash is exactly one activation near a landing; a fast
double flip is two activations within a short window).

Frames are native 30 Hz. Positions/velocities are RAW RocketSim units (uu).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from rlgym_tools.rocket_league.replays.replay_frame import ReplayFrame


def dodge_activations(frames: Sequence[ReplayFrame]) -> Dict[int, List[int]]:
    """Return, per agent id, the sorted list of frame indices where a dodge
    is activated while the car is airborne.

    A dodge activation is the frame where ``flip_time`` rises from 0 to a
    positive value with the car not on the ground.
    """
    acts: Dict[int, List[int]] = {}
    prev_flip: Dict[int, float] = {}

    for f_idx, frame in enumerate(frames):
        for agent_id, car in frame.state.cars.items():
            ft = float(getattr(car, "flip_time", 0.0) or 0.0)
            pf = prev_flip.get(agent_id, 0.0)
            prev_flip[agent_id] = ft
            if ft > 0.0 and pf <= 0.0 and not bool(car.on_ground):
                acts.setdefault(agent_id, []).append(f_idx)
    return acts
