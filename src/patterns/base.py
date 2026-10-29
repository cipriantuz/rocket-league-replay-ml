"""Shared types for mechanical pattern detectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from rlgym_tools.rocket_league.replays.replay_frame import ReplayFrame


@dataclass(frozen=True)
class PatternEvent:
    """One detected occurrence of a mechanic in a replay."""

    pattern: str        # "flip_reset" | "double_tap" | "aerial_dribble" | "ceiling_shot"
    agent_id: int       # AgentID of the car performing the mechanic
    frame_start: int    # inclusive
    frame_end: int      # inclusive
    confidence: float   # heuristic score in [0, 1] — used for QA filtering
    metadata: dict      # detector-specific extras (e.g. wall_y for double tap)

    @property
    def duration_frames(self) -> int:
        return self.frame_end - self.frame_start + 1


class PatternDetector(Protocol):
    """Each detector receives the full replay frame sequence and emits events.

    Frames are 30 Hz (replay native); detectors must convert thresholds
    explicitly if comparing to 15 Hz training observations.
    """

    name: str

    def detect(self, frames: Sequence[ReplayFrame]) -> Iterable[PatternEvent]:
        ...
