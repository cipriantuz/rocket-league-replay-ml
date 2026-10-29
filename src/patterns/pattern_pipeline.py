"""Orchestrator: runs all rule-based detectors on a ReplayFrame sequence.

Output: pandas DataFrame with one row per detected PatternEvent,
plus per-frame multi-label binary matrix for training (label_frame_matrix).

When `analyzer_hits` is provided (from parsed.analyzer['hits']), heuristic
detector events get their confidence boosted if a corresponding rocket league
analyzer hit lands within a temporal window of frame_start. This combines the
strength of geometric heuristics with the ground-truth hit detection that the
replay parser already provides.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from rlgym_tools.rocket_league.replays.replay_frame import ReplayFrame

from src.patterns.aerial_dribble import AerialDribbleDetector
from src.patterns.base import PatternEvent
from src.patterns.ceiling_shot import CeilingShotDetector
from src.patterns.demolition import DemolitionDetector
from src.patterns.double_tap import DoubleTapDetector
from src.patterns.fast_double_flip import FastDoubleFlipDetector
from src.patterns.flip_reset import FlipResetDetector
from src.patterns.musty_flick import MustyFlickDetector
from src.patterns.save import SaveDetector
from src.patterns.simple_shot import SimpleShotDetector
from src.patterns.wave_dash import WaveDashDetector

ALL_DETECTORS = [
    FlipResetDetector(),
    DoubleTapDetector(),
    AerialDribbleDetector(),
    CeilingShotDetector(),
    DemolitionDetector(),
    SimpleShotDetector(),
    WaveDashDetector(),
    FastDoubleFlipDetector(),
    MustyFlickDetector(),
    SaveDetector(),
]
PATTERN_NAMES = [d.name for d in ALL_DETECTORS]


def build_hit_frame_map(parsed_game_df, frames: Sequence[ReplayFrame]) -> np.ndarray:
    """Build mapping replay_native_frame -> rlgym_frame_idx via time alignment.

    Uses `seconds_remaining` from parsed.game_df vs scoreboard.game_timer_seconds
    in rlgym frames. Returns -1 for unmapped replay frames (pre-kickoff/replay/etc).
    """
    if parsed_game_df is None or len(frames) == 0:
        return np.array([], dtype=np.int32)
    rlgym_secs = np.array(
        [f.scoreboard.game_timer_seconds for f in frames], dtype=np.float32
    )
    # parsed.game_df['seconds_remaining'] is per native frame
    if "seconds_remaining" in parsed_game_df.columns:
        replay_secs = parsed_game_df["seconds_remaining"].to_numpy(dtype=np.float32)
    elif "replicated_game_state_time_remaining" in parsed_game_df.columns:
        replay_secs = parsed_game_df["replicated_game_state_time_remaining"].to_numpy(dtype=np.float32)
    else:
        return np.full(len(parsed_game_df), -1, dtype=np.int32)
    # For each native frame, find nearest rlgym frame by seconds_remaining
    out = np.full(len(replay_secs), -1, dtype=np.int32)
    valid = ~np.isnan(replay_secs)
    if not valid.any():
        return out
    # Use np.searchsorted (rlgym_secs is monotonically decreasing)
    rl_sorted = -rlgym_secs  # invert to ascending
    for i in np.where(valid)[0]:
        target = -replay_secs[i]
        idx = int(np.searchsorted(rl_sorted, target))
        if idx >= len(rl_sorted):
            idx = len(rl_sorted) - 1
        if idx > 0 and abs(rl_sorted[idx - 1] - target) < abs(rl_sorted[idx] - target):
            idx = idx - 1
        # Tolerance: skip if temporally too far (>1s) — replay was paused/celebrating
        if abs(rl_sorted[idx] - target) < 1.0:
            out[i] = idx
    return out


def augment_events_with_hits(
    events: list[PatternEvent],
    analyzer_hits: list[dict] | None,
    hit_to_rlgym: np.ndarray | None,
    proximity_frames: int = 6,
) -> list[PatternEvent]:
    """Boost confidence of events validated by analyzer.hits within proximity_frames."""
    if not analyzer_hits or hit_to_rlgym is None or len(hit_to_rlgym) == 0:
        return events
    # Build per-agent list of rlgym hit frames
    by_agent: dict[int, list[int]] = {}
    for h in analyzer_hits:
        fn_native = h["frame_number"]
        if fn_native >= len(hit_to_rlgym):
            continue
        rl_idx = int(hit_to_rlgym[fn_native])
        if rl_idx < 0:
            continue
        try:
            aid = int(h["player_unique_id"])
        except (TypeError, ValueError):
            continue
        by_agent.setdefault(aid, []).append(rl_idx)
    for k in by_agent:
        by_agent[k].sort()
    augmented = []
    for ev in events:
        agent_hits = by_agent.get(int(ev.agent_id), [])
        # Check if any hit falls within [frame_start - prox, frame_end + prox]
        lo, hi = ev.frame_start - proximity_frames, ev.frame_end + proximity_frames
        validated = any(lo <= h <= hi for h in agent_hits)
        new_meta = dict(ev.metadata)
        new_meta["validated_by_analyzer_hit"] = validated
        new_conf = min(1.0, ev.confidence + 0.1) if validated else max(0.0, ev.confidence - 0.15)
        augmented.append(PatternEvent(
            pattern=ev.pattern,
            agent_id=ev.agent_id,
            frame_start=ev.frame_start,
            frame_end=ev.frame_end,
            confidence=new_conf,
            metadata=new_meta,
        ))
    return augmented


def hit_summary_stats(analyzer_hits: list[dict] | None) -> dict:
    """Aggregate descriptive stats from analyzer.hits for the report."""
    if not analyzer_hits:
        return {"n_hits": 0}
    distances = [h["player_distance"] for h in analyzer_hits if "player_distance" in h]
    speed_inc = sum(1 for h in analyzer_hits if h.get("_debug_info", {}).get("speed_increased"))
    return {
        "n_hits": len(analyzer_hits),
        "mean_distance_uu": float(np.mean(distances)) if distances else 0.0,
        "median_distance_uu": float(np.median(distances)) if distances else 0.0,
        "frac_speed_increased": speed_inc / len(analyzer_hits),
    }


def detect_all_patterns(frames: Sequence[ReplayFrame]) -> list[PatternEvent]:
    """Run every detector and collect events into a single list."""
    events: list[PatternEvent] = []
    for det in ALL_DETECTORS:
        events.extend(det.detect(frames))
    events.sort(key=lambda e: (e.frame_start, e.pattern))
    return events


def events_to_dataframe(events: list[PatternEvent]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(
            columns=["pattern", "agent_id", "frame_start", "frame_end",
                     "confidence", "duration_frames", "metadata"]
        )
    return pd.DataFrame(
        [
            {
                "pattern": e.pattern,
                "agent_id": int(e.agent_id),
                "frame_start": e.frame_start,
                "frame_end": e.frame_end,
                "confidence": e.confidence,
                "duration_frames": e.duration_frames,
                "metadata": e.metadata,
            }
            for e in events
        ]
    )


def events_to_frame_label_matrix(
    events: list[PatternEvent], n_frames: int, patterns: list[str] = PATTERN_NAMES
) -> np.ndarray:
    """Return shape (n_frames, n_patterns) binary matrix: 1 if pattern active at frame f.

    Used to derive per-window multi-label targets for LSTM/Transformer training.
    """
    pattern_to_col = {p: i for i, p in enumerate(patterns)}
    mat = np.zeros((n_frames, len(patterns)), dtype=np.uint8)
    for e in events:
        col = pattern_to_col.get(e.pattern)
        if col is None:
            continue
        mat[e.frame_start : e.frame_end + 1, col] = 1
    return mat
