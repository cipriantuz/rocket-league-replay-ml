"""Windowing and weak labelling for the play-scheme classifier.

Produces (T, 133) windows plus per-window summary statistics:
  - Positives: rule-based detector events from events.parquet (ceiling_shot,
    double_tap, aerial_dribble), with the window centred on the event
  - NONE: windows from regions with no event within +-EVENT_BUFFER frames

Positives and NONE windows have the same length; the 133-dim frame vector comes from
frames_df_to_universal (src/obs/universal_frames.py).

Filters:
  - flip_reset is excluded (its detector was too noisy in manual validation)
  - an event is kept only if the replay's own touch analyzer confirms a car-ball
    contact (validated_by_analyzer_hit) AND confidence >= a per-class threshold
  - optional game-mode filter (1v1 / 2v2 / 3v3 from the number of car_* prefixes)

Public API:
    extract_positive_windows(processed_root, ...)  -> list[WindowRecord]
    extract_none_windows(processed_root, ...)       -> list[WindowRecord]
    compute_summary_stats(tensor_T133)              -> np.ndarray (shape: 133*6 = 798)
    SUMMARY_STAT_NAMES                              -> list[str] (798 names)
"""

from __future__ import annotations

import ast
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.obs.universal_frames import frames_df_to_universal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Defaults; override with env vars or function arguments
# (e.g. RL_SCHEME_WINDOW_LEN=90 RL_SCHEME_STRIDE=30 for 3 s windows at 30 Hz)
WINDOW_LENGTH: int = int(os.environ.get("RL_SCHEME_WINDOW_LEN", "45"))   # 3.0s @ 15 Hz
WINDOW_STRIDE: int = int(os.environ.get("RL_SCHEME_STRIDE", "15"))       # 1.0s @ 15 Hz
EVENT_BUFFER: int = 30       # frames excluded around any event for the NONE class

# Classes used by the classifier (flip_reset excluded)
TARGET_CLASSES: list[str] = ["ceiling_shot", "double_tap", "aerial_dribble"]

# Minimum detector confidence per class (set empirically)
CONFIDENCE_THRESHOLD: dict[str, float] = {
    "ceiling_shot":    0.80,
    "double_tap":      0.85,
    "aerial_dribble":  0.70,
}

# Summary statistics per feature (mean, std, min, max, p25, p75)
N_STATS: int = 6
STAT_SUFFIXES: list[str] = ["_mean", "_std", "_min", "_max", "_p25", "_p75"]

FEATURE_DIM: int = 133


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

@dataclass
class WindowRecord:
    """One classification window: (T, 133) tensor plus metadata."""
    replay_id: str
    frame_start: int           # index in frames.parquet (inclusive)
    frame_end: int             # index in frames.parquet (exclusive)
    label: str                 # "ceiling_shot" / "double_tap" / "aerial_dribble" / "NONE"
    confidence: float          # 1.0 for NONE
    tensor: np.ndarray         # shape (WINDOW_LENGTH, 133), float32
    summary_stats: np.ndarray  # shape (133 * N_STATS,), float32
    # optional metadata for export / diagnostics
    car_prefix: Optional[str] = None
    event_frame_start: Optional[int] = None
    event_frame_end: Optional[int] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_metadata(raw) -> dict:
    """Parse the metadata column into a dict.

    It is stored as a Python dict repr (single quotes), not JSON, so
    ast.literal_eval comes first, then json.loads, then an empty dict.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return ast.literal_eval(raw)
        except Exception:
            pass
        try:
            return json.loads(raw)
        except Exception:
            pass
    return {}


def _detect_game_mode(frames_df: pd.DataFrame) -> str:
    """Return '1v1', '2v2' or '3v3' from the car_* column prefixes."""
    prefixes = set()
    import re
    pat = re.compile(r"^(car_(?:blue|orange)\d+)_pos_x$")
    for col in frames_df.columns:
        m = pat.match(col)
        if m:
            prefixes.add(m.group(1))
    n = len(prefixes)
    if n <= 2:
        return "1v1"
    elif n <= 4:
        return "2v2"
    else:
        return "3v3"


def compute_summary_stats(tensor: np.ndarray) -> np.ndarray:
    """Summary statistics (mean/std/min/max/p25/p75) per feature.

    Args:
        tensor: shape (T, 133), float32

    Returns:
        shape (133 * 6,) = (798,), float32
        Order: [feat0_mean, feat0_std, feat0_min, feat0_max, feat0_p25, feat0_p75,
                 feat1_mean, ..., feat132_p75]
    """
    assert tensor.ndim == 2 and tensor.shape[1] == FEATURE_DIM, (
        f"Expected (T, {FEATURE_DIM}), got {tensor.shape}"
    )
    mean = tensor.mean(axis=0)                          # (133,)
    std = tensor.std(axis=0)                            # (133,)
    mn = tensor.min(axis=0)                             # (133,)
    mx = tensor.max(axis=0)                             # (133,)
    p25 = np.percentile(tensor, 25, axis=0)             # (133,)
    p75 = np.percentile(tensor, 75, axis=0)             # (133,)

    # interleaved: 6 stats per feature
    out = np.stack([mean, std, mn, mx, p25, p75], axis=1)  # (133, 6)
    return out.flatten().astype(np.float32)                 # (798,)


def _summary_stat_names_lazy() -> list[str]:
    """The 798 names of the summary-statistics vector."""
    from src.obs.universal_frames import universal_feature_names
    feat_names = universal_feature_names()
    names: list[str] = []
    for fn in feat_names:
        for suf in STAT_SUFFIXES:
            names.append(f"{fn}{suf}")
    return names


def summary_stat_names() -> list[str]:
    """The 798 names of the summary-statistics vector."""
    return _summary_stat_names_lazy()


# ---------------------------------------------------------------------------
# Positives from events.parquet
# ---------------------------------------------------------------------------

def _extract_positive_from_replay(
    frames_df: pd.DataFrame,
    events_df: pd.DataFrame,
    replay_id: str,
    game_mode_filter: Optional[list[str]] = None,
    window_len: int = WINDOW_LENGTH,
) -> list[WindowRecord]:
    """Positive windows from one replay."""
    n_frames = len(frames_df)
    mode = _detect_game_mode(frames_df)
    if game_mode_filter and mode not in game_mode_filter:
        return []

    # 133-dim universal tensor for the WHOLE replay, computed once
    universal = frames_df_to_universal(frames_df)  # (n_frames, 133)

    records: list[WindowRecord] = []

    for _, ev in events_df.iterrows():
        pattern = str(ev["pattern"])
        if pattern not in TARGET_CLASSES:
            continue

        # validated_by_analyzer_hit from the metadata dict
        meta = _parse_metadata(ev.get("metadata"))
        validated = bool(meta.get("validated_by_analyzer_hit", False))
        if not validated:
            continue

        confidence = float(ev.get("confidence", 0.0))
        thresh = CONFIDENCE_THRESHOLD.get(pattern, 0.70)
        if confidence < thresh:
            continue

        ev_start = int(ev["frame_start"])
        ev_end = int(ev["frame_end"])

        # centre the window on the middle of the event
        center = (ev_start + ev_end) // 2
        win_start = center - window_len // 2
        win_end = win_start + window_len

        # clamp to the replay bounds
        if win_start < 0:
            win_start = 0
            win_end = window_len
        if win_end > n_frames:
            win_end = n_frames
            win_start = win_end - window_len

        if win_start < 0 or win_end > n_frames or (win_end - win_start) < window_len:
            continue  # replay too short

        tensor = universal[win_start:win_end].copy()
        if tensor.shape[0] != window_len:
            continue

        stats = compute_summary_stats(tensor)
        records.append(WindowRecord(
            replay_id=replay_id,
            frame_start=win_start,
            frame_end=win_end,
            label=pattern,
            confidence=confidence,
            tensor=tensor,
            summary_stats=stats,
            event_frame_start=ev_start,
            event_frame_end=ev_end,
        ))

    return records


def extract_positive_windows(
    processed_root: str | Path = "data/processed",
    max_replays: Optional[int] = None,
    game_mode_filter: Optional[list[str]] = None,
    verbose: bool = True,
    window_len: int = WINDOW_LENGTH,
) -> list[WindowRecord]:
    """Positive windows from all processed replays.

    Keeps events with:
      - pattern in TARGET_CLASSES (ceiling_shot, double_tap, aerial_dribble)
      - validated_by_analyzer_hit == True (metadata dict)
      - confidence >= CONFIDENCE_THRESHOLD[pattern]

    Args:
        processed_root: root with one replay_id/ sub-directory per replay
        max_replays: limit for debugging / smoke tests
        game_mode_filter: ['1v1'] / ['2v2'] / ['3v3'] / None (all)
        verbose: print progress
        window_len: window length in frames (45 @ 15 Hz, 90 @ 30 Hz)

    Returns:
        WindowRecords with tensor (window_len, 133) and summary_stats (798,)
    """
    processed_root = Path(processed_root)
    all_records: list[WindowRecord] = []

    dirs = sorted(d for d in processed_root.iterdir() if d.is_dir())
    if max_replays:
        dirs = dirs[:max_replays]

    n_scanned = 0
    n_skipped_no_events = 0
    n_skipped_no_validated = 0

    for rid_dir in dirs:
        frames_pq = rid_dir / "frames.parquet"
        events_pq = rid_dir / "events.parquet"

        if not frames_pq.exists() or not events_pq.exists():
            continue

        try:
            events_df = pd.read_parquet(events_pq)
        except Exception:
            continue

        if len(events_df) == 0:
            n_skipped_no_events += 1
            continue

        # cheap filter before loading frames (saves I/O)
        target_events = events_df[events_df["pattern"].isin(TARGET_CLASSES)]
        if len(target_events) == 0:
            n_skipped_no_events += 1
            continue

        try:
            frames_df = pd.read_parquet(frames_pq)
        except Exception:
            continue

        replay_id = rid_dir.name
        n_scanned += 1

        records = _extract_positive_from_replay(
            frames_df, target_events, replay_id, game_mode_filter, window_len=window_len
        )

        if len(records) == 0:
            n_skipped_no_validated += 1

        all_records.extend(records)

        if verbose and n_scanned % 50 == 0:
            label_counts = {}
            for r in all_records:
                label_counts[r.label] = label_counts.get(r.label, 0) + 1
            print(
                f"  [positive] scanned={n_scanned}  total_windows={len(all_records)}  "
                f"per_class={label_counts}",
                flush=True,
            )

    if verbose:
        label_counts = {}
        for r in all_records:
            label_counts[r.label] = label_counts.get(r.label, 0) + 1
        print(
            f"[positive] DONE: scanned={n_scanned}  "
            f"skipped(no_events)={n_skipped_no_events}  "
            f"skipped(no_validated)={n_skipped_no_validated}  "
            f"total_positives={len(all_records)}  per_class={label_counts}",
            flush=True,
        )

    return all_records


# ---------------------------------------------------------------------------
# NONE class from event-free regions
# ---------------------------------------------------------------------------

def _build_forbidden_mask(
    events_df: pd.DataFrame,
    n_frames: int,
    window_len: int = WINDOW_LENGTH,
) -> np.ndarray:
    """Boolean mask of forbidden NONE window starts.

    A window [s, s+window_len) is forbidden if it overlaps
    [ev_start - EVENT_BUFFER, ev_end + EVENT_BUFFER] of any event.

    ALL events are used (flip_reset included) so that NONE windows do not contain
    real but unlabelled schemes.
    """
    forbidden = np.zeros(n_frames, dtype=bool)
    for _, ev in events_df.iterrows():
        ev_start = int(ev["frame_start"])
        ev_end = int(ev.get("frame_end", ev_start))
        zone_start = max(0, ev_start - EVENT_BUFFER)
        zone_end = min(n_frames, ev_end + EVENT_BUFFER + 1)
        # a window starting at s overlaps the zone if
        # s < zone_end AND s + window_len > zone_start
        win_start_lo = max(0, zone_start - window_len + 1)
        win_start_hi = min(n_frames, zone_end)
        if win_start_lo < win_start_hi:
            forbidden[win_start_lo:win_start_hi] = True
    return forbidden


def _extract_none_from_replay(
    frames_df: pd.DataFrame,
    events_df: pd.DataFrame,
    replay_id: str,
    rng: random.Random,
    max_per_replay: int = 20,
    window_len: int = WINDOW_LENGTH,
    stride: int = WINDOW_STRIDE,
) -> list[WindowRecord]:
    """NONE windows from one replay, avoiding event regions."""
    n_frames = len(frames_df)
    if n_frames < window_len:
        return []

    forbidden = _build_forbidden_mask(events_df, n_frames, window_len=window_len)
    universal = frames_df_to_universal(frames_df)  # (n_frames, 133)

    candidates: list[int] = []
    for start in range(0, n_frames - window_len + 1, stride):
        if not forbidden[start]:
            candidates.append(start)

    if not candidates:
        return []

    # random subsample for diversity
    if len(candidates) > max_per_replay:
        candidates = rng.sample(candidates, max_per_replay)

    records: list[WindowRecord] = []
    for start in candidates:
        end = start + window_len
        tensor = universal[start:end].copy()
        if tensor.shape[0] != window_len:
            continue
        stats = compute_summary_stats(tensor)
        records.append(WindowRecord(
            replay_id=replay_id,
            frame_start=start,
            frame_end=end,
            label="NONE",
            confidence=1.0,
            tensor=tensor,
            summary_stats=stats,
        ))

    return records


def extract_none_windows(
    processed_root: str | Path = "data/processed",
    max_replays: Optional[int] = None,
    target_count: int = 3000,
    max_per_replay: int = 20,
    random_seed: int = 42,
    verbose: bool = True,
    window_len: int = WINDOW_LENGTH,
    stride: int = WINDOW_STRIDE,
) -> list[WindowRecord]:
    """NONE windows from event-free regions.

    NONE windows have the SAME length as the positives; the only difference is that
    they stay EVENT_BUFFER frames away from any event, so the classes are not
    separable by a sampling artefact such as window length.

    Args:
        processed_root: root with one replay_id/ sub-directory per replay
        max_replays: limit for debugging / smoke tests
        target_count: total number of NONE windows to collect
        max_per_replay: max NONE windows per replay (diversity)
        random_seed: for reproducibility
        verbose: print progress
        window_len: window length in frames (45 @ 15 Hz, 90 @ 30 Hz)
        stride: sampling stride in frames (15 @ 15 Hz, 30 @ 30 Hz)

    Returns:
        WindowRecords with label="NONE"
    """
    processed_root = Path(processed_root)
    rng = random.Random(random_seed)
    all_records: list[WindowRecord] = []

    dirs = sorted(d for d in processed_root.iterdir() if d.is_dir())
    if max_replays:
        dirs = dirs[:max_replays]

    # shuffle replays for diversity
    rng.shuffle(dirs)

    n_scanned = 0

    for rid_dir in dirs:
        if len(all_records) >= target_count:
            break

        frames_pq = rid_dir / "frames.parquet"
        if not frames_pq.exists():
            continue

        # load events if present (without events, every frame is allowed for NONE)
        events_df = pd.DataFrame(columns=["frame_start", "frame_end"])
        events_pq = rid_dir / "events.parquet"
        if events_pq.exists():
            try:
                events_df = pd.read_parquet(events_pq)
            except Exception:
                pass

        try:
            frames_df = pd.read_parquet(frames_pq)
        except Exception:
            continue

        replay_id = rid_dir.name
        n_scanned += 1

        records = _extract_none_from_replay(
            frames_df, events_df, replay_id, rng, max_per_replay,
            window_len=window_len, stride=stride,
        )
        all_records.extend(records)

        if verbose and n_scanned % 100 == 0:
            print(
                f"  [NONE] scanned={n_scanned}  total_none={len(all_records)}",
                flush=True,
            )

    # subsample down to target_count
    if len(all_records) > target_count:
        rng.shuffle(all_records)
        all_records = all_records[:target_count]

    if verbose:
        n_replays_repr = len({r.replay_id for r in all_records})
        print(
            f"[NONE] DONE: scanned={n_scanned}  total_none={len(all_records)}  "
            f"replays_represented={n_replays_repr}",
            flush=True,
        )

    return all_records
