"""Tests for StreamingReplayDataset (LRU-based streaming loader)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.models.dataset import (
    PATTERN_COLS,
    META_COLS,
    StreamingReplayDataset,
    WindowSpec,
    _LRUReplayCache,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic parquet factories
# ---------------------------------------------------------------------------

_FEATURE_DIM = 88   # use legacy 2v2 dim for synthetic data (simpler to generate)
_FEATURE_COLS = [f"feat_{i}" for i in range(_FEATURE_DIM)]


def _write_synthetic_replay(
    root: Path,
    rid: str,
    n_frames: int = 200,
    winning_team: int = 0,
    rng: np.random.Generator | None = None,
) -> None:
    """Write a minimal (frames.parquet, labels.parquet) pair under root/rid/."""
    if rng is None:
        rng = np.random.default_rng(42)
    replay_dir = root / rid
    replay_dir.mkdir(parents=True, exist_ok=True)

    # features
    feat_data = {col: rng.standard_normal(n_frames).astype(np.float32) for col in _FEATURE_COLS}
    # required META_COLS
    meta_data = {
        "frame_idx": list(range(n_frames)),
        "src_frame_idx": list(range(n_frames)),
        "time_s": [i * 0.0667 for i in range(n_frames)],
        "episode_seconds_remaining": [300.0 - i * 0.0667 for i in range(n_frames)],
        "next_scoring_team": [winning_team] * n_frames,
        "winning_team": [winning_team] * n_frames,
        "blue_score": [0] * n_frames,
        "orange_score": [0] * n_frames,
        "goal_scored": [False] * n_frames,
    }
    # ACTION_COLS
    action_data = {
        "act_throttle": [0.0] * n_frames,
        "act_steer": [0.0] * n_frames,
        "act_pitch": [0.0] * n_frames,
        "act_yaw": [0.0] * n_frames,
        "act_roll": [0.0] * n_frames,
        "act_jump": [0.0] * n_frames,
        "act_boost": [0.0] * n_frames,
        "act_handbrake": [0.0] * n_frames,
    }
    frames_df = pl.DataFrame({**feat_data, **meta_data, **action_data})
    frames_df.write_parquet(replay_dir / "frames.parquet")

    # labels
    labels_data = {col: [0.0] * n_frames for col in PATTERN_COLS}
    pl.DataFrame(labels_data).write_parquet(replay_dir / "labels.parquet")


# ---------------------------------------------------------------------------
# LRU cache unit tests
# ---------------------------------------------------------------------------

def test_lru_evicts_oldest():
    """After capacity+N puts, cache size never exceeds capacity and oldest entry is gone."""
    cache = _LRUReplayCache(capacity=3)
    dummy = (np.zeros((10, 88), dtype=np.float32), np.zeros((10, 4), dtype=np.float32))
    for i in range(6):
        cache.put(f"rid_{i}", dummy)
    assert cache.size <= 3
    # rid_0, rid_1, rid_2 should have been evicted
    assert cache.get("rid_0") is None
    assert cache.get("rid_1") is None
    assert cache.get("rid_2") is None
    # rid_3, rid_4, rid_5 should still be present
    assert cache.get("rid_3") is not None
    assert cache.get("rid_4") is not None
    assert cache.get("rid_5") is not None


def test_lru_hit_promotes_entry():
    """Accessing an entry promotes it so it survives the next eviction."""
    cache = _LRUReplayCache(capacity=3)
    dummy = (np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32))
    for i in range(3):
        cache.put(f"rid_{i}", dummy)
    # Access rid_0 to make it MRU
    cache.get("rid_0")
    # Add one more → rid_1 (LRU) should be evicted, not rid_0
    cache.put("rid_3", dummy)
    assert cache.get("rid_0") is not None, "rid_0 was promoted and should survive"
    assert cache.get("rid_1") is None, "rid_1 was LRU and should be evicted"


# ---------------------------------------------------------------------------
# StreamingReplayDataset tests
# ---------------------------------------------------------------------------

def test_index_built_without_loading_all_features():
    """Building the index over 100 replays must NOT populate the LRU cache."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rng = np.random.default_rng(0)
        n_replays = 100
        rids = [f"replay_{i:04d}" for i in range(n_replays)]
        for rid in rids:
            _write_synthetic_replay(root, rid, n_frames=200, rng=rng)

        ds = StreamingReplayDataset(root, rids, WindowSpec(), feature_dim=_FEATURE_DIM)

        # Index should have entries
        assert len(ds) > 0, "Index should be non-empty"

        # Cache must be empty — data was NOT loaded at index time
        assert ds.cache_size == 0, (
            f"Cache should be 0 after build_index, got {ds.cache_size}. "
            "This means frames were loaded into RAM during indexing — OOM risk at 25K replays."
        )


def test_getitem_returns_correct_window():
    """__getitem__ returns (features (W,F), target long, patterns float32) with correct shapes."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        spec = WindowSpec(length=45, stride=15, min_replay_frames=90)
        rids = ["replay_000"]
        _write_synthetic_replay(root, "replay_000", n_frames=200)

        ds = StreamingReplayDataset(root, rids, spec, feature_dim=_FEATURE_DIM)
        assert len(ds) > 0

        feat, target, patterns = ds[0]

        assert feat.shape == (spec.length, _FEATURE_DIM), (
            f"Expected ({spec.length}, {_FEATURE_DIM}), got {feat.shape}"
        )
        import torch
        assert isinstance(feat, torch.Tensor)
        assert feat.dtype == torch.float32

        assert isinstance(target, torch.Tensor)
        assert target.dtype == torch.long
        assert int(target.item()) in (0, 1, 2)

        assert isinstance(patterns, torch.Tensor)
        assert patterns.shape == (len(PATTERN_COLS),)
        assert patterns.dtype == torch.float32


def test_lru_cache_evicts_oldest_dataset():
    """After fetching windows from >capacity distinct replays, cache size <= capacity."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        lru_cap = 4
        n_replays = 10
        rids = [f"replay_{i:04d}" for i in range(n_replays)]
        rng = np.random.default_rng(1)
        for rid in rids:
            _write_synthetic_replay(root, rid, n_frames=200, rng=rng)

        ds = StreamingReplayDataset(
            root, rids,
            WindowSpec(length=45, stride=15, min_replay_frames=90),
            feature_dim=_FEATURE_DIM,
            lru_capacity=lru_cap,
        )

        # Touch all replays by iterating through the index in order.
        # Each __getitem__ may or may not cause a cache miss depending on
        # window locality, but after visiting all n_replays distinct rids
        # the cache must not exceed lru_cap.
        for i in range(len(ds)):
            _ = ds[i]

        assert ds.cache_size <= lru_cap, (
            f"Cache grew to {ds.cache_size}, expected <= {lru_cap}"
        )


def test_skipped_replays_too_short():
    """Replays with fewer than min_replay_frames are excluded from the index."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rids = ["short_replay", "long_replay"]
        _write_synthetic_replay(root, "short_replay", n_frames=10)   # < 90
        _write_synthetic_replay(root, "long_replay", n_frames=200)

        ds = StreamingReplayDataset(
            root, rids,
            WindowSpec(length=45, stride=15, min_replay_frames=90),
            feature_dim=_FEATURE_DIM,
        )

        assert ds._skipped.get("too_short", 0) == 1
        # All windows must come from long_replay
        seen_rids = {ds.index[i][0] for i in range(len(ds))}
        assert "short_replay" not in seen_rids
        assert "long_replay" in seen_rids
