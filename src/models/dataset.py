"""PyTorch Dataset over processed replay parquets.

Returns sliding windows of (features, target_label, pattern_multi_label):
    features            shape (window, feature_dim) float32
    target_label        scalar long:
                          0 = next goal scored by blue
                          1 = next goal scored by orange
                          2 = no more goals in this match (clutch / final frames)
    pattern_multi_label shape (n_patterns,) float32 — 1 if any frame in
                        window has the pattern active, else 0.

Target uses `next_scoring_team` at window's last frame (learnable momentum
target), NOT `winning_team` final (unlearnable from 3s window of mid-match).

Two Dataset implementations are provided:
    ReplayWindowDataset    — legacy, loads all frames to RAM at build time.
                             Safe for ~500 replays; OOM at 25K+.
    StreamingReplayDataset — streaming with per-process LRU cache.
                             Index is ~30 bytes/entry; only replays requested
                             by __getitem__ are loaded.  Safe for 25K+ replays.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

PATTERN_COLS = ["flip_reset", "double_tap", "aerial_dribble", "ceiling_shot"]
META_COLS = [
    "frame_idx", "src_frame_idx", "time_s", "episode_seconds_remaining",
    "next_scoring_team", "winning_team", "blue_score", "orange_score", "goal_scored",
]
ACTION_COLS = [
    "act_throttle", "act_steer", "act_pitch", "act_yaw",
    "act_roll", "act_jump", "act_boost", "act_handbrake",
]
# Exclude both meta and actions from the feature vector used for training.
NON_FEATURE_COLS = set(META_COLS) | set(ACTION_COLS)


@dataclass(frozen=True)
class WindowSpec:
    length: int = 45         # 3.0s @ 15 Hz
    stride: int = 15         # 1.0s — overlapping windows
    min_replay_frames: int = 90


class ReplayWindowDataset(Dataset):
    """Loads all per-replay parquets up-front (small index, large frames cached lazily).

    For 10K+ replays this is fine — index is just (replay_id, window_offset, win_label).
    Frame data is mmap-loaded per-window via polars lazy scan (single read per replay).
    """

    def __init__(
        self,
        processed_root: Path,
        replay_ids: list[str],
        spec: WindowSpec = WindowSpec(),
        target_feature_dim: int | None = 88,  # 2v2 default; None = take whichever dim is most common
    ):
        self.processed_root = Path(processed_root)
        self.spec = spec
        self.target_feature_dim = target_feature_dim
        self.index: list[tuple[str, int, int, np.ndarray]] = []  # (replay_id, start, win, patterns_any)
        self._frame_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._skipped: dict[str, int] = {}
        self._build_index(replay_ids)

    def _build_index(self, replay_ids: list[str]) -> None:
        for rid in replay_ids:
            frames_pq = self.processed_root / rid / "frames.parquet"
            labels_pq = self.processed_root / rid / "labels.parquet"
            if not frames_pq.exists() or not labels_pq.exists():
                continue
            # Read once: meta + features, plus labels
            frames_df = pl.read_parquet(frames_pq)
            labels_df = pl.read_parquet(labels_pq)
            n = len(frames_df)
            if n < self.spec.min_replay_frames:
                self._skipped["too_short"] = self._skipped.get("too_short", 0) + 1
                continue
            feature_cols = [c for c in frames_df.columns if c not in NON_FEATURE_COLS]
            feat = frames_df.select(feature_cols).to_numpy().astype(np.float32)
            # Filter by feature dimension if requested (enforces uniform game mode)
            if self.target_feature_dim is not None and feat.shape[1] != self.target_feature_dim:
                self._skipped[f"wrong_dim_{feat.shape[1]}"] = (
                    self._skipped.get(f"wrong_dim_{feat.shape[1]}", 0) + 1
                )
                continue
            labels = labels_df.select(PATTERN_COLS).to_numpy().astype(np.float32)
            next_scoring = frames_df["next_scoring_team"].to_numpy().astype(np.int64)
            self._frame_cache[rid] = (feat, labels)
            # Build window starts
            for start in range(0, n - self.spec.length + 1, self.spec.stride):
                end = start + self.spec.length - 1
                window_labels = labels[start : start + self.spec.length].max(axis=0)
                # Target = next_scoring_team at window's last frame
                # 0 = blue scores next, 1 = orange scores next, -1 = no more goals -> map to 2
                nst = int(next_scoring[end])
                target = 2 if nst < 0 else nst
                self.index.append((rid, start, target, window_labels))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        rid, start, win, patterns_any = self.index[idx]
        feat, _labels = self._frame_cache[rid]
        x = feat[start : start + self.spec.length]
        return (
            torch.from_numpy(x),
            torch.tensor(win, dtype=torch.long),
            torch.from_numpy(patterns_any.astype(np.float32)),
        )

    @property
    def feature_dim(self) -> int:
        if not self._frame_cache:
            return 0
        return next(iter(self._frame_cache.values()))[0].shape[1]


class _LRUReplayCache:
    """Thread-safe LRU cache keyed by replay_id.

    Stores (feat: np.ndarray, labels: np.ndarray) per replay.
    Uses an OrderedDict to track insertion/access order so that
    the least-recently-used entry can be evicted in O(1).

    Thread safety note: DataLoader with num_workers>0 spawns separate
    *processes* (not threads), so each worker gets its own copy of the
    cache — no cross-process contention.  The lock is kept for safety
    when num_workers=0 (single-process iteration with potential threading).
    """

    def __init__(self, capacity: int = 512) -> None:
        self.capacity = capacity
        self._cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> tuple[np.ndarray, np.ndarray] | None:
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)   # mark as most-recently used
            return self._cache[key]

    def put(self, key: str, value: tuple[np.ndarray, np.ndarray]) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.capacity:
                    self._cache.popitem(last=False)  # evict LRU
                self._cache[key] = value

    def __len__(self) -> int:
        return len(self._cache)

    @property
    def size(self) -> int:
        return len(self._cache)


class StreamingReplayDataset(Dataset):
    """Streaming Dataset that keeps only a sliding window of replays in RAM.

    Motivation: at 25,000 replays × ~1.6 MB each = ~40 GB if all frames were
    loaded up-front (ReplayWindowDataset pattern).  StreamingReplayDataset
    builds a compact index at init time (~30 bytes/entry) and loads individual
    replay parquets on demand, with an LRU cache to amortize repeated access.

    Index entry layout: (replay_id: str, start_frame: int, target: int, pattern_bits: np.uint8)
        pattern_bits encodes the 4-bit pattern_multi_label for the window as a
        single uint8 so the index array stays small.  It is expanded back to
        float32 at __getitem__ time.

    Args:
        processed_root:   Directory containing one sub-folder per replay.
        replay_ids:       List of replay IDs (sub-folder names) to include.
        spec:             WindowSpec (length, stride, min_frames).
        feature_dim:      Expected feature dimension (133 for universal,
                          88 for legacy 2v2).  Replays with a different dimension
                          are skipped and counted in self._skipped.
        lru_capacity:     Maximum number of replays kept in the per-process
                          LRU cache.  Default 512 — each replay is ~1.6 MB so
                          512 replays ≈ 800 MB per DataLoader worker, well within
                          a 32 GB system even with 4 workers.
    """

    def __init__(
        self,
        processed_root: Path,
        replay_ids: list[str],
        spec: WindowSpec = WindowSpec(),
        feature_dim: int = 133,
        lru_capacity: int = 512,
    ) -> None:
        self.processed_root = Path(processed_root)
        self.spec = spec
        self.feature_dim = feature_dim
        self._cache = _LRUReplayCache(capacity=lru_capacity)
        self._skipped: dict[str, int] = {}
        # Compact index: list of (replay_id, start_frame_idx, target_label, pattern_bits_uint8)
        self.index: list[tuple[str, int, int, np.ndarray]] = []
        self._build_index_streaming(replay_ids)

    def _build_index_streaming(self, replay_ids: list[str]) -> None:
        """Scan metadata only — do NOT load feature arrays into RAM here."""
        print(f"[StreamingDataset] Building index over {len(replay_ids)} replays...", flush=True)
        for i, rid in enumerate(replay_ids):
            if i > 0 and i % 1000 == 0:
                print(f"[StreamingDataset]   scanned {i}/{len(replay_ids)} (indexed={len(self.index)} windows)", flush=True)
            frames_pq = self.processed_root / rid / "frames.parquet"
            labels_pq = self.processed_root / rid / "labels.parquet"
            if not frames_pq.exists() or not labels_pq.exists():
                continue

            # Lazy scan — only read the columns we need for the index.
            # polars.scan_parquet reads footer metadata without loading row data.
            try:
                lf = pl.scan_parquet(frames_pq)
                schema = lf.schema
                meta_lf = lf.select(["next_scoring_team"])
                meta_df = meta_lf.collect()
            except Exception:
                self._skipped["read_error"] = self._skipped.get("read_error", 0) + 1
                continue

            # Filter by feature dimension (enforces uniform game mode across batch)
            n_features = sum(1 for c in schema if c not in NON_FEATURE_COLS)
            if n_features != self.feature_dim:
                self._skipped[f"wrong_dim_{n_features}"] = (
                    self._skipped.get(f"wrong_dim_{n_features}", 0) + 1
                )
                continue

            n = len(meta_df)
            if n < self.spec.min_replay_frames:
                self._skipped["too_short"] = self._skipped.get("too_short", 0) + 1
                continue

            # Read pattern labels (small, 4 cols × n rows — cheap)
            try:
                labels_df = pl.read_parquet(labels_pq, columns=PATTERN_COLS)
            except Exception:
                self._skipped["labels_error"] = self._skipped.get("labels_error", 0) + 1
                continue

            labels_np = labels_df.to_numpy().astype(np.float32)   # (n, 4)
            next_scoring = meta_df["next_scoring_team"].to_numpy().astype(np.int64)

            for start in range(0, n - self.spec.length + 1, self.spec.stride):
                end = start + self.spec.length - 1
                # Pack 4-bit pattern presence into a uint8 to keep index compact
                window_any = labels_np[start : start + self.spec.length].max(axis=0)  # (4,)
                pattern_bits = np.packbits(window_any.astype(bool), bitorder="little").view(np.uint8)
                nst = int(next_scoring[end])
                target = 2 if nst < 0 else nst
                self.index.append((rid, start, target, pattern_bits))

    def _load_replay(self, rid: str) -> tuple[np.ndarray, np.ndarray]:
        """Load frames.parquet + labels.parquet for a single replay into RAM.

        This is called only on a cache MISS.  The result is stored in the LRU
        cache so subsequent windows from the same replay are served from RAM.
        """
        frames_pq = self.processed_root / rid / "frames.parquet"
        labels_pq = self.processed_root / rid / "labels.parquet"

        frames_df = pl.read_parquet(frames_pq)
        feature_cols = [c for c in frames_df.columns if c not in NON_FEATURE_COLS]
        feat = frames_df.select(feature_cols).to_numpy().astype(np.float32)

        labels_df = pl.read_parquet(labels_pq, columns=PATTERN_COLS)
        labels = labels_df.to_numpy().astype(np.float32)

        return feat, labels

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        rid, start, target, pattern_bits = self.index[idx]

        cached = self._cache.get(rid)
        if cached is None:
            feat, labels = self._load_replay(rid)
            self._cache.put(rid, (feat, labels))
        else:
            feat, labels = cached

        x = feat[start : start + self.spec.length]   # (window, feature_dim)
        # Unpack uint8 bits back to per-pattern float32 presence
        patterns_any = np.unpackbits(pattern_bits, count=len(PATTERN_COLS), bitorder="little").astype(np.float32)

        return (
            torch.from_numpy(x),
            torch.tensor(target, dtype=torch.long),
            torch.from_numpy(patterns_any),
        )

    @property
    def cache_size(self) -> int:
        """Number of replays currently held in the LRU cache."""
        return self._cache.size


def list_processed_replays(processed_root: Path) -> list[str]:
    """Return all replay IDs with both frames.parquet and labels.parquet."""
    root = Path(processed_root)
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and (p / "frames.parquet").exists() and (p / "labels.parquet").exists()
    )


def make_splits(
    replay_ids: list[str], train_frac: float = 0.7, val_frac: float = 0.15, seed: int = 42
) -> tuple[list[str], list[str], list[str]]:
    rng = np.random.default_rng(seed)
    ids = list(replay_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    return ids[:n_train], ids[n_train : n_train + n_val], ids[n_train + n_val :]
