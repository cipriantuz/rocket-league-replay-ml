"""Per-window imminence dataset for the short-horizon goal-prediction task.

Speed fix (v2)
--------------
The original implementation stored raw windows + agg features in separate
per-replay .npy files, and recomputed labels + raw frames every time a new
horizon_s was requested.  For 373K windows the pos_weight scan alone took 20+
minutes because ``materialize()`` had to read every parquet file serially.

v2 key changes:
  1. Single per-replay .npz file keyed by horizon: ``<rid>_imm_h{H}.npz``
     stores raw_windows (float32 N×45×88), agg_feats (float32 N×23), and
     both label arrays (int8 N×2) in ONE compressed archive.  No more three
     separate .npy lookups per replay.

  2. ``scripts/warm_imminence_cache.py`` pre-warms these files in parallel
     (mirrors warm_hmlt_cache.py) so the eval harness starts with 100% cache
     hits.  pos_weight is then computed by reading only the small int8 label
     slice from each .npz — no raw-frame loading, negligible RAM.

  3. ``materialize()`` detects the warm cache and loads only what it needs:
     full data for tabular arrays, labels-only for pos_weight computation.

  4. All cache classes have ``__getstate__`` / ``__setstate__`` dropping
     threading.Lock so DataLoader workers can spawn on Windows without
     EOFError (same pattern as FrameSequenceDataset._FrameLRUCache).

Design decisions
-----------------
Unlike FrameSequenceDataset (one item per MATCH), ImminenceDataset yields one
item per WINDOW.  This is necessary because:
  - Tabular baselines (LogReg, XGBoost) want a flat feature vector per window.
  - The deep model here is a per-window encoder (CNN or TransformerEncoder over
    the 45 raw frames), not a match-level sequence model.
  - Balanced sampling and PR-AUC evaluation are simpler on a flat window pool.

Each item is a named tuple of:
    raw_window     float32 (45, 88)  — raw physics frames
    agg_feats      float32 (23,)     — compute_window_match_features aggregate
    binary_label   int8              — 1 if goal within H seconds, else 0
    threeclass_label int8            — 0=blue, 1=orange scores within H, 2=no goal

Cache file layout (v2)
-----------------------
  <cache_root>/<rid>_imm_h{H}.npz

  npz keys:
    raw    float32 (n_windows, 45, 88)
    agg    float32 (n_windows, 23)
    labels int8    (n_windows, 2)  — col 0 = binary, col 1 = threeclass

Backward compatibility
-----------------------
The old three-file format (``<rid>_raw.npy``, ``<rid>_agg.npy``,
``<rid>_lbl_<h>_<hz>.npy``) is still read as a fallback if the new .npz is
absent.  This allows a smooth migration without deleting existing caches.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.models.dataset import NON_FEATURE_COLS, WindowSpec
from src.obs.frame_features import universal_feature_size
from src.obs.goal_imminence import compute_imminence_labels
from src.obs.match_features import MATCH_FEATURE_SIZE, compute_match_feature_sequence
from src.obs.universal_frames import frames_df_to_universal

# Universal feature dimension — all game modes map to this via frames_df_to_universal.
TARGET_FEATURE_DIM = universal_feature_size()  # 133


# ---------------------------------------------------------------------------
# Cache filename helper — single source of truth
# ---------------------------------------------------------------------------

def _npz_path(cache_root: Path, rid: str, horizon_s: float, hz: float) -> Path:
    """Return the canonical .npz path for a replay × horizon combination.

    The filename encodes horizon_s and hz so that changing either parameter
    produces a new file rather than reading stale data.  hz=15 is the default
    and is elided from the name when standard (for readability).

    Examples:
        rid='ABC123', horizon_s=10.0, hz=15.0 → 'ABC123_imm_h10.npz'
        rid='ABC123', horizon_s=5.0,  hz=15.0 → 'ABC123_imm_h5.npz'
        rid='ABC123', horizon_s=3.5,  hz=15.0 → 'ABC123_imm_h3.5.npz'
    """
    h = f"{horizon_s:g}"   # e.g. "10", "5", "3.5"
    if hz != 15.0:
        return cache_root / f"{rid}_imm_h{h}_{int(hz)}hz.npz"
    return cache_root / f"{rid}_imm_h{h}.npz"


# ---------------------------------------------------------------------------
# LRU cache (per-process; drops Lock on pickle for DataLoader compat)
# ---------------------------------------------------------------------------

class _ImminenceLRU:
    """Thread-safe LRU cache keyed by replay_id.

    Stores (raw_windows, agg_feats, binary_labels, threeclass_labels) tuples.
    Lock is dropped on pickling so DataLoader workers can spawn on Windows
    (each worker gets a fresh empty cache — no cross-process sharing needed).
    """

    def __init__(self, capacity: int = 128) -> None:
        self.capacity = capacity
        self._cache: OrderedDict[str, tuple] = OrderedDict()
        self._lock = threading.Lock()

    def __getstate__(self) -> dict:
        return {"capacity": self.capacity}

    def __setstate__(self, state: dict) -> None:
        self.capacity = state["capacity"]
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[tuple]:
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: str, value: tuple) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.capacity:
                    self._cache.popitem(last=False)
                self._cache[key] = value

    @property
    def size(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Low-level cache I/O
# ---------------------------------------------------------------------------

def _load_npz(npz_path: Path) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Try to load the v2 .npz cache.  Returns None on any error."""
    try:
        data = np.load(str(npz_path), allow_pickle=False)
        raw = data["raw"]     # (N, 45, 88) float32
        agg = data["agg"]     # (N, 23) float32
        labels = data["labels"]  # (N, 2) int8
        if raw.ndim == 3 and agg.ndim == 2 and labels.ndim == 2:
            return raw, agg, labels[:, 0].astype(np.int8), labels[:, 1].astype(np.int8)
    except Exception:
        pass
    return None


def _load_legacy_npy(
    cache_root: Path, rid: str, horizon_s: float, hz: float
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Try to load the old three-file .npy format.  Returns None on any error.

    Supports old caches written before the v2 .npz migration.
    """
    hz_tag = f"{int(hz)}hz"
    h_tag = f"h{horizon_s:.1f}s"
    raw_path = cache_root / f"{rid}_raw.npy"
    agg_path = cache_root / f"{rid}_agg.npy"
    lbl_path = cache_root / f"{rid}_lbl_{h_tag}_{hz_tag}.npy"
    if not (raw_path.exists() and agg_path.exists() and lbl_path.exists()):
        return None
    try:
        raw = np.load(str(raw_path), allow_pickle=False)
        agg = np.load(str(agg_path), allow_pickle=False)
        lbl = np.load(str(lbl_path), allow_pickle=False)
        if raw.ndim == 3 and agg.ndim == 2 and lbl.ndim == 2:
            return raw, agg, lbl[:, 0].astype(np.int8), lbl[:, 1].astype(np.int8)
    except Exception:
        pass
    return None


def _save_npz(
    npz_path: Path,
    raw: np.ndarray,
    agg: np.ndarray,
    binary_labels: np.ndarray,
    threeclass_labels: np.ndarray,
) -> bool:
    """Atomically save the four arrays to an .npz file.  Returns True on success.

    Uses tmp-file + os.replace for crash safety (matches warm_hmlt_cache pattern).
    """
    import os

    tmp_path = npz_path.with_suffix(".tmp.npz")
    try:
        labels = np.stack([binary_labels, threeclass_labels], axis=1)  # (N, 2)
        np.savez(str(tmp_path), raw=raw, agg=agg, labels=labels)
        os.replace(str(tmp_path), str(npz_path))
        return True
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False


# ---------------------------------------------------------------------------
# Core compute — called both by the dataset and the parallel warmer
# ---------------------------------------------------------------------------

def compute_replay_imminence(
    frames_pq: Path,
    *,
    spec: WindowSpec,
    horizon_s: float,
    hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute imminence arrays for one replay from its frames.parquet.

    Returns:
        raw_windows:      float32 (n_windows, 45, 133)  — universal 133-dim layout
        agg_feats:        float32 (n_windows, 23)
        binary_labels:    int8    (n_windows,)
        threeclass_labels:int8    (n_windows,)

    Accepts any game mode (1v1/2v2/3v3) — frames_df_to_universal maps to 133-dim.

    This function is importable by the parallel warmer (must be picklable
    — no closures, no top-level class state).
    """
    frames_df = pd.read_parquet(frames_pq)
    feat = frames_df_to_universal(frames_df)  # (n_frames, 133)

    n = feat.shape[0]
    starts = list(range(0, n - spec.length + 1, spec.stride))
    n_windows = len(starts)

    # Raw windows: (n_windows, 45, 133)
    raw_windows = np.stack(
        [feat[s : s + spec.length] for s in starts], axis=0
    )

    # Aggregate features: (n_windows, 23) using existing function
    agg_seq, _ = compute_match_feature_sequence(
        frames_df,
        window=spec.length,
        stride=spec.stride,
    )
    # Align lengths (compute_match_feature_sequence uses same formula)
    agg_seq = agg_seq[:n_windows].astype(np.float32)
    if agg_seq.shape[0] < n_windows:
        pad = np.zeros(
            (n_windows - agg_seq.shape[0], MATCH_FEATURE_SIZE), dtype=np.float32
        )
        agg_seq = np.vstack([agg_seq, pad])

    # Imminence labels
    binary_labels, threeclass_labels = compute_imminence_labels(
        frames_df,
        window=spec.length,
        stride=spec.stride,
        horizon_s=horizon_s,
        hz=hz,
    )

    return raw_windows, agg_seq, binary_labels, threeclass_labels


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ImminenceDataset(Dataset):
    """Per-window dataset for short-horizon goal-imminence prediction.

    Iterates over all 2v2 replays (88-dim), computes sliding windows with the
    same WindowSpec(length=45, stride=15) as the existing datasets, derives
    imminence labels via compute_imminence_labels, and caches results to disk.

    v2 cache (fast path)
    --------------------
    When ``cache_root`` is given and ``scripts/warm_imminence_cache.py`` has
    been run, every replay has a pre-warmed ``<rid>_imm_h{H}.npz`` file.
    ``__getitem__`` loads the npz for that replay on first access and stashes
    it in the per-process LRU.  Subsequent accesses within the same epoch hit
    the LRU in microseconds.

    Args:
        processed_root:  Directory containing one sub-folder per replay ID.
        replay_ids:      List of replay IDs to include (from make_splits).
        spec:            WindowSpec — must match existing datasets.
        horizon_s:       Imminence horizon in seconds (e.g. 5.0 or 10.0).
        hz:              Frame rate (default 15.0).
        cache_root:      Optional directory for on-disk .npz caches.
                         If None, no disk caching (recompute every epoch).
        lru_capacity:    Number of replays to keep in the per-process LRU.
        feature_dim:     Expected feature count — skip replays with other dims.
    """

    def __init__(
        self,
        processed_root: Path | str,
        replay_ids: list[str],
        *,
        spec: WindowSpec = WindowSpec(),
        horizon_s: float = 10.0,
        hz: float = 15.0,
        cache_root: Path | str | None = None,
        lru_capacity: int = 128,
        feature_dim: int = TARGET_FEATURE_DIM,
    ) -> None:
        self.processed_root = Path(processed_root)
        self.spec = spec
        self.horizon_s = horizon_s
        self.hz = hz
        self.feature_dim = feature_dim

        self.cache_root = Path(cache_root) if cache_root else None
        if self.cache_root is not None:
            self.cache_root.mkdir(parents=True, exist_ok=True)

        self._lru = _ImminenceLRU(capacity=lru_capacity)
        self.skipped: dict[str, int] = {}

        # Flat index: (replay_id, window_start) — one entry per window
        self._index: list[tuple[str, int]] = []
        # Per-replay window count (used by materialize)
        self._replay_offsets: dict[str, tuple[int, int]] = {}  # rid -> (start_idx, end_idx)

        self._build_index(replay_ids)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _npz_path(self, rid: str) -> Path:
        assert self.cache_root is not None
        return _npz_path(self.cache_root, rid, self.horizon_s, self.hz)

    def _load_or_compute(
        self, rid: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (raw_windows, agg_feats, binary_labels, threeclass_labels).

        Load order:
          1. v2 .npz cache (fast — single file, single read)
          2. Legacy .npy cache (backward compat)
          3. Compute from parquet + save as v2 .npz
        """
        # 1. v2 .npz cache (warm-cache fast path)
        if self.cache_root is not None:
            npz = self._npz_path(rid)
            if npz.exists():
                result = _load_npz(npz)
                if result is not None:
                    return result

        # 2. Legacy three-file .npy cache
        if self.cache_root is not None:
            result = _load_legacy_npy(self.cache_root, rid, self.horizon_s, self.hz)
            if result is not None:
                # Migrate to v2 format opportunistically
                raw, agg, binary, threeclass = result
                _save_npz(self._npz_path(rid), raw, agg, binary, threeclass)
                return result

        # 3. Compute from parquet
        frames_pq = self.processed_root / rid / "frames.parquet"
        raw, agg, binary, threeclass = compute_replay_imminence(
            frames_pq, spec=self.spec, horizon_s=self.horizon_s, hz=self.hz
        )

        # Save to v2 cache
        if self.cache_root is not None:
            _save_npz(self._npz_path(rid), raw, agg, binary, threeclass)

        return raw, agg, binary, threeclass

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self, replay_ids: list[str]) -> None:
        print(
            f"[ImminenceDataset] Building index over {len(replay_ids)} replays "
            f"(horizon={self.horizon_s}s)...",
            flush=True,
        )
        import polars as pl

        for i, rid in enumerate(replay_ids):
            if i > 0 and i % 500 == 0:
                print(
                    f"[ImminenceDataset]   scanned {i}/{len(replay_ids)} "
                    f"(windows={len(self._index)})",
                    flush=True,
                )

            frames_pq = self.processed_root / rid / "frames.parquet"
            if not frames_pq.exists():
                self.skipped["missing"] = self.skipped.get("missing", 0) + 1
                continue

            # NOTE: we no longer filter by raw feature dim — frames_df_to_universal
            # maps 1v1/2v2/3v3 to the fixed 133-dim universal layout automatically.

            # Count frames cheaply via parquet metadata
            try:
                n_frames = pl.scan_parquet(frames_pq).select(pl.len()).collect().item()
            except Exception:
                self.skipped["read_error"] = self.skipped.get("read_error", 0) + 1
                continue

            if n_frames < self.spec.min_replay_frames:
                self.skipped["too_short"] = self.skipped.get("too_short", 0) + 1
                continue

            n_windows = len(list(range(0, n_frames - self.spec.length + 1, self.spec.stride)))
            if n_windows == 0:
                self.skipped["no_windows"] = self.skipped.get("no_windows", 0) + 1
                continue

            offset_start = len(self._index)
            for s in range(0, n_frames - self.spec.length + 1, self.spec.stride):
                self._index.append((rid, s))
            self._replay_offsets[rid] = (offset_start, len(self._index))

        total_skipped = sum(self.skipped.values())
        print(
            f"[ImminenceDataset] Index done: {len(self._index)} windows from "
            f"{len(self._replay_offsets)} replays (skipped {total_skipped} — {self.skipped})",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rid, start = self._index[idx]

        cached = self._lru.get(rid)
        if cached is None:
            raw_windows, agg_feats, binary_labels, threeclass_labels = self._load_or_compute(rid)
            self._lru.put(rid, (raw_windows, agg_feats, binary_labels, threeclass_labels))
        else:
            raw_windows, agg_feats, binary_labels, threeclass_labels = cached

        # Determine window index within this replay
        replay_start_offset = self._replay_offsets[rid][0]
        win_idx = idx - replay_start_offset

        x_raw = torch.from_numpy(raw_windows[win_idx])           # (45, 88)
        x_agg = torch.from_numpy(agg_feats[win_idx])             # (23,)
        y_bin = torch.tensor(int(binary_labels[win_idx]), dtype=torch.long)
        y_3cl = torch.tensor(int(threeclass_labels[win_idx]), dtype=torch.long)

        return x_raw, x_agg, y_bin, y_3cl

    @property
    def lru_size(self) -> int:
        return self._lru.size


def collate_fn(
    batch: list[tuple[Tensor, Tensor, Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Stack a batch of (raw_window, agg_feats, y_bin, y_3cl) items.

    Returns:
        raw    float32 (B, 45, 88)
        agg    float32 (B, 23)
        y_bin  long    (B,)
        y_3cl  long    (B,)
    """
    raw_list, agg_list, yb_list, y3_list = zip(*batch)
    return (
        torch.stack(raw_list),
        torch.stack(agg_list),
        torch.stack(yb_list),
        torch.stack(y3_list),
    )


# ---------------------------------------------------------------------------
# materialize() — flat array accessor for tabular baselines
# ---------------------------------------------------------------------------

def _load_labels_only_from_npz(
    cache_root: Path, rid: str, horizon_s: float, hz: float
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Load only the label arrays from the v2 .npz cache (for fast pos_weight).

    Avoids loading raw windows (which are large) when only labels are needed.
    Returns (binary_labels, threeclass_labels) or None if cache miss.
    """
    npz = _npz_path(cache_root, rid, horizon_s, hz)
    if not npz.exists():
        return None
    try:
        data = np.load(str(npz), allow_pickle=False)
        labels = data["labels"]   # (N, 2) int8
        if labels.ndim == 2 and labels.shape[1] == 2:
            return labels[:, 0].astype(np.int8), labels[:, 1].astype(np.int8)
    except Exception:
        pass
    return None


def materialize(
    processed_root: Path | str,
    replay_ids: list[str],
    *,
    spec: WindowSpec = WindowSpec(),
    horizon_s: float = 10.0,
    hz: float = 15.0,
    cache_root: Path | str | None = None,
    feature_dim: int = TARGET_FEATURE_DIM,
    verbose: bool = True,
    labels_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Materialize all windows into flat numpy arrays for tabular baselines.

    When ``cache_root`` is given and the warm cache exists, loads from .npz
    files directly — no parquet reads, no feature recomputation.

    When ``labels_only=True``, raw windows and agg features are returned as
    empty arrays (shape (0,...)).  Use this for fast pos_weight computation:
    only the small int8 label column from each .npz is read, not the 45×88
    raw windows (saves ~99% of I/O).

    Returns:
        X_raw   float32 (N, 45, feature_dim) — universal physics windows (empty if labels_only)
        X_agg   float32 (N, 23)              — aggregate match features (empty if labels_only)
        y_bin   int8    (N,)                  — binary imminence label
        y_3cl   int8    (N,)                  — 3-class label
    """
    processed_root = Path(processed_root)
    cache_root_p = Path(cache_root) if cache_root else None

    raw_chunks: list[np.ndarray] = []
    agg_chunks: list[np.ndarray] = []
    yb_chunks: list[np.ndarray] = []
    y3_chunks: list[np.ndarray] = []
    skipped = 0
    cache_hits = 0

    for rid in replay_ids:
        frames_pq = processed_root / rid / "frames.parquet"
        if not frames_pq.exists():
            skipped += 1
            continue
        # NOTE: no per-replay dim filter — frames_df_to_universal handles all modes.

        loaded = False

        # Fast path: v2 .npz cache
        if cache_root_p is not None:
            if labels_only:
                # Read only the small label slice (no raw windows loaded)
                result = _load_labels_only_from_npz(cache_root_p, rid, horizon_s, hz)
                if result is not None:
                    binary, threeclass = result
                    yb_chunks.append(binary)
                    y3_chunks.append(threeclass)
                    loaded = True
                    cache_hits += 1
            else:
                npz = _npz_path(cache_root_p, rid, horizon_s, hz)
                if npz.exists():
                    result_full = _load_npz(npz)
                    if result_full is not None:
                        raw, agg, binary, threeclass = result_full
                        raw_chunks.append(raw)
                        agg_chunks.append(agg)
                        yb_chunks.append(binary)
                        y3_chunks.append(threeclass)
                        loaded = True
                        cache_hits += 1

        # Fallback: legacy three-file .npy cache
        if not loaded and cache_root_p is not None:
            result_legacy = _load_legacy_npy(cache_root_p, rid, horizon_s, hz)
            if result_legacy is not None:
                raw, agg, binary, threeclass = result_legacy
                if not labels_only:
                    raw_chunks.append(raw)
                    agg_chunks.append(agg)
                yb_chunks.append(binary)
                y3_chunks.append(threeclass)
                loaded = True
                cache_hits += 1

        # Cold path: compute from parquet (and save cache)
        if not loaded:
            try:
                raw, agg, binary, threeclass = compute_replay_imminence(
                    frames_pq, spec=spec, horizon_s=horizon_s, hz=hz
                )
                if raw.shape[0] == 0:
                    skipped += 1
                    continue
                if not labels_only:
                    raw_chunks.append(raw)
                    agg_chunks.append(agg)
                yb_chunks.append(binary)
                y3_chunks.append(threeclass)

                # Write v2 cache
                if cache_root_p is not None:
                    cache_root_p.mkdir(parents=True, exist_ok=True)
                    _save_npz(
                        _npz_path(cache_root_p, rid, horizon_s, hz),
                        raw, agg, binary, threeclass,
                    )
            except Exception:
                skipped += 1
                continue

    if verbose:
        n_loaded = len(yb_chunks)
        print(
            f"[materialize] loaded {n_loaded} replays ({skipped} skipped, "
            f"{cache_hits} cache hits)",
            flush=True,
        )

    _feat_dim = feature_dim if feature_dim is not None else TARGET_FEATURE_DIM
    if not yb_chunks:
        return (
            np.empty((0, spec.length, _feat_dim), dtype=np.float32),
            np.empty((0, MATCH_FEATURE_SIZE), dtype=np.float32),
            np.empty((0,), dtype=np.int8),
            np.empty((0,), dtype=np.int8),
        )

    y_bin = np.concatenate(yb_chunks, axis=0).astype(np.int8)
    y_3cl = np.concatenate(y3_chunks, axis=0).astype(np.int8)

    if labels_only:
        return (
            np.empty((0, spec.length, _feat_dim), dtype=np.float32),
            np.empty((0, MATCH_FEATURE_SIZE), dtype=np.float32),
            y_bin,
            y_3cl,
        )

    return (
        np.concatenate(raw_chunks, axis=0).astype(np.float32),
        np.concatenate(agg_chunks, axis=0).astype(np.float32),
        y_bin,
        y_3cl,
    )
