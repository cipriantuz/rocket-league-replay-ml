"""Per-replay FRAME-SEQUENCE dataset for HMLTv2.

WHY a new dataset (vs MatchSequenceDataset)
-------------------------------------------
MatchSequenceDataset yields (seq: T×23, length, label) where T windows of
23-dim HAND-ENGINEERED match features have already been aggregated.  HMLTv2's
L1 encoder needs the RAW FRAME PHYSICS (133 features per frame × 45 frames per
window in the universal representation), so each __getitem__ must yield
(T, 45, 133) instead of (T, 23).

This module:
  1. Reads frames.parquet per replay (any game mode: 1v1/2v2/3v3).
  2. Maps each replay through frames_df_to_universal → fixed 133-dim layout.
  3. Slices sliding windows of 45 frames with stride 15 (matching WindowSpec).
  4. Optionally computes the 23-dim match-feature sequence for the HYBRID path.
  5. Caches the raw-window tensor as <frames_cache_root>/<rid>_frames.npy to
     avoid re-reading the full parquet on every epoch (~50× speedup).
  6. Uses an LRU in-memory cache to amortize repeated access within an epoch.
     When cache_device='cuda' is requested, stores fp16 tensors on-GPU instead
     of system-RAM numpy arrays (see _VRAMCache below).

Universal 133-dim layout (from src/obs/frame_features.py):
  [0:9]      ball physics
  [9:117]    6 car slots × 18 features (blue 0-2, orange 0-2, zero-padded)
  [117:124]  boost pad summary
  [124:130]  slot presence masks
  [130:133]  n_active counts (normalised)
               index 132 = n_total/6:
                 3v3 → 1.000,  2v2 → 0.667,  1v1 → 0.333

Cache sizing note:
  One replay × 200 windows × 45 frames × 133 features × 4 bytes ≈ 4.8 MB.
  31K replays would be ~149 GB fully pre-warmed — too large for RAM.  The LRU
  cap (default 128 replays ≈ 600 MB) keeps memory sane.  The on-disk npy
  files pre-warm disk I/O; the per-epoch LRU amortizes repeated hits within a
  shuffle epoch.

  VRAM cache: fp16 halves the per-replay footprint:
    50 windows × 45 × 133 × 2 bytes = 0.60 MB/replay (vs 1.19 MB fp32).
    12 GB VRAM budget → ~20K replays; 14 GB → ~23K; 13 GB → ~21.5K.
    For 3v3 ~40K replays: use 30-window cap → 0.36 MB → 14 GB fits ~38K.

Cache layout:
  <frames_cache_root>/<rid>_frames.npy  — float32 (T_win, 45, feature_dim) array
  <hmlt_cache_root>/<rid>.npy           — float32 (T_win, 23) array (reused from v1)
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.models.dataset import NON_FEATURE_COLS, WindowSpec
from src.obs.frame_features import universal_feature_size
from src.obs.match_features import MATCH_FEATURE_SIZE, compute_match_feature_sequence
from src.obs.universal_frames import frames_df_to_universal

# Universal feature dimension — applies to ALL game modes (1v1/2v2/3v3).
# The frames_df_to_universal mapper re-slots any mode into this fixed 133-dim layout.
TARGET_FEATURE_DIM = universal_feature_size()  # 133

# Feature index used to detect game mode from the universal 133-dim vector.
# n_total/6: 3v3→1.0, 2v2→0.667, 1v1→0.333.
_N_TOTAL_NORM_IDX = 132

# Threshold for n_total/6 per mode (centred between adjacent values):
#   1v1: n_total=2 → 0.333   |< 0.5 <| 2v2: n_total=4 → 0.667   |< 0.84 <| 3v3: n_total=6 → 1.0
_MODE_THRESHOLDS: dict[str, tuple[float, float]] = {
    "1v1": (0.0, 0.50),
    "2v2": (0.50, 0.84),
    "3v3": (0.84, 1.01),
}


# ---------------------------------------------------------------------------
# VRAM-resident LRU cache — stores fp16 tensors on CUDA (replaces system-RAM LRU)
# ---------------------------------------------------------------------------

class _VRAMCache:
    """Thread-safe LRU cache that stores per-replay windows as fp16 CUDA tensors.

    Why fp16?
      - fp32 physics features are already normalised to ~[-1, 1].  fp16 keeps
        >3 decimal places of precision — ample for normalised physics inputs.
      - fp16 halves VRAM vs fp32: 50 × 45 × 133 × 2 bytes = 0.60 MB/replay.
      - The model uses torch.cuda.amp autocast so fp16 inputs are native.

    Capacity is enforced by VRAM BYTES (not entry count): entries are evicted
    LRU-style when adding a new entry would exceed vram_budget_bytes.

    __getitem__ returns the stored fp16 tensor.  The caller casts to fp32 or
    keeps fp16 depending on training mode.

    Pickling: CUDA tensors are not picklable across DataLoader worker processes.
    We drop the cache contents on __getstate__ (same pattern as _FrameLRUCache).
    With num_workers=0 (Windows default) there is no spawning, so the cache is
    shared in-process throughout the entire training run.
    """

    def __init__(self, vram_budget_bytes: int, device: torch.device) -> None:
        self.vram_budget_bytes = vram_budget_bytes
        self.device = device
        self._cache: OrderedDict[str, Tensor] = OrderedDict()
        self._used_bytes: int = 0
        self._lock = threading.Lock()

    # Pickle-safety: drop the lock + CUDA tensors (can't cross process boundary)
    def __getstate__(self) -> dict:
        return {"vram_budget_bytes": self.vram_budget_bytes, "device": self.device}

    def __setstate__(self, state: dict) -> None:
        self.vram_budget_bytes = state["vram_budget_bytes"]
        self.device = state["device"]
        self._cache = OrderedDict()
        self._used_bytes = 0
        self._lock = threading.Lock()

    def _tensor_bytes(self, t: Tensor) -> int:
        return t.numel() * t.element_size()

    def get(self, key: str) -> Optional[Tensor]:
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: str, windows_fp32: np.ndarray) -> Tensor:
        """Convert a float32 numpy array to fp16 CUDA tensor, cache + return it.

        If the entry already exists, just return it (no re-upload).
        Evicts least-recently-used entries until there is budget for the new entry.

        Returns the fp16 CUDA tensor (already on self.device).
        """
        t = torch.from_numpy(windows_fp32).to(dtype=torch.float16, device=self.device)
        entry_bytes = self._tensor_bytes(t)

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

            # Evict LRU entries until there's room
            while self._used_bytes + entry_bytes > self.vram_budget_bytes and self._cache:
                _, evicted = self._cache.popitem(last=False)
                self._used_bytes -= self._tensor_bytes(evicted)

            self._cache[key] = t
            self._used_bytes += entry_bytes
            return t

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def used_gb(self) -> float:
        return self._used_bytes / 1e9

    @property
    def budget_gb(self) -> float:
        return self.vram_budget_bytes / 1e9


# ---------------------------------------------------------------------------
# LRU memory cache — identical pattern to _SeqLRUCache in match_sequence_dataset
# ---------------------------------------------------------------------------

class _FrameLRUCache:
    """Thread-safe LRU cache.

    Stores (frame_windows: np.ndarray, match_feats: np.ndarray | None) pairs
    keyed by replay_id.  match_feats is None when with_match_feats=False.
    """

    def __init__(self, capacity: int = 128) -> None:
        self.capacity = capacity
        self._cache: OrderedDict[
            str, tuple[np.ndarray, Optional[np.ndarray]]
        ] = OrderedDict()
        self._lock = threading.Lock()

    # threading.Lock is not picklable → DataLoader worker spawn (Windows) fails
    # with EOFError. Drop the lock + cache contents on pickle; each spawned
    # worker process reconstructs a fresh, empty cache (no cross-process sharing).
    def __getstate__(self) -> dict:
        return {"capacity": self.capacity}

    def __setstate__(self, state: dict) -> None:
        self.capacity = state["capacity"]
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[tuple[np.ndarray, Optional[np.ndarray]]]:
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(
        self,
        key: str,
        value: tuple[np.ndarray, Optional[np.ndarray]],
    ) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.capacity:
                    self._cache.popitem(last=False)  # evict LRU
                self._cache[key] = value

    @property
    def size(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FrameSequenceDataset(Dataset):
    """Returns one (frame_windows, match_feats_or_None, label) per MATCH.

    Each item:
        frame_windows   float32 Tensor (T_win, 45, feature_dim)
                        T_win = min(n_windows, max_windows).
                        feature_dim defaults to 133 (universal, all game modes).
        match_feats     float32 Tensor (T_win, 23) or None (when with_match_feats=False).
        label           int {0=blue wins, 1=orange wins}.

    Variable T_win across matches is handled by the provided collate_fn.

    Args:
        processed_root:     Directory containing one sub-folder per replay ID.
        replay_ids:         Subset of replay IDs (from make_splits).
        spec:               WindowSpec(length=45, stride=15) — must match v1 for
                            identical windowing and fair comparison.
        max_windows:        Cap per-replay sequence to this many windows.
                            Default 200 ≈ 600 s; captures overtime matches.
        frames_cache_root:  If given, raw-window tensors are cached as
                            <frames_cache_root>/<rid>_frames.npy.  Speeds up
                            subsequent epochs / restarts by ~50×.
        hmlt_cache_root:    If given, the 23-dim match-feature sequences are
                            loaded from the existing v1 cache (same format as
                            MatchSequenceDataset's cache_root).  Falls back to
                            computing on-the-fly if the file does not exist.
        with_match_feats:   Whether to also compute/return the 23-dim match
                            feature sequence (required by HMLTv2 hybrid path).
        lru_capacity:       LRU in-memory (system-RAM) cache size in replays.
                            Ignored when cache_device='cuda' (VRAM cache takes over).
        feature_dim:        Output feature dimension after universal mapping
                            (default 133).  Used only to validate cached .npy
                            files; the mapper itself handles any input mode.
        cache_device:       'cpu' (default, system-RAM LRU) or 'cuda' (VRAM fp16
                            LRU — frees system RAM, eliminates CPU→GPU copy).
        vram_cache_gb:      VRAM budget in GB for the VRAM cache.  Only used when
                            cache_device='cuda'.  Entries are evicted LRU-style
                            when the budget is reached.
        mode:               If set ('1v1', '2v2', '3v3'), keep only replays whose
                            first cached-window n_total_norm value matches.
                            Detection uses feature index 132 of the npy cache:
                              3v3 → 1.000,  2v2 → 0.667,  1v1 → 0.333.
                            Falls back to a parquet column scan when npy is absent.
    """

    def __init__(
        self,
        processed_root: Path | str,
        replay_ids: list[str],
        *,
        spec: WindowSpec = WindowSpec(),
        max_windows: int = 200,
        frames_cache_root: Path | str | None = None,
        hmlt_cache_root: Path | str | None = None,
        with_match_feats: bool = True,
        lru_capacity: int = 128,
        feature_dim: int = TARGET_FEATURE_DIM,
        cache_device: str = "cpu",
        vram_cache_gb: float = 12.0,
        mode: Optional[str] = None,
    ) -> None:
        self.processed_root = Path(processed_root)
        self.spec = spec
        self.max_windows = max_windows
        self.with_match_feats = with_match_feats
        self.feature_dim = feature_dim
        self.mode = mode.lower() if mode else None

        self.frames_cache_root = Path(frames_cache_root) if frames_cache_root else None
        if self.frames_cache_root is not None:
            self.frames_cache_root.mkdir(parents=True, exist_ok=True)

        self.hmlt_cache_root = Path(hmlt_cache_root) if hmlt_cache_root else None

        # Select cache backend: VRAM (fp16 CUDA tensors) or system-RAM (fp32 numpy)
        self._use_vram_cache = (cache_device == "cuda")
        if self._use_vram_cache:
            if not torch.cuda.is_available():
                print(
                    "[FrameSequenceDataset] WARNING: cache_device='cuda' requested but "
                    "CUDA is not available — falling back to CPU RAM cache.",
                    flush=True,
                )
                self._use_vram_cache = False

        if self._use_vram_cache:
            _dev = torch.device("cuda")
            _budget = int(vram_cache_gb * 1e9)
            self._vram_cache = _VRAMCache(vram_budget_bytes=_budget, device=_dev)
            # When VRAM is used for frame windows, we still need a RAM LRU for
            # match_feats (23-dim, ~23KB/replay — negligible RAM).  Capacity should
            # match the VRAM entry count so match_feats stay warm while frame
            # windows are in VRAM. Estimate from budget and per-replay fp16 size.
            _bytes_per = max_windows * 45 * feature_dim * 2  # fp16
            _mf_lru_cap = max(lru_capacity, int(_budget / _bytes_per) + 512)
            self._mem_cache = _FrameLRUCache(capacity=_mf_lru_cap)
            print(
                f"[FrameSequenceDataset] VRAM cache enabled: "
                f"budget={vram_cache_gb:.1f} GB, device={_dev}, "
                f"match_feats RAM LRU capacity={_mf_lru_cap}",
                flush=True,
            )
        else:
            self._vram_cache = None  # type: ignore[assignment]
            self._mem_cache = _FrameLRUCache(capacity=lru_capacity)

        self._skipped: dict[str, int] = {}

        # Compact index: (replay_id, label, n_frames)
        self._index: list[tuple[str, int, int]] = []
        self._build_index(replay_ids)

    # ------------------------------------------------------------------
    # Index construction — only reads winning_team column (fast)
    # ------------------------------------------------------------------

    def _detect_mode_from_npy(self, rid: str) -> Optional[str]:
        """Detect game mode from the npy cache file (cheapest path).

        Reads only the first window's n_total_norm value (feature index 132).
        Returns '1v1', '2v2', '3v3', or None if the file is absent/unreadable.
        """
        if self.frames_cache_root is None:
            return None
        cache_file = self.frames_cache_root / f"{rid}_frames.npy"
        if not cache_file.exists():
            return None
        try:
            arr = np.load(str(cache_file), allow_pickle=False, mmap_mode="r")
            if arr.ndim != 3 or arr.shape[2] != self.feature_dim:
                return None
            n_total_norm = float(arr[0, 0, _N_TOTAL_NORM_IDX])
            for mode_name, (lo, hi) in _MODE_THRESHOLDS.items():
                if lo <= n_total_norm < hi:
                    return mode_name
            return None
        except Exception:
            return None

    def _detect_mode_from_parquet(self, frames_pq: Path) -> Optional[str]:
        """Detect game mode by counting car columns in the parquet schema.

        Fallback when no npy cache exists.  Uses parquet schema (no data read).
        """
        try:
            import pyarrow.parquet as pq
            schema = pq.read_schema(str(frames_pq))
            car_prefixes = set()
            import re as _re
            for name in schema.names:
                m = _re.match(r"^(car_(?:blue|orange)\d+)_pos_x$", name)
                if m:
                    car_prefixes.add(m.group(1))
            n_cars = len(car_prefixes)
            if n_cars == 2:
                return "1v1"
            elif n_cars == 4:
                return "2v2"
            elif n_cars == 6:
                return "3v3"
            return None
        except Exception:
            return None

    def _build_index(self, replay_ids: list[str]) -> None:
        mode_label = f" (mode={self.mode})" if self.mode else ""
        print(
            f"[FrameSequenceDataset] Building index over {len(replay_ids)} replays"
            f"{mode_label}...",
            flush=True,
        )
        skipped_label = 0
        skipped_short = 0
        skipped_missing = 0
        skipped_error = 0
        skipped_mode = 0

        for i, rid in enumerate(replay_ids):
            if i > 0 and i % 2000 == 0:
                print(
                    f"[FrameSequenceDataset]   scanned {i}/{len(replay_ids)} "
                    f"(indexed={len(self._index)})",
                    flush=True,
                )
            frames_pq = self.processed_root / rid / "frames.parquet"
            if not frames_pq.exists():
                skipped_missing += 1
                continue

            # ------ Mode filter ------
            if self.mode is not None:
                detected = self._detect_mode_from_npy(rid)
                if detected is None:
                    detected = self._detect_mode_from_parquet(frames_pq)
                if detected is not None and detected != self.mode:
                    skipped_mode += 1
                    continue
                # If detection failed (detected is None), let the replay through
                # (better to include than silently drop; it will be filtered at
                # load time by the n_total_norm check in __getitem__ if needed).

            try:
                # Read only what we need for the index — fast parquet footer scan.
                df_meta = pd.read_parquet(
                    frames_pq, columns=["winning_team"]
                )
            except Exception:
                skipped_error += 1
                continue

            n_frames = len(df_meta)
            if n_frames < self.spec.length:
                skipped_short += 1
                continue

            winning_team = int(df_meta["winning_team"].iloc[0])
            if winning_team == -1:
                skipped_label += 1
                continue

            # NOTE: we no longer filter by raw feature dimension.  The universal
            # mapper (frames_df_to_universal) accepts 1v1/2v2/3v3 and always
            # outputs self.feature_dim (133) by zero-padding absent car slots.
            self._index.append((rid, winning_team, n_frames))

        self._skipped = {
            "bad_label": skipped_label,
            "too_short": skipped_short,
            "missing": skipped_missing,
            "read_error": skipped_error,
            "wrong_mode": skipped_mode,
        }
        total_skipped = sum(self._skipped.values())
        print(
            f"[FrameSequenceDataset] Index built: {len(self._index)} replays "
            f"(skipped {total_skipped} — {self._skipped})",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _load_raw_windows(self, rid: str) -> np.ndarray:
        """Load or compute the (T_win, 45, feature_dim) raw-window array for one replay.

        Cache lookup order:
          1. On-disk npy (<frames_cache_root>/<rid>_frames.npy)  — fastest
          2. Compute from frames.parquet via frames_df_to_universal + save to disk

        The on-disk cache stores the universal 133-dim representation.  A cache
        file whose last dim does not match self.feature_dim (e.g. an old 88-dim
        cache) is ignored and recomputed.

        MEMORY NOTE: slices of a numpy array (arr[-N:]) are VIEWS that keep the
        full backing allocation alive.  We call .copy() after slicing so the LRU
        stores a compact contiguous array (max_windows rows) rather than a view
        into the full disk-loaded array (up to 200 rows × 4.8 MB).  Without this,
        each LRU entry retains the full 4.8 MB regardless of max_windows, causing
        the observed ~60 MB/replay leak when max_windows=50 (stores 4.8 MB instead
        of the expected 1.2 MB → 4× memory inflation × 5600 LRU = ~21 GB RSS).
        """
        if self.frames_cache_root is not None:
            cache_file = self.frames_cache_root / f"{rid}_frames.npy"
            if cache_file.exists():
                try:
                    arr = np.load(str(cache_file), allow_pickle=False, mmap_mode="r")
                    if arr.ndim == 3 and arr.shape[2] == self.feature_dim:
                        # Apply max_windows cap — the on-disk cache may have been
                        # saved with a larger max (e.g. 200) while this dataset
                        # instance uses a smaller train_max_windows (e.g. 50).
                        # mmap_mode='r': slicing [-N:] reads only the last N rows
                        # from disk (npy is row-major; last N windows are contiguous
                        # at file end) — ~6× less I/O than reading the full 4.8 MB.
                        # .copy() materialises only the slice and releases the mmap
                        # file handle — no file-handle leak, RAM stays bounded.
                        if arr.shape[0] > self.max_windows:
                            arr = arr[-self.max_windows :].copy()
                        else:
                            arr = arr[:].copy()
                        return arr
                    # Dimension mismatch (old 88-dim cache) — fall through to recompute.
                except Exception:
                    pass  # corrupted cache — recompute

        # Load parquet and map to universal 133-dim via frames_df_to_universal.
        frames_pq = self.processed_root / rid / "frames.parquet"
        frames_df = pd.read_parquet(frames_pq)
        feat = frames_df_to_universal(frames_df)  # (n_frames, 133)

        n = feat.shape[0]
        starts = range(0, n - self.spec.length + 1, self.spec.stride)
        windows = np.stack(
            [feat[s : s + self.spec.length] for s in starts],
            axis=0,
        )  # (n_windows, 45, 133) — np.stack always produces a contiguous copy

        # Cap to max_windows (take the last max_windows, matching v1 convention)
        if windows.shape[0] > self.max_windows:
            # np.stack already owns its memory; slice is a view — .copy() for safety
            windows = windows[-self.max_windows :].copy()

        if self.frames_cache_root is not None:
            cache_file = self.frames_cache_root / f"{rid}_frames.npy"
            try:
                np.save(str(cache_file), windows)
            except Exception:
                pass  # non-fatal

        return windows  # (T_win, 45, feature_dim) float32, contiguous

    def _load_match_feats(self, rid: str, t_win: int) -> np.ndarray:
        """Load or compute the (T_win, 23) match-feature sequence for a replay.

        Tries the v1 hmlt_cache_root first (identical format — reuse without
        recomputation).  Falls back to computing from the parquet.

        Args:
            rid:   Replay ID.
            t_win: Expected number of windows (must match raw windows).

        Returns:
            float32 array of shape (T_win, 23).
        """
        # Try existing v1 disk cache
        if self.hmlt_cache_root is not None:
            cache_file = self.hmlt_cache_root / f"{rid}.npy"
            if cache_file.exists():
                try:
                    seq = np.load(str(cache_file), allow_pickle=False, mmap_mode="r")
                    if seq.ndim == 2 and seq.shape[1] == MATCH_FEATURE_SIZE:
                        # Align length with raw windows (v1 uses same windowing)
                        # mmap_mode='r': reads only the last N rows from disk.
                        # .copy() materialises the slice and closes the mmap.
                        if seq.shape[0] > self.max_windows:
                            seq = seq[-self.max_windows :].copy()
                        else:
                            seq = seq[:].copy()
                        # If lengths still don't match (edge case), trim to t_win
                        if seq.shape[0] != t_win:
                            seq = seq[-t_win:].copy() if seq.shape[0] > t_win else seq
                        return seq.astype(np.float32)
                except Exception:
                    pass

        # Compute from parquet (slower)
        frames_pq = self.processed_root / rid / "frames.parquet"
        frames_df = pd.read_parquet(frames_pq)
        seq, _ = compute_match_feature_sequence(
            frames_df,
            window=self.spec.length,
            stride=self.spec.stride,
        )  # (n_windows, 23)
        if seq.shape[0] > self.max_windows:
            seq = seq[-self.max_windows :].copy()
        # Align length
        if seq.shape[0] != t_win:
            seq = seq[-t_win:].copy() if seq.shape[0] > t_win else seq
        return seq.astype(np.float32)

    def _load_replay(
        self, rid: str
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Load raw windows (and optionally match feats) for one replay.

        Returns:
            (frame_windows, match_feats_or_None)
            frame_windows: (T_win, 45, feature_dim)   feature_dim=133 (universal)
            match_feats:   (T_win, 23) or None
        """
        windows = self._load_raw_windows(rid)  # (T_win, 45, 88)
        t_win = windows.shape[0]

        if self.with_match_feats:
            match_feats = self._load_match_feats(rid, t_win)
        else:
            match_feats = None

        return windows, match_feats

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[Tensor, Optional[Tensor], int]:
        """Return (frame_windows_tensor, match_feats_tensor_or_None, label).

        frame_windows_tensor: float32 (T_win, 45, feature_dim)   feature_dim=133 (universal)
                              When VRAM cache is active, the tensor is already on
                              GPU (cuda device) — no CPU→GPU copy at collate time.
        match_feats_tensor:   float32 (T_win, 23) or None
        label:                int {0=blue wins, 1=orange wins}
        """
        rid, label, _n_frames = self._index[idx]

        if self._use_vram_cache:
            # VRAM path: load from npy cache (CPU) → upload once as fp16 → reuse
            fw_fp16 = self._vram_cache.get(rid)
            if fw_fp16 is None:
                windows, _mf_unused = self._load_replay(rid)
                fw_fp16 = self._vram_cache.put(rid, windows)
                # match_feats: still handled via system-RAM LRU below for hybrid
                if self.with_match_feats and _mf_unused is not None:
                    self._mem_cache.put(rid, (_mf_unused, None))
            else:
                # Already in VRAM — load match feats from RAM LRU if needed
                if self.with_match_feats:
                    _cached_mf = self._mem_cache.get(rid)
                    if _cached_mf is not None:
                        _mf_unused = _cached_mf[0]
                    else:
                        _mf_unused = self._load_match_feats(rid, fw_fp16.shape[0])
                        self._mem_cache.put(rid, (_mf_unused, None))
                else:
                    _mf_unused = None

            # Return fp32 — cast from fp16; tensor stays on CUDA device
            fw_tensor = fw_fp16.to(dtype=torch.float32)

            mf_tensor: Optional[Tensor]
            if self.with_match_feats and _mf_unused is not None:
                mf_tensor = torch.from_numpy(_mf_unused)
            else:
                mf_tensor = None

        else:
            # System-RAM path (original behaviour)
            cached = self._mem_cache.get(rid)
            if cached is not None:
                windows, match_feats = cached
            else:
                windows, match_feats = self._load_replay(rid)
                self._mem_cache.put(rid, (windows, match_feats))

            fw_tensor = torch.from_numpy(windows)  # (T_win, 45, 133) float32
            mf_tensor = (
                torch.from_numpy(match_feats) if match_feats is not None else None
            )  # (T_win, 23) float32 or None

        return fw_tensor, mf_tensor, label

    @property
    def skipped(self) -> dict[str, int]:
        return dict(self._skipped)

    @property
    def lru_size(self) -> int:
        if self._use_vram_cache:
            return self._vram_cache.size
        return self._mem_cache.size

    @property
    def vram_used_gb(self) -> float:
        """VRAM used by the cache in GB (0.0 when not using VRAM cache)."""
        if self._use_vram_cache and self._vram_cache is not None:
            return self._vram_cache.used_gb
        return 0.0


# ---------------------------------------------------------------------------
# collate_fn — pads variable-length sequences to batch max T_win
# ---------------------------------------------------------------------------

def collate_fn(
    batch: list[tuple[Tensor, Optional[Tensor], int]],
) -> tuple[Tensor, Optional[Tensor], Tensor, Tensor, Tensor]:
    """Pad a batch of (frame_windows, match_feats_or_None, label) items.

    VRAM-cache aware: when fw tensors are already on CUDA (from _VRAMCache),
    the padded output tensor is allocated on the same CUDA device — zero CPU→GPU
    copy for frame data.  match_feats remain on CPU (23-dim, negligible cost).

    Returns:
        frame_windows_padded  float32 (B, Tmax, 45, 133)  — on same device as fw_list[0]
        match_feats_padded    float32 (B, Tmax, 23) | None (if no match feats)
        lengths               long    (B,)   — actual T_win per item
        labels                long    (B,)   — {0, 1}
        pad_mask              bool    (B, Tmax) — True where position is PAD
    """
    fw_list, mf_list, label_list = zip(*batch)

    lengths = [fw.shape[0] for fw in fw_list]
    B = len(fw_list)
    T_max = max(lengths)
    n_frames = fw_list[0].shape[1]   # 45
    n_features = fw_list[0].shape[2]  # 133

    # Allocate on the same device as the input tensors (cuda if VRAM cache, else cpu)
    fw_device = fw_list[0].device
    fw_padded = torch.zeros(
        B, T_max, n_frames, n_features, dtype=torch.float32, device=fw_device
    )
    pad_mask = torch.ones(B, T_max, dtype=torch.bool, device=fw_device)

    for i, (fw, T) in enumerate(zip(fw_list, lengths)):
        fw_padded[i, :T] = fw[:T]
        pad_mask[i, :T] = False  # valid positions

    # Match feats (optional) — always CPU tensors (small, cheap to transfer)
    has_mf = mf_list[0] is not None
    if has_mf:
        n_mf = mf_list[0].shape[1]  # 23
        mf_padded: Optional[Tensor] = torch.zeros(
            B, T_max, n_mf, dtype=torch.float32
        )
        for i, (mf, T) in enumerate(zip(mf_list, lengths)):
            if mf is not None:
                mf_padded[i, :T] = mf[:T]  # type: ignore[index]
    else:
        mf_padded = None

    lengths_t = torch.tensor(lengths, dtype=torch.long)
    labels_t = torch.tensor(label_list, dtype=torch.long)

    return fw_padded, mf_padded, lengths_t, labels_t, pad_mask
