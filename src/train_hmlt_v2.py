"""HMLTv2 training script — 2-level Hierarchical Match-Level Transformer.

Trains HMLTv2 on raw per-frame physics (133-dim universal layout, any game mode)
via the FrameSequenceDataset.
The model's L1 encoder learns to embed each 3-second raw-frame window; the L2
Transformer reasons over the resulting window sequence to predict the binary
match winner.  Optionally uses the 23-dim hand-engineered match features
alongside the L1 embedding (hybrid mode).

Key design decisions carried over from train_hmlt.py (v1)
----------------------------------------------------------
  - Random 70/15/15 split at REPLAY level (seed=42): every window of a match lands
    in the same split, so no match is seen in both training and test.
  - Prefix augmentation (random fraction truncation per sample) — teaches the
    model to predict at any match progress and enables the "accuracy at 1/3
    match" dissertation metric.
  - Evaluation at configurable match-progress fractions (default: every 5%,
    0.05..1.0, 20 points).  Pass --eval-fractions to override.
  - ReduceLROnPlateau on val ROC-AUC + early stopping.

Usage:
    python -m src.train_hmlt_v2 \\
        --processed-root data/processed \\
        --limit 300 \\
        --epochs 2 \\
        --batch-size 16 \\
        --d-win 64 \\
        --d-model 96 \\
        --n-layers 2 \\
        --use-hybrid \\
        --frames-cache-root data/hmlt_frames_cache \\
        --out reports/hmlt_v2_smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

try:
    import psutil as _psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False

try:
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

from src.models.checkpoint_utils import (
    install_sigterm_handler,
    load_full_checkpoint,
    rotate_checkpoints,
    save_full_checkpoint,
)
from src.models.dataset import WindowSpec, list_processed_replays, make_splits
from src.models.frame_sequence_dataset import (
    FrameSequenceDataset,
    collate_fn as frame_collate_fn,
)
from src.models.hierarchical_winning_v2 import HMLTv2, count_params


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_hmlt_v2_{timestamp}.log"

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
    )
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)

    logger = logging.getLogger("hmlt_v2")
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers if the function is called more than once in tests
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Prefix truncation helpers — mirror train_hmlt.py exactly
# ---------------------------------------------------------------------------

def _truncate_to_fraction(
    fw: Tensor,
    fraction: float,
    mf: Optional[Tensor] = None,
) -> tuple[Tensor, Optional[Tensor]]:
    """Return the first max(1, round(fraction * T)) windows of fw (T, 45, 88).

    Also truncates match_feats (T, 23) in parallel when provided.
    """
    T = fw.shape[0]
    keep = max(1, round(fraction * T))
    return fw[:keep], (mf[:keep] if mf is not None else None)


def _prefix_collate(batch: list, rng: random.Random) -> tuple:
    """Collate with per-sample random prefix truncation for training augmentation.

    For each (fw, mf_or_None, label) sample draw p ~ Uniform(0.1, 1.0) and
    truncate the window sequence before padding.
    """
    augmented = []
    for fw, mf, label in batch:
        p = rng.uniform(0.1, 1.0)
        fw_trunc, mf_trunc = _truncate_to_fraction(fw, p, mf)
        augmented.append((fw_trunc, mf_trunc, label))
    return frame_collate_fn(augmented)


# ---------------------------------------------------------------------------
# Evaluation at fixed match-progress fractions
# ---------------------------------------------------------------------------

def _evaluate_at_fractions(
    model: HMLTv2,
    dataset: FrameSequenceDataset,
    fractions: list[float],
    device: torch.device,
    batch_size: int,
    log: logging.Logger,
) -> list[dict]:
    """For each fraction, truncate every test sequence and compute metrics.

    Returns a list of dicts (one per fraction) with keys:
        fraction, n_samples, accuracy, roc_auc, brier_score, log_loss
    (identical schema to train_hmlt.py for drop-in comparison).
    """
    if not _SKLEARN_OK:
        log.warning("scikit-learn not available — skipping metric computation")
        return []

    model.eval()

    # Pre-collect all items (no augmentation)
    all_fw: list[Tensor] = []
    all_mf: list[Optional[Tensor]] = []
    all_labels: list[int] = []
    for idx in range(len(dataset)):
        fw, mf, label = dataset[idx]
        all_fw.append(fw)
        all_mf.append(mf)
        all_labels.append(int(label))

    rows = []
    for frac in fractions:
        trunc_batch = []
        for fw, mf, lbl in zip(all_fw, all_mf, all_labels):
            fw_t, mf_t = _truncate_to_fraction(fw, frac, mf)
            trunc_batch.append((fw_t, mf_t, lbl))

        all_probs: list[float] = []
        for start in range(0, len(trunc_batch), batch_size):
            mini = trunc_batch[start : start + batch_size]
            fw_pad, mf_pad, _lengths, _labels_t, pad_mask = frame_collate_fn(mini)
            fw_pad = fw_pad.to(device)
            pad_mask = pad_mask.to(device)
            if mf_pad is not None:
                mf_pad = mf_pad.to(device)

            with torch.no_grad():
                logits = model(fw_pad, pad_mask, match_feats=mf_pad)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            all_probs.extend(probs)

        y_true = np.array(all_labels)
        y_prob = np.array(all_probs)
        y_pred = (y_prob >= 0.5).astype(int)

        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc = float("nan")

        row = {
            "fraction": frac,
            "n_samples": len(y_true),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "roc_auc": auc,
            "brier_score": float(brier_score_loss(y_true, y_prob)),
            "log_loss": float(
                log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7))
            ),
        }
        rows.append(row)
        log.info(
            f"  frac={frac:.2f}  acc={row['accuracy']:.4f}  "
            f"auc={row['roc_auc']:.4f}  brier={row['brier_score']:.4f}  "
            f"logloss={row['log_loss']:.4f}"
        )

    return rows


# ---------------------------------------------------------------------------
# Calibration helper (identical to train_hmlt.py)
# ---------------------------------------------------------------------------

def _calibration_data(
    model: HMLTv2,
    dataset: FrameSequenceDataset,
    device: torch.device,
    batch_size: int,
    n_bins: int = 10,
) -> list[dict]:
    """Collect reliability diagram data: predicted prob bins vs empirical rate."""
    model.eval()
    all_probs: list[float] = []
    all_labels: list[int] = []

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=frame_collate_fn,
        num_workers=0,
    )
    with torch.no_grad():
        for fw_pad, mf_pad, _lengths, labels_t, pad_mask in loader:
            fw_pad = fw_pad.to(device)
            pad_mask = pad_mask.to(device)
            if mf_pad is not None:
                mf_pad = mf_pad.to(device)
            logits = model(fw_pad, pad_mask, match_feats=mf_pad)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels_t.numpy().tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "bin_lo": lo,
                    "bin_hi": hi,
                    "n": 0,
                    "mean_prob": float((lo + hi) / 2),
                    "empirical_rate": float("nan"),
                }
            )
        else:
            rows.append(
                {
                    "bin_lo": float(lo),
                    "bin_hi": float(hi),
                    "n": n,
                    "mean_prob": float(y_prob[mask].mean()),
                    "empirical_rate": float(y_true[mask].mean()),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train_epoch(
    model: HMLTv2,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.BCEWithLogitsLoss,
    device: torch.device,
    train_rng: random.Random,
    use_prefix_aug: bool,
    log: logging.Logger,
    log_every: int = 50,
) -> float:
    """One training epoch; returns mean loss.

    VRAM-cache aware: when fw_pad is already on the target device (from
    _VRAMCache), fw_pad.to(device) is a no-op (same device, same dtype) —
    PyTorch skips the copy.  No special-casing needed here.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    t_batch_start = time.time()

    for raw_batch in loader:
        if use_prefix_aug:
            fw_pad, mf_pad, _lengths, labels_t, pad_mask = _prefix_collate(
                raw_batch, train_rng
            )
        else:
            fw_pad, mf_pad, _lengths, labels_t, pad_mask = frame_collate_fn(
                raw_batch
            )

        # .to(device) is a no-op when fw_pad is already on CUDA (VRAM cache path)
        fw_pad = fw_pad.to(device)
        pad_mask = pad_mask.to(device)
        labels_f = labels_t.float().to(device)
        if mf_pad is not None:
            mf_pad = mf_pad.to(device)

        optimizer.zero_grad()
        logits = model(fw_pad, pad_mask, match_feats=mf_pad)
        loss = criterion(logits, labels_f)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if n_batches % log_every == 0:
            elapsed = time.time() - t_batch_start
            log.info(
                f"    batch {n_batches}/{len(loader)} | "
                f"loss={loss.item():.4f} | "
                f"last {log_every} batches: {elapsed:.1f}s "
                f"({elapsed/log_every*1000:.0f}ms/batch)"
            )
            t_batch_start = time.time()

    return total_loss / max(n_batches, 1)


def _val_auc(
    model: HMLTv2,
    loader: DataLoader,
    criterion: nn.BCEWithLogitsLoss,
    device: torch.device,
) -> tuple[float, float]:
    """Validation pass — returns (mean_loss, roc_auc)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_probs: list[float] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for fw_pad, mf_pad, _lengths, labels_t, pad_mask in loader:
            fw_pad = fw_pad.to(device)
            pad_mask = pad_mask.to(device)
            labels_f = labels_t.float().to(device)
            if mf_pad is not None:
                mf_pad = mf_pad.to(device)
            logits = model(fw_pad, pad_mask, match_feats=mf_pad)
            loss = criterion(logits, labels_f)
            total_loss += loss.item()
            n_batches += 1
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels_t.numpy().tolist())

    mean_loss = total_loss / max(n_batches, 1)
    if _SKLEARN_OK and len(set(all_labels)) > 1:
        auc = float(roc_auc_score(np.array(all_labels), np.array(all_probs)))
    else:
        auc = float("nan")
    return mean_loss, auc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train HMLTv2 — 2-level win-prediction model over raw frames"
    )
    p.add_argument("--processed-root", default="data/processed")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Subsample N replays for fast dev runs (None = all)",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    # L1 / L2 architecture
    p.add_argument(
        "--d-win",
        type=int,
        default=128,
        help="L1 window-embedding dimension (FrameWindowEncoder output size)",
    )
    p.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="L2 Transformer hidden dimension",
    )
    p.add_argument("--n-layers", type=int, default=4, help="L2 Transformer layers")
    p.add_argument("--n-heads", type=int, default=8, help="L2 attention heads")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--l1-kernel", type=int, default=3, help="L1 CNN kernel size")
    p.add_argument(
        "--use-hybrid",
        action="store_true",
        default=True,
        help="Concat 23-dim engineered features to L1 embedding (recommended)",
    )
    p.add_argument(
        "--no-hybrid",
        dest="use_hybrid",
        action="store_false",
        help="Disable hybrid path (raw physics only)",
    )
    p.add_argument("--window-len", type=int, default=45,
                   help="Frames per window for the L1 encoder. Keep = 3s of match: 45 at 15Hz, 90 at 30Hz "
                        "(fair Hz ablation = same DURATION, different resolution).")
    p.add_argument("--stride", type=int, default=15,
                   help="Window stride in frames (15=1s at 15Hz, 30=1s at 30Hz). Use a separate --frames-cache-root per Hz.")
    p.add_argument("--max-windows", type=int, default=200,
                   help="Max windows per replay for val/test evaluation (keep 200 for full temporal coverage)")
    p.add_argument("--train-max-windows", type=int, default=50,
                   help="Max windows per replay during TRAINING only. "
                        "50 windows = 150s of match = covers all prediction fractions "
                        "including 75%% on a typical 3-min match. "
                        "Reduces per-npy slice and collation tensor size by 4x vs max=200. "
                        "Val/test use --max-windows (200) for full temporal coverage.")
    p.add_argument("--lru-capacity", type=int, default=0,
                   help="LRU in-memory cache size. 0 = auto (= len(train_ids) + 200, "
                        "covers full dataset in RAM on first epoch to eliminate cold I/O "
                        "on subsequent epochs). Set explicitly to limit RAM usage.")
    p.add_argument("--eval-lru", type=int, default=256,
                   help="LRU cache size for val AND test datasets (replays). "
                        "Each replay at max-windows=200 costs ~4.8MB; 256 = ~1.2GB. "
                        "Reduce to 128 if RAM is tight. Eval reads each replay once/epoch "
                        "so a small LRU is fine (no repeated access benefit).")
    p.add_argument("--val-max-windows", type=int, default=0,
                   help="Max windows for val/test datasets. 0 = use --max-windows value. "
                        "Set to 100 to halve eval RAM footprint (4.8MB→2.4MB/replay) "
                        "while still covering all fractions [0.1..1.0].")
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--cache-root",
        default=None,
        help="v1 hmlt_cache dir for 23-dim match-feature npy files (hybrid reuse)",
    )
    p.add_argument(
        "--frames-cache-root",
        default="data/hmlt_frames_cache_univ",
        help="Dir for universal-frame (T,45,input_dim) npy cache files",
    )
    p.add_argument("--out", default="reports/hmlt_v2")
    p.add_argument("--seed", type=int, default=42)
    # --- Resumability ---
    p.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from last_checkpoint.pt in --ckpt-dir",
    )
    p.add_argument(
        "--ckpt-dir",
        default=None,
        help="Directory for checkpoints (default: <--out>/checkpoints)",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument(
        "--eval-fractions",
        default=None,
        help=(
            "Comma-separated match-progress fractions at which to evaluate "
            "accuracy/AUC on the test set.  E.g. '0.1,0.5,1.0'. "
            "Default: every 5%% from 0.05 to 1.0 (20 points)."
        ),
    )
    p.add_argument(
        "--input-dim",
        type=int,
        default=133,
        help="L1 input feature dimension per frame (133=universal default; 88=old 2v2-only)",
    )
    # VRAM cache
    p.add_argument(
        "--cache-device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Where to store the frame-window cache: 'cpu' (system-RAM numpy LRU, default) "
             "or 'cuda' (fp16 VRAM LRU — frees system RAM, eliminates CPU→GPU copy). "
             "Requires CUDA. Use with --vram-cache-gb to set the VRAM budget.",
    )
    p.add_argument(
        "--vram-cache-gb",
        type=float,
        default=12.0,
        help="VRAM budget in GB for the fp16 VRAM cache (used when --cache-device=cuda). "
             "Capacity: budget / (max_windows×45×133×2 bytes). "
             "E.g. 13 GB + 30 windows ≈ 38K 3v3 replays. Default 12.0.",
    )
    # Mode filter
    p.add_argument(
        "--mode",
        default=None,
        choices=["1v1", "2v2", "3v3"],
        help="Keep only replays of the specified game mode. "
             "Detection uses feature index 132 (n_total_norm) from the npy cache "
             "(3v3→1.0, 2v2→0.667, 1v1→0.333), with parquet schema fallback. "
             "Default: all modes.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_DEFAULT_EVAL_FRACTIONS: list[float] = [round(i * 0.05, 10) for i in range(1, 21)]
"""Default evaluation fractions: every 5% from 0.05 to 1.0 (20 points)."""


def _parse_eval_fractions(raw: str | None) -> list[float]:
    """Parse --eval-fractions string to a sorted list of floats in (0, 1].

    Accepts comma-separated floats, e.g. '0.05,0.1,0.5,1.0'.
    Returns the default 20-point list when raw is None or empty.
    """
    if not raw:
        return list(_DEFAULT_EVAL_FRACTIONS)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    fractions = []
    for p in parts:
        try:
            v = float(p)
        except ValueError as exc:
            raise ValueError(f"--eval-fractions: invalid float '{p}'") from exc
        if not (0.0 < v <= 1.0):
            raise ValueError(
                f"--eval-fractions: fraction {v} out of range (must be in (0, 1])"
            )
        fractions.append(v)
    return sorted(set(fractions))


def main() -> None:
    args = _parse_args()

    script_dir = Path(__file__).resolve().parent.parent
    processed_root = script_dir / args.processed_root
    out_dir = script_dir / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = script_dir / "logs"

    log = _setup_logging(log_dir)
    log.info("=" * 70)
    log.info("HMLTv2 training — 2-level raw-frame Hierarchical Match Transformer")
    log.info(f"args: {vars(args)}")

    # ---- Seed ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    train_rng = random.Random(args.seed)

    # ---- Device ----
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(
            f"GPU: {torch.cuda.get_device_name(device)} | "
            f"VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB"
        )

    if not _SKLEARN_OK:
        log.error("scikit-learn not found")
        sys.exit(1)

    # ---- Discover replays ----
    all_ids = list_processed_replays(processed_root)
    log.info(f"Found {len(all_ids)} replays with frames+labels parquets")

    if args.limit is not None:
        rng_limit = np.random.default_rng(args.seed)
        idx_sub = rng_limit.choice(
            len(all_ids), size=min(args.limit, len(all_ids)), replace=False
        )
        all_ids = [all_ids[i] for i in sorted(idx_sub)]
        log.info(f"Subsampled to {len(all_ids)} replays (--limit {args.limit})")

    # Identical split to v1 — same seed, same function → same train/val/test IDs
    log.info("Using random 70/15/15 split (seed=42, identical to v1/baselines)")
    train_ids, val_ids, test_ids = make_splits(
        all_ids, train_frac=0.70, val_frac=0.15, seed=args.seed
    )
    log.info(
        f"Split: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}"
    )

    spec = WindowSpec(length=args.window_len, stride=args.stride)
    # --frames-cache-root none|"" => no on-disk npy cache (VRAM only). Avoids filling the
    # disk: the npy is float32 (T,45,133) ~9 MB/replay x 15K = ~138 GB. With
    # cache-device=cuda everything stays in VRAM after epoch 1 (epoch 1 reads the parquet).
    if str(args.frames_cache_root).strip().lower() in ("", "none"):
        frames_cache_root = None
    else:
        frames_cache_root = script_dir / args.frames_cache_root
    hmlt_cache_root = (script_dir / args.cache_root) if args.cache_root else None

    # ---- LRU capacity ----
    # Auto = full train set + headroom so epoch 1 warms the entire cache and
    # subsequent epochs have zero cold I/O (zero cold disk reads).
    # In-memory size per replay: train_max_windows * 45 * 133 * 4 bytes.
    # With train_max_windows=50 (default): 50*45*133*4 = 1.2 MB per replay.
    # 20000 replays * 1.2 MB = 24 GB — fits in 32 GB RAM with headroom.
    # Cap auto at 20000 to be safe.  Disk read is always 4.7 MB (full npy), but
    # in-memory footprint is 1.2 MB after slicing to train_max_windows.
    if args.lru_capacity == 0:
        lru_mb_per_replay = (args.train_max_windows * 45 * args.input_dim * 4) / 1e6
        auto_lru = min(len(train_ids) + 200, 20000)
        log.info(
            f"LRU capacity: auto={auto_lru} "
            f"(covers {'full' if auto_lru >= len(train_ids) else 'partial'} train set; "
            f"~{auto_lru * lru_mb_per_replay:.0f} MB peak RAM at "
            f"{lru_mb_per_replay:.1f} MB/replay)"
        )
        lru_cap = auto_lru
    else:
        lru_cap = args.lru_capacity
        log.info(f"LRU capacity: {lru_cap} (explicit)")

    train_max_win = args.train_max_windows
    eval_max_win = args.max_windows if args.val_max_windows == 0 else args.val_max_windows
    eval_lru = args.eval_lru
    log.info(
        f"max_windows: train={train_max_win}  val/test={eval_max_win}  "
        f"(train cap reduces tensor size by {eval_max_win // train_max_win}x)"
    )
    eval_mb_per_replay = (eval_max_win * 45 * args.input_dim * 4) / 1e6
    log.info(
        f"eval LRU: {eval_lru} replays × {eval_mb_per_replay:.1f} MB = "
        f"~{eval_lru * eval_mb_per_replay:.0f} MB per split (val + test = "
        f"~{2 * eval_lru * eval_mb_per_replay:.0f} MB)"
    )

    # ---- VRAM cache config ----
    use_vram = (args.cache_device == "cuda")
    if use_vram and args.num_workers > 0:
        # VRAM-resident cache (main process) + forked DataLoader workers on Linux
        # => "Cannot re-initialize CUDA in forked subprocess". Workers are useless with
        # a VRAM cache anyway (no CPU I/O after warm-up), so force serial loading.
        log.warning(
            "cache-device=cuda is incompatible with num_workers>0 (CUDA+fork). "
            f"Forcing num_workers=0 (was {args.num_workers})."
        )
        args.num_workers = 0
    if use_vram:
        mb_per_replay_fp16 = (train_max_win * 45 * args.input_dim * 2) / 1e6
        bytes_per_replay_fp16 = train_max_win * 45 * args.input_dim * 2
        theoretical_capacity = int(args.vram_cache_gb * 1e9 / bytes_per_replay_fp16)
        log.info(
            f"VRAM cache: budget={args.vram_cache_gb:.1f} GB | "
            f"fp16 per-replay={mb_per_replay_fp16:.2f} MB "
            f"({train_max_win} windows × 45 × {args.input_dim} × 2 bytes) | "
            f"theoretical capacity={theoretical_capacity:,} replays at train_max_windows={train_max_win}"
        )
    else:
        log.info("Cache: system-RAM LRU (CPU numpy fp32)")

    # ---- Datasets ----
    def _make_ds(
        ids: list[str],
        split: str,
        max_win: int,
        lru: int,
        use_vram_cache: bool = False,
        vram_gb: float = 12.0,
    ) -> FrameSequenceDataset:
        _cache_dev = "cuda" if use_vram_cache else "cpu"
        log.info(
            f"Building {split} dataset ({len(ids)} replays, max_windows={max_win}, "
            f"lru={lru}, cache={'VRAM-fp16' if use_vram_cache else 'RAM-fp32'}"
            f"{', mode='+args.mode if args.mode else ''})..."
        )
        ds = FrameSequenceDataset(
            processed_root=processed_root,
            replay_ids=ids,
            spec=spec,
            max_windows=max_win,
            frames_cache_root=frames_cache_root,
            hmlt_cache_root=hmlt_cache_root,
            with_match_feats=args.use_hybrid,
            lru_capacity=lru,
            cache_device=_cache_dev,
            vram_cache_gb=vram_gb,
            mode=args.mode,
        )
        log.info(
            f"  {split} dataset: {len(ds)} replays | skipped={ds.skipped}"
        )
        return ds

    train_ds = _make_ds(train_ids, "train", train_max_win, lru_cap,
                        use_vram_cache=use_vram, vram_gb=args.vram_cache_gb)
    # Val/test: always CPU cache — VRAM budget is reserved for train data
    val_ds = _make_ds(val_ids, "val", eval_max_win, eval_lru)
    test_ds = _make_ds(test_ids, "test", eval_max_win, eval_lru)

    if len(train_ds) == 0:
        log.error(
            "Train dataset is empty — check processed_root and feature_dim filter."
        )
        sys.exit(1)

    # Baseline memory snapshot (before any training — cache is empty here)
    if _PSUTIL_OK:
        rss_baseline = _psutil.Process().memory_info().rss / 1e9
        log.info(f"Baseline sys RAM (RSS) after dataset init: {rss_baseline:.2f} GB")
    if device.type == "cuda":
        log.info(
            f"Baseline VRAM allocated: "
            f"{torch.cuda.memory_allocated(device)/1e9:.3f} GB  "
            f"reserved: {torch.cuda.memory_reserved(device)/1e9:.3f} GB"
        )

    # Class balance
    train_labels = [lbl for _, lbl, _ in train_ds._index]
    blue_frac = float(np.mean([l == 0 for l in train_labels]))
    log.info(
        f"Train class balance: blue_wins={blue_frac:.3f} "
        f"orange_wins={1-blue_frac:.3f}"
    )
    test_labels_all = [lbl for _, lbl, _ in test_ds._index]
    test_blue_frac = float(np.mean([l == 0 for l in test_labels_all]))
    log.info(
        f"Test  class balance: blue_wins={test_blue_frac:.3f} "
        f"orange_wins={1-test_blue_frac:.3f}"
    )

    # ---- DataLoaders ----
    # Training: raw list (no collate) so _prefix_collate can do per-sample truncation.
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: b,  # raw list — collated inside _train_epoch
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=frame_collate_fn,
        num_workers=args.num_workers,
    )

    # ---- Model ----
    # max_len for sinusoidal PE must cover the longest sequence seen at eval time
    # (val/test use eval_max_win; train uses train_max_win which is typically smaller).
    pe_max_len = max(eval_max_win, train_max_win) + 16
    model = HMLTv2(
        n_frames=spec.length,           # 45
        n_features=args.input_dim,      # 133 universal (default) or 88 for old ckpt compat
        d_win=args.d_win,
        n_match_feats=23,
        use_hybrid=args.use_hybrid,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        max_len=pe_max_len,
        l1_kernel=args.l1_kernel,
    ).to(device)
    log.info(f"Model input_dim={args.input_dim} (L1 n_features)")

    n_params = count_params(model)
    log.info(f"HMLTv2 parameters: {n_params:,}")
    log.info(f"  use_hybrid={args.use_hybrid}  d_win={args.d_win}  "
             f"d_model={args.d_model}  n_layers={args.n_layers}  "
             f"n_heads={args.n_heads}")

    # ---- Optimizer + scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    criterion = nn.BCEWithLogitsLoss()

    # ---- Checkpoint dir ----
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt_path = ckpt_dir / "last_checkpoint.pt"
    best_model_path = out_dir / "hmlt_v2_best.pt"

    # ---- Resume ----
    start_epoch = 1
    best_val_auc = -1.0
    best_epoch = -1
    patience_counter = 0

    if args.resume and last_ckpt_path.exists():
        log.info(f"--resume: loading checkpoint from {last_ckpt_path}")
        start_epoch, loaded_best, extra = load_full_checkpoint(
            last_ckpt_path, model, optimizer, device
        )
        if loaded_best is not None:
            best_val_auc = loaded_best
        if extra:
            best_epoch = extra.get("best_epoch", best_epoch)
            patience_counter = extra.get("patience_counter", 0)
        log.info(
            f"  Resumed at epoch {start_epoch}  best_val_auc={best_val_auc:.4f}  "
            f"patience={patience_counter}"
        )
    elif args.resume:
        log.info(
            f"--resume set but no checkpoint found at {last_ckpt_path}. Starting from scratch."
        )

    # ---- SIGTERM handler — saves last_checkpoint before kill ----
    # _cur tracks the epoch variable inside the loop via a mutable container
    _cur: list[int] = [start_epoch - 1]

    def _emergency_save() -> None:
        log.info("Emergency checkpoint save triggered by signal...")
        save_full_checkpoint(
            last_ckpt_path, model, optimizer,
            epoch=_cur[0],
            best_metric=best_val_auc,
            extra={"best_epoch": best_epoch, "patience_counter": patience_counter},
        )

    install_sigterm_handler(_emergency_save)

    for epoch in range(start_epoch, args.epochs + 1):
        _cur[0] = epoch
        t0 = time.time()
        train_loss = _train_epoch(
            model, train_loader, optimizer, criterion, device,
            train_rng, use_prefix_aug=True, log=log,
        )
        val_loss, val_auc = _val_auc(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        if device.type == "cuda":
            gpu_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
            gpu_reserved_gb = torch.cuda.memory_reserved(device) / 1e9
            gpu_note = f"  peak_gpu_alloc={gpu_mem_gb:.2f}GB reserved={gpu_reserved_gb:.2f}GB"
        else:
            gpu_note = ""

        # System RAM (RSS) — sanity check: should stay LOW when VRAM cache is active
        if _PSUTIL_OK:
            rss_gb = _psutil.Process().memory_info().rss / 1e9
            ram_note = f"  sys_ram={rss_gb:.2f}GB"
        else:
            ram_note = ""

        # VRAM cache utilization
        if use_vram and hasattr(train_ds, "vram_used_gb"):
            cache_note = (
                f"  vram_cache={train_ds.vram_used_gb:.2f}/"
                f"{args.vram_cache_gb:.1f}GB "
                f"({train_ds.lru_size} replays)"
            )
        else:
            cache_note = f"  lru={train_ds.lru_size}"

        log.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_auc={val_auc:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"[{elapsed:.1f}s]{gpu_note}{ram_note}{cache_note}"
        )

        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            log.info(f"  -> New best val AUC {best_val_auc:.4f} — model saved")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info(f"Early stop at epoch {epoch} (patience={args.patience})")
                # Save before early-stop exit so --resume can continue from here
                save_full_checkpoint(
                    last_ckpt_path, model, optimizer, epoch=epoch,
                    best_metric=best_val_auc,
                    extra={"best_epoch": best_epoch, "patience_counter": patience_counter},
                )
                break

        # ---- Per-epoch checkpoint (full, resumable) ----
        save_full_checkpoint(
            last_ckpt_path, model, optimizer, epoch=epoch,
            best_metric=best_val_auc,
            extra={"best_epoch": best_epoch, "patience_counter": patience_counter},
        )
        # Also save a named epoch file for rotate logic
        epoch_ckpt = ckpt_dir / f"epoch_{epoch:04d}.pt"
        save_full_checkpoint(
            epoch_ckpt, model, optimizer, epoch=epoch,
            best_metric=best_val_auc,
            extra={"best_epoch": best_epoch, "patience_counter": patience_counter},
        )
        rotate_checkpoints(ckpt_dir, keep_last=2)
        log.info(f"  Checkpoint saved (epoch {epoch})")

    log.info(f"Training done. Best val AUC={best_val_auc:.4f} at epoch {best_epoch}")

    # ---- Reload best model for evaluation ----
    model.load_state_dict(
        torch.load(best_model_path, map_location=device, weights_only=True)
    )
    model.eval()

    # ---- Evaluation at configurable match-progress fractions ----
    eval_fractions = _parse_eval_fractions(args.eval_fractions)
    log.info(
        f"Evaluating on test set at {len(eval_fractions)} match-progress fractions "
        f"(first={eval_fractions[0]:.2f}, last={eval_fractions[-1]:.2f}):"
    )
    rows = _evaluate_at_fractions(
        model, test_ds, eval_fractions, device, args.batch_size, log
    )

    # Save CSV
    csv_path = out_dir / "accuracy_vs_progress.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"Saved accuracy_vs_progress.csv -> {csv_path}")

        # Print table
        header = (
            f"{'frac':>6}  {'n':>6}  {'acc':>8}  "
            f"{'auc':>8}  {'brier':>8}  {'logloss':>8}"
        )
        log.info("\n" + header)
        log.info("-" * len(header))
        for r in rows:
            log.info(
                f"{r['fraction']:6.2f}  {r['n_samples']:6d}  "
                f"{r['accuracy']:8.4f}  {r['roc_auc']:8.4f}  "
                f"{r['brier_score']:8.4f}  {r['log_loss']:8.4f}"
            )

    # ---- Calibration ----
    calib_rows = _calibration_data(model, test_ds, device, args.batch_size)
    calib_path = out_dir / "calibration.csv"
    if calib_rows:
        with open(calib_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=calib_rows[0].keys())
            writer.writeheader()
            writer.writerows(calib_rows)
        log.info(f"Saved calibration.csv -> {calib_path}")

    # ---- Peak GPU memory note ----
    if device.type == "cuda":
        peak_gpu_gb = torch.cuda.max_memory_allocated(device) / 1e9
        log.info(f"Peak GPU memory: {peak_gpu_gb:.2f} GB")
    else:
        peak_gpu_gb = None

    # ---- metrics.json ----
    metrics: dict = {
        "model": "HMLTv2",
        "best_val_auc": best_val_auc,
        "best_epoch": best_epoch,
        "n_params": n_params,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "train_blue_win_frac": blue_frac,
        "test_blue_win_frac": test_blue_frac,
        "peak_gpu_gb": peak_gpu_gb,
        "args": vars(args),
        "accuracy_vs_progress": rows,
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Saved metrics.json -> {metrics_path}")
    log.info("Done.")


if __name__ == "__main__":
    main()
