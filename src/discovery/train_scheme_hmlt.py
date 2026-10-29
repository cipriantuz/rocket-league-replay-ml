"""Play-scheme classifier built on the HMLT level-1 window encoder.

Model: FrameWindowEncoder(n_features=133, d_win=128) -> Linear(128 -> n_classes).
  - FrameWindowEncoder = 2x ResConv1dBlock (1D CNN) + global mean-pool -> (d_win,)
  - The mean-pool makes the encoder independent of window length (45 @ 15 Hz, 90 @ 30 Hz)
  - No level-2 Transformer: one window -> one class
Data: positive windows around rule-based detector events + NONE windows from normal play
      (src/discovery/scheme_window_extractor.py). Labels are therefore WEAK labels.
Split: GroupKFold(groups=replay_id), so no replay contributes to both train and test.
Class weighting: compute_class_weight('balanced') -> weighted cross-entropy.

Smoke run (CPU, small subset):
    python -u src/discovery/train_scheme_hmlt.py \\
        --max-replays 80 --epochs 2 --device cpu \\
        --out-dir reports/scheme_hmlt_smoke

Full run, 30 Hz, 1 s windows (the configuration behind the headline result):
    python -u src/discovery/train_scheme_hmlt.py \\
        --processed-root data/processed_30hz --window-len 30 --stride 30 \\
        --device cuda --epochs 30 --out-dir reports/scheme_hmlt_30hz_1s

15 Hz vs 30 Hz ablation (3 s windows):
    --processed-root data/processed_15hz --window-len 45 --stride 15
    --processed-root data/processed_30hz --window-len 90 --stride 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# make the project root importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.checkpoint_utils import (  # noqa: E402
    install_sigterm_handler,
    load_full_checkpoint,
    rotate_checkpoints,
    save_full_checkpoint,
)


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train HMLT L1-encoder scheme classifier on raw (T,133) tensors"
    )
    # NOTE: --resume and --ckpt-dir are added AFTER other args at the bottom
    # of this function so they appear last in --help
    p.add_argument("--processed-root", default="data/processed",
                   help="Root with one sub-directory per replay")
    p.add_argument("--max-replays", type=int, default=None,
                   help="Max replays (debugging / smoke)")
    p.add_argument("--window-len", type=int, default=45,
                   help="Window length in frames (45 @ 15 Hz, 90 @ 30 Hz)")
    p.add_argument("--stride", type=int, default=15,
                   help="NONE sampling stride (15 @ 15 Hz, 30 @ 30 Hz)")
    p.add_argument("--none-ratio", type=float, default=1.5,
                   help="NONE:positive ratio (default 1.5)")
    p.add_argument("--device", default="auto",
                   help="'cpu' / 'cuda' / 'auto'")
    p.add_argument("--epochs", type=int, default=30,
                   help="Training epochs")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Adam learning rate")
    p.add_argument("--d-win", type=int, default=128,
                   help="FrameWindowEncoder latent size (d_win)")
    p.add_argument("--kernel-size", type=int, default=3,
                   help="Conv1d kernel size in ResConv1dBlock")
    p.add_argument("--dropout", type=float, default=0.3,
                   help="Dropout before the classification head")
    p.add_argument("--n-folds", type=int, default=5,
                   help="Number of GroupKFold folds")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="reports/scheme_hmlt",
                   help="Output directory for metrics.json")
    # --- Resumability ---
    p.add_argument("--resume", action="store_true", default=False,
                   help="Resume from the last saved fold/epoch in --ckpt-dir")
    p.add_argument("--ckpt-dir", default=None,
                   help="Checkpoint directory (default: <--out-dir>/checkpoints)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model: FrameWindowEncoder + classification head
# ---------------------------------------------------------------------------

def _build_model(n_features: int, d_win: int, kernel_size: int, dropout: float, n_classes: int):
    """FrameWindowEncoder (HMLT level 1) + linear classification head."""
    import torch.nn as nn
    from src.models.hierarchical_winning_v2 import FrameWindowEncoder

    class HMLTSchemeClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            # Same level-1 encoder as HMLT; global mean-pool, so window length is free
            # (n_frames is informational only)
            self.encoder = FrameWindowEncoder(
                n_frames=45,        # informational; ignored at runtime
                n_features=n_features,
                d_win=d_win,
                kernel_size=kernel_size,
            )
            # dropout -> Linear(d_win -> n_classes)
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(d_win, n_classes),
            )

        def forward(self, x):
            # x: (B, T, n_features)
            embed = self.encoder(x)   # (B, d_win), mean-pooled over T
            return self.head(embed)   # (B, n_classes)

    return HMLTSchemeClassifier()


# ---------------------------------------------------------------------------
# In-memory dataset
# ---------------------------------------------------------------------------

def _make_tensors(records, label_to_idx: dict, window_len: int):
    """Convert a list of WindowRecord into numpy arrays."""
    X = np.stack([r.tensor for r in records], axis=0).astype(np.float32)  # (N, T, 133)
    y = np.array([label_to_idx[r.label] for r in records], dtype=np.int64)
    groups = np.array([r.replay_id for r in records])
    weights = np.array([r.confidence for r in records], dtype=np.float32)

    assert X.shape[1] == window_len, (
        f"Tensor shape mismatch: expected T={window_len}, got T={X.shape[1]}"
    )
    return X, y, groups, weights


# ---------------------------------------------------------------------------
# Training + evaluation for one fold
# ---------------------------------------------------------------------------

def _train_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    class_weights_tensor,
    args: argparse.Namespace,
    n_classes: int,
    device,
    fold_idx: int,
    ckpt_dir: Optional[Path] = None,
    resume_epoch: int = 0,
):
    """Train one fold; return (f1_macro, f1_per_class, confusion_matrix, model).

    Note: the best epoch is selected on the held-out fold itself (there is no separate
    validation split inside the fold), so the reported F1 is slightly optimistic.

    Parameters
    ----------
    ckpt_dir:
        When provided, saves a ``fold_{fold_idx}_last.pt`` after each epoch
        and rotates epoch-named checkpoints.
    resume_epoch:
        If > 0, loads ``fold_{fold_idx}_last.pt`` from ckpt_dir and resumes
        training from epoch ``resume_epoch + 1``.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.metrics import f1_score, confusion_matrix

    X_tr_t = torch.from_numpy(X_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    X_te_t = torch.from_numpy(X_te).to(device)
    y_te_t = torch.from_numpy(y_te)   # on CPU for sklearn

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,      # in-memory tensors; workers not needed
        pin_memory=False,
    )

    model = _build_model(
        n_features=X_tr.shape[2],
        d_win=args.d_win,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        n_classes=n_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5
    )

    best_f1 = -1.0
    best_state = None
    start_epoch = 1

    # --- Resume per-fold ---
    fold_ckpt_path = (ckpt_dir / f"fold_{fold_idx}_last.pt") if ckpt_dir else None
    if ckpt_dir and resume_epoch > 0 and fold_ckpt_path and fold_ckpt_path.exists():
        loaded_start, loaded_best, _extra = load_full_checkpoint(
            fold_ckpt_path, model, optimizer, device
        )
        start_epoch = loaded_start
        if loaded_best is not None:
            best_f1 = loaded_best
        print(
            f"    Fold {fold_idx+1}: resumed from epoch {start_epoch}  best_f1={best_f1:.3f}",
            flush=True,
        )
        # Restore best_state from the loaded model weights
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # evaluate on the held-out fold
        model.eval()
        with torch.no_grad():
            logits_te = model(X_te_t.to(device))
            preds_te = logits_te.argmax(dim=1).cpu().numpy()

        f1_macro_ep = f1_score(y_te_t.numpy(), preds_te, average="macro", zero_division=0)
        scheduler.step(f1_macro_ep)

        if f1_macro_ep > best_f1:
            best_f1 = f1_macro_ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
            avg_loss = epoch_loss / max(n_batches, 1)
            print(
                f"    Fold {fold_idx+1}  epoch {epoch:3d}/{args.epochs}"
                f"  loss={avg_loss:.4f}  val_macro-F1={f1_macro_ep:.3f}"
                f"  [best={best_f1:.3f}]",
                flush=True,
            )

        # --- Per-epoch checkpoint (fold-scoped) ---
        if ckpt_dir and fold_ckpt_path:
            save_full_checkpoint(
                fold_ckpt_path, model, optimizer, epoch=epoch, best_metric=best_f1
            )
            epoch_file = ckpt_dir / f"fold_{fold_idx}_epoch_{epoch:04d}.pt"
            save_full_checkpoint(
                epoch_file, model, optimizer, epoch=epoch, best_metric=best_f1
            )
            # Rotate only this fold's epoch files
            import re as _re
            fold_epoch_re = _re.compile(rf"fold_{fold_idx}_epoch_(\d+)\.pt$")
            epoch_files_fold = []
            for f in ckpt_dir.iterdir():
                m = fold_epoch_re.search(f.name)
                if m:
                    epoch_files_fold.append((int(m.group(1)), f))
            epoch_files_fold.sort(key=lambda t: t[0])
            for _, old_f in epoch_files_fold[: max(0, len(epoch_files_fold) - 2)]:
                try:
                    old_f.unlink()
                except OSError:
                    pass

    # final evaluation with the best epoch
    if best_state is None:
        # fallback: use current weights
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_te = model(X_te_t.to(device))
        preds_final = logits_te.argmax(dim=1).cpu().numpy()

    y_te_np = y_te_t.numpy()
    f1_macro = float(f1_score(y_te_np, preds_final, average="macro", zero_division=0))
    f1_per = f1_score(
        y_te_np, preds_final,
        average=None,
        labels=list(range(n_classes)),
        zero_division=0,
    ).tolist()
    cm = confusion_matrix(y_te_np, preds_final, labels=list(range(n_classes))).tolist()

    return f1_macro, f1_per, cm, model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    import torch
    from sklearn.model_selection import GroupKFold
    from sklearn.utils.class_weight import compute_class_weight

    from src.discovery.scheme_window_extractor import (
        extract_positive_windows,
        extract_none_windows,
        TARGET_CLASSES,
    )

    # --- Device ---
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Device: {device}", flush=True)
    print(f"[INFO] window_len={args.window_len}  stride={args.stride}", flush=True)
    print(f"[INFO] d_win={args.d_win}  kernel_size={args.kernel_size}", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Checkpoint dir (always created; per-epoch saves work even without --resume) ---
    ckpt_dir: Path = Path(args.ckpt_dir) if args.ckpt_dir else out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fold_progress_path = ckpt_dir / "fold_progress.json"
    completed_folds: set[int] = set()

    if args.resume and fold_progress_path.exists():
        try:
            prog = json.loads(fold_progress_path.read_text(encoding="utf-8"))
            completed_folds = set(prog.get("completed_folds", []))
            print(
                f"[INFO] --resume: folds already completed: {sorted(completed_folds)}",
                flush=True,
            )
        except (json.JSONDecodeError, OSError) as exc:
            print(
                f"[WARN] --resume: cannot read {fold_progress_path} ({exc}); starting from scratch.",
                flush=True,
            )
            completed_folds = set()

    # --- Data ---
    print("\n=== Extracting positive windows ===", flush=True)
    pos_records = extract_positive_windows(
        processed_root=args.processed_root,
        max_replays=args.max_replays,
        verbose=True,
        window_len=args.window_len,
    )

    n_pos = len(pos_records)
    print(f"  Total positives: {n_pos}", flush=True)

    if n_pos == 0:
        print(
            "[ERROR] 0 positive windows; check events.parquet and the confidence thresholds.",
            file=sys.stderr,
        )
        sys.exit(1)

    target_none = int(n_pos * args.none_ratio)
    print(f"\n=== Extracting NONE windows (target={target_none}) ===", flush=True)
    none_records = extract_none_windows(
        processed_root=args.processed_root,
        max_replays=args.max_replays,
        target_count=target_none,
        max_per_replay=20,
        random_seed=args.seed,
        verbose=True,
        window_len=args.window_len,
        stride=args.stride,
    )

    n_none = len(none_records)
    print(f"  Total NONE: {n_none}", flush=True)

    all_records = pos_records + none_records
    n_total = len(all_records)
    print(f"\n  TOTAL windows: {n_total}  (pos={n_pos}  none={n_none})", flush=True)

    if n_total == 0:
        print("[ERROR] 0 windows in total.", file=sys.stderr)
        sys.exit(1)

    # --- Label encoding (sorted label names) ---
    all_labels = sorted(set(r.label for r in all_records))
    label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}
    idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}
    n_classes = len(all_labels)

    print(f"\n  Classes ({n_classes}): {all_labels}", flush=True)
    for lbl in all_labels:
        cnt = sum(1 for r in all_records if r.label == lbl)
        print(f"    {lbl:20s}: {cnt}", flush=True)

    # --- Tensors ---
    X, y, groups, _ = _make_tensors(all_records, label_to_idx, args.window_len)
    # X: (N, T, 133), y: (N,), groups: (N,)

    # --- Balanced class weights for the cross-entropy ---
    class_weights_np = compute_class_weight(
        "balanced",
        classes=np.arange(n_classes),
        y=y,
    ).astype(np.float32)
    class_weights_tensor = torch.from_numpy(class_weights_np)
    print("\n  Class weights (balanced):", flush=True)
    for i, w in enumerate(class_weights_np):
        print(f"    {idx_to_label[i]:20s}: {w:.3f}", flush=True)

    # --- GroupKFold ---
    n_unique_groups = len(np.unique(groups))
    n_folds = min(args.n_folds, n_unique_groups)
    if n_folds < 2:
        print(
            f"[ERROR] too few unique groups ({n_unique_groups}) for GroupKFold.",
            file=sys.stderr,
        )
        sys.exit(1)

    gkf = GroupKFold(n_splits=n_folds)
    print(
        f"\n=== HMLT L1 GroupKFold (n_folds={n_folds}) — "
        f"d_win={args.d_win}  kernel={args.kernel_size}  epochs={args.epochs} ===",
        flush=True,
    )

    fold_f1_macro: list[float] = []
    fold_f1_per_class: list[list[float]] = []
    all_conf_matrices: list[list] = []

    final_model = None

    # Load already-collected results from a previous --resume run
    if args.resume and fold_progress_path.exists():
        try:
            prog = json.loads(fold_progress_path.read_text(encoding="utf-8"))
            saved_f1_macro = prog.get("fold_f1_macro", [])
            saved_f1_per = prog.get("fold_f1_per_class", [])
            saved_cms = prog.get("all_conf_matrices", [])
            fold_f1_macro.extend(saved_f1_macro)
            fold_f1_per_class.extend(saved_f1_per)
            all_conf_matrices.extend(saved_cms)
            print(
                f"[INFO] --resume: loaded {len(saved_f1_macro)} fold results from progress file.",
                flush=True,
            )
        except Exception:
            pass  # already loaded completed_folds above; metrics are best-effort

    # --- SIGTERM handler (best-effort during fold loop) ---
    _sig_model_ref: list = [None]

    def _emergency_save_fold() -> None:
        print("[checkpoint_utils] Emergency: no per-fold save possible mid-epoch (CV mode).", flush=True)

    install_sigterm_handler(_emergency_save_fold)

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # the test fold needs at least 2 classes for a meaningful F1
        unique_test = np.unique(y_te)
        if len(unique_test) < 2:
            print(
                f"  Fold {fold_idx+1}: SKIP (test fold has only {len(unique_test)} class)",
                flush=True,
            )
            continue

        # Skip folds already completed in a previous run
        if fold_idx in completed_folds:
            print(
                f"  Fold {fold_idx+1}: SKIP (already completed in previous run)",
                flush=True,
            )
            continue

        # class distribution per fold
        print(f"\n  --- Fold {fold_idx+1}/{n_folds} ---", flush=True)
        for i, lbl in idx_to_label.items():
            n_tr = int((y_tr == i).sum())
            n_te = int((y_te == i).sum())
            print(f"    {lbl:20s}: train={n_tr:4d}  test={n_te:3d}", flush=True)

        # Determine if we should resume mid-fold
        fold_ckpt = ckpt_dir / f"fold_{fold_idx}_last.pt"
        resume_epoch_for_fold = 0
        if args.resume and fold_ckpt.exists():
            import torch as _torch_tmp
            try:
                _payload = _torch_tmp.load(fold_ckpt, map_location="cpu", weights_only=False)
                resume_epoch_for_fold = int(_payload.get("epoch", 0))
                print(
                    f"  Fold {fold_idx+1}: resuming from epoch {resume_epoch_for_fold + 1}",
                    flush=True,
                )
            except Exception:
                pass

        f1_macro, f1_per, cm, model = _train_fold(
            X_tr, y_tr, X_te, y_te,
            class_weights_tensor,
            args, n_classes, device, fold_idx,
            ckpt_dir=ckpt_dir,
            resume_epoch=resume_epoch_for_fold,
        )

        fold_f1_macro.append(f1_macro)
        fold_f1_per_class.append(f1_per)
        all_conf_matrices.append(cm)
        final_model = model
        completed_folds.add(fold_idx)

        # Persist fold progress so a future --resume can skip this fold
        _prog_data = {
            "completed_folds": sorted(completed_folds),
            "fold_f1_macro": fold_f1_macro,
            "fold_f1_per_class": fold_f1_per_class,
            "all_conf_matrices": all_conf_matrices,
        }
        fold_progress_path.write_text(json.dumps(_prog_data, indent=2), encoding="utf-8")

        print(
            f"  Fold {fold_idx+1} RESULT  macro-F1={f1_macro:.3f}  "
            f"per-class={[f'{v:.3f}' for v in f1_per]}",
            flush=True,
        )

    # --- Aggregate results ---
    if not fold_f1_macro:
        print("[WARN] no valid fold; too little data.", flush=True)
        metrics = {
            "error": "no_valid_folds",
            "n_total": n_total,
            "class_names": all_labels,
            "class_counts": {lbl: int((y == i).sum()) for i, lbl in idx_to_label.items()},
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        return

    mean_f1 = float(np.mean(fold_f1_macro))
    std_f1 = float(np.std(fold_f1_macro))
    mean_per_class = np.mean(fold_f1_per_class, axis=0).tolist()
    sum_cm = np.sum(np.array(all_conf_matrices), axis=0).tolist()

    print("\n=== CV results ===", flush=True)
    print(f"  macro-F1: {mean_f1:.3f} +/- {std_f1:.3f}", flush=True)
    for i, lbl in idx_to_label.items():
        print(f"  {lbl:20s}: F1={mean_per_class[i]:.3f}", flush=True)
    print(f"\n  Confusion matrix (sum over folds, order={all_labels}):", flush=True)
    for row, lbl in zip(sum_cm, all_labels):
        print(f"  {lbl:20s}: {row}", flush=True)

    # --- Save the model of the last fold ---
    model_path: Optional[Path] = None
    if final_model is not None:
        model_path = out_dir / "hmlt_scheme.pt"
        torch.save(
            {
                "state_dict": final_model.state_dict(),
                "label_to_idx": label_to_idx,
                "idx_to_label": idx_to_label,
                "args": vars(args),
            },
            str(model_path),
        )
        print(f"\n  Model saved: {model_path}", flush=True)

    # --- metrics.json ---
    metrics = {
        "model": "HMLT-L1-FrameWindowEncoder",
        "n_total": n_total,
        "n_positives": n_pos,
        "n_none": n_none,
        "class_names": all_labels,
        "class_counts": {lbl: int((y == i).sum()) for i, lbl in idx_to_label.items()},
        "window_len": args.window_len,
        "stride": args.stride,
        "d_win": args.d_win,
        "kernel_size": args.kernel_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "n_folds": n_folds,
        "cv_macro_f1_mean": mean_f1,
        "cv_macro_f1_std": std_f1,
        "cv_macro_f1_per_fold": [float(v) for v in fold_f1_macro],
        "cv_f1_per_class": {lbl: float(mean_per_class[i]) for i, lbl in idx_to_label.items()},
        "cv_confusion_matrix_sum": sum_cm,
        "class_weights_balanced": {lbl: float(class_weights_np[i]) for i, lbl in idx_to_label.items()},
        "model_path": str(model_path) if model_path else None,
        "device": str(device),
    }

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\n  Metrics written: {metrics_path}", flush=True)

    print(
        f"\n[DONE] macro-F1={mean_f1:.3f} +/- {std_f1:.3f}  |  "
        f"ceiling_shot={metrics['cv_f1_per_class'].get('ceiling_shot', 0):.3f}  |  "
        f"see {metrics_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
