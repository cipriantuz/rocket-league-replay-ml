"""Checkpoint utilities — save/load/rotate + SIGTERM handler.

Designed for multi-night training runs where the GPU is shared and jobs
can be killed at any time via SIGTERM (Linux containers) or SIGINT.

Public API
----------
save_full_checkpoint(path, model, optimizer, epoch, best_metric, extra)
    Save a complete resumable checkpoint.

load_full_checkpoint(path, model, optimizer, device) -> (start_epoch, best_metric)
    Load checkpoint if it exists, otherwise return (0, None).

rotate_checkpoints(ckpt_dir, keep_last=2)
    Delete old epoch checkpoints; keep last ``keep_last`` + any "best" files.

install_sigterm_handler(save_fn)
    Register SIGTERM (Linux) and SIGINT (both) handlers that call save_fn()
    then exit cleanly.
"""

from __future__ import annotations

import os
import re
import signal
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import torch


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_full_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: Optional[float],
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Save a full resumable checkpoint.

    Saves model state_dict, optimizer state_dict, epoch number, best metric
    so far, Python/numpy/torch RNG states, and any extra dict the caller
    wants to persist (e.g. scheduler state, patience counter, fold_idx).

    The file is written atomically (via a temp name + rename) so a mid-write
    SIGTERM never leaves a corrupt checkpoint.
    """
    path = Path(path)
    tmp_path = path.with_suffix(".tmp")

    payload: dict[str, Any] = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        # RNG states — needed for deterministic resume
        "rng_cpu": torch.get_rng_state(),
        "rng_numpy": None,   # filled below
        "rng_python": None,  # filled below
    }

    try:
        import random as _random
        import numpy as _np
        payload["rng_numpy"] = _np.random.get_state()
        payload["rng_python"] = _random.getstate()
    except Exception:
        pass  # best-effort; resume still works without RNG parity

    if torch.cuda.is_available():
        payload["rng_cuda"] = torch.cuda.get_rng_state_all()
    else:
        payload["rng_cuda"] = None

    if extra:
        payload["extra"] = extra

    torch.save(payload, tmp_path)
    tmp_path.replace(path)   # atomic on POSIX; near-atomic on Windows NTFS


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_full_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, Optional[float], Optional[dict]]:
    """Load a checkpoint produced by save_full_checkpoint.

    Returns
    -------
    (start_epoch, best_metric)
        start_epoch is the epoch AFTER the saved one (i.e. training resumes
        at ``for epoch in range(start_epoch, args.epochs + 1)``).
        best_metric is None when no checkpoint was found.
    """
    path = Path(path)
    if not path.exists():
        return 0, None

    payload = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])

    # Restore RNG states if available
    try:
        import random as _random
        import numpy as _np
        if payload.get("rng_cpu") is not None:
            torch.set_rng_state(payload["rng_cpu"])
        if payload.get("rng_numpy") is not None:
            _np.random.set_state(payload["rng_numpy"])
        if payload.get("rng_python") is not None:
            _random.setstate(payload["rng_python"])
        if payload.get("rng_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["rng_cuda"])
    except Exception:
        pass  # best-effort; tolerate version mismatches

    start_epoch = int(payload["epoch"]) + 1
    best_metric = payload.get("best_metric")
    extra = payload.get("extra")
    return start_epoch, best_metric, extra  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Rotate
# ---------------------------------------------------------------------------

_EPOCH_CKPT_RE = re.compile(r"epoch_(\d+)\.pt$")


def rotate_checkpoints(ckpt_dir: str | Path, keep_last: int = 2) -> None:
    """Delete old epoch checkpoints, keep last ``keep_last`` + any "best" files.

    Files matching the pattern ``epoch_NNN.pt`` are considered epoch
    checkpoints.  Files containing "best" in their name are always kept.
    ``last_checkpoint.pt`` is also always kept (it's not epoch-named).

    Parameters
    ----------
    ckpt_dir:
        Directory that contains checkpoints.
    keep_last:
        How many of the most-recent epoch_NNN.pt files to keep.
    """
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return

    epoch_files: list[tuple[int, Path]] = []
    for f in ckpt_dir.iterdir():
        m = _EPOCH_CKPT_RE.search(f.name)
        if m:
            epoch_files.append((int(m.group(1)), f))

    # Sort ascending by epoch number
    epoch_files.sort(key=lambda t: t[0])

    # Keep the last keep_last; delete the rest
    to_delete = epoch_files[: max(0, len(epoch_files) - keep_last)]
    for _, f in to_delete:
        try:
            f.unlink()
        except OSError:
            pass  # already deleted or locked


# ---------------------------------------------------------------------------
# SIGTERM / SIGINT handler
# ---------------------------------------------------------------------------

def install_sigterm_handler(save_fn: Callable[[], None]) -> None:
    """Register signal handlers that call save_fn() then exit.

    On Linux/Mac: handles both SIGTERM and SIGINT.
    On Windows: SIGTERM is not supported by Python; only SIGINT is registered.
    The container admin typically uses ``docker stop`` (SIGTERM) on Linux, so
    on a shared machine this will trigger save_fn before the 10 s kill timeout.
    """

    def _handler(signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        print(
            f"\n[checkpoint_utils] Received {sig_name} — saving checkpoint before exit...",
            flush=True,
        )
        try:
            save_fn()
            print("[checkpoint_utils] Checkpoint saved. Exiting.", flush=True)
        except Exception as exc:
            print(f"[checkpoint_utils] WARNING: save_fn() raised {exc}", flush=True)
        sys.exit(0)

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)
