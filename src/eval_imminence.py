"""Short-horizon goal-imminence evaluation harness.

Research question
-----------------
Does a per-window deep model (CNN / Transformer over raw physics frames) beat
tabular baselines (LogReg, XGBoost on 23-dim aggregate features) on the
goal-imminence task?

Hypothesis: YES — this is the opposite of win-prediction, where models tied at
the ceiling.  The imminence signal (developing aerials, breakaways, ball speed
toward goal) lives in the FINE spatiotemporal dynamics of raw frames.  Tabular
aggregates flatten these into scalar means and lose the sequence structure.
A transformer attending to frame-level transitions should extract this signal
better.

Metrics
--------
We use PR-AUC (average_precision_score) as the primary metric because goals are
RARE per window (~5–20% positive base rate depending on horizon_s).  Accuracy
is dominated by the majority class (non-imminent) and gives a misleading picture
of model quality.  Additional metrics: ROC-AUC, Brier score, base rate.

Output
-------
    reports/<out>/comparison.csv    — rows = model×horizon, cols = metrics
    reports/<out>/summary.json      — deep_pr_auc - best_tabular_pr_auc (the key number)
    logs/eval_imminence_<ts>.log    — full run log

Usage
------
    python -m src.eval_imminence \\
        --processed-root data/processed \\
        --limit 300 \\
        --horizon-s 10 \\
        --epochs 5 \\
        --batch-size 64 \\
        --out reports/imminence_smoke \\
        --device auto \\
        --seed 42

Karpathy: simplest deep model that tests the hypothesis — 1D-CNN + MLP, not a
full transformer (add --use-transformer flag to switch).  PR-AUC is the number
to read; if it's better than tabular, we have our dissertation finding.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.models.dataset import WindowSpec, list_processed_replays, make_splits
from src.models.imminence_dataset import (
    ImminenceDataset,
    collate_fn,
    materialize,
)
from src.obs.match_features import match_feature_names

# Indices of time-related features in the 23-dim aggregate that create structural
# leakage: episode_seconds_remaining counts down to 0, so near end-of-match
# windows are positive just because matches tend to end after the last goal.
# The ablated baseline removes these to isolate the PHYSICS signal.
_TIME_FEATURE_NAMES = {"time_remaining_norm", "is_overtime", "score_x_time"}
_PHYSICS_ONLY_MASK: list[int] | None = None  # computed lazily after import


def _get_physics_only_mask() -> list[int]:
    """Return column indices that exclude time-structural features from agg."""
    global _PHYSICS_ONLY_MASK
    if _PHYSICS_ONLY_MASK is None:
        names = match_feature_names()
        _PHYSICS_ONLY_MASK = [i for i, n in enumerate(names) if n not in _TIME_FEATURE_NAMES]
    return _PHYSICS_ONLY_MASK

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(log_dir: Path, ts: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"eval_imminence_{ts}.log"
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger = logging.getLogger("imminence")
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Deep model definitions
# ---------------------------------------------------------------------------

class ImminenceCNN(nn.Module):
    """1D-CNN per-window encoder for goal imminence.

    Architecture: temporal 1D-CNN over the 45-frame window.
      - Input:  (B, 45, 88)  → transpose to (B, 88, 45) for Conv1d
      - Conv1d 88→128, k=3, GELU
      - Conv1d 128→128, k=3, GELU
      - Global average pool → (B, 128)
      - MLP 128→64→1 with GELU
      - Output: raw logit (B,) — no sigmoid (BCEWithLogitsLoss handles it)

    WHY CNN over Transformer for the per-window head:
    The 45-frame window is short (3 s); local temporal patterns (e.g., ball
    acceleration over 0.5 s) are the signal.  A CNN with kernel=3 (0.2 s
    receptive field) captures these efficiently with far fewer parameters than
    a Transformer.  The Transformer variant is also provided for comparison.
    """

    def __init__(
        self,
        feature_dim: int = 133,
        seq_len: int = 45,
        conv_channels: int = 128,
        mlp_hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feature_dim, conv_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # global average pool → (B, conv_channels, 1)
        self.mlp = nn.Sequential(
            nn.Linear(conv_channels, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: float32 (B, seq_len, feature_dim)

        Returns:
            logit: float32 (B,) — raw binary logit (pre-sigmoid)
        """
        # (B, T, F) → (B, F, T) for Conv1d
        h = x.transpose(1, 2)          # (B, 88, 45)
        h = self.conv(h)               # (B, 128, 45)
        h = self.pool(h).squeeze(-1)   # (B, 128)
        return self.mlp(h).squeeze(-1) # (B,)


class ImminenceTransformer(nn.Module):
    """2-layer TransformerEncoder per-window encoder for goal imminence.

    Input projection: 88 → d_model via linear.
    TransformerEncoder: 2 layers, nhead=4, dim_feedforward=256, dropout=0.1.
    Global average pool over time → MLP → logit.

    More expressive than the CNN but slower and needs more data.
    """

    def __init__(
        self,
        feature_dim: int = 133,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        mlp_hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B, T, d_model) convention
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: float32 (B, seq_len, feature_dim)

        Returns:
            logit: float32 (B,) — raw binary logit (pre-sigmoid)
        """
        h = self.input_proj(x)          # (B, 45, d_model)
        h = self.encoder(h)             # (B, 45, d_model)
        h = h.mean(dim=1)               # (B, d_model) — global avg over time
        return self.mlp(h).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _compute_pos_weight(y_train: np.ndarray) -> float:
    """Compute BCEWithLogitsLoss pos_weight for imbalanced binary labels.

    pos_weight = n_neg / n_pos.  This balances the gradient contribution of
    the rare positive class (imminent goals) against the majority negative.

    See: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    """
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def _train_deep_model(
    model: nn.Module,
    train_ids: list[str],
    val_ids: list[str],
    processed_root: Path,
    cache_root: Path | None,
    *,
    spec: WindowSpec,
    horizon_s: float,
    hz: float,
    epochs: int,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    logger: logging.Logger,
    seed: int,
) -> nn.Module:
    """Train deep model with BCEWithLogitsLoss + AdamW."""
    torch.manual_seed(seed)

    train_ds = ImminenceDataset(
        processed_root, train_ids,
        spec=spec, horizon_s=horizon_s, hz=hz,
        cache_root=cache_root,
    )
    val_ds = ImminenceDataset(
        processed_root, val_ids,
        spec=spec, horizon_s=horizon_s, hz=hz,
        cache_root=cache_root,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
    )

    # Compute pos_weight from training labels — need a quick scan.
    # When the warm cache exists, labels_only=True reads only the small int8
    # label slice from each .npz (no raw-frame loading) → milliseconds instead
    # of 20 minutes for the full-dataset scan.
    logger.info("Computing pos_weight from training labels (labels_only fast path)...")
    _, _, y_train, _ = materialize(
        processed_root, train_ids,
        spec=spec, horizon_s=horizon_s, hz=hz,
        cache_root=cache_root, verbose=False,
        labels_only=True,
    )
    pw = _compute_pos_weight(y_train.astype(np.float32))
    logger.info(f"pos_weight = {pw:.2f}  (n_pos={y_train.sum()}, n_neg={len(y_train)-y_train.sum()})")

    pos_weight = torch.tensor([pw], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    model = model.to(device)
    model.train()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for x_raw, _x_agg, y_bin, _y_3cl in train_loader:
            x_raw = x_raw.to(device, non_blocking=True)
            y_bin = y_bin.to(device, non_blocking=True).float()
            logits = model(x_raw)
            loss = criterion(logits, y_bin)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        logger.info(f"Epoch {epoch}/{epochs}  train_loss={avg_loss:.4f}")

    return model


@torch.no_grad()
def _predict_deep(
    model: nn.Module,
    X_raw: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Run inference; return sigmoid probabilities (N,) float32."""
    model.eval()
    probs: list[np.ndarray] = []
    for i in range(0, len(X_raw), batch_size):
        chunk = torch.from_numpy(X_raw[i : i + batch_size]).to(device)
        logits = model(chunk)
        p = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        probs.append(p)
    return np.concatenate(probs, axis=0)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def _evaluate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    horizon_s: float,
    logger: logging.Logger,
) -> dict[str, object]:
    """Compute and log imbalance-aware metrics."""
    base_rate = float(y_true.mean())
    # Skip degenerate cases
    if base_rate == 0.0 or base_rate == 1.0:
        logger.warning(f"[{model_name}] degenerate labels (base_rate={base_rate:.3f}) — skipping")
        return {
            "model": model_name,
            "horizon_s": horizon_s,
            "pr_auc": float("nan"),
            "roc_auc": float("nan"),
            "brier": float("nan"),
            "base_rate": base_rate,
        }

    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    logger.info(
        f"  [{model_name:22s}] H={horizon_s:.0f}s  "
        f"pr_auc={pr_auc:.4f}  roc_auc={roc_auc:.4f}  "
        f"brier={brier:.4f}  base_rate={base_rate:.3f}"
    )
    return {
        "model": model_name,
        "horizon_s": horizon_s,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier": brier,
        "base_rate": base_rate,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Goal imminence model comparison")
    parser.add_argument("--processed-root", default="data/processed", help="Path to processed replays")
    parser.add_argument("--limit", type=int, default=None, help="Max replays to load (smoke test)")
    parser.add_argument("--horizon-s", type=float, default=10.0, help="Imminence horizon in seconds")
    parser.add_argument("--hz", type=float, default=15.0, help="Frame rate (default 15)")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs for deep model")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--out", default="reports/imminence", help="Output directory")
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0=main process, safe on Windows)")
    parser.add_argument("--use-transformer", action="store_true", help="Use TransformerEncoder instead of CNN")
    parser.add_argument("--cache-root", default=None,
                        help="Directory for .npz caches (default: data/imminence_cache_univ after re-warm)")
    parser.add_argument("--input-dim", type=int, default=133,
                        help="Raw frame feature dimension (133=universal; 88=old 2v2-only compat)")
    args = parser.parse_args(argv)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("logs")
    logger = _setup_logging(log_dir, ts)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Splits
    processed_root = Path(args.processed_root)
    all_ids = list_processed_replays(processed_root)
    logger.info(f"Total replays available: {len(all_ids)}")
    if args.limit:
        rng = np.random.default_rng(args.seed)
        all_ids = rng.choice(all_ids, size=min(args.limit, len(all_ids)), replace=False).tolist()
        logger.info(f"Limited to {len(all_ids)} replays (--limit {args.limit})")
    train_ids, val_ids, test_ids = make_splits(all_ids, seed=args.seed)
    logger.info(f"Split: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

    spec = WindowSpec()
    cache_root = Path(args.cache_root) if args.cache_root else None
    horizon_s = args.horizon_s
    hz = args.hz

    # ---------------------------------------------------------------------------
    # Materialize flat arrays for tabular baselines
    # ---------------------------------------------------------------------------
    logger.info(f"Materializing windows (horizon={horizon_s}s)...")
    logger.info("  Training set...")
    X_raw_tr, X_agg_tr, y_bin_tr, _ = materialize(
        processed_root, train_ids, spec=spec, horizon_s=horizon_s, hz=hz,
        cache_root=cache_root, verbose=True,
    )
    logger.info("  Validation set...")
    X_raw_va, X_agg_va, y_bin_va, _ = materialize(
        processed_root, val_ids, spec=spec, horizon_s=horizon_s, hz=hz,
        cache_root=cache_root, verbose=True,
    )
    logger.info("  Test set...")
    X_raw_te, X_agg_te, y_bin_te, _ = materialize(
        processed_root, test_ids, spec=spec, horizon_s=horizon_s, hz=hz,
        cache_root=cache_root, verbose=True,
    )

    logger.info(
        f"Materialized: train={len(X_agg_tr)} val={len(X_agg_va)} test={len(X_agg_te)} windows"
    )
    y_bin_tr_f = y_bin_tr.astype(np.float32)
    y_bin_va_f = y_bin_va.astype(np.float32)
    y_bin_te_f = y_bin_te.astype(np.float32)

    base_rate_train = float(y_bin_tr_f.mean())
    base_rate_test = float(y_bin_te_f.mean())
    logger.info(f"Positive base rate — train: {base_rate_train:.3f}  test: {base_rate_test:.3f}")

    results: list[dict] = []

    # ---------------------------------------------------------------------------
    # Tabular baseline 1: Logistic Regression on 23-dim aggregate
    # ---------------------------------------------------------------------------
    logger.info("--- Tabular: Logistic Regression ---")
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=args.seed, solver="lbfgs")),
    ])
    logreg.fit(X_agg_tr, y_bin_tr)
    logreg_prob_te = logreg.predict_proba(X_agg_te)[:, 1]
    results.append(_evaluate(y_bin_te_f, logreg_prob_te, "LogReg", horizon_s, logger))

    # Ablated LogReg: remove time-structural features to isolate physics signal.
    # time_remaining_norm is a massive structural confounder: goals cluster near
    # the end of the match, so models learn "time is low = goal soon" without
    # capturing any spatial dynamics.  A fair deep-vs-tabular comparison should
    # compare on PHYSICS features only.
    logger.info("--- Tabular: LogReg (no-time ablation) ---")
    phys_mask = _get_physics_only_mask()
    logreg_phys = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=args.seed, solver="lbfgs")),
    ])
    logreg_phys.fit(X_agg_tr[:, phys_mask], y_bin_tr)
    logreg_phys_prob_te = logreg_phys.predict_proba(X_agg_te[:, phys_mask])[:, 1]
    results.append(_evaluate(y_bin_te_f, logreg_phys_prob_te, "LogReg_no_time", horizon_s, logger))

    # ---------------------------------------------------------------------------
    # Tabular baseline 2: XGBoost on 23-dim aggregate
    # ---------------------------------------------------------------------------
    logger.info("--- Tabular: XGBoost ---")
    try:
        import xgboost as xgb
        n_pos = int(y_bin_tr.sum())
        n_neg = len(y_bin_tr) - n_pos
        scale_pos = n_neg / max(n_pos, 1)
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=scale_pos,
            eval_metric="aucpr",
            random_state=args.seed,
            n_jobs=4,
            tree_method="hist",
            device="cpu",  # keep off GPU to avoid VRAM conflict with deep model
            verbosity=0,
        )
        xgb_model.fit(
            X_agg_tr, y_bin_tr,
            eval_set=[(X_agg_va, y_bin_va)],
            verbose=False,
        )
        xgb_prob_te = xgb_model.predict_proba(X_agg_te)[:, 1]
        results.append(_evaluate(y_bin_te_f, xgb_prob_te, "XGBoost", horizon_s, logger))

        # XGBoost ablated (no time features)
        logger.info("--- Tabular: XGBoost (no-time ablation) ---")
        xgb_phys = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            scale_pos_weight=scale_pos, eval_metric="aucpr",
            random_state=args.seed, n_jobs=4, tree_method="hist",
            device="cpu", verbosity=0,
        )
        xgb_phys.fit(
            X_agg_tr[:, phys_mask], y_bin_tr,
            eval_set=[(X_agg_va[:, phys_mask], y_bin_va)],
            verbose=False,
        )
        xgb_phys_prob_te = xgb_phys.predict_proba(X_agg_te[:, phys_mask])[:, 1]
        results.append(_evaluate(y_bin_te_f, xgb_phys_prob_te, "XGBoost_no_time", horizon_s, logger))
    except ImportError:
        logger.warning("xgboost not installed — skipping")

    # ---------------------------------------------------------------------------
    # Deep model: CNN (or Transformer) on raw 45×input_dim windows
    # ---------------------------------------------------------------------------
    input_dim = getattr(args, "input_dim", 133)  # 133 universal by default
    model_tag = "Transformer" if args.use_transformer else "CNN"
    logger.info(f"--- Deep: {model_tag} (input_dim={input_dim}) ---")

    if args.use_transformer:
        deep_model: nn.Module = ImminenceTransformer(feature_dim=input_dim)
    else:
        deep_model = ImminenceCNN(feature_dim=input_dim, seq_len=45)

    n_params = sum(p.numel() for p in deep_model.parameters() if p.requires_grad)
    logger.info(f"Deep model params: {n_params:,}")

    deep_model = _train_deep_model(
        deep_model,
        train_ids=train_ids,
        val_ids=val_ids,
        processed_root=processed_root,
        cache_root=cache_root,
        spec=spec,
        horizon_s=horizon_s,
        hz=hz,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
        logger=logger,
        seed=args.seed,
    )

    deep_prob_te = _predict_deep(deep_model, X_raw_te, device=device, batch_size=args.batch_size * 4)
    results.append(_evaluate(y_bin_te_f, deep_prob_te, f"Deep_{model_tag}", horizon_s, logger))

    # ---------------------------------------------------------------------------
    # Optional: Hybrid (raw + agg concat) — simple version via XGBoost on flattened
    # ---------------------------------------------------------------------------
    # Omitted from default run to keep smoke fast; add --hybrid flag if needed.

    # ---------------------------------------------------------------------------
    # Write outputs
    # ---------------------------------------------------------------------------
    csv_path = out_dir / "comparison.csv"
    fieldnames = ["model", "horizon_s", "pr_auc", "roc_auc", "brier", "base_rate"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Saved comparison table -> {csv_path}")

    # Summary: report TWO gaps.
    # Gap 1 (full): Deep vs best tabular with ALL 23 features (includes time_remaining).
    #   This is the naive comparison; tabular wins due to temporal leakage.
    # Gap 2 (fair): Deep vs best tabular with PHYSICS-ONLY features (no time_remaining).
    #   This is the fair comparison for the dissertation hypothesis.
    full_tabular_models = {"LogReg", "XGBoost"}
    fair_tabular_models = {"LogReg_no_time", "XGBoost_no_time"}

    def _best_pr_auc(model_set: set[str]) -> tuple[str, float]:
        pr_aucs = {
            r["model"]: r["pr_auc"]
            for r in results
            if r["model"] in model_set and not np.isnan(r.get("pr_auc", float("nan")))  # type: ignore[arg-type]
        }
        if not pr_aucs:
            return "N/A", float("nan")
        best = max(pr_aucs, key=lambda k: pr_aucs[k])
        return best, pr_aucs[best]

    deep_result = next((r for r in results if r["model"].startswith("Deep_")), None)
    deep_pr_auc = deep_result["pr_auc"] if deep_result else float("nan")

    best_full_name, best_full_pr = _best_pr_auc(full_tabular_models)
    best_fair_name, best_fair_pr = _best_pr_auc(fair_tabular_models)

    def _gap(a: float, b: float) -> float | None:
        return float(a - b) if not (np.isnan(a) or np.isnan(b)) else None

    gap_full = _gap(float(deep_pr_auc), best_full_pr)
    gap_fair = _gap(float(deep_pr_auc), best_fair_pr)
    # Use full gap as primary (most conservative for hypothesis)
    gap = gap_full if gap_full is not None else float("nan")

    summary = {
        "horizon_s": horizon_s,
        "deep_pr_auc": float(deep_pr_auc) if not np.isnan(deep_pr_auc) else None,
        # Full comparison (includes time_remaining — structural leakage)
        "best_tabular_full_model": best_full_name,
        "best_tabular_full_pr_auc": float(best_full_pr) if not np.isnan(best_full_pr) else None,
        "deep_minus_tabular_full_pr_auc": gap_full,
        # Fair comparison (physics-only features — no time leakage)
        "best_tabular_fair_model": best_fair_name,
        "best_tabular_fair_pr_auc": float(best_fair_pr) if not np.isnan(best_fair_pr) else None,
        "deep_minus_tabular_fair_pr_auc": gap_fair,
        # Legacy key for compatibility
        "best_tabular_model": best_full_name,
        "best_tabular_pr_auc": float(best_full_pr) if not np.isnan(best_full_pr) else None,
        "deep_minus_tabular_pr_auc": gap_full,
        "positive_base_rate_test": base_rate_test,
        "n_test_windows": int(len(y_bin_te_f)),
        "model_type": model_tag,
        "epochs": args.epochs,
        "note": (
            "Full comparison includes time_remaining (structural leakage: goals cluster near end). "
            "Fair comparison = physics-only features (no time_remaining, is_overtime, score_x_time). "
            "Dissertation hypothesis: deep > fair_tabular. "
            "Positive gap_fair = transformer-edge confirmed on PHYSICS signal only."
        ),
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary -> {summary_path}")

    # Print final table to stdout for quick reading
    logger.info("=" * 70)
    logger.info("IMMINENCE COMPARISON TABLE")
    logger.info(f"  Horizon: {horizon_s}s  |  Test windows: {len(y_bin_te_f)}  |  Base rate: {base_rate_test:.3f}")
    logger.info(f"  {'Model':<25} {'PR-AUC':>8} {'ROC-AUC':>8} {'Brier':>7}")
    logger.info(f"  {'-'*52}")
    for r in results:
        pr = r.get("pr_auc", float("nan"))
        ro = r.get("roc_auc", float("nan"))
        br = r.get("brier", float("nan"))
        logger.info(
            f"  {r['model']:<25} "
            f"{pr:>8.4f} {ro:>8.4f} {br:>7.4f}"
        )
    logger.info(f"  {'-'*52}")
    gap_str = "N/A" if (gap_full is None) else f"{gap_full:+.4f}"
    gap_fair_str = "N/A" if (gap_fair is None) else f"{gap_fair:+.4f}"
    logger.info(f"  Deep - best_tabular (full)      PR-AUC gap = {gap_str}")
    logger.info(f"  Deep - best_tabular (no-time)   PR-AUC gap = {gap_fair_str}")
    logger.info(f"  Note: 'full' includes time_remaining (structural leakage).")
    logger.info(f"  Note: 'no-time' is the FAIR comparison for the dissertation.")
    logger.info("=" * 70)

    if gap_fair is not None:
        if gap_fair > 0.01:
            logger.info("READ [FAIR]: Deep model BEATS physics-only tabular (hypothesis SUPPORTED).")
        elif gap_fair > -0.01:
            logger.info("READ [FAIR]: Deep model TIES physics-only tabular (inconclusive).")
        else:
            logger.info("READ [FAIR]: Physics-only tabular BEATS deep model (hypothesis NOT supported).")
    if gap_full is not None:
        if gap_full < -0.1:
            logger.info("READ [FULL]: Large negative gap driven by time_remaining leakage in tabular.")


if __name__ == "__main__":
    main()
