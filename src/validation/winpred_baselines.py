"""Tabular win-prediction baselines at a fixed fraction of the match.

Level 0: majority class. Level 1: logistic regression. Level 1b: gradient boosting
(HistGradientBoostingClassifier). Every level is reported on TWO feature sets
(with score / physics only), so the score signal is kept separate from the physics
signal. All cross-validation is GroupKFold(groups=replay_id).

Example:
    python -u -m src.validation.winpred_baselines \
        --processed-root data/processed_15hz \
        --limit 15000 \
        --out reports/winpred_baselines.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

# Make the project root importable when the file is run directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.dataset import list_processed_replays          # noqa: E402
from src.obs.match_features import (                           # noqa: E402
    compute_match_feature_sequence,
    match_feature_names,
    MATCH_FEATURE_SIZE,
)

# ---------------------------------------------------------------------------
# Score-derived features (removed for the physics-only variant)
# ---------------------------------------------------------------------------
# Indices in the 23-dim vector returned by match_feature_names():
#   idx 0: goal_diff
#   idx 1: total_goals
#   idx 3: score_x_time
#   idx 15: goals_this_window
#   idx 16: blue_goal_momentum
_SCORE_DERIVED_NAMES: frozenset[str] = frozenset([
    "goal_diff",
    "total_goals",
    "score_x_time",
    "goals_this_window",
    "blue_goal_momentum",
    "is_blue_leading",  # also score-derived (who leads)
])

# Score columns expected in frames.parquet (with fallbacks)
_SCORE_COL_CANDIDATES = {
    "blue":   ["blue_score",   "score_blue",   "team_blue_score",   "blue_goals"],
    "orange": ["orange_score", "score_orange",  "team_orange_score", "orange_goals"],
}


def _find_score_cols(df: pd.DataFrame) -> tuple[str, str]:
    """Return (blue_col, orange_col) from a frames DataFrame; ValueError if missing."""
    blue_col = next((c for c in _SCORE_COL_CANDIDATES["blue"] if c in df.columns), None)
    orange_col = next((c for c in _SCORE_COL_CANDIDATES["orange"] if c in df.columns), None)

    if blue_col is None:
        available = [c for c in df.columns if "score" in c.lower() or "goal" in c.lower()]
        raise ValueError(
            f"[SCORE] blue score column not found. Tried: "
            f"{_SCORE_COL_CANDIDATES['blue']}. Available 'score'/'goal' columns: {available}"
        )
    if orange_col is None:
        available = [c for c in df.columns if "score" in c.lower() or "goal" in c.lower()]
        raise ValueError(
            f"[SCORE] orange score column not found. Tried: "
            f"{_SCORE_COL_CANDIDATES['orange']}. Available 'score'/'goal' columns: {available}"
        )
    return blue_col, orange_col


def _load_frames(replay_dir: Path) -> pd.DataFrame | None:
    """Read frames.parquet from a replay directory; None on error."""
    frames_pq = replay_dir / "frames.parquet"
    if not frames_pq.exists():
        return None
    try:
        # polars is faster; fall back to pandas
        try:
            import polars as pl
            return pl.read_parquet(frames_pq).to_pandas()
        except Exception:
            return pd.read_parquet(frames_pq)
    except Exception as exc:
        print(f"[WARN] cannot read {frames_pq}: {exc}", file=sys.stderr, flush=True)
        return None


# ---------------------------------------------------------------------------
# Per-match extraction
# ---------------------------------------------------------------------------

def extract_match_record(
    processed_root: Path,
    replay_id: str,
    at_fraction: float,
) -> dict[str, Any] | None:
    """Extract (features_23, winner_label) for one replay.

    Args:
        processed_root: root of the processed data
        replay_id:      replay id (sub-directory name)
        at_fraction:    fraction of the match used as input (0.34 = first third)

    Returns:
        dict with replay_id, features (23,), winner (0=orange / 1=blue), or None.
    """
    replay_dir = processed_root / replay_id
    df = _load_frames(replay_dir)
    if df is None or df.empty:
        return None

    n_frames = len(df)

    # --- Target: FINAL score (last frame) ---
    try:
        blue_col, orange_col = _find_score_cols(df)
    except ValueError as exc:
        print(f"[WARN] {replay_id}: {exc}", file=sys.stderr, flush=True)
        return None

    last_row = df.iloc[-1]
    blue_final = float(last_row[blue_col])
    orange_final = float(last_row[orange_col])

    if blue_final == orange_final:
        # Draws are excluded (no binary target)
        return None

    winner = 1 if blue_final > orange_final else 0  # 1 = blue wins, 0 = orange wins

    # --- Features: only the first at_fraction of the match (in-play, not end-of-match) ---
    cutoff = max(1, int(n_frames * at_fraction))
    df_partial = df.iloc[:cutoff]

    # window features over the partial match only
    seq, _end_indices = compute_match_feature_sequence(df_partial)

    if seq.shape[0] == 0:
        # Too short for one full window at this fraction
        return None

    # Mean over the available windows -> one 23-dim vector per match
    features = seq.mean(axis=0)  # shape (23,)

    return {
        "replay_id": replay_id,
        "features": features,
        "winner": winner,
    }


# ---------------------------------------------------------------------------
# Full dataset
# ---------------------------------------------------------------------------

def build_dataset(
    processed_root: Path,
    replay_ids: list[str],
    at_fraction: float,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build X (N, 23), y (N,), groups (N,) for all replays.

    groups holds the replay_id of each match, used by GroupKFold.
    """
    all_features: list[np.ndarray] = []
    all_labels: list[int] = []
    all_groups: list[str] = []

    skipped_draw = 0
    skipped_short = 0
    skipped_error = 0

    for i, rid in enumerate(replay_ids):
        if verbose and i > 0 and i % 500 == 0:
            print(
                f"[DATA] processed {i}/{len(replay_ids)} replays "
                f"(valid={len(all_labels)}, draw={skipped_draw}, "
                f"short={skipped_short}, err={skipped_error})",
                flush=True,
            )

        rec = extract_match_record(processed_root, rid, at_fraction)
        if rec is None:
            # Re-read frames to tell the skip reason apart
            replay_dir = processed_root / rid
            df = _load_frames(replay_dir)
            if df is None:
                skipped_error += 1
            else:
                try:
                    blue_col, orange_col = _find_score_cols(df)
                    b = float(df.iloc[-1][blue_col])
                    o = float(df.iloc[-1][orange_col])
                    if b == o:
                        skipped_draw += 1
                    else:
                        skipped_short += 1
                except ValueError:
                    skipped_error += 1
            continue

        all_features.append(rec["features"])
        all_labels.append(rec["winner"])
        all_groups.append(rec["replay_id"])

    print(
        f"[DATA] Final: {len(all_labels)} valid matches "
        f"| draws excluded={skipped_draw} "
        f"| too short/empty={skipped_short} "
        f"| errors={skipped_error}",
        flush=True,
    )

    if not all_labels:
        return (
            np.empty((0, MATCH_FEATURE_SIZE), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            [],
        )

    X = np.stack(all_features).astype(np.float32)   # (N, 23)
    y = np.array(all_labels, dtype=np.int64)
    return X, y, all_groups


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: list[str],
    n_splits: int = 5,
    seed: int = 42,
    model_kind: str = "logreg",
) -> dict[str, float]:
    """LogReg or gradient boosting, cross-validated with GroupKFold(replay_id).

    model_kind: "logreg" (linear) or "gbm" (HistGradientBoosting, non-linear).
    Returns:
        dict with acc, auc, logloss, n_folds_ok
    """
    group_arr = np.array(groups)

    # GroupKFold needs at least n_splits unique groups
    n_unique = len(set(groups))
    actual_splits = min(n_splits, n_unique)
    if actual_splits < 2:
        return {"acc": float("nan"), "auc": float("nan"), "logloss": float("nan"), "n_folds_ok": 0}

    gkf = GroupKFold(n_splits=actual_splits)

    acc_list: list[float] = []
    auc_list: list[float] = []
    logloss_list: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, group_arr)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            # Degenerate fold (single class): skip
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if model_kind == "gbm":
                from sklearn.ensemble import HistGradientBoostingClassifier
                clf = HistGradientBoostingClassifier(
                    max_iter=300, learning_rate=0.05, max_depth=None,
                    l2_regularization=1.0, random_state=seed,
                )
            else:
                clf = LogisticRegression(
                    max_iter=1000, solver="lbfgs", random_state=seed, C=1.0,
                )
            clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_te)
        y_proba = clf.predict_proba(X_te)[:, 1]

        acc_list.append(float(np.mean(y_pred == y_te)))
        try:
            auc_list.append(float(roc_auc_score(y_te, y_proba)))
        except Exception:
            auc_list.append(float("nan"))
        try:
            logloss_list.append(float(log_loss(y_te, y_proba)))
        except Exception:
            logloss_list.append(float("nan"))

    def _safe_mean(lst: list[float]) -> float:
        valid = [v for v in lst if not np.isnan(v)]
        return float(np.mean(valid)) if valid else float("nan")

    return {
        "acc": _safe_mean(acc_list),
        "auc": _safe_mean(auc_list),
        "logloss": _safe_mean(logloss_list),
        "n_folds_ok": len(acc_list),
    }


def compute_random_baseline(y: np.ndarray) -> dict[str, float]:
    """Level 0: majority class. acc_random = max(p_blue, p_orange)."""
    if len(y) == 0:
        return {"acc": float("nan"), "base_rate": float("nan")}
    base_rate = float(np.mean(y))          # P(blue wins)
    acc_random = max(base_rate, 1.0 - base_rate)
    return {"acc": acc_random, "base_rate": base_rate}


# ---------------------------------------------------------------------------
# Physics-only feature selection
# ---------------------------------------------------------------------------

def get_physics_mask() -> np.ndarray:
    """Boolean mask (23,), True = physics (non-score) feature.

    Excluded (score-derived): goal_diff, total_goals, score_x_time,
    goals_this_window, blue_goal_momentum, is_blue_leading.
    Note: the physics-only set still contains the MMR prior features.
    """
    names = match_feature_names()
    mask = np.array([name not in _SCORE_DERIVED_NAMES for name in names], dtype=bool)
    return mask


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _fmt(v: float, decimals: int = 4) -> str:
    if np.isnan(v):
        return "N/A"
    return f"{v:.{decimals}f}"


def print_summary_table(results: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print(f"  WIN-PREDICTION BASELINES — at_fraction={results['at_fraction']:.2f}")
    print(f"  N matches={results['n_matches']}  |  base_rate(blue)={results['base_rate']:.4f}")
    print(f"  Random acc (majority)={results['random_acc']:.4f}")
    print("=" * 72)
    print(f"  {'Variant':<22} {'Acc':>8} {'AUC':>8} {'LogLoss':>9} {'Lift/Rnd':>10}")
    print("-" * 72)

    for variant_key, label in [("with_score", "With score (23 feat)"), ("physics_only", "Physics only (17 feat)")]:
        m = results.get(variant_key, {})
        acc = m.get("acc", float("nan"))
        lift = acc - results["random_acc"] if not np.isnan(acc) else float("nan")
        print(
            f"  {label:<22} "
            f"{_fmt(acc):>8} "
            f"{_fmt(m.get('auc', float('nan'))):>8} "
            f"{_fmt(m.get('logloss', float('nan'))):>9} "
            f"{_fmt(lift):>10}"
        )

    print("=" * 72)
    print(
        "  Note: GroupKFold(replay_id), no replay in both train and test.\n"
        "  'Lift/Rnd' = acc_model - acc_random.\n"
        "  The headline variant is physics only (no score)."
    )
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Win-prediction baselines: majority, logistic regression, gradient boosting"
    )
    parser.add_argument(
        "--processed-root",
        required=True,
        type=Path,
        help="Processed replays (<root>/<replay_id>/frames.parquet)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of replays (debugging)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/winpred_baselines.json"),
        help="Output metrics JSON",
    )
    parser.add_argument(
        "--at-fraction",
        type=float,
        default=0.34,
        help="Fraction of the match used as input (default 0.34 = first third)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of GroupKFold folds",
    )
    args = parser.parse_args(argv)

    np.random.seed(args.seed)

    # 1. List processed replays
    print(f"[INFO] scanning {args.processed_root} ...", flush=True)
    replay_ids = list_processed_replays(args.processed_root)
    print(f"[INFO] {len(replay_ids)} replays found.", flush=True)

    if args.limit is not None and args.limit < len(replay_ids):
        rng = np.random.default_rng(args.seed)
        replay_ids = list(rng.choice(replay_ids, size=args.limit, replace=False))
        print(f"[INFO] limited to {args.limit} replays (--limit).", flush=True)

    if not replay_ids:
        sys.exit("[ERROR] no processed replays found; check --processed-root.")

    # 2. Sanity check: 23 feature names
    feature_names = match_feature_names()
    assert len(feature_names) == MATCH_FEATURE_SIZE, (
        f"MATCH_FEATURE_SIZE={MATCH_FEATURE_SIZE} but match_feature_names() returns "
        f"{len(feature_names)} items."
    )

    # 3. Build dataset (features aggregated up to at_fraction)
    print(
        f"[INFO] extracting features at at_fraction={args.at_fraction:.2f} ...",
        flush=True,
    )
    X_full, y, groups = build_dataset(
        args.processed_root, replay_ids, args.at_fraction, verbose=True
    )

    n_matches = len(y)
    if n_matches == 0:
        sys.exit("[ERROR] zero valid matches after filtering (draws + too short).")

    print(f"[INFO] Dataset: {n_matches} matches, {X_full.shape[1]} features.", flush=True)

    # 4. Level 0: majority class
    rand_baseline = compute_random_baseline(y)
    print(
        f"[INFO] Level 0 (majority): acc={rand_baseline['acc']:.4f}  "
        f"base_rate_blue={rand_baseline['base_rate']:.4f}",
        flush=True,
    )

    # 5. Physics-only mask
    physics_mask = get_physics_mask()
    physics_feature_names = [n for n, m in zip(feature_names, physics_mask) if m]
    score_feature_names = [n for n, m in zip(feature_names, ~physics_mask) if m]

    print(
        f"[INFO] excluded (score-derived) features: {score_feature_names}",
        flush=True,
    )
    print(
        f"[INFO] physics-only features ({physics_mask.sum()} features): {physics_feature_names}",
        flush=True,
    )

    X_physics = X_full[:, physics_mask]   # (N, 17)

    # 6. Level 1: logistic regression with GroupKFold
    print(
        f"[INFO] Level 1 LogReg (with score) — GroupKFold(n_splits={args.n_splits}) ...",
        flush=True,
    )
    metrics_with_score = compute_metrics_cv(
        X_full, y, groups, n_splits=args.n_splits, seed=args.seed
    )
    print(
        f"[INFO]   acc={_fmt(metrics_with_score['acc'])}  "
        f"auc={_fmt(metrics_with_score['auc'])}  "
        f"logloss={_fmt(metrics_with_score['logloss'])}  "
        f"folds_ok={metrics_with_score['n_folds_ok']}",
        flush=True,
    )

    print(
        f"[INFO] Level 1 LogReg (physics only) — GroupKFold(n_splits={args.n_splits}) ...",
        flush=True,
    )
    metrics_physics = compute_metrics_cv(
        X_physics, y, groups, n_splits=args.n_splits, seed=args.seed
    )
    print(
        f"[INFO]   acc={_fmt(metrics_physics['acc'])}  "
        f"auc={_fmt(metrics_physics['auc'])}  "
        f"logloss={_fmt(metrics_physics['logloss'])}  "
        f"folds_ok={metrics_physics['n_folds_ok']}",
        flush=True,
    )

    # 6b. Level 1b: gradient boosting on the same aggregates. Controls for model capacity:
    # the frame-level model only "wins" if it also beats a strong non-linear tabular model.
    print(f"[INFO] Level 1b GBM (with score) — GroupKFold(n_splits={args.n_splits}) ...", flush=True)
    metrics_gbm_score = compute_metrics_cv(
        X_full, y, groups, n_splits=args.n_splits, seed=args.seed, model_kind="gbm"
    )
    print(f"[INFO]   acc={_fmt(metrics_gbm_score['acc'])} auc={_fmt(metrics_gbm_score['auc'])}", flush=True)
    print(f"[INFO] Level 1b GBM (physics only) — GroupKFold(n_splits={args.n_splits}) ...", flush=True)
    metrics_gbm_physics = compute_metrics_cv(
        X_physics, y, groups, n_splits=args.n_splits, seed=args.seed, model_kind="gbm"
    )
    print(f"[INFO]   acc={_fmt(metrics_gbm_physics['acc'])} auc={_fmt(metrics_gbm_physics['auc'])}", flush=True)

    # 7. Lift over the majority baseline
    def with_lift(m: dict[str, float]) -> dict[str, float]:
        acc = m.get("acc", float("nan"))
        lift = acc - rand_baseline["acc"] if not np.isnan(acc) else float("nan")
        return {**m, "lift_over_random": float(lift)}

    # 8. Assemble the metrics JSON
    results: dict[str, Any] = {
        "n_matches": int(n_matches),
        "base_rate": float(rand_baseline["base_rate"]),
        "random_acc": float(rand_baseline["acc"]),
        "at_fraction": float(args.at_fraction),
        "n_splits": args.n_splits,
        "seed": args.seed,
        "score_features_excluded": score_feature_names,
        "physics_features_used": physics_feature_names,
        "blue_col_used": None,   # filled in below from a sample replay
        "orange_col_used": None,
        "with_score": with_lift(metrics_with_score),
        "physics_only": with_lift(metrics_physics),
        "gbm_with_score": with_lift(metrics_gbm_score),
        "gbm_physics_only": with_lift(metrics_gbm_physics),
    }

    # Record the score column names actually found
    for rid in replay_ids[:20]:
        df_sample = _load_frames(args.processed_root / rid)
        if df_sample is not None:
            try:
                bc, oc = _find_score_cols(df_sample)
                results["blue_col_used"] = bc
                results["orange_col_used"] = oc
                break
            except ValueError:
                pass

    # 9. Write the JSON output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] metrics written to {args.out}", flush=True)

    # 10. Console summary
    print_summary_table(results)


if __name__ == "__main__":
    main()
