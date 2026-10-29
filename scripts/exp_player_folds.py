"""Leakage check: win prediction with player-disjoint folds.

Question: does GroupKFold(replay_id) in winpred_baselines.py underestimate leakage
because the SAME players appear in several matches? Here no player appears in both
train and test.

Method:
  1. Read agent_id from events.parquet (stable uint64 player id).
  2. Two replays are connected if they share at least one player.
  3. Connected components via Union-Find -> groups of replays.
  4. GroupKFold(groups) with LogReg + gradient boosting, physics only at 1/3 of the
     match (same features as winpred_baselines.py), and compare the AUC.

Caveat: the co-appearance graph can be dominated by one giant component, which then
lands in a single fold; read the group_stats in the output before trusting the delta.

Example:
    python -u scripts/exp_player_folds.py \\
        --processed-root data/processed --limit 15000 \\
        --out reports/player_folds.json
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.dataset import list_processed_replays          # noqa: E402
from src.validation.winpred_baselines import (                  # noqa: E402
    build_dataset,
    compute_random_baseline,
    get_physics_mask,
)


# ---------------------------------------------------------------------------
# Union-Find for connected components
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1


# ---------------------------------------------------------------------------
# agent_ids per replay from events.parquet
# ---------------------------------------------------------------------------

def get_replay_agents(processed_root: Path, replay_id: str) -> set[int] | None:
    """Set of agent_id values in a replay's events.parquet.

    None if events.parquet is missing or has no agent_id column; an empty set if the
    file exists but has no detections.
    """
    evp = processed_root / replay_id / "events.parquet"
    if not evp.exists():
        return None
    try:
        ev = pd.read_parquet(evp)
    except Exception:
        return None
    if "agent_id" not in ev.columns:
        return None
    return set(int(x) for x in ev["agent_id"].dropna().unique())


# ---------------------------------------------------------------------------
# Player-disjoint groups via Union-Find
# ---------------------------------------------------------------------------

def build_player_groups(
    processed_root: Path,
    replay_ids: list[str],
    verbose: bool = True,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Group replays by player co-appearance.

    Returns:
        replay_to_group: dict[replay_id -> group_id (component root)]
        stats: component count, coverage, group-size distribution
    """
    uf = UnionFind()
    # player_id -> first replay_id seen (for union)
    player_first_replay: dict[int, str] = {}

    n_events_coverage = 0
    n_no_events = 0
    n_empty_events = 0
    n_agents_total: set[int] = set()

    for i, rid in enumerate(replay_ids):
        if verbose and i > 0 and i % 2000 == 0:
            print(f"[GROUPS] {i}/{len(replay_ids)} processed ...", flush=True)

        # make sure the replay is in the forest
        uf.find(rid)

        agents = get_replay_agents(processed_root, rid)
        if agents is None:
            n_no_events += 1
            continue
        if len(agents) == 0:
            n_empty_events += 1
            continue

        n_events_coverage += 1
        n_agents_total.update(agents)

        for agent_id in agents:
            if agent_id in player_first_replay:
                # join this replay with the player's first replay
                uf.union(rid, player_first_replay[agent_id])
            else:
                player_first_replay[agent_id] = rid

    # replay_id -> group_root
    replay_to_group: dict[str, str] = {rid: uf.find(rid) for rid in replay_ids}
    unique_groups = set(replay_to_group.values())
    n_groups = len(unique_groups)

    # group-size distribution
    from collections import Counter
    group_sizes = Counter(replay_to_group.values())
    sizes = sorted(group_sizes.values(), reverse=True)
    size_stats = {
        "max_group": int(sizes[0]) if sizes else 0,
        "median_group": int(np.median(sizes)) if sizes else 0,
        "mean_group": float(np.mean(sizes)) if sizes else 0.0,
        "singleton_groups": int(sum(1 for s in sizes if s == 1)),
        "n_groups": n_groups,
    }

    print(
        f"[GROUPS] {len(replay_ids)} replays -> {n_groups} components",
        flush=True,
    )
    print(
        f"[GROUPS] events coverage: {n_events_coverage}/{len(replay_ids)} "
        f"| no events: {n_no_events} | empty events: {n_empty_events}",
        flush=True,
    )
    print(
        f"[GROUPS] unique players: {len(n_agents_total)} "
        f"| largest group: {size_stats['max_group']} replays "
        f"| singletons: {size_stats['singleton_groups']}",
        flush=True,
    )

    stats = {
        "n_replays": len(replay_ids),
        "n_unique_players": len(n_agents_total),
        "n_player_groups": n_groups,
        "coverage_with_events": n_events_coverage,
        "coverage_no_events": n_no_events,
        "coverage_empty_events": n_empty_events,
        **size_stats,
    }
    return replay_to_group, stats


# ---------------------------------------------------------------------------
# CV with player-disjoint groups
# ---------------------------------------------------------------------------

def compute_metrics_player_cv(
    X: np.ndarray,
    y: np.ndarray,
    replay_to_group: dict[str, str],
    groups_order: list[str],   # replay_ids aligned with X/y
    n_splits: int = 5,
    seed: int = 42,
    model_kind: str = "logreg",
) -> dict[str, float]:
    """GroupKFold over player groups instead of replay ids."""
    group_arr = np.array([replay_to_group[rid] for rid in groups_order])
    n_unique = len(set(replay_to_group[rid] for rid in groups_order))
    actual_splits = min(n_splits, n_unique)

    if actual_splits < 2:
        return {"acc": float("nan"), "auc": float("nan"), "logloss": float("nan"), "n_folds_ok": 0}

    gkf = GroupKFold(n_splits=actual_splits)
    acc_list, auc_list, ll_list = [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, group_arr)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if model_kind == "gbm":
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
            ll_list.append(float(log_loss(y_te, y_proba)))
        except Exception:
            ll_list.append(float("nan"))

    def _safe_mean(lst: list[float]) -> float:
        valid = [v for v in lst if not np.isnan(v)]
        return float(np.mean(valid)) if valid else float("nan")

    return {
        "acc": _safe_mean(acc_list),
        "auc": _safe_mean(auc_list),
        "logloss": _safe_mean(ll_list),
        "n_folds_ok": len(acc_list),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Win prediction with player-disjoint folds")
    ap.add_argument("--processed-root", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=15000)
    ap.add_argument("--out", type=Path, default=Path("reports/player_folds.json"))
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--at-fraction", type=float, default=0.34,
                    help="Fraction of the match used as input (same as winpred_baselines)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    np.random.seed(args.seed)

    # -----------------------------------------------------------------------
    # 1. List replays
    # -----------------------------------------------------------------------
    print(f"[INFO] scanning {args.processed_root} ...", flush=True)
    replay_ids = list_processed_replays(args.processed_root)
    print(f"[INFO] {len(replay_ids)} replays available.", flush=True)

    if args.limit and args.limit < len(replay_ids):
        rng = np.random.default_rng(args.seed)
        replay_ids = list(rng.choice(replay_ids, size=args.limit, replace=False))
        print(f"[INFO] limited to {args.limit} (seed={args.seed}).", flush=True)

    # -----------------------------------------------------------------------
    # 2. Build player groups BEFORE extracting features
    # -----------------------------------------------------------------------
    print("\n[GROUPS] building player-disjoint groups via Union-Find ...", flush=True)
    replay_to_group, group_stats = build_player_groups(
        args.processed_root, replay_ids, verbose=True
    )

    # -----------------------------------------------------------------------
    # 3. Are player ids available?
    # -----------------------------------------------------------------------
    n_singletons = group_stats["singleton_groups"]
    n_groups = group_stats["n_player_groups"]
    coverage_pct = group_stats["coverage_with_events"] / len(replay_ids) * 100

    print(f"\n[INFO] agent_id coverage: {coverage_pct:.1f}%", flush=True)

    if group_stats["n_unique_players"] == 0:
        # no player ids at all
        print("[INFO] agent_id not available in this data.", flush=True)
        results = {
            "player_ids_available": False,
            "reason": (
                "events.parquet has no agent_id column or no detections; "
                "player-disjoint folds cannot be built."
            ),
            "group_stats": group_stats,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[INFO] wrote {args.out} (blocked: no player ids)", flush=True)
        return

    # -----------------------------------------------------------------------
    # 4. Build dataset (features at at_fraction, same as winpred_baselines)
    # -----------------------------------------------------------------------
    print(f"\n[DATA] extracting features at at_fraction={args.at_fraction} ...", flush=True)
    X_full, y, groups_replay = build_dataset(
        args.processed_root, replay_ids, args.at_fraction, verbose=True
    )

    if len(y) == 0:
        sys.exit("[ERROR] zero valid matches.")

    print(f"[INFO] Dataset: {len(y)} matches, {X_full.shape[1]} features.", flush=True)

    # physics only (no score-derived features)
    physics_mask = get_physics_mask()
    X_physics = X_full[:, physics_mask]

    rand = compute_random_baseline(y)
    print(f"[INFO] Random acc: {rand['acc']:.4f}", flush=True)

    # -----------------------------------------------------------------------
    # 5. Every replay in the dataset needs a group
    # -----------------------------------------------------------------------
    for rid in groups_replay:
        if rid not in replay_to_group:
            # valid replay without a group: treat as a singleton
            replay_to_group[rid] = rid

    # -----------------------------------------------------------------------
    # 6. Reference: replay-grouped folds on the same data
    # -----------------------------------------------------------------------
    print("\n[BASELINE] replay-grouped GroupKFold (reference) ...", flush=True)

    def _cv_replay(X: np.ndarray, kind: str) -> dict:
        from src.validation.winpred_baselines import compute_metrics_cv
        return compute_metrics_cv(X, y, groups_replay, n_splits=args.n_splits,
                                  seed=args.seed, model_kind=kind)

    base_lr = _cv_replay(X_physics, "logreg")
    base_gbm = _cv_replay(X_physics, "gbm")
    print(f"[BASELINE] LogReg physics AUC={base_lr['auc']:.4f}  GBM AUC={base_gbm['auc']:.4f}", flush=True)

    # -----------------------------------------------------------------------
    # 7. Player-disjoint folds
    # -----------------------------------------------------------------------
    print(f"\n[PLAYER] Player-disjoint GroupKFold({args.n_splits}) ...", flush=True)
    print(f"[PLAYER] effective groups: {n_groups} (from {len(replay_ids)} replays)", flush=True)

    player_lr = compute_metrics_player_cv(
        X_physics, y, replay_to_group, groups_replay,
        n_splits=args.n_splits, seed=args.seed, model_kind="logreg",
    )
    player_gbm = compute_metrics_player_cv(
        X_physics, y, replay_to_group, groups_replay,
        n_splits=args.n_splits, seed=args.seed, model_kind="gbm",
    )
    print(f"[PLAYER] LogReg physics AUC={player_lr['auc']:.4f}  GBM AUC={player_gbm['auc']:.4f}", flush=True)

    # -----------------------------------------------------------------------
    # 8. Results
    # -----------------------------------------------------------------------
    def lift(m: dict) -> float:
        return float(m["acc"] - rand["acc"]) if not np.isnan(m["acc"]) else float("nan")

    results = {
        "player_ids_available": True,
        "player_id_source": "events.parquet / agent_id (ballchasing player UID, uint64)",
        "config": {
            "processed_root": str(args.processed_root),
            "at_fraction": args.at_fraction,
            "limit": args.limit,
            "seed": args.seed,
            "n_splits": args.n_splits,
        },
        "group_stats": group_stats,
        "n_matches_dataset": int(len(y)),
        "random_acc": float(rand["acc"]),
        "replay_grouped_folds": {
            "logreg": {**base_lr, "lift_over_random": lift(base_lr)},
            "gbm": {**base_gbm, "lift_over_random": lift(base_gbm)},
        },
        "player_grouped_folds": {
            "n_effective_groups": n_groups,
            "singleton_groups": n_singletons,
            "logreg": {**player_lr, "lift_over_random": lift(player_lr)},
            "gbm": {**player_gbm, "lift_over_random": lift(player_gbm)},
        },
        "delta_replay_vs_player_auc": {
            "logreg": round(float(base_lr["auc"] - player_lr["auc"]), 4)
            if not np.isnan(base_lr["auc"]) and not np.isnan(player_lr["auc"]) else None,
            "gbm": round(float(base_gbm["auc"] - player_gbm["auc"]), 4)
            if not np.isnan(base_gbm["auc"]) and not np.isnan(player_gbm["auc"]) else None,
        },
        "interpretation": (
            "A positive delta means replay-grouped folds underestimated leakage. "
            "Delta close to 0 means shared players across matches do not add "
            "meaningful leakage over GroupKFold(replay_id)."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] wrote {args.out}", flush=True)

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Player-disjoint folds for win prediction")
    print(f"  N matches: {len(y)} | player groups: {n_groups} | singletons: {n_singletons}")
    print(f"  Random acc: {rand['acc']:.4f}")
    print("  --- Replay-grouped folds (reference) ---")
    print(f"  LogReg physics AUC = {base_lr['auc']:.4f}")
    print(f"  GBM    physics AUC = {base_gbm['auc']:.4f}")
    print("  --- Player-grouped folds ---")
    print(f"  LogReg physics AUC = {player_lr['auc']:.4f}")
    print(f"  GBM    physics AUC = {player_gbm['auc']:.4f}")
    delta_lr = base_lr["auc"] - player_lr["auc"]
    delta_gbm = base_gbm["auc"] - player_gbm["auc"]
    if not np.isnan(delta_lr):
        print(f"  Delta AUC (replay - player): LogReg {delta_lr:+.4f} | GBM {delta_gbm:+.4f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
