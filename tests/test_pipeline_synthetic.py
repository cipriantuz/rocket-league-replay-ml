"""End-to-end checks on the synthetic processed-replay tree (see conftest.py).

Covers the parts the per-module tests do not: replay-level splits, the tabular
baselines with GroupKFold, player-disjoint grouping, scheme windowing / weak labels,
and smoke runs of the two training entry points on CPU.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from src.discovery import scheme_window_extractor as swe
from src.models.dataset import list_processed_replays, make_splits
from src.models.hierarchical_winning_v2 import HMLTv2, count_params
from src.obs.universal_frames import frames_df_to_universal
from src.validation import winpred_baselines as wb

# ---------------------------------------------------------------------------
# Parsing output -> model input
# ---------------------------------------------------------------------------


def test_frame_table_maps_to_133_dim_universal(processed_root):
    df = pd.read_parquet(processed_root / "replay_00" / "frames.parquet")
    u = frames_df_to_universal(df)
    assert u.shape == (len(df), 133)
    assert u.dtype == np.float32
    # 2v2: four presence-mask slots on (blue0, blue1, orange0, orange1), two off
    assert u[:, 124:130].sum(axis=1).tolist() == [4.0] * len(df)


# ---------------------------------------------------------------------------
# Replay-level split
# ---------------------------------------------------------------------------


def test_replay_level_split_is_disjoint_complete_and_deterministic(processed_root):
    ids = list_processed_replays(processed_root)
    assert len(ids) == 12
    tr, va, te = make_splits(ids, seed=42)
    assert set(tr).isdisjoint(va) and set(tr).isdisjoint(te) and set(va).isdisjoint(te)
    assert sorted(tr + va + te) == sorted(ids)
    assert make_splits(ids, seed=42) == (tr, va, te)


# ---------------------------------------------------------------------------
# Tabular baselines (gradient boosting / logistic regression)
# ---------------------------------------------------------------------------


def test_physics_mask_drops_every_score_feature():
    from src.obs.match_features import match_feature_names

    names = np.array(match_feature_names())
    kept = set(names[wb.get_physics_mask()])
    assert kept.isdisjoint(wb._SCORE_DERIVED_NAMES)
    assert len(kept) == 17


def test_baseline_features_only_use_the_first_fraction(processed_root):
    # Goals happen after half time in the fixture, so at 1/3 the score is still 0-0
    # even though the final score decides the label.
    from src.obs.match_features import match_feature_names

    X, y, groups = wb.build_dataset(processed_root, list_processed_replays(processed_root), 0.34)
    assert X.shape == (12, 23) and len(groups) == 12
    names = match_feature_names()
    for f in ("goal_diff", "total_goals", "goals_this_window"):
        assert np.all(X[:, names.index(f)] == 0.0), f
    assert np.all(X[:, names.index("is_blue_leading")] == 0.5)  # 0.5 encodes a tie


def test_groupkfold_baselines_run(processed_root):
    X, y, groups = wb.build_dataset(processed_root, list_processed_replays(processed_root), 0.34)
    for kind in ("logreg", "gbm"):
        m = wb.compute_metrics_cv(X, y, groups, n_splits=3, model_kind=kind)
        assert m["n_folds_ok"] == 3
        assert 0.0 <= m["auc"] <= 1.0


def test_winpred_baselines_cli_writes_metrics(processed_root, tmp_path):
    out = tmp_path / "winpred.json"
    wb.main(["--processed-root", str(processed_root), "--out", str(out), "--n-splits", "3"])
    res = json.loads(out.read_text())
    assert res["n_matches"] == 12
    assert {"physics_only", "gbm_physics_only", "with_score", "gbm_with_score"} <= res.keys()
    assert "goal_diff" in res["score_features_excluded"]


# ---------------------------------------------------------------------------
# Player-disjoint folds (leakage check)
# ---------------------------------------------------------------------------


def test_player_groups_join_replays_that_share_a_player(processed_root):
    from scripts.exp_player_folds import build_player_groups

    ids = list_processed_replays(processed_root)
    groups, stats = build_player_groups(processed_root, ids, verbose=False)
    assert groups["replay_00"] == groups["replay_01"]  # both contain agent 7
    assert len({groups[r] for r in ids}) == 11
    assert stats["n_unique_players"] == 23


# ---------------------------------------------------------------------------
# Play-scheme windows (weak labels) and GroupKFold
# ---------------------------------------------------------------------------


def test_positive_windows_are_centred_on_validated_events(processed_root):
    recs = swe.extract_positive_windows(processed_root, verbose=False, window_len=30)
    assert len(recs) == 12  # one validated ceiling shot per replay; simple_shot is not a target
    for r in recs:
        assert r.label == "ceiling_shot"
        assert r.tensor.shape == (30, 133)
        assert r.frame_start <= r.event_frame_start and r.event_frame_end <= r.frame_end


def test_none_windows_stay_away_from_every_event(processed_root):
    recs = swe.extract_none_windows(
        processed_root, target_count=100, verbose=False, window_len=30, stride=10
    )
    assert recs, "expected some NONE windows"
    events = [(100, 110), (180, 185)]
    for r in recs:
        assert r.label == "NONE"
        for s, e in events:
            # window [start, end) must not touch [s - buffer, e + buffer]
            assert r.frame_end <= s - swe.EVENT_BUFFER or r.frame_start > e + swe.EVENT_BUFFER


def test_groupkfold_never_puts_one_replay_in_train_and_test(processed_root):
    from sklearn.model_selection import GroupKFold

    recs = swe.extract_positive_windows(processed_root, verbose=False, window_len=30)
    recs += swe.extract_none_windows(processed_root, target_count=60, verbose=False,
                                     window_len=30, stride=10)
    groups = np.array([r.replay_id for r in recs])
    for tr, te in GroupKFold(n_splits=4).split(np.zeros(len(recs)), groups=groups):
        assert set(groups[tr]).isdisjoint(groups[te])


def test_scheme_classifier_cli_smoke(processed_root, tmp_path, monkeypatch):
    from src.discovery import train_scheme_hmlt

    out = tmp_path / "scheme"
    monkeypatch.setattr(sys, "argv", [
        "train_scheme_hmlt", "--processed-root", str(processed_root), "--window-len", "30",
        "--stride", "10", "--epochs", "1", "--n-folds", "3", "--device", "cpu",
        "--out-dir", str(out),
    ])
    train_scheme_hmlt.main()
    m = json.loads((out / "metrics.json").read_text())
    assert m["class_names"] == ["NONE", "ceiling_shot"]
    assert m["n_folds"] == 3 and len(m["cv_macro_f1_per_fold"]) == 3


# ---------------------------------------------------------------------------
# Model size claimed in the README
# ---------------------------------------------------------------------------


def test_hmlt_default_physics_only_config_has_1_035_009_parameters():
    model = HMLTv2(n_features=133, use_hybrid=False)  # d_model=128, 4 layers, 8 heads
    assert count_params(model) == 1_035_009
    assert len(model.transformer.layers) == 4


def test_hmlt_forward_on_a_variable_length_match():
    model = HMLTv2(n_features=133, use_hybrid=False, d_win=32, d_model=32, n_layers=1,
                   n_heads=4).eval()
    fw = torch.randn(2, 7, 45, 133)
    pad_mask = torch.zeros(2, 7, dtype=torch.bool)
    pad_mask[1, 5:] = True  # second match is shorter (True = padding)
    with torch.no_grad():
        logits = model(fw, pad_mask)
    assert logits.shape == (2,) and torch.isfinite(logits).all()
