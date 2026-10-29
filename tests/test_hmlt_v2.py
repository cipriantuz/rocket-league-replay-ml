"""Tests for HMLTv2 — 2-level Hierarchical Match-Level Transformer.

Covers:
  1. HMLTv2 forward pass (hybrid + non-hybrid): shape, finiteness, no NaN.
  2. FrameWindowEncoder (L1) output shape.
  3. Padding invariance: padded windows do not affect masked output (< 1e-4).
  4. collate_fn: correct padding shapes, pad_mask correctness.
  5. FrameSequenceDataset on synthetic processed dir:
       - 2 valid 88-dim replays with >= 90 frames → len == 2
       - 1 skipped (winning_team == -1)
       - 1 skipped (wrong feature dim — 52-dim)
       - __getitem__ shapes: (T, 45, 88) frames + (T, 23) match feats + label
  6. count_params returns a positive int.

All tests are synthetic — no real parquet data required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.hierarchical_winning_v2 import (
    FrameWindowEncoder,
    HMLTv2,
    count_params,
)
from src.models.frame_sequence_dataset import (
    FrameSequenceDataset,
    collate_fn as frame_collate_fn,
)


# ---------------------------------------------------------------------------
# Synthetic frame DataFrame builder (mirrors test_hmlt.py for reuse in v2)
# ---------------------------------------------------------------------------

_META_COLS = [
    "frame_idx", "src_frame_idx", "time_s", "episode_seconds_remaining",
    "next_scoring_team", "winning_team", "blue_score", "orange_score", "goal_scored",
]
_ACTION_COLS = [
    "act_throttle", "act_steer", "act_pitch", "act_yaw",
    "act_roll", "act_jump", "act_boost", "act_handbrake",
]


def _make_synthetic_frames(n: int, winning_team: int, n_cars: int = 4) -> pd.DataFrame:
    """Build a synthetic frames DataFrame with the correct 2v2 88-feature schema.

    Args:
        n:            Number of frames.
        winning_team: 0 = blue, 1 = orange, -1 = unknown.
        n_cars:       2 = 1v1 (2 total, 52 features), 4 = 2v2 (88 features).
    """
    rng = np.random.default_rng(0)
    rows: dict[str, np.ndarray] = {}

    # Meta
    rows["frame_idx"] = np.arange(n, dtype=np.int64)
    rows["src_frame_idx"] = np.arange(n, dtype=np.int64)
    rows["time_s"] = np.linspace(0.0, n / 15.0, n, dtype=np.float32)
    rows["episode_seconds_remaining"] = np.linspace(300.0, 0.0, n, dtype=np.float32)
    rows["next_scoring_team"] = np.full(n, -1, dtype=np.int64)
    rows["winning_team"] = np.full(n, winning_team, dtype=np.int64)
    rows["blue_score"] = np.zeros(n, dtype=np.int64)
    rows["orange_score"] = np.zeros(n, dtype=np.int64)
    rows["goal_scored"] = np.zeros(n, dtype=np.int64)

    # Actions
    for col in _ACTION_COLS:
        rows[col] = np.zeros(n, dtype=np.float32)

    # Ball physics (9 features)
    rows["ball_pos_x"] = np.zeros(n, dtype=np.float32)
    rows["ball_pos_y"] = rng.uniform(-0.5, 0.5, n).astype(np.float32)
    rows["ball_pos_z"] = np.full(n, 0.05, dtype=np.float32)
    rows["ball_linvel_x"] = rng.uniform(-0.3, 0.3, n).astype(np.float32)
    rows["ball_linvel_y"] = rng.uniform(-0.3, 0.3, n).astype(np.float32)
    rows["ball_linvel_z"] = np.zeros(n, dtype=np.float32)
    rows["ball_angvel_x"] = np.zeros(n, dtype=np.float32)
    rows["ball_angvel_y"] = np.zeros(n, dtype=np.float32)
    rows["ball_angvel_z"] = np.zeros(n, dtype=np.float32)

    # Cars: each car = 18 features (9 physics + 9 state); n_cars total
    teams_cars: list[tuple[str, int]] = []
    blue_count = n_cars // 2
    orange_count = n_cars // 2
    for i in range(blue_count):
        teams_cars.append(("blue", i))
    for i in range(orange_count):
        teams_cars.append(("orange", i))

    for colour, i in teams_cars:
        prefix = f"car_{colour}{i}"
        rows[f"{prefix}_pos_x"] = rng.uniform(-0.5, 0.5, n).astype(np.float32)
        rows[f"{prefix}_pos_y"] = np.full(
            n, -0.5 if colour == "blue" else 0.5, dtype=np.float32
        )
        rows[f"{prefix}_pos_z"] = np.full(n, 0.008, dtype=np.float32)
        for ax in ("x", "y", "z"):
            rows[f"{prefix}_linvel_{ax}"] = np.zeros(n, dtype=np.float32)
            rows[f"{prefix}_angvel_{ax}"] = np.zeros(n, dtype=np.float32)
        rows[f"{prefix}_boost"] = np.full(n, 0.5, dtype=np.float32)
        rows[f"{prefix}_on_ground"] = np.ones(n, dtype=np.float32)
        rows[f"{prefix}_has_flip"] = np.ones(n, dtype=np.float32)
        rows[f"{prefix}_has_jumped"] = np.zeros(n, dtype=np.float32)
        rows[f"{prefix}_is_supersonic"] = np.zeros(n, dtype=np.float32)
        rows[f"{prefix}_air_time"] = np.zeros(n, dtype=np.float32)
        rows[f"{prefix}_flip_time"] = np.zeros(n, dtype=np.float32)
        rows[f"{prefix}_demo_timer"] = np.zeros(n, dtype=np.float32)
        rows[f"{prefix}_is_orange"] = np.full(
            n, 1.0 if colour == "orange" else 0.0, dtype=np.float32
        )

    # Boost pads (6 large + 1 small-frac summary = 7 features)
    for i in range(6):
        rows[f"boost_pad_large_{i}_timer"] = np.zeros(n, dtype=np.float32)
    rows["boost_pad_small_ready_frac"] = np.ones(n, dtype=np.float32)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1 — HMLTv2 forward pass: hybrid and non-hybrid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_hybrid", [True, False])
def test_hmltv2_forward_shape_and_finite(use_hybrid: bool) -> None:
    """Forward pass returns logit Tensor of shape (B,) with no NaN/Inf."""
    B, T, W, F = 2, 10, 45, 88
    model = HMLTv2(
        n_frames=W,
        n_features=F,
        d_win=32,
        n_match_feats=23,
        use_hybrid=use_hybrid,
        d_model=32,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
        max_len=64,
    )
    model.eval()

    frame_windows = torch.randn(B, T, W, F)
    match_feats = torch.randn(B, T, 23) if use_hybrid else None
    pad_mask = torch.zeros(B, T, dtype=torch.bool)  # no padding

    with torch.no_grad():
        logit = model(frame_windows, pad_mask, match_feats=match_feats)

    assert logit.shape == (B,), f"Expected shape ({B},), got {logit.shape}"
    assert not torch.isnan(logit).any(), "Output contains NaN"
    assert not torch.isinf(logit).any(), "Output contains Inf"


# ---------------------------------------------------------------------------
# Test 2 — FrameWindowEncoder (L1) output shape
# ---------------------------------------------------------------------------

def test_frame_window_encoder_output_shape() -> None:
    """FrameWindowEncoder maps (B_eff, 45, 88) → (B_eff, d_win)."""
    B_eff = 20
    n_frames, n_features, d_win = 45, 88, 64
    encoder = FrameWindowEncoder(
        n_frames=n_frames, n_features=n_features, d_win=d_win, kernel_size=3
    )
    encoder.eval()

    x = torch.randn(B_eff, n_frames, n_features)
    with torch.no_grad():
        out = encoder(x)

    assert out.shape == (B_eff, d_win), (
        f"Expected ({B_eff}, {d_win}), got {out.shape}"
    )
    assert not torch.isnan(out).any(), "L1 encoder output contains NaN"


# ---------------------------------------------------------------------------
# Test 3 — Padding invariance: padded windows must not affect output
# ---------------------------------------------------------------------------

def test_hmltv2_padding_invariance() -> None:
    """Output is the same regardless of what values sit in pad positions (< 1e-4)."""
    B, T_valid, T_total, W, F = 1, 5, 10, 45, 88
    model = HMLTv2(
        n_frames=W, n_features=F, d_win=32, n_match_feats=23,
        use_hybrid=True, d_model=32, n_layers=2, n_heads=4,
        dropout=0.0, max_len=64,
    )
    model.eval()

    torch.manual_seed(7)
    fw_valid = torch.randn(B, T_valid, W, F)
    mf_valid = torch.randn(B, T_valid, 23)

    def _make_inputs(garbage_fw: float, garbage_mf: float):
        fw = torch.zeros(B, T_total, W, F)
        fw[:, :T_valid, :, :] = fw_valid
        fw[:, T_valid:, :, :] = garbage_fw
        mf = torch.zeros(B, T_total, 23)
        mf[:, :T_valid, :] = mf_valid
        mf[:, T_valid:, :] = garbage_mf
        mask = torch.zeros(B, T_total, dtype=torch.bool)
        mask[:, T_valid:] = True  # pad
        return fw, mf, mask

    fw_a, mf_a, mask = _make_inputs(0.0, 0.0)
    fw_b, mf_b, _ = _make_inputs(999.0, -999.0)

    with torch.no_grad():
        out_a = model(fw_a, mask, match_feats=mf_a)
        out_b = model(fw_b, mask, match_feats=mf_b)

    assert torch.allclose(out_a, out_b, atol=1e-4), (
        f"Padding invariance violated: out_a={out_a.item():.6f}, "
        f"out_b={out_b.item():.6f}, diff={abs(out_a - out_b).item():.6e}"
    )


# ---------------------------------------------------------------------------
# Test 4 — collate_fn pads (T, 45, 88) windows correctly; pad_mask correct
# ---------------------------------------------------------------------------

def test_collate_fn_frame_sequence() -> None:
    """Two replays of T_win=3 and T_win=5 → padded (2, 5, 45, 88), correct mask."""
    W, F, M = 45, 88, 23
    fw3 = torch.ones(3, W, F) * 1.0
    mf3 = torch.ones(3, M) * 1.0
    fw5 = torch.ones(5, W, F) * 2.0
    mf5 = torch.ones(5, M) * 2.0

    batch = [(fw3, mf3, 0), (fw5, mf5, 1)]
    fw_pad, mf_pad, lengths, labels, pad_mask = frame_collate_fn(batch)

    assert fw_pad.shape == (2, 5, W, F), (
        f"Expected (2, 5, {W}, {F}), got {fw_pad.shape}"
    )
    assert mf_pad is not None
    assert mf_pad.shape == (2, 5, M), (
        f"Expected (2, 5, {M}), got {mf_pad.shape}"
    )
    assert lengths.tolist() == [3, 5], f"Expected [3, 5], got {lengths.tolist()}"
    assert labels.tolist() == [0, 1], f"Expected [0, 1], got {labels.tolist()}"

    # First item: positions 0-2 valid, 3-4 pad
    assert pad_mask[0, :3].sum().item() == 0, "Valid positions must be False in pad_mask"
    assert pad_mask[0, 3:].sum().item() == 2, "Pad positions must be True"
    # Second item: all valid
    assert pad_mask[1, :].sum().item() == 0, "Second item has no padding"

    # Zero-fill in pad region
    assert fw_pad[0, 3:, :, :].abs().max().item() == 0.0, "Pad region not zero-filled"
    assert mf_pad[0, 3:, :].abs().max().item() == 0.0, "Match-feat pad region not zero-filled"


def test_collate_fn_no_match_feats() -> None:
    """collate_fn returns mf_padded=None when match_feats are None."""
    W, F = 45, 88
    batch = [
        (torch.ones(3, W, F), None, 0),
        (torch.ones(4, W, F), None, 1),
    ]
    fw_pad, mf_pad, lengths, labels, pad_mask = frame_collate_fn(batch)

    assert fw_pad.shape == (2, 4, W, F)
    assert mf_pad is None, "Expected mf_padded=None when no match feats"
    assert lengths.tolist() == [3, 4]


# ---------------------------------------------------------------------------
# Test 5 — FrameSequenceDataset on synthetic processed dir
# ---------------------------------------------------------------------------

def test_frame_sequence_dataset_synthetic(tmp_path: pytest.TempPath) -> None:
    """Four replays: 1 bad label → 3 valid; universal mapper accepts all game modes.

    With the 88→133 migration, the universal mapper (frames_df_to_universal)
    handles any game mode (1v1/2v2/3v3) without a per-mode dim filter.
    Previously 1v1 replays were filtered as "wrong_dim_52" — now they are accepted
    and zero-padded to 133 dims.  Only bad-label replays are skipped.
    """
    processed_root = tmp_path / "processed"
    processed_root.mkdir()

    def _write_replay(rid: str, n_frames: int, winning_team: int, n_cars: int = 4) -> None:
        d = processed_root / rid
        d.mkdir()
        df = _make_synthetic_frames(n_frames, winning_team, n_cars=n_cars)
        df.to_parquet(d / "frames.parquet", index=False)
        ldf = pd.DataFrame(
            {
                "flip_reset": np.zeros(n_frames, dtype=np.float32),
                "double_tap": np.zeros(n_frames, dtype=np.float32),
                "aerial_dribble": np.zeros(n_frames, dtype=np.float32),
                "ceiling_shot": np.zeros(n_frames, dtype=np.float32),
            }
        )
        ldf.to_parquet(d / "labels.parquet", index=False)

    # Valid 2v2 replays
    _write_replay("replay_blue_wins", n_frames=90, winning_team=0, n_cars=4)
    _write_replay("replay_orange_wins", n_frames=120, winning_team=1, n_cars=4)

    # Bad label → should be skipped
    _write_replay("replay_unknown_label", n_frames=90, winning_team=-1, n_cars=4)

    # 1v1 replay — was previously filtered as "wrong_dim" but is now accepted
    # because the universal mapper handles all game modes.
    _write_replay("replay_1v1_all_modes", n_frames=90, winning_team=0, n_cars=2)

    from src.models.dataset import WindowSpec
    ds = FrameSequenceDataset(
        processed_root=processed_root,
        replay_ids=[
            "replay_blue_wins",
            "replay_orange_wins",
            "replay_unknown_label",
            "replay_1v1_all_modes",
        ],
        spec=WindowSpec(length=45, stride=15),
        max_windows=200,
        frames_cache_root=None,
        hmlt_cache_root=None,
        with_match_feats=True,
        feature_dim=133,  # universal
    )

    # 3 valid: 2 × 2v2 + 1 × 1v1 (all accepted by universal mapper; only bad_label skipped)
    assert len(ds) == 3, (
        f"Expected 3 valid replays (all-modes accepted), got {len(ds)}. skipped={ds.skipped}"
    )
    assert ds.skipped.get("bad_label", 0) == 1, (
        f"Expected 1 bad_label skip, got {ds.skipped}"
    )

    # Validate __getitem__ contracts
    for idx in range(len(ds)):
        fw_tensor, mf_tensor, label = ds[idx]

        assert isinstance(fw_tensor, torch.Tensor), "fw must be a Tensor"
        assert fw_tensor.dtype == torch.float32, f"Expected float32, got {fw_tensor.dtype}"
        assert fw_tensor.ndim == 3, f"Expected 3-D (T, 45, 133), got {fw_tensor.ndim}-D"
        T_win = fw_tensor.shape[0]
        assert T_win > 0, "At least one window required"
        assert fw_tensor.shape[1] == 45, f"Expected 45 frames, got {fw_tensor.shape[1]}"
        assert fw_tensor.shape[2] == 133, f"Expected 133 universal features, got {fw_tensor.shape[2]}"
        assert not torch.isnan(fw_tensor).any(), "frame_windows contains NaN"
        assert not torch.isinf(fw_tensor).any(), "frame_windows contains Inf"

        assert mf_tensor is not None, "match_feats expected (with_match_feats=True)"
        assert mf_tensor.dtype == torch.float32
        assert mf_tensor.ndim == 2, f"Expected 2-D (T, 23), got {mf_tensor.ndim}-D"
        assert mf_tensor.shape[0] == T_win, (
            f"match_feats T {mf_tensor.shape[0]} != frame_windows T {T_win}"
        )
        assert mf_tensor.shape[1] == 23, f"Expected 23 match features, got {mf_tensor.shape[1]}"
        assert not torch.isnan(mf_tensor).any(), "match_feats contains NaN"

        assert label in (0, 1), f"label must be 0 or 1, got {label}"


# ---------------------------------------------------------------------------
# Test 6 — count_params returns a positive integer
# ---------------------------------------------------------------------------

def test_count_params_hmltv2() -> None:
    """count_params returns a positive integer for both hybrid and non-hybrid."""
    for use_hybrid in (True, False):
        model = HMLTv2(
            n_frames=45, n_features=88, d_win=32, n_match_feats=23,
            use_hybrid=use_hybrid, d_model=32, n_layers=2, n_heads=4, dropout=0.0,
        )
        n = count_params(model)
        assert isinstance(n, int) and n > 0, (
            f"count_params should return positive int, got {n} "
            f"(use_hybrid={use_hybrid})"
        )
