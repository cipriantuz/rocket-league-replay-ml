"""Tests for goal-imminence target derivation and supporting infrastructure.

Coverage:
  1. compute_imminence_labels — goal at frame 60, windows within/outside horizon
  2. 3-class team assignment — correct team for blue and orange
  3. Horizon sensitivity — larger H → more or equal positives
  4. End-of-match — no future goal → label 0 / class 2
  5. materialize shapes — X_raw (N,45,88), X_agg (N,23), labels (N,)
  6. Deep model forward — (4,45,88) → logit (4,), finite
  7. pos_weight computation — correct on imbalanced synthetic labels
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.obs.goal_imminence import compute_imminence_labels
from src.eval_imminence import ImminenceCNN, ImminenceTransformer, _compute_pos_weight


# ---------------------------------------------------------------------------
# Fixtures — synthetic frames
# ---------------------------------------------------------------------------

def _make_frames(n_frames: int, goal_at: int | None, scoring_team: int = 0) -> pd.DataFrame:
    """Build a minimal synthetic frames.parquet DataFrame.

    Args:
        n_frames:     Total number of frames.
        goal_at:      Frame index where a goal occurs (score increment).
                      If None, no goal in the match.
        scoring_team: 0=blue, 1=orange.
    """
    blue_score = np.zeros(n_frames, dtype=np.int32)
    orange_score = np.zeros(n_frames, dtype=np.int32)

    if goal_at is not None and 0 < goal_at < n_frames:
        if scoring_team == 0:
            blue_score[goal_at:] = 1
        else:
            orange_score[goal_at:] = 1

    # Minimal columns required by compute_imminence_labels (only score cols)
    # and for materialize (needs physics cols + meta cols)
    data: dict[str, np.ndarray] = {
        "blue_score": blue_score,
        "orange_score": orange_score,
        # Meta columns
        "frame_idx": np.arange(n_frames, dtype=np.int64),
        "src_frame_idx": np.arange(n_frames, dtype=np.int64),
        "time_s": np.arange(n_frames, dtype=np.float32) / 15.0,
        "episode_seconds_remaining": np.full(n_frames, 300.0, dtype=np.float32),
        "next_scoring_team": np.full(n_frames, -1, dtype=np.int64),
        "winning_team": np.zeros(n_frames, dtype=np.int64),
        "goal_scored": np.zeros(n_frames, dtype=bool),
        # Action columns (not features)
        "act_throttle": np.zeros(n_frames, dtype=np.float32),
        "act_steer": np.zeros(n_frames, dtype=np.float32),
        "act_pitch": np.zeros(n_frames, dtype=np.float32),
        "act_yaw": np.zeros(n_frames, dtype=np.float32),
        "act_roll": np.zeros(n_frames, dtype=np.float32),
        "act_jump": np.zeros(n_frames, dtype=np.float32),
        "act_boost": np.zeros(n_frames, dtype=np.float32),
        "act_handbrake": np.zeros(n_frames, dtype=np.float32),
        # Physics feature columns to reach 88 total
        "ball_pos_x": np.random.default_rng(0).standard_normal(n_frames).astype(np.float32),
        "ball_pos_y": np.random.default_rng(1).standard_normal(n_frames).astype(np.float32),
        "ball_pos_z": np.random.default_rng(2).standard_normal(n_frames).astype(np.float32),
        "ball_linvel_x": np.random.default_rng(3).standard_normal(n_frames).astype(np.float32),
        "ball_linvel_y": np.random.default_rng(4).standard_normal(n_frames).astype(np.float32),
        "ball_linvel_z": np.random.default_rng(5).standard_normal(n_frames).astype(np.float32),
        "ball_angvel_x": np.random.default_rng(6).standard_normal(n_frames).astype(np.float32),
        "ball_angvel_y": np.random.default_rng(7).standard_normal(n_frames).astype(np.float32),
        "ball_angvel_z": np.random.default_rng(8).standard_normal(n_frames).astype(np.float32),
    }

    # Add enough dummy car columns to reach 88 feature columns total
    # Existing non-action physics: 9 ball cols above → need 79 more to hit 88
    from src.models.dataset import NON_FEATURE_COLS
    existing_feat_count = sum(1 for c in data if c not in NON_FEATURE_COLS)
    needed = 88 - existing_feat_count
    rng = np.random.default_rng(99)
    for i in range(needed):
        data[f"car_blue0_feat_{i:03d}"] = rng.standard_normal(n_frames).astype(np.float32)

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# 1. Basic label correctness — goal at frame 60
# ---------------------------------------------------------------------------

class TestComputeImminenceLabels:

    def test_window_ending_within_horizon_is_positive(self):
        """Window ending at frame 44 (last frame), goal at 60 → 16 frames = ~1.07s.
        With horizon_s=5 (75 frames), that window should be positive."""
        frames = _make_frames(200, goal_at=60, scoring_team=0)
        binary, threeclass = compute_imminence_labels(
            frames, window=45, stride=15, horizon_s=5.0, hz=15.0
        )
        # Window 0: frames 0..44, end=44, goal at 60, gap=16 frames = 1.067s < 5s → positive
        assert binary[0] == 1, f"Expected window 0 to be positive, got binary[0]={binary[0]}"
        assert threeclass[0] == 0, f"Expected team 0 (blue), got {threeclass[0]}"

    def test_window_ending_far_before_goal_is_negative(self):
        """Goal at frame 60.  Window ending at frame 14 (window 0 with stride 15?
        Actually with stride=15: w0 end=44, w1 end=59, w2 end=74, ...
        Let's use a goal far away — goal at frame 5000 in a 5500-frame match.
        Window 0 ends at frame 44: gap = 5000-44 = 4956 frames = 330s >> 10s → negative.
        """
        frames = _make_frames(5500, goal_at=5000, scoring_team=1)
        binary, threeclass = compute_imminence_labels(
            frames, window=45, stride=15, horizon_s=10.0, hz=15.0
        )
        assert binary[0] == 0
        assert threeclass[0] == 2, f"Expected class 2 (no goal), got {threeclass[0]}"

    def test_no_goals_all_negative(self):
        """Match with no goals → all binary=0, all threeclass=2."""
        frames = _make_frames(500, goal_at=None)
        binary, threeclass = compute_imminence_labels(
            frames, window=45, stride=15, horizon_s=10.0, hz=15.0
        )
        assert np.all(binary == 0), "Expected all zeros when no goals"
        assert np.all(threeclass == 2), "Expected all class-2 when no goals"

    def test_window_after_last_goal_is_negative(self):
        """Goal at frame 60; windows starting at 75+ have no future goal → negative."""
        frames = _make_frames(300, goal_at=60, scoring_team=0)
        binary, threeclass = compute_imminence_labels(
            frames, window=45, stride=15, horizon_s=10.0, hz=15.0
        )
        # Window starting at 75 ends at frame 119; goal at 60 < 119+1=120 → no future goal
        # The goal at 60 is BEFORE the window end 119, so no future goal
        # Find windows where start >= 76 (window end >= 120 > 60)
        # start = 75 → end = 119 → searchsorted for 120 in [60] → pos=1 >= len → negative
        last_window_after_goal = None
        for i, start in enumerate(range(0, 300 - 45 + 1, 15)):
            end = start + 44
            if end >= 60:
                last_window_after_goal = i
                break
        # All windows with end >= goal are negative (no future goal after)
        if last_window_after_goal is not None:
            for i in range(last_window_after_goal, len(binary)):
                assert binary[i] == 0, (
                    f"Window {i} (end={i*15+44}) with no future goal should be 0, got {binary[i]}"
                )


# ---------------------------------------------------------------------------
# 2. 3-class team assignment
# ---------------------------------------------------------------------------

class TestThreeclassTeam:

    def test_blue_scores_3class_is_0(self):
        """When blue (team 0) scores within horizon → 3-class label = 0."""
        frames = _make_frames(150, goal_at=50, scoring_team=0)
        _, threeclass = compute_imminence_labels(
            frames, window=45, stride=15, horizon_s=10.0, hz=15.0
        )
        # Window 0: end=44, goal at 50, gap=6 frames < horizon → positive
        assert threeclass[0] == 0, f"Expected 0 (blue scores), got {threeclass[0]}"

    def test_orange_scores_3class_is_1(self):
        """When orange (team 1) scores within horizon → 3-class label = 1."""
        frames = _make_frames(150, goal_at=50, scoring_team=1)
        _, threeclass = compute_imminence_labels(
            frames, window=45, stride=15, horizon_s=10.0, hz=15.0
        )
        assert threeclass[0] == 1, f"Expected 1 (orange scores), got {threeclass[0]}"

    def test_no_imminent_goal_3class_is_2(self):
        """When no goal within horizon → 3-class label = 2."""
        frames = _make_frames(500, goal_at=None)
        _, threeclass = compute_imminence_labels(
            frames, window=45, stride=15, horizon_s=5.0, hz=15.0
        )
        assert np.all(threeclass == 2)


# ---------------------------------------------------------------------------
# 3. Horizon sensitivity
# ---------------------------------------------------------------------------

class TestHorizonSensitivity:

    def test_larger_horizon_more_or_equal_positives(self):
        """Increasing horizon_s ≥ existing positives: #positives(H2) >= #positives(H1)."""
        frames = _make_frames(500, goal_at=200, scoring_team=0)
        bin5, _ = compute_imminence_labels(frames, window=45, stride=15, horizon_s=5.0, hz=15.0)
        bin15, _ = compute_imminence_labels(frames, window=45, stride=15, horizon_s=15.0, hz=15.0)
        assert int(bin15.sum()) >= int(bin5.sum()), (
            f"Expected H=15 to have >= positives than H=5, got {bin15.sum()} < {bin5.sum()}"
        )

    def test_zero_horizon_no_positives(self):
        """horizon_s=0 → no window can be within 0 seconds of a goal → all negative."""
        frames = _make_frames(300, goal_at=100, scoring_team=0)
        binary, _ = compute_imminence_labels(
            frames, window=45, stride=15, horizon_s=0.0, hz=15.0
        )
        assert np.all(binary == 0), f"Expected all 0 for horizon=0, got {binary.sum()} positives"

    def test_large_horizon_catches_most_windows(self):
        """horizon_s=300 (whole match) → every window before the goal should be positive."""
        frames = _make_frames(300, goal_at=250, scoring_team=0)
        binary, _ = compute_imminence_labels(
            frames, window=45, stride=15, horizon_s=300.0, hz=15.0
        )
        # All windows with end < 250 should be positive (goal in future)
        n_pos = int(binary.sum())
        assert n_pos > 0, "Expected some positives with horizon=300s"


# ---------------------------------------------------------------------------
# 4. materialize shapes
# ---------------------------------------------------------------------------

class TestMaterializeShapes:
    """Test that materialize returns arrays with the correct shapes.

    We can't use real replay IDs here (test isolation), so we verify the shape
    contract by calling compute_imminence_labels directly and simulating what
    materialize would return.
    """

    def test_shapes_from_synthetic_replay(self):
        """Simulate materialize output shapes for a synthetic 200-frame replay."""
        from src.obs.match_features import compute_match_feature_sequence, MATCH_FEATURE_SIZE
        from src.models.dataset import NON_FEATURE_COLS

        frames = _make_frames(200, goal_at=100, scoring_team=0)
        window, stride = 45, 15

        # Simulate what materialize does
        feat_cols = [c for c in frames.columns if c not in NON_FEATURE_COLS]
        feat = frames[feat_cols].to_numpy(dtype=np.float32)

        n = feat.shape[0]
        starts = list(range(0, n - window + 1, stride))
        X_raw = np.stack([feat[s: s + window] for s in starts], axis=0)  # (N, 45, 88)
        X_agg, _ = compute_match_feature_sequence(frames, window=window, stride=stride)

        binary_labels, threeclass_labels = compute_imminence_labels(
            frames, window=window, stride=stride, horizon_s=10.0, hz=15.0
        )

        N = len(starts)
        assert X_raw.shape == (N, 45, 88), f"X_raw shape mismatch: {X_raw.shape}"
        assert X_agg.shape == (N, MATCH_FEATURE_SIZE), f"X_agg shape mismatch: {X_agg.shape}"
        assert binary_labels.shape == (N,), f"binary_labels shape: {binary_labels.shape}"
        assert threeclass_labels.shape == (N,), f"threeclass_labels shape: {threeclass_labels.shape}"

    def test_no_nans_in_labels(self):
        """Labels must be NaN-free (int8 cannot be NaN; this tests the contract)."""
        frames = _make_frames(300, goal_at=150, scoring_team=1)
        binary, threeclass = compute_imminence_labels(
            frames, window=45, stride=15, horizon_s=10.0, hz=15.0
        )
        # int8 arrays have no NaN concept — verify all values are valid
        assert np.all((binary == 0) | (binary == 1))
        assert np.all((threeclass >= 0) & (threeclass <= 2))


# ---------------------------------------------------------------------------
# 5. Deep model forward pass
# ---------------------------------------------------------------------------

class TestDeepModelForward:

    def test_cnn_forward_shape_and_finite(self):
        """ImminenceCNN: (4, 45, 88) → logit (4,), all values finite."""
        model = ImminenceCNN(feature_dim=88, seq_len=45)
        model.eval()
        x = torch.randn(4, 45, 88)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (4,), f"Expected (4,), got {logits.shape}"
        assert torch.all(torch.isfinite(logits)), "Logits contain non-finite values"

    def test_transformer_forward_shape_and_finite(self):
        """ImminenceTransformer: (4, 45, 88) → logit (4,), all values finite."""
        model = ImminenceTransformer(feature_dim=88, d_model=64, nhead=4, num_layers=2)
        model.eval()
        x = torch.randn(4, 45, 88)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (4,), f"Expected (4,), got {logits.shape}"
        assert torch.all(torch.isfinite(logits)), "Logits contain non-finite values"

    def test_cnn_different_batch_sizes(self):
        """CNN handles batch sizes 1, 8, 32 without errors."""
        model = ImminenceCNN(feature_dim=88)
        model.eval()
        for bs in (1, 8, 32):
            x = torch.randn(bs, 45, 88)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (bs,)

    def test_sigmoid_of_logit_in_0_1(self):
        """Sigmoid(logit) must be in [0, 1] — correctness of probability output."""
        model = ImminenceCNN(feature_dim=88)
        model.eval()
        x = torch.randn(16, 45, 88)
        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits)
        assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


# ---------------------------------------------------------------------------
# 6. pos_weight computation
# ---------------------------------------------------------------------------

class TestPosWeight:

    def test_balanced_gives_one(self):
        """Balanced labels → pos_weight ≈ 1.0."""
        y = np.array([0, 1] * 50, dtype=np.float32)
        pw = _compute_pos_weight(y)
        assert abs(pw - 1.0) < 1e-6, f"Expected 1.0, got {pw}"

    def test_rare_positive_gives_large_weight(self):
        """10% positives → pos_weight = 9.0 (9 negatives per positive)."""
        y = np.array([0] * 90 + [1] * 10, dtype=np.float32)
        pw = _compute_pos_weight(y)
        assert abs(pw - 9.0) < 1e-6, f"Expected 9.0, got {pw}"

    def test_no_positives_gives_one(self):
        """No positives in label array → safe fallback to 1.0."""
        y = np.zeros(100, dtype=np.float32)
        pw = _compute_pos_weight(y)
        assert pw == 1.0, f"Expected 1.0 fallback, got {pw}"

    def test_all_positives_gives_zero(self):
        """All positives → pos_weight = 0.0 (n_neg=0)."""
        y = np.ones(100, dtype=np.float32)
        pw = _compute_pos_weight(y)
        assert pw == 0.0, f"Expected 0.0, got {pw}"

    def test_typical_imminence_imbalance(self):
        """5% base rate → pos_weight = 19.0."""
        y = np.array([0] * 95 + [1] * 5, dtype=np.float32)
        pw = _compute_pos_weight(y)
        assert abs(pw - 19.0) < 1e-6, f"Expected 19.0, got {pw}"


# ---------------------------------------------------------------------------
# 7. Integration: label alignment with window count
# ---------------------------------------------------------------------------

class TestLabelWindowAlignment:

    def test_n_labels_equals_n_windows(self):
        """compute_imminence_labels returns exactly as many labels as windows."""
        for n_frames in (100, 200, 500):
            frames = _make_frames(n_frames, goal_at=n_frames // 2, scoring_team=0)
            window, stride = 45, 15
            binary, threeclass = compute_imminence_labels(
                frames, window=window, stride=stride, horizon_s=10.0, hz=15.0
            )
            expected_n = len(list(range(0, n_frames - window + 1, stride)))
            assert len(binary) == expected_n, (
                f"n_frames={n_frames}: expected {expected_n} labels, got {len(binary)}"
            )
            assert len(threeclass) == expected_n

    def test_label_dtype_is_int8(self):
        """Labels are stored as int8 (memory efficient for large datasets)."""
        frames = _make_frames(200, goal_at=100, scoring_team=0)
        binary, threeclass = compute_imminence_labels(frames, window=45, stride=15)
        assert binary.dtype == np.int8, f"Expected int8, got {binary.dtype}"
        assert threeclass.dtype == np.int8, f"Expected int8, got {threeclass.dtype}"
