"""Shared synthetic fixture: a small processed-replay tree, no real .replay needed.

Mimics what src/ingestion/replay_pipeline.py writes per replay:
    <root>/<replay_id>/frames.parquet  — meta columns + 88 physics features (2v2)
    <root>/<replay_id>/labels.parquet  — per-frame pattern flags
    <root>/<replay_id>/events.parquet  — rule-based detector events
Replays are real .replay files from ballchasing.com and are not redistributed, so the
tests start from the frame table that the parser produces.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.obs.frame_features import ExtractorConfig, feature_names

N_REPLAYS = 12
N_FRAMES = 240  # 16 s at 15 Hz


def _frames_table(rng: np.random.Generator, blue_wins: bool) -> pd.DataFrame:
    names = feature_names(ExtractorConfig(n_cars=4))
    X = rng.normal(0.0, 0.3, size=(N_FRAMES, len(names))).astype(np.float32)
    df = pd.DataFrame(X, columns=names)
    # Binary-ish columns stay in {0, 1}
    for c in names:
        if c.endswith(("_on_ground", "_has_flip", "_has_jumped", "_is_supersonic")):
            df[c] = (rng.random(N_FRAMES) > 0.5).astype(np.float32)
        if c.endswith("_is_orange"):
            df[c] = 1.0 if "orange" in c else 0.0
    # A weak, learnable signal: the winning side keeps the ball in the other half
    sign = 1.0 if blue_wins else -1.0
    df["ball_pos_y"] = sign * 0.4 + rng.normal(0.0, 0.2, N_FRAMES)

    t = np.arange(N_FRAMES)
    final_blue, final_orange = (2, 1) if blue_wins else (1, 2)
    meta = pd.DataFrame({
        "frame_idx": t,
        "src_frame_idx": t * 2,
        "time_s": t / 15.0,
        "episode_seconds_remaining": 300.0 - t / 15.0,
        "next_scoring_team": 0 if blue_wins else 1,
        "winning_team": 0 if blue_wins else 1,
        "blue_score": np.where(t > N_FRAMES // 2, final_blue, 0),
        "orange_score": np.where(t > N_FRAMES // 2, final_orange, 0),
        "goal_scored": np.isin(t, [N_FRAMES // 2 + 1]),
    })
    return pd.concat([meta, df], axis=1)


def _events_table(replay_idx: int) -> pd.DataFrame:
    # One validated ceiling shot in the middle of every replay; players 1000+idx and a
    # player shared by replays 0 and 1 (agent 7), used by the player-fold test.
    agents = [1000 + replay_idx, 7 if replay_idx in (0, 1) else 2000 + replay_idx]
    return pd.DataFrame({
        "pattern": ["ceiling_shot", "simple_shot"],
        "frame_start": [100, 180],
        "frame_end": [110, 185],
        "confidence": [0.95, 0.9],
        "agent_id": agents,
        "metadata": ["{'validated_by_analyzer_hit': True}", ""],
    })


@pytest.fixture()
def processed_root(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    root = tmp_path / "processed"
    for i in range(N_REPLAYS):
        d = root / f"replay_{i:02d}"
        d.mkdir(parents=True)
        frames = _frames_table(rng, blue_wins=(i % 2 == 0))
        frames.to_parquet(d / "frames.parquet", index=False)
        labels = pd.DataFrame({"frame_idx": np.arange(N_FRAMES, dtype=np.int32)})
        labels.to_parquet(d / "labels.parquet", index=False)
        _events_table(i).to_parquet(d / "events.parquet", index=False)
    return root
