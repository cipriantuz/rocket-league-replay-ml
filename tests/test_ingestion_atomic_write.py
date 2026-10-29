"""Tests for atomic cache check in replay_pipeline.process_replay (Bug #2)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.replay_pipeline import process_replay


class TestCacheCheckValidatesAllThreeParquets:
    """Bug #2: cache check must require all 3 parquets to exist, not just frames.parquet."""

    def test_only_frames_parquet_present_triggers_reprocessing(self, tmp_path):
        """If only frames.parquet exists (labels/events missing), must NOT return 'cached'.

        Simulates a partial write scenario where the process died mid-way.
        We mock the actual parsing so we don't need a real replay file.
        """
        replay_id = "test_replay_abc"
        out_dir = tmp_path / replay_id
        out_dir.mkdir(parents=True)

        # Create ONLY frames.parquet — labels and events are absent
        (out_dir / "frames.parquet").touch()

        replay_path = tmp_path / f"{replay_id}.replay"
        replay_path.touch()  # empty file — parsing will fail

        result = process_replay(replay_path, tmp_path, force=False)

        # Must NOT be cached — it must attempt processing (and fail on parse, status=error)
        assert result["status"] != "cached", (
            "process_replay should NOT return 'cached' when labels.parquet or "
            "events.parquet are missing, even if frames.parquet exists"
        )

    def test_all_three_parquets_present_returns_cached(self, tmp_path):
        """If all 3 parquets exist and force=False, must return 'cached' immediately."""
        replay_id = "test_replay_xyz"
        out_dir = tmp_path / replay_id
        out_dir.mkdir(parents=True)

        (out_dir / "frames.parquet").touch()
        (out_dir / "labels.parquet").touch()
        (out_dir / "events.parquet").touch()

        replay_path = tmp_path / f"{replay_id}.replay"
        replay_path.touch()

        result = process_replay(replay_path, tmp_path, force=False)

        assert result["status"] == "cached"
        assert result["replay_id"] == replay_id

    def test_force_true_bypasses_cache_even_when_all_three_present(self, tmp_path):
        """force=True must bypass the cache check regardless of parquet presence."""
        replay_id = "test_replay_force"
        out_dir = tmp_path / replay_id
        out_dir.mkdir(parents=True)

        (out_dir / "frames.parquet").touch()
        (out_dir / "labels.parquet").touch()
        (out_dir / "events.parquet").touch()

        replay_path = tmp_path / f"{replay_id}.replay"
        replay_path.touch()

        result = process_replay(replay_path, tmp_path, force=True)

        # force=True: must NOT be cached (will error on parse of empty file — that's fine)
        assert result["status"] != "cached"
