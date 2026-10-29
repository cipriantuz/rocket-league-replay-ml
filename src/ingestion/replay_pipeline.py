"""Replay → parquet ingestion pipeline.

Per replay produces THREE parquet files in data/processed/<replay_id>/:
    frames.parquet  — one row per frame at 15 Hz (decimated from 30 Hz; RL_DECIMATE_EVERY=1
                      keeps 30 Hz), columns = feature_size() floats (88 in 2v2 mode) +
                      frame_idx + time_s + score + winning_team + next_scoring_team +
                      episode_seconds_remaining + mean controller actions
    labels.parquet  — per-frame 0/1 flags for each rule-based pattern
    events.parquet  — one row per detected PatternEvent (flip_reset, double_tap, etc.)

Pipeline supports multiprocessing for batch parsing of large replay sets.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rlgym_tools.rocket_league.replays.convert import replay_to_rlgym
from rlgym_tools.rocket_league.replays.parsed_replay import ParsedReplay

from src.obs.frame_features import (
    ExtractorConfig,
    extract_frame_features,
    feature_names,
    feature_size,
)
from src.patterns.pattern_pipeline import (
    PATTERN_NAMES,
    augment_events_with_hits,
    build_hit_frame_map,
    detect_all_patterns,
    events_to_dataframe,
    events_to_frame_label_matrix,
    hit_summary_stats,
)

log = logging.getLogger(__name__)

# Replay native rate = 30 Hz. We keep every 2nd frame → unified 15 Hz training rate.
# Configurable via env RL_DECIMATE_EVERY: 2=15Hz (default), 1=30Hz (finer res for scheme test).
import os as _os
DECIMATE_EVERY = int(_os.environ.get("RL_DECIMATE_EVERY", "2"))
TRAIN_HZ = 30 // DECIMATE_EVERY  # 15 Hz at DECIMATE_EVERY=2, 30 Hz at =1


def _detect_n_cars(state) -> int:
    """Round n_cars up to the nearest even value (we pad missing cars)."""
    n = len(state.cars)
    if n <= 2:
        return 2
    if n <= 4:
        return 4
    return 6


def process_replay(replay_path: Path, out_root: Path, force: bool = False) -> dict:
    """Parse one replay, extract features + pattern events, save to parquet.

    Returns a summary dict with timings and counts (used for batch reporting).
    """
    replay_id = replay_path.stem
    out_dir = out_root / replay_id
    frames_pq = out_dir / "frames.parquet"
    events_pq = out_dir / "events.parquet"
    labels_pq = out_dir / "labels.parquet"

    if frames_pq.exists() and labels_pq.exists() and events_pq.exists() and not force:
        return {"replay_id": replay_id, "status": "cached"}

    summary: dict = {"replay_id": replay_id, "status": "ok"}
    t0 = time.time()

    try:
        parsed = ParsedReplay.load(str(replay_path))
        summary["parse_load_s"] = round(time.time() - t0, 2)

        t1 = time.time()
        frames = list(replay_to_rlgym(parsed, interpolation="rocketsim"))
        summary["replay_to_rlgym_s"] = round(time.time() - t1, 2)
        summary["n_frames_native"] = len(frames)

        if len(frames) < 30:
            summary["status"] = "skipped_too_short"
            return summary

        n_cars = _detect_n_cars(frames[0].state)
        cfg = ExtractorConfig(n_cars=n_cars)

        # Detect patterns on native 30 Hz frames + augment with analyzer hits
        t2 = time.time()
        events = detect_all_patterns(frames)
        analyzer_hits = parsed.analyzer.get("hits", []) if hasattr(parsed, "analyzer") else []
        hit_map = build_hit_frame_map(getattr(parsed, "game_df", None), frames)
        events = augment_events_with_hits(events, analyzer_hits, hit_map)
        summary["detect_s"] = round(time.time() - t2, 2)
        summary["n_events"] = len(events)
        summary["hit_stats"] = hit_summary_stats(analyzer_hits)
        summary["validated_events"] = sum(
            1 for e in events if e.metadata.get("validated_by_analyzer_hit")
        )

        # Decimate to 15 Hz and build feature matrix + action matrix (per-car mean)
        t3 = time.time()
        decimated_indices = list(range(0, len(frames), DECIMATE_EVERY))
        feat_dim = feature_size(cfg)
        X = np.zeros((len(decimated_indices), feat_dim), dtype=np.float32)
        # Action mean across all cars in frame (8-dim) — high-level style signal
        action_means = np.zeros((len(decimated_indices), 8), dtype=np.float32)
        meta_rows = []
        for new_idx, src_idx in enumerate(decimated_indices):
            f = frames[src_idx]
            X[new_idx] = extract_frame_features(f.state, config=cfg)
            if f.actions:
                acts = np.stack(list(f.actions.values()))
                action_means[new_idx] = acts.mean(axis=0)
            meta_rows.append(
                {
                    "frame_idx": new_idx,
                    "src_frame_idx": src_idx,
                    "time_s": src_idx / 30.0,
                    "episode_seconds_remaining": f.episode_seconds_remaining,
                    "next_scoring_team": f.next_scoring_team if f.next_scoring_team is not None else -1,
                    "winning_team": f.winning_team if f.winning_team is not None else -1,
                    "blue_score": f.scoreboard.blue_score,
                    "orange_score": f.scoreboard.orange_score,
                    "goal_scored": f.state.goal_scored,
                }
            )
        summary["feature_extract_s"] = round(time.time() - t3, 2)

        # Decimate pattern labels matching the same indices
        label_mat_native = events_to_frame_label_matrix(events, n_frames=len(frames))
        label_mat = label_mat_native[decimated_indices]

        # Persist
        out_dir.mkdir(parents=True, exist_ok=True)
        names = feature_names(cfg)
        feat_df = pd.DataFrame(X, columns=names)
        action_cols = ["act_throttle", "act_steer", "act_pitch", "act_yaw",
                       "act_roll", "act_jump", "act_boost", "act_handbrake"]
        action_df = pd.DataFrame(action_means, columns=action_cols)
        meta_df = pd.DataFrame(meta_rows)
        frames_df = pd.concat([
            meta_df.reset_index(drop=True),
            feat_df.reset_index(drop=True),
            action_df.reset_index(drop=True),
        ], axis=1)
        frames_df.to_parquet(frames_pq, compression="zstd", index=False)

        labels_df = pd.DataFrame(label_mat, columns=PATTERN_NAMES, dtype=np.uint8)
        labels_df.insert(0, "frame_idx", np.arange(len(label_mat), dtype=np.int32))
        labels_df.to_parquet(labels_pq, compression="zstd", index=False)

        events_df = events_to_dataframe(events)
        # Attribute each event to a team (0=blue, 1=orange) via the car's team_num.
        # agent_id->team is stable across the replay; read it from the first frame's
        # car states. Needed by team_profiles.py for per-team mechanic rates.
        agent_team_map = {
            int(aid): int(car.team_num)
            for aid, car in frames[0].state.cars.items()
        }
        if "agent_id" in events_df.columns:
            events_df["agent_team"] = events_df["agent_id"].map(
                lambda a: agent_team_map.get(int(a), -1)
            ).astype("int8")
        # Drop the metadata dict column (not parquet-friendly) into stringified JSON.
        if "metadata" in events_df.columns:
            events_df["metadata"] = events_df["metadata"].apply(lambda d: str(d) if d else "")
        events_df.to_parquet(events_pq, compression="zstd", index=False)

        summary["n_frames_decimated"] = len(decimated_indices)
        summary["n_cars"] = n_cars
        summary["total_s"] = round(time.time() - t0, 2)
        return summary

    except Exception as e:
        summary["status"] = "error"
        summary["error"] = f"{type(e).__name__}: {e}"
        summary["traceback"] = traceback.format_exc().splitlines()[-3:]
        log.error("Failed %s: %s", replay_id, e)
        return summary


def _process_one_worker(args):
    import gc
    replay_path, out_root, force = args
    result = process_replay(Path(replay_path), Path(out_root), force=force)
    # Native parser (rocketsim/boxcars) doesn't always return freed memory to the
    # OS; force a collection each replay so per-worker RSS doesn't creep upward.
    gc.collect()
    return result


def batch_process(
    replay_paths: Iterable[Path],
    out_root: Path,
    *,
    n_workers: int | None = None,
    force: bool = False,
    maxtasksperchild: int = 40,
) -> pd.DataFrame:
    """Process many replays in parallel via multiprocessing.Pool.

    Returns a summary DataFrame with one row per replay.

    maxtasksperchild: recycle each worker after this many replays so the native
    replay parser's (rocketsim/boxcars) un-returned memory is released back to the
    OS. WITHOUT this, long-lived workers leak monotonically -> 32GB OOM on big runs.
    """
    replay_paths = list(replay_paths)
    n_workers = n_workers or max(1, (os.cpu_count() or 4) - 1)
    log.info("Batch processing %d replays with %d workers (maxtasksperchild=%d)",
             len(replay_paths), n_workers, maxtasksperchild)
    args = [(str(p), str(out_root), force) for p in replay_paths]
    results: list[dict] = []

    # Sequential fallback if n_workers == 1
    if n_workers == 1:
        for i, a in enumerate(args, 1):
            r = _process_one_worker(a)
            results.append(r)
            if i % 10 == 0:
                log.info("  Progress: %d/%d, last=%s", i, len(args), r.get("status"))
    else:
        with mp.Pool(processes=n_workers, maxtasksperchild=maxtasksperchild) as pool:
            for i, r in enumerate(pool.imap_unordered(_process_one_worker, args), 1):
                results.append(r)
                if i % 10 == 0 or i == len(args):
                    statuses = pd.Series([x.get("status") for x in results]).value_counts().to_dict()
                    log.info("  Progress: %d/%d, statuses=%s", i, len(args), statuses)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Parse replays → parquet (frames + events)")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--workers", type=int, default=None, help="Default: cpu_count-1")
    parser.add_argument("--limit", type=int, default=None, help="Process only N replays")
    parser.add_argument("--force", action="store_true", help="Re-process even if parquet exists")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Recursive: works for both flat layouts (data/raw/*.replay) and nested
    # dataset trees (data/kaggle_raw/Season-*/SSL/*.replay).
    replays = sorted(args.raw_dir.rglob("*.replay"))
    if args.limit:
        replays = replays[: args.limit]
    if not replays:
        log.error("No replays found (recursive) in %s", args.raw_dir)
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summary_df = batch_process(replays, args.out_dir, n_workers=args.workers, force=args.force)
    elapsed = time.time() - t0

    log.info("=== Batch processing done in %.1fs ===", elapsed)
    log.info("Status counts: %s", summary_df["status"].value_counts().to_dict())
    if "total_s" in summary_df.columns:
        ok = summary_df[summary_df["status"] == "ok"]
        if len(ok) > 0:
            log.info("Per-replay timing: median=%.2fs p95=%.2fs",
                     ok["total_s"].median(), ok["total_s"].quantile(0.95))
    summary_csv = args.out_dir / "_ingest_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    log.info("Summary saved to %s", summary_csv)


if __name__ == "__main__":
    main()
