"""BallChasing.com replay downloader — rate-limited + retry + local cache.

Free tier: 2 req/s, max 50 req/min
Premium ($5/mo): ~16 req/s, 1000 req/min, unlimited downloads
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

log = logging.getLogger(__name__)

BALLCHASING_API = "https://ballchasing.com/api"


@dataclass
class BallChasingConfig:
    token: str
    cache_dir: Path
    rate_limit_rps: float = 2.0  # free=2, premium can push higher
    timeout: float = 30.0


class BallChasingClient:
    """Rate-limited BallChasing.com API client with retry."""

    def __init__(self, config: BallChasingConfig):
        self.config = config
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"Authorization": config.token})
        self._last_request_ts = 0.0

    def _throttle(self) -> None:
        min_interval = 1.0 / self.config.rate_limit_rps
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_ts = time.monotonic()

    @retry(
        retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.Timeout)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get(self, url: str, **kwargs) -> requests.Response:
        self._throttle()
        resp = self.session.get(url, timeout=self.config.timeout, **kwargs)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", "10"))
            log.warning("Rate limited, waiting %ds", retry_after)
            time.sleep(retry_after)
            raise requests.exceptions.RequestException("429 backoff")
        resp.raise_for_status()
        return resp

    def list_replays(
        self,
        *,
        playlist: str = "ranked-doubles",
        min_rank: str = "champion-1",
        max_rank: str | None = None,
        season: str | None = None,
        count: int = 200,
        page_after: str | None = None,
    ) -> Iterator[dict]:
        """Yield replay metadata. BallChasing paginates via `next` absolute URL.

        Each response contains a `next` field with a fully-qualified URL that
        ALREADY encodes all the filter parameters from the original request.
        We MUST follow that URL verbatim — passing a separate params dict to
        the next call would either (a) duplicate query params or (b) drop
        filters entirely (latter being the previous bug).
        """
        url = f"{BALLCHASING_API}/replays"
        params: dict | None = {
            "playlist": playlist,
            "min-rank": min_rank,
            "count": min(count, 200),
            "sort-by": "replay-date",
            "sort-dir": "desc",
        }
        if max_rank:
            params["max-rank"] = max_rank
        if season:
            params["season"] = season
        if page_after:
            params["after"] = page_after

        fetched = 0
        seen_ids: set[str] = set()  # defensive: skip duplicates across pages
        while fetched < count:
            resp = self._get(url, params=params)
            data = resp.json()
            replays = data.get("list", [])
            if not replays:
                break
            new_in_page = 0
            for r in replays:
                rid = r.get("id")
                if rid is None or rid in seen_ids:
                    continue
                seen_ids.add(rid)
                new_in_page += 1
                yield r
                fetched += 1
                if fetched >= count:
                    return
            if new_in_page == 0:
                # Pagination cursor stuck — bail to avoid infinite loop
                break
            next_url = data.get("next")
            if not next_url:
                break
            # next URL is absolute and already encodes all filters/cursor
            url = next_url
            params = None

    def list_group_replays(self, group_id: str, count: int = 1000) -> Iterator[dict]:
        """List all replays inside a BallChasing group (e.g. an RLCS event)."""
        params: dict = {"group": group_id, "count": 200}
        fetched = 0
        while fetched < count:
            resp = self._get(f"{BALLCHASING_API}/replays", params=params)
            data = resp.json()
            replays = data.get("list", [])
            if not replays:
                break
            for r in replays:
                yield r
                fetched += 1
                if fetched >= count:
                    return
            next_url = data.get("next")
            if not next_url:
                break
            # next_url is a fully-qualified URL; switch the URL but drop params for the next call
            params = {}

    def download_replay(self, replay_id: str, *, force: bool = False) -> Path:
        """Download a single .replay file to cache_dir. Returns local path."""
        out_path = self.config.cache_dir / f"{replay_id}.replay"
        if out_path.exists() and not force:
            return out_path
        resp = self._get(f"{BALLCHASING_API}/replays/{replay_id}/file", stream=True)
        tmp = out_path.with_suffix(".replay.partial")
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.rename(out_path)
        return out_path

    def batch_download(
        self,
        replay_ids: list[str],
        *,
        max_failures: int = 50,
    ) -> dict[str, Path | Exception]:
        """Download many replays sequentially with progress bar."""
        results: dict[str, Path | Exception] = {}
        failures = 0
        for rid in tqdm(replay_ids, desc="Downloading replays", unit="replay"):
            try:
                results[rid] = self.download_replay(rid)
            except Exception as e:
                results[rid] = e
                failures += 1
                if failures >= max_failures:
                    log.error("Reached %d failures, aborting", max_failures)
                    break
        return results


def load_token(env_var: str = "BALLCHASING_TOKEN") -> str:
    """Read the ballchasing.com API key from an environment variable (never from a file)."""
    token = os.environ.get(env_var)
    if token:
        return token.strip()
    raise RuntimeError(
        f"BallChasing API key not found. Set the {env_var} environment variable. "
        "Get your own key at https://ballchasing.com/upload (Settings -> API Key)."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download replays from BallChasing.com")
    parser.add_argument("--count", type=int, default=20, help="Number of replays to download")
    parser.add_argument("--min-rank", default="grand-champion-1", help="Minimum rank filter")
    parser.add_argument("--playlist", default="ranked-doubles", help="Playlist filter")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw"), help="Output dir")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    token = load_token()
    cfg = BallChasingConfig(token=token, cache_dir=args.cache_dir)
    client = BallChasingClient(cfg)

    log.info("Listing %d replays (min_rank=%s, playlist=%s)", args.count, args.min_rank, args.playlist)
    replay_meta = list(
        client.list_replays(
            playlist=args.playlist,
            min_rank=args.min_rank,
            count=args.count,
        )
    )
    log.info("Got %d replay metadata entries", len(replay_meta))

    if replay_meta:
        ids = [r["id"] for r in replay_meta]
        results = client.batch_download(ids)
        ok = sum(1 for v in results.values() if isinstance(v, Path))
        log.info("Downloaded %d/%d replays to %s", ok, len(results), args.cache_dir)
