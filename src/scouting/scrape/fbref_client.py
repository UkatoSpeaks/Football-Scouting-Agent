from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from scouting.config import Config
from scouting.utils.logging import get_logger

log = get_logger(__name__)

_last_request_at: float | None = None


class FetchError(RuntimeError):
    pass


def _cache_key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def _cache_paths(cache_dir: Path, url: str) -> tuple[Path, Path]:
    key = _cache_key(url)
    return cache_dir / f"{key}.html", cache_dir / f"{key}.meta.json"


def _read_cache(cache_dir: Path, url: str, ttl_days: int) -> str | None:
    html_path, meta_path = _cache_paths(cache_dir, url)
    if not html_path.exists() or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fetched_at = datetime.fromisoformat(meta["fetched_at"])
    if datetime.now(timezone.utc) - fetched_at > timedelta(days=ttl_days):
        return None
    return html_path.read_text(encoding="utf-8")


def _write_cache(cache_dir: Path, url: str, html: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    html_path, meta_path = _cache_paths(cache_dir, url)
    html_path.write_text(html, encoding="utf-8")
    meta_path.write_text(json.dumps({"url": url, "fetched_at": datetime.now(timezone.utc).isoformat()}), encoding="utf-8")


def _throttle(min_delay_seconds: float) -> None:
    global _last_request_at
    if _last_request_at is not None:
        elapsed = time.monotonic() - _last_request_at
        remaining = min_delay_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)
    _last_request_at = time.monotonic()


def fetch(url: str, config: Config, *, force_refresh: bool = False, max_retries: int = 3) -> str:
    if not force_refresh:
        cached = _read_cache(config.cache_dir, url, config.scrape.cache_ttl_days)
        if cached is not None:
            log.info("cache hit: %s", url)
            return cached

    headers = {"User-Agent": config.scrape.user_agent}
    backoff = config.scrape.min_delay_seconds

    for attempt in range(1, max_retries + 1):
        _throttle(config.scrape.min_delay_seconds)
        log.info("fetching (attempt %d/%d): %s", attempt, max_retries, url)
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code == 200:
            _write_cache(config.cache_dir, url, response.text)
            return response.text

        if response.status_code == 429 and attempt < max_retries:
            log.warning("429 rate-limited, backing off %.0fs", backoff)
            time.sleep(backoff)
            backoff *= 2
            continue

        raise FetchError(f"GET {url} failed with status {response.status_code}")

    raise FetchError(f"GET {url} failed after {max_retries} attempts")
