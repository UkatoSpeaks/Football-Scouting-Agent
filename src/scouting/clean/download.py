"""Download the raw player table from Kaggle into data/raw/<season>/."""
from __future__ import annotations

import shutil
from pathlib import Path

from dotenv import load_dotenv

from scouting.config import PROJECT_ROOT, Config, load_config
from scouting.utils.logging import get_logger

log = get_logger(__name__)


def download_raw(config: Config, *, force: bool = False) -> Path:
    dest = config.raw_dir / config.data.raw_filename
    if dest.exists() and not force:
        log.info("raw file already present: %s", dest)
        return dest

    # kagglehub reads KAGGLE_API_TOKEN from the environment.
    load_dotenv(PROJECT_ROOT / ".env")
    import kagglehub

    cache_path = Path(kagglehub.dataset_download(config.data.kaggle_dataset))
    src = cache_path / config.data.raw_filename
    if not src.exists():
        raise FileNotFoundError(f"{config.data.raw_filename} not found in {cache_path}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    log.info("saved raw file: %s", dest)
    return dest


if __name__ == "__main__":
    download_raw(load_config())
