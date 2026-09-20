from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class DataConfig:
    kaggle_dataset: str
    raw_filename: str


@dataclass
class ScrapeConfig:
    base_url: str
    min_delay_seconds: float
    cache_ttl_days: int
    user_agent: str


@dataclass
class FeatureConfig:
    min_minutes: int
    per90_stats: list[str] = field(default_factory=list)
    keep_as_is_stats: list[str] = field(default_factory=list)
    ratios: dict[str, list[str]] = field(default_factory=dict)
    ratio_prior_strength: float = 10.0


@dataclass
class ModelingConfig:
    k_range: list[int]
    random_state: int
    position_feature_lists: dict[str, list[str]] = field(default_factory=dict)
    log1p_features: dict[str, list[str]] = field(default_factory=dict)
    log1p_skew_threshold: float = 1.0
    kmeans_n_init: int = 20
    chosen_k: dict[str, int] = field(default_factory=dict)


@dataclass
class Config:
    season: str
    data: DataConfig
    scrape: ScrapeConfig
    feature_engineering: FeatureConfig
    modeling: ModelingConfig

    @property
    def raw_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "raw" / self.season

    @property
    def interim_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "interim" / self.season

    @property
    def processed_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "processed" / self.season

    @property
    def cache_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "cache" / "http_cache"

    @property
    def models_dir(self) -> Path:
        return PROJECT_ROOT / "models" / self.season

    @property
    def metadata_dir(self) -> Path:
        """Hand-maintained presentation metadata (for example player photo URLs); not a model input."""
        return PROJECT_ROOT / "data" / "metadata"

    @property
    def figures_dir(self) -> Path:
        return PROJECT_ROOT / "reports" / self.season


def load_config(path: Path = CONFIG_PATH) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(
        season=raw["season"],
        data=DataConfig(**raw["data"]),
        scrape=ScrapeConfig(**raw["scrape"]),
        feature_engineering=FeatureConfig(**raw["feature_engineering"]),
        modeling=ModelingConfig(**raw["modeling"]),
    )
