"""Load the saved pipeline outputs the dashboard needs. Nothing is retrained here.

``load_dashboard_data`` reads the processed tables, the similarity engine, the saved
K-Means and PCA models, and checks that they all describe the same players and
clusters. A missing or stale artifact raises ``DashboardArtifactsError`` with the
command that rebuilds it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from scouting.config import Config, load_config
from scouting.dashboard.photos import load_photo_map
from scouting.similarity import SimilarityEngine

POSITION_GROUPS = ["GK", "DEF", "MID", "FWD"]

REQUIRED_FILES = {
    "players_features.csv": "python -m scouting.features.engineering",
    "cluster_assignments.csv": "python -m scouting.clustering fit",
    "cluster_profiles.csv": "python -m scouting.clustering fit",
}


class DashboardArtifactsError(RuntimeError):
    """A pipeline output is missing or does not match the others."""


@dataclass
class DashboardData:
    config: Config
    engine: SimilarityEngine
    players: pd.DataFrame          # feature table (per-90 features + display totals), indexed by player_id
    assignments: pd.DataFrame      # cluster assignment per player, indexed by player_id
    profiles: pd.DataFrame         # cluster profiles (long format)
    pca_coords: dict[str, pd.DataFrame]
    pca_variance: dict[str, np.ndarray]
    percentiles: dict[str, pd.DataFrame]   # feature percentiles within the position group, indexed by player_id
    leagues: list[str] = field(default_factory=list)
    season: str = ""
    photos: dict[int, str] = field(default_factory=dict)          # player_id -> photo URL (optional metadata)
    photo_problems: list[str] = field(default_factory=list)       # entries skipped while loading photos


def _require_files(config: Config) -> None:
    missing = []
    for name, command in REQUIRED_FILES.items():
        if not (config.processed_dir / name).exists():
            missing.append(f"{config.processed_dir / name} (build with: {command})")
    for group in config.modeling.position_feature_lists:
        for path, command in [
            (config.processed_dir / f"model_matrix_{group}.csv", "python -m scouting.features.scaling"),
            (config.processed_dir / f"pca_{group}.csv", "python -m scouting.clustering fit"),
        ]:
            if not path.exists():
                missing.append(f"{path} (build with: {command})")
    for path, command in [
        (config.models_dir / "scalers.joblib", "python -m scouting.features.scaling"),
        (config.models_dir / "kmeans.joblib", "python -m scouting.clustering fit"),
        (config.models_dir / "pca.joblib", "python -m scouting.clustering fit"),
    ]:
        if not path.exists():
            missing.append(f"{path} (build with: {command})")
    if missing:
        raise DashboardArtifactsError("Pipeline outputs are missing:\n- " + "\n- ".join(missing))


def _check_consistency(config, engine, assignments, profiles, kmeans) -> None:
    """The saved models, assignments and profiles must describe the same players."""
    problems = []
    if set(assignments.index) != set(engine.players.index):
        raise DashboardArtifactsError(
            "Pipeline outputs are out of sync (rerun `python -m scouting.clustering fit`):\n"
            "- cluster_assignments.csv covers different players than the similarity pool"
        )
    for group, art in kmeans.items():
        chosen = config.modeling.chosen_k.get(group)
        if art["k"] != chosen:
            problems.append(f"{group}: saved K-Means has K={art['k']} but config chosen_k is {chosen}")
        ids = engine.matrices[group]["player_id"].to_numpy()
        saved = assignments.loc[ids, "cluster_id"].to_numpy()
        if not np.array_equal(saved, art["model"].labels_):
            problems.append(f"{group}: cluster_assignments.csv does not match the saved K-Means labels")
        n_profiled = profiles.loc[profiles["position_group"] == group, "cluster_id"].nunique()
        if n_profiled != art["k"]:
            problems.append(f"{group}: cluster_profiles.csv has {n_profiled} clusters, model has {art['k']}")
    if problems:
        raise DashboardArtifactsError(
            "Pipeline outputs are out of sync (rerun `python -m scouting.clustering fit`):\n- " + "\n- ".join(problems)
        )


def load_dashboard_data(config: Config | None = None) -> DashboardData:
    config = config or load_config()
    _require_files(config)

    engine = SimilarityEngine.from_config(config)          # also verifies the saved scalers
    players = pd.read_csv(config.processed_dir / "players_features.csv").set_index("player_id")
    assignments = pd.read_csv(config.processed_dir / "cluster_assignments.csv").set_index("player_id")
    profiles = pd.read_csv(config.processed_dir / "cluster_profiles.csv")
    kmeans = joblib.load(config.models_dir / "kmeans.joblib")
    pcas = joblib.load(config.models_dir / "pca.joblib")
    _check_consistency(config, engine, assignments, profiles, kmeans)

    groups = list(config.modeling.position_feature_lists)
    pca_coords = {g: pd.read_csv(config.processed_dir / f"pca_{g}.csv") for g in groups}
    for g, coords in pca_coords.items():
        if set(coords["player_id"]) != set(engine.matrices[g]["player_id"]):
            raise DashboardArtifactsError(f"pca_{g}.csv covers different players than the {g} matrix")

    percentiles = {}
    for g in groups:
        feats = config.modeling.position_feature_lists[g]
        ids = engine.matrices[g]["player_id"].to_numpy()
        percentiles[g] = players.loc[ids, feats].rank(pct=True) * 100

    photos, photo_problems = load_photo_map(players, config.metadata_dir / "player_photos.csv")

    return DashboardData(
        config=config,
        engine=engine,
        players=players,
        assignments=assignments,
        profiles=profiles,
        pca_coords=pca_coords,
        pca_variance={g: pcas[g].explained_variance_ratio_ for g in groups},
        percentiles=percentiles,
        leagues=sorted(players["league"].dropna().unique()),
        season=config.season,
        photos=photos,
        photo_problems=photo_problems,
    )
