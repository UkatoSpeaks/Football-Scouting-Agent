"""K-Means player clustering, one model per position group.

Each group is clustered on its own standardized matrix (see
scouting.features.scaling); a goalkeeper never shares a feature space with a
forward. K is NOT picked automatically: ``evaluate_k`` produces inertia,
silhouette, cluster-size and stability diagnostics, K is chosen by a human from
those plus cluster interpretability, and the choice is stored in config.yaml
(``modeling.chosen_k``).

Cluster labels are reordered so cluster 0 is the largest, 1 the next, and so on,
which keeps ids stable and readable. Clusters are described from their
statistics only (see ``describe_cluster``); nothing is named by hand.
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score

from scouting.config import Config, load_config
from scouting.labels import label
from scouting.utils.logging import get_logger

log = get_logger(__name__)

STABILITY_SEEDS = 4
ASSIGNMENT_COLUMNS = [
    "player_id", "player_name", "league", "club", "position", "position_group", "minutes",
    "cluster_id", "cluster_key",
]


# ------------------------------------------------------------------- loading
def load_matrices(config: Config) -> dict[str, pd.DataFrame]:
    return {
        g: pd.read_csv(config.processed_dir / f"model_matrix_{g}.csv")
        for g in config.modeling.position_feature_lists
    }


def load_players(config: Config) -> pd.DataFrame:
    return pd.read_csv(config.processed_dir / "players_features.csv").set_index("player_id")


def feature_columns(matrix: pd.DataFrame) -> list[str]:
    return [c for c in matrix.columns if c != "player_id"]


# ---------------------------------------------------------------- evaluation
def evaluate_k(
    Z: np.ndarray, k_values, random_state: int = 42, n_init: int = 20, group: str = ""
) -> pd.DataFrame:
    """Inertia, silhouette, size and stability diagnostics for each K.

    stability_ari = mean adjusted Rand index between the reference solution and
    solutions from other random seeds (1.0 = the same clusters every time).
    min_cluster_silhouette = mean silhouette of the worst cluster (low or
    negative means one cluster is poorly separated).
    """
    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state).fit(Z)
        sizes = np.bincount(km.labels_, minlength=k)
        per_sample = silhouette_samples(Z, km.labels_)
        aris = [
            adjusted_rand_score(
                km.labels_,
                KMeans(n_clusters=k, n_init=n_init, random_state=random_state + 1 + i).fit(Z).labels_,
            )
            for i in range(STABILITY_SEEDS)
        ]
        rows.append({
            "position_group": group,
            "k": k,
            "inertia": km.inertia_,
            "silhouette": float(per_sample.mean()),
            "min_cluster_size": int(sizes.min()),
            "max_cluster_size": int(sizes.max()),
            "min_cluster_share": float(sizes.min() / len(Z)),
            "min_cluster_silhouette": float(min(per_sample[km.labels_ == c].mean() for c in range(k))),
            "stability_ari": float(np.mean(aris)),
        })
    return pd.DataFrame(rows)


def evaluate_all(matrices: dict[str, pd.DataFrame], config: Config) -> pd.DataFrame:
    lo, hi = config.modeling.k_range
    frames = []
    for group, matrix in matrices.items():
        Z = matrix[feature_columns(matrix)].to_numpy()
        log.info("evaluating K=%d..%d for %s (%d players)", lo, hi, group, len(Z))
        frames.append(evaluate_k(Z, range(lo, hi + 1), config.modeling.random_state,
                                 config.modeling.kmeans_n_init, group))
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------- fitting
def fit_kmeans(Z: np.ndarray, k: int, random_state: int = 42, n_init: int = 20) -> KMeans:
    """Fit K-Means, then relabel so cluster 0 is the largest (ties: lower old label)."""
    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state).fit(Z)
    order = np.argsort(-np.bincount(km.labels_, minlength=k), kind="stable")
    # Refit from the reordered centres: converges immediately to the same partition
    # but yields a genuine fitted model whose labels are already size-ordered.
    ordered = KMeans(n_clusters=k, init=km.cluster_centers_[order], n_init=1,
                     random_state=random_state).fit(Z)
    if not np.array_equal(ordered.labels_, np.argsort(order, kind="stable")[km.labels_]):
        raise RuntimeError("relabeling changed the partition")
    return ordered


# ----------------------------------------------------------------- profiling
def cluster_profiles(
    matrix: pd.DataFrame, labels: np.ndarray, raw: pd.DataFrame, group: str
) -> pd.DataFrame:
    """Per cluster and feature: mean/median z-score and mean raw value (long format)."""
    feats = feature_columns(matrix)
    Z = matrix[feats].reset_index(drop=True)
    R = raw.loc[matrix["player_id"], feats].reset_index(drop=True)
    rows = []
    for c in sorted(np.unique(labels)):
        mask = labels == c
        for f in feats:
            rows.append({
                "position_group": group, "cluster_id": int(c), "n_players": int(mask.sum()),
                "feature": f, "mean_z": Z.loc[mask, f].mean(), "median_z": Z.loc[mask, f].median(),
                "mean_raw": R.loc[mask, f].mean(),
            })
    return pd.DataFrame(rows)


def describe_cluster(profile: pd.DataFrame, threshold: float = 0.5, top: int = 5) -> dict[str, list[tuple[str, float]]]:
    """High/low traits of one cluster, straight from its mean z-scores.

    profile: the rows of one cluster (columns feature, mean_z). A feature is
    "high" at mean_z >= threshold and "low" at mean_z <= -threshold; at most
    ``top`` of each, strongest first.
    """
    p = profile.set_index("feature")["mean_z"]
    high = p[p >= threshold].sort_values(ascending=False).head(top)
    low = p[p <= -threshold].sort_values().head(top)
    return {
        "high": [(label(f), float(v)) for f, v in high.items()],
        "low": [(label(f), float(v)) for f, v in low.items()],
    }


def representatives(Z: np.ndarray, km: KMeans, cluster: int, n: int = 5) -> np.ndarray:
    """Row indices of the players closest to a cluster's centre."""
    idx = np.where(km.labels_ == cluster)[0]
    d = np.linalg.norm(Z[idx] - km.cluster_centers_[cluster], axis=1)
    return idx[np.argsort(d, kind="stable")[:n]]


# -------------------------------------------------------------- assignments
def build_assignments(matrices: dict[str, pd.DataFrame], models: dict[str, KMeans], players: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for group, matrix in matrices.items():
        info = players.loc[matrix["player_id"]]
        frames.append(pd.DataFrame({
            "player_id": matrix["player_id"].to_numpy(),
            "player_name": info["player"].to_numpy(),
            "league": info["league"].to_numpy(),
            "club": info["squad"].to_numpy(),
            "position": info["pos_raw"].to_numpy(),
            "position_group": group,
            "minutes": info["minutes"].to_numpy(),
            "cluster_id": models[group].labels_,
            "cluster_key": [f"{group}-{c}" for c in models[group].labels_],
        }))
    return pd.concat(frames, ignore_index=True)[ASSIGNMENT_COLUMNS]


# ----------------------------------------------------------------------- PCA
def fit_pca(Z: np.ndarray, random_state: int = 42) -> PCA:
    return PCA(n_components=2, random_state=random_state).fit(Z)


def pca_coordinates(matrix: pd.DataFrame, pca: PCA, assignments: pd.DataFrame) -> pd.DataFrame:
    coords = pca.transform(matrix[feature_columns(matrix)].to_numpy())
    by_id = assignments.set_index("player_id")
    ids = matrix["player_id"].to_numpy()
    return pd.DataFrame({
        "player_id": ids,
        "player_name": by_id.loc[ids, "player_name"].to_numpy(),
        "cluster_id": by_id.loc[ids, "cluster_id"].to_numpy(),
        "pc1": coords[:, 0],
        "pc2": coords[:, 1],
    })


# ------------------------------------------------------------------ pipeline
def run_evaluation(config: Config) -> pd.DataFrame:
    from scouting.visualization import plot_elbow

    table = evaluate_all(load_matrices(config), config)
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.processed_dir / "kmeans_evaluation.csv", index=False)
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    for group, sub in table.groupby("position_group", sort=False):
        plot_elbow(sub, group, config.figures_dir / f"elbow_{group}.png", config.modeling.chosen_k.get(group))
    log.info("wrote kmeans_evaluation.csv and elbow plots")
    return table


def run_fit(config: Config) -> None:
    from scouting.visualization import plot_pca

    chosen = config.modeling.chosen_k
    missing = [g for g in config.modeling.position_feature_lists if g not in chosen]
    if missing:
        raise ValueError(f"modeling.chosen_k has no value for {missing}; run 'evaluate' and choose K first")

    matrices, players = load_matrices(config), load_players(config)
    rs, n_init = config.modeling.random_state, config.modeling.kmeans_n_init

    models, pcas, profiles = {}, {}, []
    for group, matrix in matrices.items():
        Z = matrix[feature_columns(matrix)].to_numpy()
        models[group] = fit_kmeans(Z, chosen[group], rs, n_init)
        pcas[group] = fit_pca(Z, rs)
        profiles.append(cluster_profiles(matrix, models[group].labels_, players, group))
        log.info("%s: K=%d sizes=%s", group, chosen[group], np.bincount(models[group].labels_).tolist())

    assignments = build_assignments(matrices, models, players)
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(config.processed_dir / "cluster_assignments.csv", index=False)
    pd.concat(profiles, ignore_index=True).to_csv(config.processed_dir / "cluster_profiles.csv", index=False)

    config.figures_dir.mkdir(parents=True, exist_ok=True)
    for group, matrix in matrices.items():
        coords = pca_coordinates(matrix, pcas[group], assignments)
        coords.to_csv(config.processed_dir / f"pca_{group}.csv", index=False)
        plot_pca(coords, group, pcas[group].explained_variance_ratio_, config.figures_dir / f"pca_{group}.png")

    config.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {g: {"model": models[g], "features": feature_columns(matrices[g]), "k": chosen[g], "random_state": rs}
         for g in models},
        config.models_dir / "kmeans.joblib",
    )
    joblib.dump(pcas, config.models_dir / "pca.joblib")
    log.info("saved assignments, profiles, PCA coordinates, plots and models")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["evaluate", "fit"])
    args = parser.parse_args(argv)
    config = load_config()
    run_evaluation(config) if args.step == "evaluate" else run_fit(config)


if __name__ == "__main__":
    main()
