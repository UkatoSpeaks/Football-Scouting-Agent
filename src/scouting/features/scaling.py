"""Feature table -> per-position-group standardized model matrices.

For each position group: select its feature list, apply log1p to the configured
skewed features, then z-score every feature with a StandardScaler fitted on that
group's players (>= min_minutes). No feature weights are applied.

The scaler is fitted on the full player pool because the downstream models
(similarity, K-Means) are unsupervised and there is no held-out target, so there
is nothing to leak. The fitted scalers are saved so the app can transform
players the same way without refitting.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from scouting.config import Config, load_config
from scouting.utils.logging import get_logger

log = get_logger(__name__)


def log1p_transform(table: pd.DataFrame, pos_group: str, config: Config) -> pd.DataFrame:
    """Selected features of one position group, with log1p on the configured ones."""
    features = config.modeling.position_feature_lists[pos_group]
    to_log = config.modeling.log1p_features.get(pos_group, [])
    unknown = [f for f in to_log if f not in features]
    if unknown:
        raise KeyError(f"{pos_group}: log1p features not in the feature list: {unknown}")

    X = table.loc[table["pos_group"] == pos_group, features].astype("float64").copy()
    if (X[to_log] < 0).any().any():
        raise ValueError(f"{pos_group}: log1p needs non-negative values")
    X[to_log] = np.log1p(X[to_log])
    return X


def build_model_matrix(table: pd.DataFrame, pos_group: str, config: Config) -> tuple[pd.DataFrame, StandardScaler]:
    X = log1p_transform(table, pos_group, config)
    scaler = StandardScaler().fit(X)
    Z = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
    Z.insert(0, "player_id", table.loc[X.index, "player_id"])
    return Z, scaler


def build_all(table: pd.DataFrame, config: Config) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    matrices, artifacts = {}, {}
    for group in config.modeling.position_feature_lists:
        Z, scaler = build_model_matrix(table, group, config)
        matrices[group] = Z
        artifacts[group] = {
            "scaler": scaler,
            "features": config.modeling.position_feature_lists[group],
            "log1p_features": config.modeling.log1p_features.get(group, []),
        }
        log.info("%s: %d players x %d features", group, len(Z), Z.shape[1] - 1)
    return matrices, artifacts


def main() -> None:
    config = load_config()
    table = pd.read_csv(config.processed_dir / "players_features.csv")
    matrices, artifacts = build_all(table, config)

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    for group, Z in matrices.items():
        Z.to_csv(config.processed_dir / f"model_matrix_{group}.csv", index=False)

    config.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, config.models_dir / "scalers.joblib")
    log.info("saved matrices to %s and scalers to %s", config.processed_dir, config.models_dir / "scalers.joblib")


if __name__ == "__main__":
    main()
