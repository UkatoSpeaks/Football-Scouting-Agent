import joblib
import numpy as np
import pandas as pd
import pytest

from scouting.config import load_config
from scouting.features.engineering import skewed_features
from scouting.features.scaling import build_model_matrix, log1p_transform

CONFIG = load_config()
FEATURES_CSV = CONFIG.processed_dir / "players_features.csv"


def _toy_config():
    cfg = load_config()
    cfg.modeling.position_feature_lists = {"FWD": ["a_p90", "b_pct"]}
    cfg.modeling.log1p_features = {"FWD": ["a_p90"]}
    return cfg


def _toy_table():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "player_id": range(6),
        "pos_group": ["FWD"] * 5 + ["DEF"],
        "a_p90": rng.exponential(1.0, 6),
        "b_pct": rng.normal(60, 10, 6),
    })


def test_log1p_only_touches_configured_features():
    cfg, table = _toy_config(), _toy_table()
    X = log1p_transform(table, "FWD", cfg)
    fwd = table[table.pos_group == "FWD"]
    assert np.allclose(X["a_p90"], np.log1p(fwd["a_p90"]))
    assert np.allclose(X["b_pct"], fwd["b_pct"])
    assert len(X) == 5  # DEF row excluded


def test_log1p_rejects_negative_values():
    cfg, table = _toy_config(), _toy_table()
    table.loc[0, "a_p90"] = -0.5
    with pytest.raises(ValueError):
        log1p_transform(table, "FWD", cfg)


def test_log1p_feature_must_be_in_feature_list():
    cfg = _toy_config()
    cfg.modeling.log1p_features = {"FWD": ["not_a_feature"]}
    with pytest.raises(KeyError):
        log1p_transform(_toy_table(), "FWD", cfg)


def test_matrix_is_standardized_per_group():
    cfg, table = _toy_config(), _toy_table()
    Z, scaler = build_model_matrix(table, "FWD", cfg)
    feats = ["a_p90", "b_pct"]
    assert np.allclose(Z[feats].mean(), 0, atol=1e-9)
    assert np.allclose(Z[feats].std(ddof=0), 1, atol=1e-9)
    assert Z["player_id"].tolist() == [0, 1, 2, 3, 4]
    assert scaler.n_features_in_ == 2


@pytest.mark.skipif(not FEATURES_CSV.exists(), reason="run scouting.features.engineering first")
def test_configured_log1p_lists_match_the_data():
    table = pd.read_csv(FEATURES_CSV)
    for group, features in CONFIG.modeling.position_feature_lists.items():
        expected = skewed_features(table, group, features, CONFIG.modeling.log1p_skew_threshold)
        assert sorted(CONFIG.modeling.log1p_features.get(group, [])) == sorted(expected), group


@pytest.mark.skipif(not FEATURES_CSV.exists(), reason="run scouting.features.engineering first")
def test_saved_scaler_reproduces_the_matrix():
    saved = CONFIG.models_dir / "scalers.joblib"
    if not saved.exists():
        pytest.skip("run scouting.features.scaling first")
    table = pd.read_csv(FEATURES_CSV)
    artifacts = joblib.load(saved)
    for group, art in artifacts.items():
        Z, _ = build_model_matrix(table, group, CONFIG)
        X = log1p_transform(table, group, CONFIG)
        assert np.allclose(art["scaler"].transform(X), Z[art["features"]].to_numpy())
