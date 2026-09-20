import joblib
import numpy as np
import pandas as pd
import pytest

from scouting.clustering import (
    ASSIGNMENT_COLUMNS,
    build_assignments,
    cluster_profiles,
    describe_cluster,
    evaluate_k,
    feature_columns,
    fit_kmeans,
    fit_pca,
    pca_coordinates,
    representatives,
    run_fit,
)
from scouting.config import load_config

CONFIG = load_config()


def _blobs(sizes=(60, 40, 20), spread=0.25, seed=0):
    """Three well-separated blobs in 3 dimensions, standardized like the real matrices."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0, 0], [5, 0, 0], [0, 5, 5]], dtype=float)
    X = np.vstack([rng.normal(c, spread, size=(n, 3)) for c, n in zip(centers, sizes)])
    return (X - X.mean(axis=0)) / X.std(axis=0)


def _matrix(Z, first_id=100):
    m = pd.DataFrame(Z, columns=["a_p90", "b_pct", "c_p90"])
    m.insert(0, "player_id", range(first_id, first_id + len(Z)))
    return m


# ---------------------------------------------------------------- evaluation
def test_evaluate_k_table_shape_and_ranges():
    t = evaluate_k(_blobs(), range(2, 6), random_state=0, n_init=5, group="FWD")
    assert t["k"].tolist() == [2, 3, 4, 5]
    assert (t["position_group"] == "FWD").all()
    assert t["silhouette"].between(-1, 1).all()
    assert t["stability_ari"].between(-1, 1).all()
    assert (t["min_cluster_size"] <= t["max_cluster_size"]).all()
    assert (t["min_cluster_share"] > 0).all()


def test_inertia_never_increases_with_k_and_true_k_wins_silhouette_on_clean_blobs():
    t = evaluate_k(_blobs(), range(2, 7), random_state=0, n_init=10)
    assert np.all(np.diff(t["inertia"]) <= 1e-9)
    assert t.loc[t["silhouette"].idxmax(), "k"] == 3
    assert t.loc[t["k"] == 3, "stability_ari"].iloc[0] > 0.99


def test_evaluation_is_reproducible():
    a = evaluate_k(_blobs(), [2, 3], random_state=1, n_init=5)
    b = evaluate_k(_blobs(), [2, 3], random_state=1, n_init=5)
    pd.testing.assert_frame_equal(a, b)


# ------------------------------------------------------------------- fitting
def test_clusters_are_ordered_by_size_and_recover_the_blobs():
    km = fit_kmeans(_blobs((20, 60, 40)), k=3, random_state=0, n_init=5)
    sizes = np.bincount(km.labels_)
    assert sizes.tolist() == [60, 40, 20]          # cluster 0 = largest
    assert np.array_equal(km.predict(_blobs((20, 60, 40))), km.labels_)


def test_fit_is_deterministic_for_a_seed():
    Z = _blobs()
    a = fit_kmeans(Z, 3, random_state=7, n_init=5)
    b = fit_kmeans(Z, 3, random_state=7, n_init=5)
    assert np.array_equal(a.labels_, b.labels_) and np.allclose(a.cluster_centers_, b.cluster_centers_)


def test_size_ties_keep_a_stable_valid_labeling():
    km = fit_kmeans(_blobs((30, 30, 30)), 3, random_state=0, n_init=5)
    assert sorted(np.bincount(km.labels_).tolist()) == [30, 30, 30]
    assert set(km.labels_) == {0, 1, 2}


# ----------------------------------------------------------------- profiling
def test_profiles_weighted_mean_of_standardized_features_is_zero():
    Z = _blobs()
    km = fit_kmeans(Z, 3, 0, 5)
    matrix = _matrix(Z)
    raw = pd.DataFrame(np.exp(Z), index=matrix["player_id"], columns=feature_columns(matrix))
    prof = cluster_profiles(matrix, km.labels_, raw, "FWD")
    assert set(prof["cluster_id"]) == {0, 1, 2}
    for f, sub in prof.groupby("feature"):
        weighted = (sub["mean_z"] * sub["n_players"]).sum() / sub["n_players"].sum()
        assert weighted == pytest.approx(0, abs=1e-9)
    assert prof.groupby("cluster_id")["n_players"].first().sum() == len(Z)
    assert (prof["mean_raw"] > 0).all()   # raw values come from the raw table, not the z matrix


def test_describe_cluster_uses_only_strong_traits_and_sorts_them():
    prof = pd.DataFrame({
        "feature": ["tackles_p90", "key_passes_p90", "clearances_p90", "shots_p90", "blocks_p90"],
        "mean_z": [1.4, 0.6, -0.9, 0.3, -0.5],
    })
    d = describe_cluster(prof, threshold=0.5)
    assert [n for n, _ in d["high"]] == ["tackles", "key passes"]
    assert [n for n, _ in d["low"]] == ["clearances", "blocks"]
    assert d["high"][0][1] == pytest.approx(1.4)
    assert describe_cluster(prof, threshold=2.0) == {"high": [], "low": []}
    assert len(describe_cluster(prof, threshold=0.0, top=1)["high"]) == 1


def test_representatives_are_members_closest_to_the_centre():
    Z = _blobs()
    km = fit_kmeans(Z, 3, 0, 5)
    for c in range(3):
        reps = representatives(Z, km, c, n=5)
        assert len(reps) == 5 and (km.labels_[reps] == c).all()
        d = np.linalg.norm(Z[reps] - km.cluster_centers_[c], axis=1)
        assert np.all(np.diff(d) >= 0)


# ------------------------------------------------------- assignments and PCA
def _players(ids):
    return pd.DataFrame(
        {"player": [f"P{i}" for i in ids], "squad": "Club", "league": "L", "pos_raw": "FW", "pos_group": "FWD",
         "minutes": 1000}, index=pd.Index(ids, name="player_id"))


def test_assignments_cover_every_player_once_with_the_expected_columns():
    Z = _blobs()
    matrix = _matrix(Z)
    km = fit_kmeans(Z, 3, 0, 5)
    out = build_assignments({"FWD": matrix}, {"FWD": km}, _players(matrix["player_id"].tolist()))
    assert list(out.columns) == ASSIGNMENT_COLUMNS
    assert out["player_id"].is_unique and len(out) == len(Z)
    assert (out["position_group"] == "FWD").all()
    assert out["cluster_key"].tolist() == [f"FWD-{c}" for c in km.labels_]
    assert out["cluster_id"].tolist() == km.labels_.tolist()


def test_pca_coordinates_align_with_players_and_report_variance():
    Z = _blobs()
    matrix = _matrix(Z)
    km = fit_kmeans(Z, 3, 0, 5)
    assign = build_assignments({"FWD": matrix}, {"FWD": km}, _players(matrix["player_id"].tolist()))
    pca = fit_pca(Z, 0)
    coords = pca_coordinates(matrix, pca, assign)
    assert list(coords.columns) == ["player_id", "player_name", "cluster_id", "pc1", "pc2"]
    assert coords["player_id"].tolist() == matrix["player_id"].tolist()
    assert coords["cluster_id"].tolist() == km.labels_.tolist()
    assert 0 < pca.explained_variance_ratio_.sum() <= 1
    assert coords[["pc1", "pc2"]].corr().iloc[0, 1] == pytest.approx(0, abs=1e-9)


def test_fit_refuses_to_run_without_a_chosen_k_for_every_group():
    cfg = load_config()
    cfg.modeling.chosen_k = {"GK": 2}
    with pytest.raises(ValueError, match="chosen_k"):
        run_fit(cfg)


# ------------------------------------------------------------ saved artifacts
PROCESSED = CONFIG.processed_dir
READY = all((PROCESSED / f).exists() for f in ["cluster_assignments.csv", "cluster_profiles.csv"]) and (
    CONFIG.models_dir / "kmeans.joblib"
).exists()
real = pytest.mark.skipif(not READY, reason="run: python -m scouting.clustering fit")


@real
def test_saved_models_match_config_and_reproduce_their_labels():
    saved = joblib.load(CONFIG.models_dir / "kmeans.joblib")
    assert set(saved) == set(CONFIG.modeling.position_feature_lists)
    for group, art in saved.items():
        matrix = pd.read_csv(PROCESSED / f"model_matrix_{group}.csv")
        assert art["k"] == CONFIG.modeling.chosen_k[group] == art["model"].n_clusters
        assert art["features"] == feature_columns(matrix) == CONFIG.modeling.position_feature_lists[group]
        assert np.array_equal(art["model"].predict(matrix[art["features"]].to_numpy()), art["model"].labels_)
        assert np.all(np.diff(np.bincount(art["model"].labels_)) <= 0)  # sizes descending


@real
def test_saved_assignments_cover_all_modeled_players_and_stay_within_group():
    out = pd.read_csv(PROCESSED / "cluster_assignments.csv")
    feats = pd.read_csv(PROCESSED / "players_features.csv")
    assert list(out.columns) == ASSIGNMENT_COLUMNS
    assert out["player_id"].is_unique
    assert sorted(out["player_id"]) == sorted(feats["player_id"])
    merged = out.merge(feats[["player_id", "pos_group"]], on="player_id", suffixes=("", "_feat"))
    assert (merged["position_group"] == merged["pos_group"]).all()
    for group, sub in out.groupby("position_group"):
        assert set(sub["cluster_id"]) == set(range(CONFIG.modeling.chosen_k[group]))


@real
def test_saved_profiles_and_pca_files_are_consistent():
    prof = pd.read_csv(PROCESSED / "cluster_profiles.csv")
    for group, k in CONFIG.modeling.chosen_k.items():
        sub = prof[prof["position_group"] == group]
        assert sub["cluster_id"].nunique() == k
        feats = CONFIG.modeling.position_feature_lists[group]
        assert set(sub["feature"]) == set(feats)
        one = sub[sub["feature"] == feats[0]]
        assert (one["mean_z"] * one["n_players"]).sum() == pytest.approx(0, abs=1e-6)
        pca = pd.read_csv(PROCESSED / f"pca_{group}.csv")
        assert len(pca) == one["n_players"].sum() and pca[["pc1", "pc2"]].notna().all().all()
