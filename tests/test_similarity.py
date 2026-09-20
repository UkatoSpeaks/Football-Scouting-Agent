import numpy as np
import pandas as pd
import pytest

from scouting.config import load_config
from scouting.similarity import PlayerNotFoundError, SimilarityEngine, find_similar_players


def _engine(minutes: dict[int, int] | None = None) -> SimilarityEngine:
    # FWD ids 1-6 (2 features), DEF ids 10-12. DEF 10 has the same vector as FWD 1
    # so that any cross-group leak would show up as a perfect match.
    fwd = pd.DataFrame({
        "player_id": [1, 2, 3, 4, 5, 6],
        "f1": [1.0, 2.0, 0.0, -1.0, 1.0, 0.0],
        "f2": [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],   # id 6 is the zero vector
    })
    dfn = pd.DataFrame({"player_id": [10, 11, 12], "f1": [1.0, 0.5, -1.0], "f2": [0.0, 1.0, 0.2]})
    players = pd.DataFrame(
        {
            "player": [f"P{i}" for i in (1, 2, 3, 4, 5, 6, 10, 11, 12)],
            "squad": "Club", "league": ["A", "B", "A", "B", "A", "B", "A", "B", "A"],
            "pos_raw": ["FW"] * 6 + ["DF"] * 3,
            "pos_group": ["FWD"] * 6 + ["DEF"] * 3,
            "minutes": 900,
        },
        index=[1, 2, 3, 4, 5, 6, 10, 11, 12],
    )
    for pid, m in (minutes or {}).items():
        players.loc[pid, "minutes"] = m
    return SimilarityEngine(matrices={"FWD": fwd, "DEF": dfn}, players=players)


def test_player_is_not_their_own_recommendation():
    out = _engine().find_similar_players(1, n=10)
    assert 1 not in out["player_id"].tolist()


def test_returns_n_when_enough_candidates_and_fewer_when_not():
    engine = _engine()
    assert len(engine.find_similar_players(1, n=3)) == 3
    assert len(engine.find_similar_players(1, n=100)) == 5      # 6 FWD minus the query
    assert len(engine.find_similar_players(10, n=100)) == 2     # 3 DEF minus the query


def test_only_same_position_group_is_returned():
    engine = _engine()
    out = engine.find_similar_players(1, n=100)
    assert set(out["position_group"]) == {"FWD"}
    assert 10 not in out["player_id"].tolist()  # DEF twin with the identical vector
    assert set(engine.find_similar_players(10, n=100)["position_group"]) == {"DEF"}


def test_cosine_scores_are_valid_and_sorted_descending():
    out = _engine().find_similar_players(1, n=100)
    s = out["similarity"].to_numpy()
    assert ((s >= -1) & (s <= 1)).all() and np.isfinite(s).all()
    assert (np.diff(s) <= 0).all()
    assert out["rank"].tolist() == list(range(1, len(out) + 1))


def test_known_cosine_values():
    out = _engine().find_similar_players(1, n=100).set_index("player_id")["similarity"]
    assert out[2] == pytest.approx(1.0)                 # same direction, bigger magnitude
    assert out[3] == pytest.approx(0.0)                 # orthogonal
    assert out[4] == pytest.approx(-1.0)                # opposite
    assert out[5] == pytest.approx(1 / np.sqrt(2))      # 45 degrees
    assert out[6] == 0.0                                # zero vector: defined as 0, never NaN


def test_euclidean_is_sorted_ascending_and_nonnegative():
    out = _engine().find_similar_players(1, n=100, metric="euclidean")
    d = out["distance"].to_numpy()
    assert (d >= 0).all() and (np.diff(d) >= 0).all()
    assert "similarity" not in out.columns
    assert out.set_index("player_id")["distance"][2] == pytest.approx(1.0)


def test_cosine_ignores_magnitude_but_euclidean_does_not():
    # query (1, 0). A = (3, 0): same direction, further away. B = (1, 0.8): closer, different angle.
    group = pd.DataFrame({"player_id": [1, 2, 3], "f1": [1.0, 3.0, 1.0], "f2": [0.0, 0.0, 0.8]})
    players = pd.DataFrame({"player": ["q", "A", "B"], "squad": "x", "league": "L", "pos_raw": "FW",
                            "pos_group": "FWD", "minutes": 900}, index=[1, 2, 3])
    engine = SimilarityEngine(matrices={"FWD": group}, players=players)
    assert engine.find_similar_players(1, n=1, metric="cosine")["player_name"].iloc[0] == "A"
    assert engine.find_similar_players(1, n=1, metric="euclidean")["player_name"].iloc[0] == "B"
    cmp = engine.compare_metrics(1, n=1)
    assert cmp["overlap"] == 0 and cmp["only_cosine"] == [2] and cmp["only_euclidean"] == [3]


def test_missing_player_id_is_handled_cleanly():
    engine = _engine()
    for bad in (999, -1, None, "abc"):
        with pytest.raises(PlayerNotFoundError):
            engine.find_similar_players(bad)
    assert issubclass(PlayerNotFoundError, KeyError)


def test_position_group_must_match_the_players_group():
    engine = _engine()
    assert len(engine.find_similar_players(1, position_group="FWD", n=2)) == 2
    with pytest.raises(ValueError):
        engine.find_similar_players(1, position_group="DEF")


def test_invalid_arguments():
    engine = _engine()
    for n in (0, -3, 2.5, "x"):
        with pytest.raises(ValueError):
            engine.find_similar_players(1, n=n)
    with pytest.raises(ValueError):
        engine.find_similar_players(1, metric="manhattan")


def test_zero_vector_query_has_undefined_cosine():
    with pytest.raises(ValueError):
        _engine().find_similar_players(6, metric="cosine")
    assert len(_engine().find_similar_players(6, n=3, metric="euclidean")) == 3


def test_output_columns_and_league_are_preserved():
    out = _engine().find_similar_players(1, n=3)
    assert list(out.columns) == ["rank", "player_id", "player_name", "league", "club", "position",
                                 "position_group", "minutes", "similarity"]
    assert set(out["league"]) <= {"A", "B"}


def test_ties_are_broken_deterministically_by_player_id():
    fwd = pd.DataFrame({"player_id": [1, 2, 3], "f": [1.0, 1.0, 1.0]})
    players = pd.DataFrame({"player": ["a", "b", "c"], "squad": "x", "league": "A", "pos_raw": "FW",
                            "pos_group": "FWD", "minutes": 900}, index=[1, 2, 3])
    out = SimilarityEngine(matrices={"FWD": fwd}, players=players).find_similar_players(3, n=2)
    assert out["player_id"].tolist() == [1, 2]


def test_explain_match_sorted_by_absolute_difference_and_rejects_cross_group():
    engine = _engine()
    e = engine.explain_match(1, 5, top=2)
    assert list(e.columns) == ["feature", "z_query", "z_match", "z_diff"]
    assert e["z_diff"].abs().is_monotonic_decreasing
    with pytest.raises(ValueError):
        engine.explain_match(1, 10)


def test_stale_scaler_artifacts_are_rejected():
    class FakeScaler:
        n_samples_seen_ = 6

    engine = _engine()
    with pytest.raises(ValueError, match="do not match"):
        engine.check_scalers({"FWD": {"scaler": FakeScaler(), "features": ["other", "cols"]}})
    FakeScaler.n_samples_seen_ = 99
    with pytest.raises(ValueError, match="different player pool"):
        engine.check_scalers({"FWD": {"scaler": FakeScaler(), "features": ["f1", "f2"]}})


# ------------------------------------------------- candidate minimum minutes
# FWD ids 2..6 as candidates of query 1; minutes chosen around the 900 default.
MINUTES = {1: 450, 2: 899, 3: 900, 4: 1500, 5: 2500, 6: 300}


def test_default_candidate_minimum_is_900_and_is_inclusive():
    out = _engine(MINUTES).find_similar_players(1, n=100)
    assert sorted(out["player_id"]) == [3, 4, 5]     # 2 (899) and 6 (300) dropped, 3 (900) kept
    assert (out["minutes"] >= 900).all()
    assert 899 not in out["minutes"].tolist()


def test_query_player_below_the_candidate_minimum_still_works():
    out = _engine(MINUTES).find_similar_players(1, n=10)   # query has 450 minutes
    assert len(out) == 3 and 1 not in out["player_id"].tolist()


def test_minimum_can_be_overridden_down_up_or_disabled():
    engine = _engine(MINUTES)
    assert len(engine.find_similar_players(1, n=100, min_candidate_minutes=0)) == 5
    assert len(engine.find_similar_players(1, n=100, min_candidate_minutes=None)) == 5
    assert sorted(engine.find_similar_players(1, n=100, min_candidate_minutes=300)["player_id"]) == [2, 3, 4, 5, 6]
    assert sorted(engine.find_similar_players(1, n=100, min_candidate_minutes=1500)["player_id"]) == [4, 5]


def test_fewer_than_n_when_the_filter_leaves_few_candidates():
    out = _engine(MINUTES).find_similar_players(1, n=10, min_candidate_minutes=2000)
    assert out["player_id"].tolist() == [5]


def test_no_eligible_candidates_returns_an_empty_frame_not_an_error():
    out = _engine(MINUTES).find_similar_players(1, n=10, min_candidate_minutes=10_000)
    assert out.empty
    assert list(out.columns) == ["rank", "player_id", "player_name", "league", "club", "position",
                                 "position_group", "minutes", "similarity"]


def test_filter_does_not_change_scores_or_relative_order():
    engine = _engine(MINUTES)
    everyone = engine.find_similar_players(1, n=100, min_candidate_minutes=0).set_index("player_id")["similarity"]
    filtered = engine.find_similar_players(1, n=100, min_candidate_minutes=900).set_index("player_id")["similarity"]
    assert np.allclose(filtered, everyone.loc[filtered.index])
    assert filtered.index.tolist() == [i for i in everyone.index if i in filtered.index]


def test_filter_applies_to_euclidean_and_compare_metrics():
    engine = _engine(MINUTES)
    eu = engine.find_similar_players(1, n=100, metric="euclidean")
    assert sorted(eu["player_id"]) == [3, 4, 5] and eu["distance"].is_monotonic_increasing
    cmp = engine.compare_metrics(1, n=10)
    assert set(cmp["top_cosine"]["player_id"]) == {3, 4, 5} == set(cmp["top_euclidean"]["player_id"])


def test_filter_keeps_group_and_self_exclusion_rules():
    out = _engine({10: 2000, 11: 2000, 12: 2000, **MINUTES}).find_similar_players(1, n=100, min_candidate_minutes=0)
    assert set(out["position_group"]) == {"FWD"} and 1 not in out["player_id"].tolist()


def test_invalid_min_candidate_minutes():
    engine = _engine(MINUTES)
    for bad in (-1, "900", True, [900]):
        with pytest.raises(ValueError):
            engine.find_similar_players(1, min_candidate_minutes=bad)


def test_filtered_results_are_deterministic():
    engine = _engine(MINUTES)
    a = engine.find_similar_players(1, n=5, min_candidate_minutes=900)
    b = engine.find_similar_players(1, n=5, min_candidate_minutes=900)
    pd.testing.assert_frame_equal(a, b)


# ------------------------------------------------------- candidate league filter
# synthetic leagues: ids 1,3,5 -> "A"; ids 2,4,6 -> "B" (query 1 is in A)
def test_league_filter_keeps_only_candidates_from_that_league():
    engine = _engine()
    out = engine.find_similar_players(1, n=100, leagues="A")
    assert sorted(out["player_id"]) == [3, 5] and set(out["league"]) == {"A"}
    other = engine.find_similar_players(1, n=100, leagues="B")
    assert sorted(other["player_id"]) == [2, 4, 6] and set(other["league"]) == {"B"}


def test_league_filter_accepts_several_leagues_and_none_means_all():
    engine = _engine()
    assert len(engine.find_similar_players(1, n=100, leagues=["A", "B"])) == 5
    assert len(engine.find_similar_players(1, n=100, leagues=None)) == 5


def test_query_player_may_come_from_a_different_league_than_the_filter():
    out = _engine().find_similar_players(2, n=100, leagues="A")   # query 2 is in B
    assert sorted(out["player_id"]) == [1, 3, 5]


def test_league_filter_does_not_change_scores_and_combines_with_minutes():
    engine = _engine(MINUTES)
    everyone = engine.find_similar_players(1, n=100, min_candidate_minutes=0).set_index("player_id")["similarity"]
    a_only = engine.find_similar_players(1, n=100, min_candidate_minutes=0, leagues="A").set_index("player_id")["similarity"]
    assert np.allclose(a_only, everyone.loc[a_only.index])
    both = engine.find_similar_players(1, n=100, min_candidate_minutes=1000, leagues="A")
    assert both["player_id"].tolist() == [5]                       # 3 has 900 minutes, 5 has 2500


def test_league_filter_with_no_matches_returns_empty_and_unknown_league_raises():
    engine = _engine(MINUTES)
    assert engine.find_similar_players(1, n=10, min_candidate_minutes=10_000, leagues="A").empty
    with pytest.raises(ValueError, match="unknown league"):
        engine.find_similar_players(1, leagues="Eredivisie")
    with pytest.raises(ValueError):
        engine.find_similar_players(1, leagues=["A", "Nope"])


def test_league_filter_works_for_euclidean_and_keeps_group_rule():
    out = _engine().find_similar_players(1, n=100, metric="euclidean", leagues="A")
    assert sorted(out["player_id"]) == [3, 5] and out["distance"].is_monotonic_increasing
    assert 10 not in out["player_id"].tolist()   # DEF player in league A never leaks in


# ------------------------------------------------------------------ real data
CONFIG = load_config()
DATA_READY = all(
    (CONFIG.processed_dir / f).exists()
    for f in ["players_features.csv", "model_matrix_FWD.csv"]
) and (CONFIG.models_dir / "scalers.joblib").exists()
real = pytest.mark.skipif(not DATA_READY, reason="run the feature and scaling steps first")


@pytest.fixture(scope="module")
def real_engine():
    return SimilarityEngine.from_config(CONFIG)


@real
@pytest.mark.parametrize("name, group", [
    ("Bukayo Saka", "FWD"), ("Erling Haaland", "FWD"), ("Declan Rice", "MID"),
    ("Virgil van Dijk", "DEF"), ("Alisson", "GK"),
])
def test_real_players(real_engine, name, group):
    pid = real_engine.find_id(name)
    out = real_engine.find_similar_players(pid, n=10)
    assert len(out) == 10
    assert pid not in out["player_id"].tolist()
    assert (out["position_group"] == group).all()
    assert out["similarity"].between(-1, 1).all()
    assert out["similarity"].is_monotonic_decreasing
    assert out["minutes"].min() >= 900   # default candidate minimum
    loose = real_engine.find_similar_players(pid, n=10, min_candidate_minutes=CONFIG.feature_engineering.min_minutes)
    assert loose["minutes"].min() >= CONFIG.feature_engineering.min_minutes


@real
def test_real_results_span_leagues_and_wrapper_works(real_engine):
    pid = real_engine.find_id("Bukayo Saka")
    out = find_similar_players(pid, n=10)
    assert out["league"].nunique() > 1
    assert out["player_id"].tolist() == real_engine.find_similar_players(pid, n=10)["player_id"].tolist()
    assert find_similar_players(pid, n=10, min_candidate_minutes=2500)["minutes"].min() >= 2500


@real
def test_real_league_filter(real_engine):
    pid = real_engine.find_id("Bukayo Saka")
    out = real_engine.find_similar_players(pid, n=10, leagues="La Liga")
    assert len(out) == 10 and set(out["league"]) == {"La Liga"} and (out["position_group"] == "FWD").all()


@real
def test_real_missing_id_and_ambiguous_name(real_engine):
    with pytest.raises(PlayerNotFoundError):
        real_engine.find_similar_players(10**9)
    with pytest.raises(PlayerNotFoundError):
        real_engine.find_id("Nobody Atall")
