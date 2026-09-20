"""Dashboard tests: pure logic (no Streamlit) and headless UI runs with Streamlit's AppTest.

These use the real pipeline outputs and are skipped if they have not been built.
"""
import json
import re
from dataclasses import replace
from html import escape as html_escape
from html import unescape as html_unescape
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scouting.config import load_config
from scouting.dashboard import charts, logic
from scouting.dashboard.data import DashboardArtifactsError, _check_consistency, load_dashboard_data
from scouting.labels import CATEGORY_ORDER, FEATURE_CATEGORIES, FEATURE_LABELS, display_name, label
from scouting.similarity import PlayerNotFoundError

CONFIG = load_config()
READY = (CONFIG.models_dir / "kmeans.joblib").exists() and (CONFIG.processed_dir / "cluster_assignments.csv").exists()
real = pytest.mark.skipif(not READY, reason="run the pipeline first (features, scaling, clustering fit)")
APP = str(Path(__file__).resolve().parents[1] / "app.py")
FIVE = ["Bukayo Saka", "Erling Haaland", "Declan Rice", "Virgil van Dijk", "Alisson"]


@pytest.fixture(scope="module")
def data():
    from scouting.dashboard.views import get_data

    return get_data()


def _app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(APP, default_timeout=120)


def _select(at, name, data):
    at.sidebar.selectbox(key="player_id").set_value(data.engine.find_id(name)).run()
    return at


# ------------------------------------------------------------ labels / config
def test_every_configured_feature_has_a_label_category_and_display_name():
    for group, feats in CONFIG.modeling.position_feature_lists.items():
        for f in feats:
            assert f in FEATURE_LABELS, f"{group}: no label for {f}"
            assert FEATURE_CATEGORIES[f] in CATEGORY_ORDER, f"{group}: no category for {f}"
            assert display_name(f)


def test_display_names_keep_acronyms_and_do_not_repeat_units():
    assert display_name("npxg_p90") == "npxG (per 90)"
    assert display_name("xag_p90") == "xAG (per 90)"
    assert display_name("gk_save_pct") == "Save %"            # no "Save % (%)"
    assert display_name("pass_completion_pct") == "Pass completion (%)"
    assert label("not_a_feature") == "not_a_feature"          # unknown features fall back, never crash


# ---------------------------------------------------------------- data layer
def test_missing_artifacts_raise_a_helpful_error():
    cfg = load_config()
    cfg.season = "9999-9999"
    with pytest.raises(DashboardArtifactsError, match="python -m scouting"):
        load_dashboard_data(cfg)


@real
def test_inconsistent_artifacts_are_rejected(data):
    import joblib

    kmeans = joblib.load(CONFIG.models_dir / "kmeans.joblib")
    bad = data.assignments.copy()
    bad.loc[bad.index[0], "cluster_id"] = (int(bad["cluster_id"].iloc[0]) + 1) % 2
    with pytest.raises(DashboardArtifactsError, match="do not match|does not match"):
        _check_consistency(CONFIG, data.engine, bad, data.profiles, kmeans)
    with pytest.raises(DashboardArtifactsError, match="different players"):
        _check_consistency(CONFIG, data.engine, data.assignments.iloc[:-1], data.profiles, kmeans)


@real
def test_loaded_data_is_consistent(data):
    assert set(data.assignments.index) == set(data.engine.players.index) == set(data.players.index)
    assert set(data.leagues) == {"Bundesliga", "La Liga", "Ligue 1", "Premier League", "Serie A"}
    for g, pca in data.pca_coords.items():
        assert len(pca) == len(data.engine.matrices[g]) and len(data.pca_variance[g]) == 2


# ------------------------------------------------------------------- picker
@real
def test_picker_lists_unique_players_and_respects_the_position_filter(data):
    everyone = logic.picker_labels(data)
    assert len(everyone) == len(data.players) and len(set(everyone.values())) == len(everyone)
    gks = logic.picker_labels(data, "GK")
    assert set(gks) == set(data.players.index[data.players["pos_group"] == "GK"])
    names = [v.split(" — ")[0] for v in everyone.values()]
    assert names == sorted(names)                              # alphabetical, so search results are predictable
    assert all("—" in v and "(" in v for v in list(everyone.values())[:50])


# ----------------------------------------------------------------- overview
@real
@pytest.mark.parametrize("name", FIVE)
def test_overview_for_the_five_test_players(data, name):
    o = logic.overview(data, data.engine.find_id(name))
    assert o["name"] == name and o["position_group"] in {"GK", "DEF", "MID", "FWD"}
    assert o["cluster_key"] == f"{o['position_group']}-{o['cluster_id']}"
    assert o["minutes"] >= CONFIG.feature_engineering.min_minutes and o["club"] and o["league"]


@real
def test_overview_survives_missing_optional_metadata(data):
    missing = data.players.index[data.players["age"].isna() | data.players["nation"].isna()]
    assert len(missing) > 0
    o = logic.overview(data, int(missing[0]))
    assert o["age"] is None or o["nation"] is None
    assert o["name"] and o["cluster_key"]


# ------------------------------------------------------------ similar players
@real
@pytest.mark.parametrize("name", FIVE)
def test_similar_table_for_the_five_players(data, name):
    pid = data.engine.find_id(name)
    t = logic.similar_table(data, pid, n=10)
    group = data.players.loc[pid, "pos_group"]
    assert list(t.columns) == ["player_id", *logic.SIMILAR_COLUMNS]
    assert len(t) == 10 and pid not in t["player_id"].tolist()
    assert (data.players.loc[t["player_id"], "pos_group"] == group).all()
    assert (t["Minutes"] >= 900).all()
    assert t["Cosine similarity"].between(-1, 1).all() and t["Cosine similarity"].is_monotonic_decreasing
    assert t["Rank"].tolist() == list(range(1, 11))
    assert t["Cluster"].str.startswith(group).all()


@real
def test_similar_table_filters(data):
    pid = data.engine.find_id("Bukayo Saka")
    la_liga = logic.similar_table(data, pid, n=10, league="La Liga")
    assert len(la_liga) == 10 and set(la_liga["League"]) == {"La Liga"}
    long_minutes = logic.similar_table(data, pid, n=10, min_minutes=2500)
    assert (long_minutes["Minutes"] >= 2500).all()
    assert len(logic.similar_table(data, pid, n=5)) == 5
    both = logic.similar_table(data, pid, n=10, min_minutes=3300, league="Ligue 1")
    assert (both["League"] == "Ligue 1").all() and (both["Minutes"] >= 3300).all()
    assert logic.similar_table(data, pid, n=10, min_minutes=10_000).empty


@real
def test_similar_table_rejects_unknown_players(data):
    with pytest.raises(PlayerNotFoundError):
        logic.similar_table(data, 10**9)


@real
def test_similarity_scores_come_from_the_engine_not_the_dashboard(data):
    pid = data.engine.find_id("Declan Rice")
    direct = data.engine.find_similar_players(pid, n=10, min_candidate_minutes=900)
    ours = logic.similar_table(data, pid, n=10)
    assert ours["player_id"].tolist() == direct["player_id"].tolist()
    assert np.allclose(ours["Cosine similarity"], direct["similarity"])


# --------------------------------------------------------------- statistics
@real
@pytest.mark.parametrize("name, expected, absent", [
    ("Alisson", {"Passing", "Goalkeeping"}, {"Attacking", "Defending", "Chance creation"}),
    ("Bukayo Saka", {"Attacking", "Chance creation", "Passing", "Carrying & take-ons", "Defending"}, {"Goalkeeping"}),
    ("Virgil van Dijk", {"Passing", "Defending"}, {"Goalkeeping", "Attacking"}),
    ("Declan Rice", {"Passing", "Defending", "Attacking"}, {"Goalkeeping"}),
])
def test_stats_are_position_specific(data, name, expected, absent):
    pid = data.engine.find_id(name)
    cats = dict(logic.player_stats(data, pid))
    assert expected <= set(cats) and not (absent & set(cats))
    group = data.players.loc[pid, "pos_group"]
    shown = [f for df in cats.values() for f in df["feature"]]
    assert sorted(shown) == sorted(CONFIG.modeling.position_feature_lists[group])
    for df in cats.values():
        assert df["value"].notna().all() and df["percentile"].between(0, 100).all() and (df["text"] != "—").all()
    assert list(cats) == [c for c in CATEGORY_ORDER if c in cats]     # category order


def test_value_formatting_and_ordinals():
    assert logic.format_value("pass_completion_pct", 83.456) == "83.5%"
    assert logic.format_value("npxg_per_shot", 0.1234) == "0.123"
    assert logic.format_value("goals_p90", 0.309) == "0.31"
    assert logic.format_value("goals_p90", float("nan")) == "—"
    assert [logic.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 56, 91, 100)] == \
        ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "56th", "91st", "100th"]


# --------------------------------------------------------------- comparison
@real
def test_comparison_uses_the_position_features_and_rejects_cross_group(data):
    saka = data.engine.find_id("Bukayo Saka")
    other = int(logic.similar_table(data, saka, n=3)["player_id"].iloc[0])
    cmp = logic.comparison_table(data, saka, other)
    assert cmp["feature"].tolist() == sorted(CONFIG.modeling.position_feature_lists["FWD"],
                                             key=lambda f: CATEGORY_ORDER.index(FEATURE_CATEGORIES[f]))
    assert cmp[["value_a", "value_b", "z_a", "z_b"]].notna().all().all()
    assert np.allclose(cmp["z_diff"], cmp["z_b"] - cmp["z_a"])
    assert np.allclose(cmp["value_a"], data.players.loc[saka, cmp["feature"]].to_numpy(dtype=float))
    with pytest.raises(ValueError):
        logic.comparison_table(data, saka, data.engine.find_id("Alisson"))
    with pytest.raises(PlayerNotFoundError):
        logic.comparison_table(data, saka, 10**9)
    gaps = logic.biggest_differences(cmp, 3)
    assert len(gaps) == 3 and gaps["z_diff"].abs().is_monotonic_decreasing


@real
def test_comparison_chart_has_two_named_series_and_a_bar_per_feature(data):
    saka = data.engine.find_id("Bukayo Saka")
    other = int(logic.similar_table(data, saka, n=1)["player_id"].iloc[0])
    cmp = logic.comparison_table(data, saka, other)
    fig = charts.comparison_chart(cmp, "A", "B", "FWD")
    assert [t.name for t in fig.data] == ["A", "B"] and all(len(t.x) == len(cmp) for t in fig.data)


# ---------------------------------------------------------------- clusters
@real
def test_cluster_descriptions_are_derived_from_the_profiles_not_hard_coded(data):
    info = logic.cluster_info(data, "FWD", 4)
    prof = data.profiles[(data.profiles.position_group == "FWD") & (data.profiles.cluster_id == 4)]
    strong = prof[prof["mean_z"] >= 0.5]
    assert len(info.high) == min(5, len(strong)) and all(z >= 0.5 for _, z in info.high)
    assert all(z <= -0.5 for _, z in info.low)
    before = logic.cluster_sentence(info)
    # flip the profile's sign: the generated text must change accordingly
    flipped = replace(data, profiles=data.profiles.assign(mean_z=-data.profiles["mean_z"]))
    after = logic.cluster_sentence(logic.cluster_info(flipped, "FWD", 4))
    assert before != after
    assert [n for n, _ in logic.cluster_info(flipped, "FWD", 4).low] == [n for n, _ in info.high]


@real
def test_cluster_overview_matches_the_saved_assignments(data):
    for group, k in CONFIG.modeling.chosen_k.items():
        infos = logic.cluster_overview(data, group)
        assert [i.cluster_id for i in infos] == list(range(k))
        sizes = data.assignments[data.assignments.position_group == group]["cluster_id"].value_counts().sort_index()
        assert [i.n_players for i in infos] == sizes.tolist()
        assert sum(i.share for i in infos) == pytest.approx(1)


@real
def test_cluster_members_filters_and_sorting(data):
    everyone = logic.cluster_members(data, "MID", 0)
    assert len(everyone) == int(data.profiles[(data.profiles.position_group == "MID") & (data.profiles.cluster_id == 0)].n_players.iloc[0])
    assert everyone["Minutes"].is_monotonic_decreasing
    sub = logic.cluster_members(data, "MID", 0, league="Serie A", min_minutes=1500)
    assert set(sub["League"]) <= {"Serie A"} and (sub["Minutes"] >= 1500).all()
    assert logic.cluster_members(data, "MID", 0, min_minutes=10_000).empty


@real
def test_profile_matrix_and_heatmap(data):
    z, raw = logic.profile_matrix(data, "DEF")
    assert z.shape == (len(CONFIG.modeling.position_feature_lists["DEF"]), CONFIG.modeling.chosen_k["DEF"])
    assert z.shape == raw.shape and z.notna().all().all()
    fig = charts.profile_heatmap(z, raw)
    assert fig.data[0].zmid == 0 and fig.data[0].zmin == -fig.data[0].zmax


# ---------------------------------------------------------------- PCA / league
@real
def test_pca_frame_and_chart_use_saved_coordinates(data):
    frame = logic.pca_frame(data, "MID")
    saved = data.pca_coords["MID"]
    assert np.allclose(frame[["pc1", "pc2"]].to_numpy(), saved[["pc1", "pc2"]].to_numpy())
    assert {"club", "league", "cluster_key"} <= set(frame.columns) and frame["club"].notna().all()
    rice = data.engine.find_id("Declan Rice")
    fig = charts.pca_chart(frame, data.pca_variance["MID"], "MID", selected_id=rice, similar_ids=[])
    assert [t.name for t in fig.data][-1] == "Selected player"
    star = fig.data[-1]
    row = frame[frame["player_id"] == rice].iloc[0]
    assert (star.x[0], star.y[0]) == (row["pc1"], row["pc2"])


@real
def test_league_counts_include_every_league_and_sum_to_the_table(data):
    pid = data.engine.find_id("Bukayo Saka")
    t = logic.similar_table(data, pid, n=10)
    counts = logic.league_counts(t, data.leagues)
    assert counts["League"].tolist() == data.leagues and counts["Players"].sum() == 10
    assert logic.league_counts(t.iloc[0:0], data.leagues)["Players"].sum() == 0
    assert logic.main_league(data, pid) == "Premier League"


# ------------------------------------------------------------- UI (AppTest)
def _hero_html(at) -> str:
    """The profile header markdown (empty string when no player is selected)."""
    return " ".join(m.value for m in at.markdown if 'class="sc-hero"' in m.value)


def _similar_cards_html(at) -> list[str]:
    """One markdown value per rendered similar-player card (Phase 2: cards, not a dataframe)."""
    return [m.value for m in at.markdown if 'class="sc-sim-card"' in m.value]


def _card_names(at) -> list[str]:
    return [html_unescape(n) for n in re.findall(r'sc-sim-name">([^<]+)</div>', "".join(_similar_cards_html(at)))]


def _limitations_text(at):
    return " ".join(m.value for m in at.markdown if "Team style affects per-90" in m.value)


@real
def test_landing_page_has_no_player_and_no_errors(data):
    at = _app().run()
    assert not at.exception
    assert at.title[0].value == "European Football Player Scouting"
    text = " ".join(m.value for m in at.markdown)
    assert "Find statistically similar players across Europe's top five leagues." in text
    assert "Select a player to begin scouting." in text
    assert _hero_html(at) == "" and len(at.dataframe) == 0             # nothing selected, no profile
    assert at.sidebar.selectbox(key="player_id").value is None
    assert at.selectbox(key="landing_player").placeholder == "Search for a player"
    stats = " ".join(m.value for m in at.markdown if "sc-stats" in m.value)
    n_groups = data.players["pos_group"].nunique()
    assert f"{len(data.players):,}" in stats and f">{len(data.leagues)}<" in stats and f">{n_groups}<" in stats


@real
def test_team_style_limitation_is_documented_on_every_page(data):
    from streamlit.testing.v1 import AppTest

    def explorer():
        from scouting.dashboard import views

        views.cluster_explorer_page(views.get_data())

    landing = _app().run()
    player = _select(_app().run(), "Declan Rice", data)
    clusters = AppTest.from_function(explorer, default_timeout=120).run()
    for at in (landing, player, clusters):
        text = _limitations_text(at)
        flat = " ".join(text.split())
        assert "Team style affects per-90" in flat and "does not correct for this" in flat
        assert "proxy for playing style" in text and "2025-26" in text


@real
@pytest.mark.parametrize("name", FIVE)
def test_full_page_for_the_five_players(data, name):
    at = _select(_app().run(), name, data)
    assert not at.exception
    pid = data.engine.find_id(name)
    group = data.players.loc[pid, "pos_group"]
    hero = _hero_html(at)
    o = logic.overview(data, pid)
    assert html_escape(o["name"]) in hero
    assert html_escape(o["club"]) in hero and html_escape(o["league"]) in hero
    assert f">{o['cluster_key']}<" in hero and f"{o['minutes']:,}" in hero
    assert [s.value for s in at.subheader] == [
        "Player statistics", "Find similar players", "Player comparison", "Playing style", "Player style map"]
    expected = logic.similar_table(data, pid, n=10)
    cards = _similar_cards_html(at)
    assert len(cards) == 10 and name not in _card_names(at)
    assert set(_card_names(at)) == set(expected["Player"])
    assert len(at.get("plotly_chart")) == 4                       # radar, comparison, league mix, PCA map
    assert any('class="sc-style-id"' in m.value and f">{o['cluster_key']}<" in m.value for m in at.markdown)
    scores = [float(s) for s in re.findall(r"<b>(-?\d+\.\d+)</b><span>similarity</span>", "".join(cards))]
    assert len(scores) == 10 and all(-1 <= s <= 1 for s in scores)          # a number in [-1, 1], never a "87%" string


@real
def test_similarity_is_not_presented_as_a_percentage(data):
    at = _select(_app().run(), "Bukayo Saka", data)
    cards = "".join(_similar_cards_html(at))
    score_spans = re.findall(r"<b>-?\d+\.\d+</b><span>similarity</span>", cards)
    assert len(score_spans) == 10 and "%" not in "".join(score_spans)
    explanation = " ".join(m.value for m in at.markdown if "Cosine similarity" in m.value).lower()
    assert "not a percentage or a probability" in explanation


@real
def test_league_position_minutes_and_count_filters(data):
    at = _select(_app().run(), "Bukayo Saka", data)
    saka = data.engine.find_id("Bukayo Saka")
    at.sidebar.selectbox(key="league").set_value("La Liga").run()
    assert not at.exception
    assert set(_card_names(at)) == set(logic.similar_table(data, saka, n=10, league="La Liga")["Player"])
    at.sidebar.selectbox(key="league").set_value(logic.ALL_LEAGUES).run()
    at.sidebar.slider(key="min_minutes").set_value(2500).run()
    assert not at.exception
    assert set(_card_names(at)) == set(logic.similar_table(data, saka, n=10, min_minutes=2500)["Player"])
    at.sidebar.slider(key="n_similar").set_value(5).run()
    assert not at.exception and len(_similar_cards_html(at)) == 5
    at.sidebar.slider(key="n_similar").set_value(20).run()
    assert len(_similar_cards_html(at)) == 20


@real
def test_position_filter_narrows_the_player_list(data):
    at = _app().run()
    everyone = len(at.sidebar.selectbox(key="player_id").options)
    at.sidebar.selectbox(key="position").set_value("GK").run()
    gk_options = at.sidebar.selectbox(key="player_id").options
    assert not at.exception and len(gk_options) == int((data.players["pos_group"] == "GK").sum()) < everyone
    alisson = data.engine.find_id("Alisson")
    at.sidebar.selectbox(key="player_id").set_value(alisson).run()
    assert not at.exception
    assert set(_card_names(at)) == set(logic.similar_table(data, alisson, n=10)["Player"])


@real
def test_partial_results_are_explained(data):
    at = _select(_app().run(), "Alisson", data)
    at.sidebar.selectbox(key="league").set_value("Bundesliga").run()
    at.sidebar.slider(key="min_minutes").set_value(3000).run()
    expected = len(logic.similar_table(data, data.engine.find_id("Alisson"), n=10, min_minutes=3000, league="Bundesliga"))
    assert 0 < expected < 10 and not at.exception
    assert len(_similar_cards_html(at)) == expected
    assert any(c.value == f"Only {expected} players meet the filters." for c in at.caption)


@real
def test_no_matching_players_shows_a_message_not_an_error():
    def render():
        from scouting.dashboard import views

        d = views.get_data()
        pid = d.engine.find_id("Alisson")
        f = views.Filters(pid, "All Leagues", "All", 10_000, 10)   # beyond the slider range on purpose
        o = {"player_id": pid, "name": "Alisson", "position_group": "GK"}
        views.similar_players(d, o, f)

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(render, default_timeout=120).run()
    assert not at.exception and len(at.dataframe) == 0 and len(_similar_cards_html(at)) == 0
    assert len(at.info) == 1 and "No players match" in at.info[0].value


@real
def test_comparison_follows_the_selected_similar_player(data):
    at = _select(_app().run(), "Bukayo Saka", data)
    saka = data.engine.find_id("Bukayo Saka")
    table = logic.similar_table(data, saka, n=10)
    pid = int(table["player_id"].iloc[2])
    at.selectbox(key="compare_id").set_value(pid).run()
    assert not at.exception
    other_name = data.players.loc[pid, "player"]
    view = at.dataframe[0].value                                  # the similar-player results are now cards, not a dataframe
    assert list(view.columns) == ["Category", "Statistic", "Bukayo Saka", other_name]
    assert len(view) == len(CONFIG.modeling.position_feature_lists["FWD"])


@real
def test_player_with_missing_optional_metadata_renders(data):
    missing = data.players.index[data.players["age"].isna()][0]
    at = _app().run()
    at.sidebar.selectbox(key="player_id").set_value(int(missing)).run()
    assert not at.exception
    hero = _hero_html(at)
    assert html_escape(data.players.loc[missing, "player"]) in hero
    visible = re.sub(r"<[^>]+>", " ", hero)                       # text a user can see, not attributes
    assert not re.search(r"(None|nan|NaN)", visible)
    assert all("None" not in str(c.value) and "nan" not in str(c.value).lower() for c in at.caption)


@real
def test_shareable_link_selects_a_player_and_bad_links_are_handled(data):
    pid = data.engine.find_id("Erling Haaland")
    at = _app()
    at.query_params["player_id"] = str(pid)
    at.run()
    assert not at.exception and "Erling Haaland" in _hero_html(at)
    for bad in ("999999", "abc", ""):
        at = _app()
        at.query_params["player_id"] = bad
        at.run()
        assert not at.exception and len(at.dataframe) == 0


@real
def test_invalid_player_id_is_reported_in_the_view_not_raised(data):
    def render():
        from scouting.dashboard import views

        d = views.get_data()
        f = views.Filters(None, "All Leagues", "All", 900, 10)
        o = {"player_id": 10**9, "name": "Nobody", "position_group": "FWD"}
        views.similar_players(d, o, f)

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(render, default_timeout=120).run()
    assert not at.exception and len(at.warning) == 1 and "not in the modeled pool" in at.warning[0].value


@real
def test_cluster_explorer_page_for_every_group(data):
    from streamlit.testing.v1 import AppTest

    def explorer():
        from scouting.dashboard import views

        views.cluster_explorer_page(views.get_data())

    at = AppTest.from_function(explorer, default_timeout=120).run()
    assert not at.exception
    for group, k in CONFIG.modeling.chosen_k.items():
        at.radio(key="explorer_group").set_value(group).run()
        assert not at.exception
        bars = json.loads(at.get("plotly_chart")[0].proto.spec)["data"][0]
        assert list(bars["x"]) == [f"{group}-{c}" for c in range(k)]
        option_keys = [o.split(" ")[0] for o in at.selectbox(key="explorer_cluster").options]
        assert option_keys == [f"{group}-{c}" for c in range(k)]
    at.radio(key="explorer_group").set_value("DEF").run()
    at.selectbox(key="explorer_cluster").set_value("DEF-1").run()
    assert not at.exception
    opac = [t["marker"]["opacity"] for t in json.loads(at.get("plotly_chart")[2].proto.spec)["data"]]
    assert opac.count(0.85) == 1 and opac.count(0.12) == CONFIG.modeling.chosen_k["DEF"] - 1
    at.selectbox(key="explorer_league").set_value("Bundesliga").run()
    assert not at.exception
