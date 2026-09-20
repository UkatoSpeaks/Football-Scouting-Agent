"""Phase 2 UI tests: percentile stat cards, similar-player scouting cards, View Profile,
Compare, the comparison header and the presentation-only radar chart."""
import re
from pathlib import Path

import pytest

from scouting import labels
from scouting.config import load_config
from scouting.dashboard import charts, components, logic

CONFIG = load_config()
READY = (CONFIG.models_dir / "kmeans.joblib").exists() and (CONFIG.processed_dir / "cluster_assignments.csv").exists()
real = pytest.mark.skipif(not READY, reason="run the pipeline first")
APP = str(Path(__file__).resolve().parents[1] / "app.py")


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


def _hero(at) -> str:
    return " ".join(m.value for m in at.markdown if 'class="sc-hero"' in m.value)


def _similar_cards_html(at) -> list[str]:
    return [m.value for m in at.markdown if 'class="sc-sim-card"' in m.value]


def _card_names(at) -> list[str]:
    return re.findall(r'sc-sim-name">([^<]+)</div>', "".join(_similar_cards_html(at)))


# ------------------------------------------------------------------- labels
def test_every_configured_feature_has_a_radar_category():
    for group, feats in CONFIG.modeling.position_feature_lists.items():
        for f in feats:
            assert f in labels.RADAR_CATEGORIES, f"{group}: no radar category for {f}"
            assert labels.RADAR_CATEGORIES[f] in labels.RADAR_CATEGORY_ORDER


# --------------------------------------------------------------- radar_scores
@real
def test_radar_categories_are_goalkeeper_specific_for_goalkeepers(data):
    alisson = data.engine.find_id("Alisson")
    cats = dict(logic.radar_scores(data, alisson))
    assert set(cats) == {"Shot stopping", "Cross claiming", "Sweeping", "Passing"}
    assert all(0 <= v <= 100 for v in cats.values())


@real
def test_radar_categories_for_outfield_players_follow_the_fixed_order(data):
    saka = data.engine.find_id("Bukayo Saka")
    cats = [c for c, _ in logic.radar_scores(data, saka)]
    assert cats == [c for c in labels.RADAR_CATEGORY_ORDER if c in cats]           # fixed order, not dict order
    assert set(cats) <= {"Shooting", "Chance creation", "Passing", "Progression", "Carrying & take-ons", "Defending"}
    assert len(cats) >= 4                                                          # enough axes for a readable radar


@real
def test_radar_score_is_the_mean_percentile_of_its_categorys_features(data):
    saka = data.engine.find_id("Bukayo Saka")
    group = data.players.loc[saka, "pos_group"]
    feats = data.config.modeling.position_feature_lists[group]
    shooting_feats = [f for f in feats if labels.RADAR_CATEGORIES[f] == "Shooting"]
    expected = data.percentiles[group].loc[saka, shooting_feats].mean()
    assert dict(logic.radar_scores(data, saka))["Shooting"] == pytest.approx(expected)


@real
def test_radar_scores_do_not_touch_similarity_or_clustering(data):
    """Presentation only: radar_scores must not mutate the shared data object."""
    saka = data.engine.find_id("Bukayo Saka")
    before = data.percentiles["FWD"].copy()
    logic.radar_scores(data, saka)
    assert before.equals(data.percentiles["FWD"])


# --------------------------------------------------------------- components
def test_stat_card_html_keeps_the_value_prominent_and_the_percentile_secondary():
    html = components.stat_card_html("Goals (per 90)", "0.96", 99.4)
    assert "0.96" in html and "99th percentile" in html and "width:99.4%" in html
    assert "<script" not in html


def test_stat_card_html_clamps_out_of_range_percentiles():
    assert "width:100.0%" in components.stat_card_html("x", "1", 137)
    assert "width:0.0%" in components.stat_card_html("x", "1", -5)


_ROW = {"Player": "Michael Olise", "Club": "Bayern Munich", "League": "Bundesliga",
        "Position": "FW,MF", "Minutes": 1850, "Cosine similarity": 0.924, "Cluster": "FWD-4"}


def test_similar_card_html_never_shows_similarity_as_a_percentage():
    html = components.similar_card_html(_ROW)
    assert "<b>0.924</b><span>similarity</span>" in html
    assert "92.4%" not in html and "92%" not in html
    assert "Michael Olise" in html and "1,850" in html and "Bundesliga" in html and "FWD-4" in html


def test_similar_card_html_escapes_player_derived_text():
    row = {**_ROW, "Player": "<script>alert(1)</script>", "Club": "A & B"}
    html = components.similar_card_html(row)
    assert "<script>" not in html and "A &amp; B" in html


def test_comparison_header_html_shows_both_players_and_escapes_names():
    html = components.comparison_header_html("Mbappé <b>", "Real Madrid", "La Liga", None, "Player X", "Club Y", "Serie A", None)
    assert "VS" in html and "Real Madrid" in html and "Club Y" in html and "Serie A" in html
    assert "&lt;b&gt;" in html and "<b>" not in html.split('class="sc-vs-name"')[1][:20]


# -------------------------------------------------------------------- charts
def test_radar_chart_has_two_named_closed_traces():
    fig = charts.radar_chart(["A", "B", "C"], [10, 20, 30], "X", [40, 50, 60], "Y")
    assert [t.name for t in fig.data] == ["X", "Y"]
    assert list(fig.data[0].theta) == ["A", "B", "C", "A"]                # closed into a loop
    assert list(fig.data[0].r) == [10, 20, 30, 10]
    assert fig.layout.polar.radialaxis.range == (0, 100)


# ----------------------------------------------------------------- UI runs
@real
def test_statistics_cards_show_the_percentile_bar_not_a_metric_delta(data):
    at = _select(_app().run(), "Bukayo Saka", data)
    stat_cards = [m.value for m in at.markdown if 'class="sc-stat-card"' in m.value]
    assert len(stat_cards) > 0
    assert any("percentile" in c and "sc-bar-fill" in c for c in stat_cards)
    assert len(at.metric) == 1                                            # only "Matches from other leagues" remains


@real
def test_similar_players_render_as_cards_with_actions(data):
    at = _select(_app().run(), "Bukayo Saka", data)
    cards = _similar_cards_html(at)
    assert len(cards) == 10
    view_buttons = [b for b in at.button if b.key and b.key.startswith("view_")]
    compare_buttons = [b for b in at.button if b.key and b.key.startswith("cmp_")]
    assert len(view_buttons) == 10 and len(compare_buttons) == 10


@real
def test_view_profile_makes_the_clicked_player_active_everywhere(data):
    at = _select(_app().run(), "Bukayo Saka", data)
    saka = data.engine.find_id("Bukayo Saka")
    target = int(logic.similar_table(data, saka, n=10)["player_id"].iloc[0])
    at.button(key=f"view_{target}").click().run()
    assert not at.exception
    assert at.sidebar.selectbox(key="player_id").value == target
    assert data.players.loc[target, "player"] in _hero(at)
    assert at.query_params["player_id"] == [str(target)]          # AppTest re-parses the query string into lists


@real
def test_compare_button_preselects_that_player_in_the_comparison_section(data):
    at = _select(_app().run(), "Bukayo Saka", data)
    saka = data.engine.find_id("Bukayo Saka")
    target = int(logic.similar_table(data, saka, n=10)["player_id"].iloc[3])
    at.button(key=f"cmp_{target}").click().run()
    assert not at.exception
    assert at.selectbox(key="compare_id").value == target
    view = at.dataframe[0].value
    assert data.players.loc[target, "player"] in list(view.columns)
    assert len(at.get("plotly_chart")) == 4                               # radar chart included


@real
def test_compare_selection_resets_when_the_active_player_changes(data):
    at = _select(_app().run(), "Bukayo Saka", data)
    saka = data.engine.find_id("Bukayo Saka")
    table = logic.similar_table(data, saka, n=10)
    compare_target = int(table["player_id"].iloc[3])
    view_target = int(table["player_id"].iloc[0])
    at.button(key=f"cmp_{compare_target}").click().run()
    assert at.selectbox(key="compare_id").value == compare_target
    at.button(key=f"view_{view_target}").click().run()
    assert not at.exception
    new_default = int(logic.similar_table(data, view_target, n=10)["player_id"].iloc[0])
    assert at.selectbox(key="compare_id").value == new_default            # not the stale Saka-side selection


@real
def test_comparison_header_and_radar_appear_for_a_goalkeeper_too(data):
    at = _select(_app().run(), "Alisson", data)
    assert not at.exception
    vs = " ".join(m.value for m in at.markdown if 'class="sc-vs"' in m.value)
    assert "Alisson" in vs and "VS" in vs
    assert len(at.get("plotly_chart")) == 4                               # radar, comparison, league mix, PCA map
