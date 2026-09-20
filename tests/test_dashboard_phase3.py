"""Phase 3 UI tests: the playing-style section, the Player Style Map (renamed PCA chart)
and the restructured Cluster Explorer."""
import re
from pathlib import Path

import pytest

from scouting.config import load_config
from scouting.dashboard import charts, components, logic
from scouting.labels import RADAR_CATEGORY_ORDER

CONFIG = load_config()
READY = (CONFIG.models_dir / "kmeans.joblib").exists() and (CONFIG.processed_dir / "cluster_assignments.csv").exists()
real = pytest.mark.skipif(not READY, reason="run the pipeline first")
APP = str(Path(__file__).resolve().parents[1] / "app.py")

BANNED_PHRASES = [
    "best player", "worst player", "perfect replacement", "ideal replacement", "guaranteed replacement",
    "best cluster", "elite player", "definitely a", "plays exactly like",
]


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


def _all_text(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    return " ".join(parts).lower()


# --------------------------------------------------------------------- logic
@real
def test_style_traits_use_the_players_own_percentile_not_the_cluster_mean(data):
    saka = data.engine.find_id("Bukayo Saka")
    # a synthetic ClusterInfo isolates style_traits from whichever cluster Saka is actually in
    info = logic.ClusterInfo("FWD-4", "FWD", 4, 30, 0.07, high=[("npxG", 1.5)], low=[("crossing", -0.6)])
    traits = logic.style_traits(data, saka, info)
    assert traits["high"] == [("npxG", pytest.approx(float(data.percentiles["FWD"].loc[saka, "npxg_p90"])))]
    assert traits["low"] == [("Crossing", pytest.approx(float(data.percentiles["FWD"].loc[saka, "crosses_p90"])))]


@real
def test_style_traits_works_for_the_goalkeepers_smaller_feature_set(data):
    alisson = data.engine.find_id("Alisson")
    info = logic.cluster_info(data, "GK", int(data.assignments.loc[alisson, "cluster_id"]))
    traits = logic.style_traits(data, alisson, info)
    assert all(0 <= pct <= 100 for _, pct in traits["high"] + traits["low"])


def test_style_category_highlights_are_derived_from_radar_categories_not_invented():
    info = logic.ClusterInfo("FWD-4", "FWD", 4, 30, 0.07,
                              high=[("xAG", 2.1), ("key passes", 1.6), ("progressive passes received", 1.2)], low=[])
    cats = logic.style_category_highlights(info)
    assert cats and set(cats) <= set(RADAR_CATEGORY_ORDER)
    assert "Chance creation" in cats                          # xAG and key passes are both chance creation


def test_style_category_highlights_is_empty_when_the_cluster_has_no_high_traits():
    info = logic.ClusterInfo("GK-0", "GK", 0, 86, 0.56, high=[], low=[("passing volume", -0.6)])
    assert logic.style_category_highlights(info) == []


# ---------------------------------------------------------------- components
def test_style_identity_html_names_the_cluster_size_and_categories():
    html = components.style_identity_html("FWD-4", 30, 0.0667, "FWD", ["Chance creation", "Progression"])
    assert ">FWD-4<" in html and "30 players in this cluster" in html and "7%" in html
    assert "Chance creation, Progression" in html


def test_style_identity_html_says_so_plainly_with_no_high_categories():
    html = components.style_identity_html("GK-0", 86, 0.56, "GK", [])
    assert "close to the position average" in html.lower()
    assert "Higher than position average:" not in html          # no dangling "Higher than...:" with nothing after it


def test_trait_percentile_rows_html_shows_ordinal_percentiles():
    html = components.trait_percentile_rows_html([("xAG", 96.4), ("Key passes", 89.1)])
    assert "xAG" in html and "96th" in html and "Key passes" in html and "89th" in html


def test_trait_percentile_rows_html_handles_no_traits_without_claiming_none_exist():
    html = components.trait_percentile_rows_html([])
    assert "0.5 SD" in html or "None" in html


def test_style_profile_bars_html_clamps_and_escapes():
    html = components.style_profile_bars_html([("<script>", 137), ("Defending", -5)])
    assert "<script>" not in html and "width:100.0%" in html and "width:0.0%" in html


def test_map_legend_html_names_every_marker():
    html = components.map_legend_html()
    assert "Cluster members" in html and "Selected player" in html and "Similar players" in html


# -------------------------------------------------------------------- charts
def test_profile_heatmap_title_and_axes_state_the_units():
    import pandas as pd

    z = pd.DataFrame({"C0": [1.2, -0.3]}, index=["Goals (per 90)", "Tackles (per 90)"])
    raw = pd.DataFrame({"C0": [0.5, 1.1]}, index=z.index)
    fig = charts.profile_heatmap(z, raw, group="FWD")
    assert "standard deviation" in fig.layout.title.text.lower()
    assert "FWD" in fig.layout.title.text
    assert "standard deviation" in fig.layout.yaxis.title.text.lower() or "statistic" in fig.layout.yaxis.title.text.lower()
    assert "position average" in fig.data[0].colorbar.title.text.lower()


# ----------------------------------------------------------------- UI runs
@real
def test_playing_style_section_shows_the_players_own_percentiles(data):
    at = _select(_app().run(), "Bukayo Saka", data)
    assert not at.exception
    style_blocks = [m.value for m in at.markdown if 'class="sc-style-id"' in m.value]
    assert len(style_blocks) == 1
    trait_blocks = [m.value for m in at.markdown if 'class="sc-trait-list"' in m.value]
    assert len(trait_blocks) == 2                                       # higher- and lower-than-average columns
    assert re.search(r"\d+(st|nd|rd|th)", "".join(trait_blocks))        # ordinal percentiles, not raw z-scores
    profile_bars = [m.value for m in at.markdown if 'class="sc-profile-bars"' in m.value]
    assert len(profile_bars) == 1


@real
def test_playing_style_language_stays_statistical(data):
    at = _select(_app().run(), "Bukayo Saka", data)
    text = _all_text(at)
    for phrase in BANNED_PHRASES:
        assert phrase not in text, f"found discouraged phrase: {phrase!r}"
    assert "statistical profile" in text or "position average" in text


@real
def test_player_style_map_has_a_legend_and_a_how_to_read_expander(data):
    at = _select(_app().run(), "Bukayo Saka", data)
    assert not at.exception
    assert any(e.label == "How to read this map" for e in at.expander)
    legend = " ".join(m.value for m in at.markdown if 'class="sc-legend"' in m.value)
    assert "Cluster members" in legend and "Selected player" in legend and "Similar players" in legend
    assert [s.value for s in at.subheader][-1] == "Player style map"


@real
def test_cluster_explorer_selecting_a_cluster_updates_its_profile_and_map_focus(data):
    def explorer():
        from scouting.dashboard import views

        views.cluster_explorer_page(views.get_data())

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(explorer, default_timeout=120).run()
    at.radio(key="explorer_group").set_value("FWD").run()
    at.selectbox(key="explorer_cluster").set_value("FWD-4").run()
    assert not at.exception
    identity = " ".join(m.value for m in at.markdown if 'class="sc-style-id"' in m.value)
    assert ">FWD-4<" in identity
    info = logic.cluster_info(data, "FWD", 4)
    assert f"{info.n_players:,} players in this cluster" in identity
    at.selectbox(key="explorer_cluster").set_value("FWD-0").run()
    assert not at.exception
    identity_0 = " ".join(m.value for m in at.markdown if 'class="sc-style-id"' in m.value)
    assert ">FWD-0<" in identity_0 and identity_0 != identity


@real
def test_cluster_explorer_language_stays_statistical(data):
    def explorer():
        from scouting.dashboard import views

        views.cluster_explorer_page(views.get_data())

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(explorer, default_timeout=120).run()
    text = _all_text(at)
    for phrase in BANNED_PHRASES:
        assert phrase not in text, f"found discouraged phrase: {phrase!r}"
