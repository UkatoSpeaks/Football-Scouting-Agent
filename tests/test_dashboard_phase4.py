"""Phase 4 UI tests: the Player Database page and the scouting Shortlist (session state only,
no persistence). The shortlist workflow test drives the real multipage app end to end."""
from pathlib import Path

import pytest

from scouting.config import load_config
from scouting.dashboard import components, logic

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


def _run(render, args=None):
    from streamlit.testing.v1 import AppTest

    return AppTest.from_function(render, default_timeout=120, args=args).run()


def _render_shortlist_page():
    from scouting.dashboard import views

    views.shortlist_page(views.get_data())


def _shortlist_at(*player_ids):
    """A shortlist-page AppTest seeded with ``player_ids`` already shortlisted.

    Session state is set *before* the first ``run()``, not by calling ``_add_to_shortlist``
    inside the rendered function - that function's body reruns on every interaction (just like
    a real script), so seeding there would re-add the ids after every click, masking a removal.
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_render_shortlist_page, default_timeout=120)
    at.session_state["shortlist"] = list(player_ids)
    return at.run()


# =================================================================== 4A: Player Database
# ------------------------------------------------------------------------- pure logic
@real
def test_database_table_filters_league_position_cluster_minutes_and_search(data):
    everyone = logic.database_table(data)
    assert len(everyone) == len(data.players)                                  # the whole modelled pool
    assert list(everyone.columns) == logic.DATABASE_COLUMNS

    la_liga = logic.database_table(data, league="La Liga")
    assert set(la_liga["League"]) == {"La Liga"}

    fwd = logic.database_table(data, position="FWD")
    assert set(fwd["Position"]) == {"FWD"}

    cluster = logic.database_table(data, position="FWD", cluster="FWD-4")
    assert set(cluster["Cluster"]) == {"FWD-4"}

    high_minutes = logic.database_table(data, min_minutes=3000)
    assert (high_minutes["Minutes"] >= 3000).all()

    saka = logic.database_table(data, search="bukayo sak")                     # case-insensitive substring
    assert list(saka["Player"]) == ["Bukayo Saka"]

    assert logic.database_table(data, search="zzzznobodyhasthisname").empty
    assert everyone["Minutes"].is_monotonic_decreasing                          # most minutes first


@real
def test_database_clusters_are_scoped_to_the_position_filter(data):
    assert logic.database_clusters(data, "GK") == ["GK-0", "GK-1"]
    all_clusters = logic.database_clusters(data, logic.ALL_POSITIONS)
    assert len(all_clusters) == sum(CONFIG.modeling.chosen_k.values())
    assert all_clusters == sorted(all_clusters, key=lambda k: (k.rsplit("-", 1)[0], int(k.rsplit("-", 1)[1])))


def test_database_card_html_has_no_similarity_score_and_reuses_the_card_language():
    row = {"Player": "Michael Olise", "Club": "Bayern Munich", "League": "Bundesliga",
           "Position": "FWD", "Minutes": 1850, "Cluster": "FWD-4"}
    html = components.database_card_html(row)
    assert 'class="sc-sim-card"' in html                        # same visual language as similar-player cards
    assert "similarity" not in html.lower()
    assert "Michael Olise" in html and "1,850" in html and "FWD-4" in html


def test_database_card_html_escapes_player_derived_text():
    row = {"Player": "<script>", "Club": "C", "League": "L", "Position": "P", "Minutes": 900, "Cluster": "FWD-0"}
    assert "<script>" not in components.database_card_html(row)


# ------------------------------------------------------------------------------ UI runs
def _database_render():
    from scouting.dashboard import views

    views.player_database_page(views.get_data())


@real
def test_database_page_renders_with_stats_and_the_full_pool(data):
    at = _run(_database_render)
    assert not at.exception
    assert at.header[0].value == "Player database"
    stats = " ".join(m.value for m in at.markdown if "sc-stats" in m.value)
    n_groups = data.players["pos_group"].nunique()
    assert f"{len(data.players):,}" in stats and f">{len(data.leagues)}<" in stats and f">{n_groups}<" in stats
    # the minimum-minutes filter defaults to 900 (matching the sidebar's similar-player default), so the
    # count shown is that subset of the modelled pool, not the full 1,959 - the stats line above still is
    default_shown = len(logic.database_table(data, min_minutes=900))
    assert any(c.value.startswith(f"{default_shown:,} of {len(data.players):,}") for c in at.caption)
    assert len(at.dataframe) == 1


@real
def test_database_league_filter_narrows_the_table(data):
    at = _run(_database_render)
    at.selectbox(key="db_league").set_value("La Liga").run()
    assert not at.exception
    assert set(at.dataframe[0].value["League"]) == {"La Liga"}


@real
def test_database_position_filter_narrows_the_table_and_cluster_options(data):
    at = _run(_database_render)
    at.selectbox(key="db_position").set_value("GK").run()
    assert not at.exception
    assert set(at.dataframe[0].value["Position"]) == {"GK"}
    assert at.selectbox(key="db_cluster").options == ["All Clusters", "GK-0", "GK-1"]


@real
def test_database_cluster_filter_narrows_the_table(data):
    at = _run(_database_render)
    at.selectbox(key="db_position").set_value("FWD").run()
    at.selectbox(key="db_cluster").set_value("FWD-4").run()
    assert not at.exception
    assert set(at.dataframe[0].value["Cluster"]) == {"FWD-4"}


@real
def test_database_minutes_filter_narrows_the_table(data):
    at = _run(_database_render)
    at.slider(key="db_min_minutes").set_value(3000).run()
    assert not at.exception
    assert (at.dataframe[0].value["Minutes"] >= 3000).all()


@real
def test_database_search_matches_by_name(data):
    at = _run(_database_render)
    at.text_input(key="db_search").set_value("bukayo").run()
    assert not at.exception
    assert list(at.dataframe[0].value["Player"]) == ["Bukayo Saka"]


@real
def test_database_empty_results_show_a_message_not_an_error(data):
    at = _run(_database_render)
    at.text_input(key="db_search").set_value("zzzz-nobody-zzzz").run()
    assert not at.exception
    assert len(at.dataframe) == 0
    assert any("No players match" in i.value for i in at.info)


@real
def test_database_table_has_a_row_selection_enabled_dataframe(data):
    """``st.dataframe``'s own row-selection grid interaction can't be driven through AppTest
    (no such API on the test proxy); the action panel it reveals is covered directly below."""
    at = _run(_database_render)
    assert not at.exception
    assert list(at.dataframe[0].proto.selection_mode) == [0]     # 0 == SelectionMode.SINGLE_ROW


def _render_row_actions():
    from scouting.dashboard import views

    row = {"player_id": 900, "Player": "Bukayo Saka", "Club": "Arsenal", "League": "Premier League",
           "Position": "FWD", "Minutes": 1729, "Cluster": "FWD-4"}
    views._database_row_actions("dbsel", row)


def test_database_row_actions_wire_up_view_profile_and_shortlist():
    """The action panel shown for whichever row is selected - tested directly, since the
    dataframe's own row-click can't be simulated through AppTest."""
    at = _run(_render_row_actions)
    assert not at.exception
    assert {b.key for b in at.button} == {"dbsel_view_900", "dbsel_short_900"}
    assert at.button(key="dbsel_short_900").label == "+ Shortlist"
    at.button(key="dbsel_short_900").click().run()
    assert not at.exception
    assert at.session_state["shortlist"] == [900]
    assert at.button(key="dbsel_short_900").label == "✓ Shortlisted"
    assert "player_id" not in at.session_state                    # View Profile not clicked - nothing navigated yet


@real
def test_database_card_view_toggle_shows_cards_with_actions(data):
    at = _run(_database_render)
    at.selectbox(key="db_position").set_value("GK").run()
    at.slider(key="db_min_minutes").set_value(3000).run()           # small enough group to stay under the card cap
    at.toggle(key="db_card_view").set_value(True).run()
    assert not at.exception
    expected = len(logic.database_table(data, position="GK", min_minutes=3000))
    assert 0 < expected < 60
    cards = [m.value for m in at.markdown if 'class="sc-sim-card"' in m.value]
    assert len(cards) == expected
    assert len(at.dataframe) == 0                                   # table replaced, not duplicated
    action_buttons = [b for b in at.button if b.key and b.key.startswith("dbcard_short_")]
    assert len(action_buttons) == len(cards)


@real
def test_database_card_view_caps_a_large_result_set(data):
    at = _run(_database_render)
    at.slider(key="db_min_minutes").set_value(0).run()              # the whole 1,959-player pool
    at.toggle(key="db_card_view").set_value(True).run()
    assert not at.exception
    cards = [m.value for m in at.markdown if 'class="sc-sim-card"' in m.value]
    assert len(cards) == 60
    assert any("first 60" in c.value for c in at.caption)


# =================================================================== 4B/4C: Shortlist
@real
def test_empty_shortlist_state_is_clean(data):
    at = _shortlist_at()
    assert not at.exception
    assert at.header[0].value == "My scouting shortlist"
    text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "No players shortlisted yet" in text
    assert "Similar players" in text and "Player database" in text
    assert len(at.button) == 0                                      # no compare/view/remove controls to show


@real
def test_shortlist_page_lists_players_with_view_profile_and_remove(data):
    at = _shortlist_at(900, 23)                            # Saka, Van Dijk
    assert not at.exception
    assert any(c.value == "2 players" for c in at.caption)
    cards = [m.value for m in at.markdown if 'class="sc-sim-card"' in m.value]
    assert len(cards) == 2 and "Bukayo Saka" in "".join(cards) and "Virgil van Dijk" in "".join(cards)
    assert {b.key for b in at.button} >= {"sl_view_900", "sl_remove_900", "sl_view_23", "sl_remove_23"}


@real
def test_removing_a_shortlisted_player_updates_the_page(data):
    at = _shortlist_at(900, 23)
    at.button(key="sl_remove_900").click().run()
    assert not at.exception
    assert at.session_state["shortlist"] == [23]
    cards = [m.value for m in at.markdown if 'class="sc-sim-card"' in m.value]
    assert len(cards) == 1 and "Virgil van Dijk" in cards[0] and "Bukayo Saka" not in cards[0]


def test_duplicate_add_does_not_create_duplicates():
    def render():
        from scouting.dashboard import views

        views._add_to_shortlist(900)
        views._add_to_shortlist(900)

    at = _run(render)
    assert not at.exception
    assert at.session_state["shortlist"] == [900]


def test_shortlist_survives_a_rerun_triggered_by_an_unrelated_widget():
    def render():
        import streamlit as st

        from scouting.dashboard import views

        views._add_to_shortlist(900)
        st.checkbox("unrelated", key="unrelated_widget")

    at = _run(render)
    assert at.session_state["shortlist"] == [900]
    at.checkbox(key="unrelated_widget").set_value(True).run()
    assert not at.exception
    assert at.session_state["shortlist"] == [900]


def test_toggling_shortlist_twice_removes_it_again():
    def render():
        from scouting.dashboard import views

        views._toggle_shortlist(900)

    at = _run(render)
    assert at.session_state["shortlist"] == [900]


# =================================================================== 4D: shortlist compare
@real
def test_shortlist_compare_multiselect_is_capped_at_three(data):
    at = _shortlist_at(900, 147, 23, 383)
    assert not at.exception
    el = at.multiselect(key="shortlist_compare")
    assert el.proto.max_selections == 3
    names = data.assignments.loc[[900, 147, 23, 383], "player_name"]
    joined = " ".join(el.options)
    assert all(name in joined for name in names)


@real
def test_shortlist_compare_reuses_the_single_comparison_implementation(data):
    """Two same-group shortlisted players -> the exact same comparison block as the
    Player scouting page's Player comparison section (radar + bar chart + detail table)."""
    at = _shortlist_at(900, 147)                            # Saka + Mbappe, both FWD
    at.multiselect(key="shortlist_compare").set_value([900, 147]).run()
    assert not at.exception
    assert len(at.get("plotly_chart")) == 2                           # radar + bar chart
    assert len(at.dataframe) == 1                                     # the detail table
    view = at.dataframe[0].value
    assert list(view.columns)[:2] == ["Category", "Statistic"]
    assert len(view) == len(CONFIG.modeling.position_feature_lists["FWD"])


@real
def test_shortlist_compare_explains_incompatible_position_groups(data):
    at = _shortlist_at(900, 23)                              # Saka (FWD) + Van Dijk (DEF)
    at.multiselect(key="shortlist_compare").set_value([900, 23]).run()
    assert not at.exception
    assert len(at.get("plotly_chart")) == 0 and len(at.dataframe) == 0  # no comparison rendered
    assert any("different position groups" in i.value for i in at.info)


@real
def test_shortlist_compare_needs_at_least_two_selections(data):
    at = _shortlist_at(900, 147)
    at.multiselect(key="shortlist_compare").set_value([900]).run()
    assert not at.exception
    assert any("at least two" in c.value for c in at.caption)


# =================================================================== end-to-end workflow
@real
def test_full_shortlist_workflow_search_shortlist_view_and_compare(data):
    """Search Saka -> shortlist a similar player from a card -> sidebar reflects it ->
    open the shortlist -> view that player's profile -> back on Player scouting with them
    active. Drives the real multipage app (app.py), not an isolated page function."""
    saka = data.engine.find_id("Bukayo Saka")
    target = int(logic.similar_table(data, saka, n=10)["player_id"].iloc[0])
    target_name = data.players.loc[target, "player"]

    at = _app().run()
    at.sidebar.selectbox(key="player_id").set_value(saka).run()
    at.button(key=f"short_{target}").click().run()
    assert not at.exception
    assert at.session_state["shortlist"] == [target]

    sidebar_text = " ".join(m.value for m in at.sidebar.markdown)
    assert target_name in sidebar_text

    at.sidebar.button(key="goto_shortlist").click().run()
    assert not at.exception
    assert at.header[0].value == "My scouting shortlist"
    assert any(b.key == f"sl_view_{target}" for b in at.button)

    at.button(key=f"sl_view_{target}").click().run()
    assert not at.exception
    assert at.sidebar.selectbox(key="player_id").value == target
    hero = " ".join(m.value for m in at.markdown if 'class="sc-hero"' in m.value)
    assert target_name in hero
