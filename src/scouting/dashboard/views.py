"""Streamlit rendering for the scouting dashboard.

All computation lives in ``scouting.dashboard.logic`` and the ML modules; this
module only lays results out. Models and tables are loaded once with
``st.cache_resource`` (nothing is retrained when the app starts).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import streamlit as st

from scouting.dashboard import charts, components, logic
from scouting.dashboard.data import POSITION_GROUPS, DashboardArtifactsError, DashboardData, load_dashboard_data
from scouting.similarity import DEFAULT_MIN_CANDIDATE_MINUTES, PlayerNotFoundError

TITLE = "European Football Player Scouting"
SUBTITLE = "Find statistically similar players across Europe's top five leagues."
DASH = "—"


@st.cache_resource(show_spinner="Loading player data and models…")
def get_data() -> DashboardData:
    return load_dashboard_data()


def load_data_or_stop() -> DashboardData:
    try:
        return get_data()
    except DashboardArtifactsError as exc:
        st.error(f"The scouting data is not ready.\n\n{exc}")
        st.stop()


def page_header(title: str) -> None:
    """Small brand line above a page title (used by pages that have no hero panel)."""
    st.markdown(f'<div class="sc-kicker">{TITLE.upper()}</div>', unsafe_allow_html=True)
    st.header(title)


# ------------------------------------------------------------------ sidebar
@dataclass
class Filters:
    player_id: int | None
    league: str
    position: str
    min_minutes: int
    n_similar: int


def _query_player(data: DashboardData) -> None:
    """Preselect a player from ?player_id=... on first load (shareable links)."""
    if "player_id" in st.session_state or "player_id" not in st.query_params:
        return
    try:
        pid = int(st.query_params["player_id"])
    except (TypeError, ValueError):
        pid = None
    if pid in data.players.index:
        st.session_state["player_id"] = pid
    else:
        st.sidebar.warning(f"Unknown player_id in the link: {st.query_params['player_id']!r}")


FILTER_DEFAULTS = {
    "league": logic.ALL_LEAGUES,
    "position": logic.ALL_POSITIONS,
    "min_minutes": DEFAULT_MIN_CANDIDATE_MINUTES,
    "n_similar": 10,
}


def _reset_filters() -> None:
    """Button callback: put every filter back to its default (the selected player is kept)."""
    for key, value in FILTER_DEFAULTS.items():
        st.session_state[key] = value


def _sidebar_label(text: str) -> None:
    st.sidebar.markdown(components.sidebar_label(text), unsafe_allow_html=True)


# ---------------------------------------------------------------- navigation
def _switch_to(page_key: str) -> None:
    """Jump to another page of the app (set up once in ``app.py``'s ``st.session_state["_nav_pages"]``).

    A no-op outside the full app (for example a test that renders one page function directly
    with ``AppTest.from_function``), so nothing here ever requires the real multipage shell.
    """
    pages = st.session_state.get("_nav_pages")
    if pages and page_key in pages:
        st.switch_page(pages[page_key])


def _goto_profile(player_id: int) -> None:
    """'View Profile' from the Player database or the shortlist: same navigation as everywhere else."""
    st.session_state["player_id"] = player_id
    _switch_to("scouting")


# ------------------------------------------------------------------ shortlist
def _shortlist_ids() -> list[int]:
    return st.session_state.setdefault("shortlist", [])


def _is_shortlisted(player_id: int) -> bool:
    return player_id in st.session_state.get("shortlist", [])


def _shortlist_label(player_id: int) -> str:
    return "✓ Shortlisted" if _is_shortlisted(player_id) else "+ Shortlist"


def _add_to_shortlist(player_id: int) -> None:
    ids = st.session_state.setdefault("shortlist", [])
    if player_id not in ids:
        ids.append(player_id)


def _remove_from_shortlist(player_id: int) -> None:
    ids = st.session_state.get("shortlist", [])
    if player_id in ids:
        ids.remove(player_id)


def _toggle_shortlist(player_id: int) -> None:
    if _is_shortlisted(player_id):
        _remove_from_shortlist(player_id)
    else:
        _add_to_shortlist(player_id)


def _shortlist_sidebar(data: DashboardData) -> None:
    _sidebar_label("SHORTLIST")
    ids = _shortlist_ids()
    if not ids:
        st.sidebar.caption("No players shortlisted yet.")
    else:
        names = [data.assignments.loc[i, "player_name"] for i in ids if i in data.assignments.index]
        st.sidebar.caption(f"{len(names)} player{'' if len(names) == 1 else 's'}")
        preview = names[:5]
        text = "\n".join(f"- {n}" for n in preview)
        if len(names) > len(preview):
            text += f"\n- …and {len(names) - len(preview)} more"
        st.sidebar.markdown(text)
    st.sidebar.button("View shortlist", key="goto_shortlist", on_click=_switch_to, args=("shortlist",), width="stretch")


def sidebar(data: DashboardData) -> Filters:
    _query_player(data)

    # The position filter narrows the player list; it sits further down but is read from
    # session state so that the list is already correct on the same run.
    position_now = st.session_state.get("position", logic.ALL_POSITIONS)
    labels = logic.picker_labels(data, position_now)

    _sidebar_label("PLAYER")
    player_id = st.sidebar.selectbox(
        "Player",
        options=list(labels),
        format_func=labels.get,
        index=None,
        placeholder="Search for a player…",
        key="player_id",
        label_visibility="collapsed",
        help="Type to search. Only players with enough minutes to be modelled are listed.",
    )

    _sidebar_label("PROFILE")
    league = st.sidebar.selectbox(
        "League",
        [logic.ALL_LEAGUES, *data.leagues],
        key="league",
        help="Restricts the similar players to this league. The selected player can be from any league.",
    )
    position = st.sidebar.selectbox(
        "Position",
        [logic.ALL_POSITIONS, *POSITION_GROUPS],
        key="position",
        help="Narrows the player list. Similar players are always taken from the selected player's own "
        "position group (a goalkeeper is never compared with a forward).",
    )

    _sidebar_label("SCOUTING")
    min_minutes = st.sidebar.slider(
        "Minimum candidate minutes",
        min_value=int(data.config.feature_engineering.min_minutes),
        max_value=3000,
        value=FILTER_DEFAULTS["min_minutes"],
        step=50,
        key="min_minutes",
        help="Similar players must have played at least this many minutes. Higher values mean more reliable comparisons.",
    )
    n_similar = st.sidebar.slider("Number of similar players", 5, 20, FILTER_DEFAULTS["n_similar"], key="n_similar")

    _sidebar_label("ACTIONS")
    st.sidebar.button("Reset filters", key="reset_filters", on_click=_reset_filters, width="stretch")

    _shortlist_sidebar(data)
    return Filters(player_id, league, position, min_minutes, n_similar)


# ------------------------------------------------------------- limitations
LIMITATIONS = """
- **Team style affects per-90 numbers.** Counts such as touches, passes, tackles and interceptions depend on how
  much of the ball a player's team has and how it plays. A midfielder at a high-possession club tends to record
  fewer defensive actions per 90 than an equally good defender at a team that defends more. The analysis does
  not correct for this, so some similarities and clusters partly reflect team style rather than the player.
- **Statistical similarity is a proxy for playing style.** It only sees the statistics in the dataset (no video,
  off-ball movement, tactical role or quality of opposition) and it says nothing about how good a player is.
- **One season, and no advanced data for 2025-26.** Results use the complete 2024-25 season. FBref removed its
  advanced statistics in January 2026, so a newer season with the same features is not available.
- **Small samples are noisy.** Players need at least 450 minutes to be modelled and similar players default to
  900+, but season statistics for a few hundred minutes can still swing a lot.
- **Position groups are broad.** Players are grouped as GK, DEF, MID or FWD from their first listed position;
  there is no separate winger or full-back label. Similarity is only computed within a group.
- **Clusters are soft.** Player styles form a continuum, so cluster boundaries are fuzzy and a player near an
  edge could sit in a neighbouring cluster. The goalkeeper model uses only six statistics and is the weakest.
- **Leagues are not adjusted.** Statistics are standardised across all five leagues together, so league-level
  differences in style and intensity are still present.
"""


def limitations() -> None:
    with st.expander("Limitations of this analysis"):
        st.markdown(LIMITATIONS)


# ---------------------------------------------------------------- landing
def _pick_from_landing() -> None:
    """Landing search callback: select the player everywhere (the sidebar selector is the source of truth)."""
    pid = st.session_state.get("landing_player")
    if pid is not None:
        st.session_state["player_id"] = pid
    st.session_state["landing_player"] = None


def landing(data: DashboardData) -> None:
    """First screen: no player selected yet."""
    labels = logic.picker_labels(data, st.session_state.get("position", logic.ALL_POSITIONS))
    with st.container(key="landing_hero"):
        st.title(TITLE)
        st.markdown(SUBTITLE)
        st.selectbox(
            "Search for a player",
            options=list(labels),
            format_func=labels.get,
            index=None,
            placeholder="Search for a player",
            key="landing_player",
            label_visibility="collapsed",
            on_change=_pick_from_landing,
        )
        st.markdown('<div class="sc-hint">Select a player to begin scouting.</div>', unsafe_allow_html=True)
        st.markdown(
            components.landing_stats_html(len(data.players), len(data.leagues), int(data.players["pos_group"].nunique())),
            unsafe_allow_html=True,
        )
    limitations()


# --------------------------------------------------------------- overview
def player_overview(data: DashboardData, player_id: int) -> dict:
    """Player profile header: photo (or placeholder), identity, cluster badge and key facts."""
    o = logic.overview(data, player_id)
    info = logic.cluster_info(data, o["position_group"], o["cluster_id"])
    st.markdown(
        components.profile_hero_html(
            o, logic.cluster_headline(info), logic.hero_chips(data, o), photo_url=data.photos.get(int(player_id))
        ),
        unsafe_allow_html=True,
    )
    return o


# --------------------------------------------------------------- league mix
def league_mix(data: DashboardData, o: dict, table, f: Filters) -> None:
    counts = logic.league_counts(table, data.leagues)
    home = logic.main_league(data, o["player_id"])
    outside = int((table["League"] != home).sum())
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(charts.league_chart(counts, highlight=home), width="stretch")
    with right:
        if f.league == logic.ALL_LEAGUES:
            st.metric("Matches from other leagues", f"{outside} of {len(table)}", border=True,
                      help=f"{o['name']} plays in the {home}; the darker bar marks that league.")
        else:
            st.caption(f"Results are limited to {f.league}. Choose 'All Leagues' to see the cross-league mix.")


# -------------------------------------------------------------- statistics
def player_statistics(data: DashboardData, o: dict) -> None:
    st.subheader("Player statistics")
    group = o["position_group"]
    totals = logic.season_totals(data, o["player_id"])
    st.caption(
        f"Per-90 rates and ratios used to model {group} players. Percentile = rank among the "
        f"{logic.group_size(data, group):,} {group}s with enough minutes (higher value = higher percentile)."
        + (f" Season totals: {totals['goals']} goals · {totals['assists']} assists · {totals['minutes']:,} minutes."
           if {"goals", "assists", "minutes"} <= totals.keys() else "")
    )
    categories = logic.player_stats(data, o["player_id"])
    tabs = st.tabs([name for name, _ in categories])
    for tab, (_, rows) in zip(tabs, categories):
        with tab:
            for start in range(0, len(rows), 4):
                cols = st.columns(4)
                for col, (_, r) in zip(cols, rows.iloc[start:start + 4].iterrows()):
                    with col:
                        st.markdown(components.stat_card_html(r["name"], r["text"], r["percentile"]),
                                    unsafe_allow_html=True)


# --------------------------------------------------------- similar players
def _view_profile(player_id: int) -> None:
    """'View Profile' callback: the clicked player becomes the active player everywhere."""
    st.session_state["player_id"] = player_id


def _set_compare(player_id: int) -> None:
    """'Compare' callback: preselect this player in the comparison section below."""
    st.session_state["compare_id"] = player_id


def _similar_player_cards(data: DashboardData, table) -> None:
    """A responsive grid of scouting cards (photo, identity, similarity, cluster, actions)."""
    records = table.to_dict("records")
    cols_per_row = 3
    for start in range(0, len(records), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, row in zip(cols, records[start:start + cols_per_row]):
            pid = int(row["player_id"])
            with col, st.container(border=True):
                st.markdown(components.similar_card_html(row, data.photos.get(pid)), unsafe_allow_html=True)
                b1, b2 = st.columns(2)
                b1.button("View Profile", key=f"view_{pid}", width="stretch", on_click=_view_profile, args=(pid,))
                b2.button("Compare", key=f"cmp_{pid}", width="stretch", on_click=_set_compare, args=(pid,))
                st.button(_shortlist_label(pid), key=f"short_{pid}", width="stretch",
                          on_click=_toggle_shortlist, args=(pid,))


def similar_players(data: DashboardData, o: dict, f: Filters):
    st.subheader("Find similar players")
    st.markdown(f"**Which players statistically resemble {o['name']}?**")
    scope = (
        f"Compared with {o['position_group']} players who have played at least {f.min_minutes:,} minutes"
        f" · league: {f.league}"
    )
    st.caption(scope)
    try:
        table = logic.similar_table(data, o["player_id"], n=f.n_similar, min_minutes=f.min_minutes, league=f.league)
    except PlayerNotFoundError as exc:
        st.warning(str(exc))
        return None
    if table.empty:
        st.info("No players match these filters. Try a lower minimum, or set the league to 'All Leagues'.")
        return table
    if len(table) < f.n_similar:
        st.caption(f"Only {len(table)} players meet the filters.")
    _similar_player_cards(data, table)
    st.markdown(
        "**Cosine similarity** compares the shape of two players' standardised statistical profiles (0 = the "
        "position average): 1 means the same direction, 0 unrelated, -1 opposite. It is a similarity score, "
        "**not a percentage or a probability**, and says nothing about how good either player is."
    )
    with st.expander("League mix of these results"):
        league_mix(data, o, table, f)
    return table


# ---------------------------------------------------------- comparison
def _comparison_block(data: DashboardData, player_a: int, player_b: int) -> None:
    """The comparison header, radar, bar chart and detail table for two players.

    The single implementation behind both the Player scouting page's "Player comparison"
    (picked from a dropdown of similar players) and the shortlist's "Compare selected
    players" (picked from a multiselect) - there is only one comparison renderer in the app.
    Raises ``ValueError`` for two players of different position groups and
    ``PlayerNotFoundError`` for an id outside the modelled pool; callers decide how to show that.
    """
    o = logic.overview(data, player_a)
    name_b = data.players.loc[player_b, "player"]
    cmp = logic.comparison_table(data, player_a, player_b)  # raises ValueError / PlayerNotFoundError
    group = o["position_group"]

    st.markdown(
        components.comparison_header_html(
            o["name"], o["club"], o["league"], data.photos.get(player_a),
            name_b, data.players.loc[player_b, "squads"], data.players.loc[player_b, "leagues"],
            data.photos.get(int(player_b)),
        ),
        unsafe_allow_html=True,
    )

    radar_a = logic.radar_scores(data, player_a)
    if len(radar_a) >= 3:
        radar_b = dict(logic.radar_scores(data, player_b))
        cats = [c for c, _ in radar_a]
        st.markdown("**Statistical profile**")
        st.caption(
            "Each axis is that category's model features (the same categories as Player statistics above), "
            "averaged to one percentile within the position group. Presentation only — not used for similarity or clustering."
        )
        st.plotly_chart(
            charts.radar_chart(cats, [v for _, v in radar_a], o["name"], [radar_b[c] for c in cats], name_b),
            width="stretch",
        )

    st.caption(
        f"Both are {group}s, so the same {len(cmp)} statistics apply. Bars show standard deviations from the "
        f"{group} average (0 = average), which puts statistics with different units on one scale; hover for the real values."
    )
    st.plotly_chart(charts.comparison_chart(cmp, o["name"], name_b, group), width="stretch")
    gaps = logic.biggest_differences(cmp)
    st.caption("Largest differences: " + "; ".join(
        f"{r['name']} ({r['text_a']} vs {r['text_b']})" for _, r in gaps.iterrows()))
    view = cmp.rename(columns={"category": "Category", "name": "Statistic", "text_a": o["name"], "text_b": name_b})
    st.dataframe(view[["Category", "Statistic", o["name"], name_b]], hide_index=True, width="stretch")


def player_comparison(data: DashboardData, o: dict, table) -> None:
    st.subheader("Player comparison")
    if table is None or table.empty:
        st.caption("Select a player with similar-player results to compare.")
        return
    labels = {
        int(r.player_id): f"{r.Rank}. {r.Player} — {r.Club} ({r.League})" for r in table.itertuples()
    }
    if st.session_state.get("compare_id") not in labels:
        st.session_state.pop("compare_id", None)
    other = st.selectbox(f"Compare {o['name']} with", options=list(labels), format_func=labels.get, key="compare_id")
    try:
        _comparison_block(data, o["player_id"], other)
    except (ValueError, PlayerNotFoundError) as exc:
        st.warning(str(exc))


# ------------------------------------------------------------ playing style
def playing_style(data: DashboardData, o: dict) -> None:
    st.subheader("Playing style")
    info = logic.cluster_info(data, o["position_group"], o["cluster_id"])
    categories = logic.style_category_highlights(info)
    st.markdown(
        components.style_identity_html(info.key, info.n_players, info.share, info.group, categories),
        unsafe_allow_html=True,
    )
    st.caption(logic.cluster_sentence(info))

    traits = logic.style_traits(data, o["player_id"], info)
    left, right = st.columns(2)
    with left:
        st.markdown("**Higher than position average**")
        st.markdown(components.trait_percentile_rows_html(traits["high"]), unsafe_allow_html=True)
    with right:
        st.markdown("**Lower than position average**")
        st.markdown(components.trait_percentile_rows_html(traits["low"]), unsafe_allow_html=True)
    st.caption("Numbers above are this player's own percentile within the position group.")

    radar = logic.radar_scores(data, o["player_id"])
    if radar:
        st.markdown("**Statistical profile**")
        st.markdown(components.style_profile_bars_html(radar), unsafe_allow_html=True)
        st.caption(
            "This player's percentile within the position group, by statistical category (the same categories "
            "used in the comparison radar above)."
        )

    st.caption(
        "The high/lower traits come from the cluster's average standardised statistics (K-Means within the "
        "position group), not from this player alone. **Statistical profile only** - not a tactical role, and "
        "not a claim about how good the player is."
    )


# ------------------------------------------------------------- PCA map
def cluster_map(data: DashboardData, o: dict, table) -> None:
    st.subheader("Player style map")
    st.caption(
        "Players closer together on this map have more similar statistical profiles within their position "
        "group. Colors represent K-Means clusters."
    )
    group = o["position_group"]
    show_similar = st.checkbox("Highlight the similar players", value=True, key="show_similar")
    similar_ids = table["player_id"].tolist() if (show_similar and table is not None and not table.empty) else []
    st.markdown(components.map_legend_html(), unsafe_allow_html=True)
    st.plotly_chart(
        charts.pca_chart(logic.pca_frame(data, group), data.pca_variance[group], group,
                         selected_id=o["player_id"], similar_ids=similar_ids),
        width="stretch",
    )
    with st.expander("How to read this map"):
        st.markdown(
            "- **PCA** reduces the model's many standardised statistics to the two dimensions that capture the "
            f"most variation ({data.pca_variance[group][:2].sum():.0%} of it here), so the whole position group "
            "can be shown on one map.\n"
            "- Nearby points generally have more similar statistical profiles.\n"
            "- **Distance on this map is an approximation.** It is not the cosine similarity score used elsewhere "
            "in this dashboard - two points drawn close together are not guaranteed to be top similarity matches, "
            "and vice versa.\n"
            "- Colors represent the K-Means clusters, the same ones used throughout the dashboard."
        )


# ---------------------------------------------------------- cluster explorer
def _trait_columns(info: logic.ClusterInfo) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown("**Higher than position average**")
        st.markdown("\n".join(f"- {t}" for t in logic.trait_lines(info.high)) or "- none above +0.5")
    with right:
        st.markdown("**Lower than position average**")
        st.markdown("\n".join(f"- {t}" for t in logic.trait_lines(info.low)) or "- none below -0.5")


def cluster_explorer_page(data: DashboardData) -> None:
    page_header("Cluster explorer")
    st.markdown("Explore statistical player profiles by position.")
    st.caption(
        "K-Means clusters group players with similar statistical profiles, separately for each position group. "
        "Cluster numbers only mean something within their own group (0 = largest)."
    )
    group = st.radio("Position group", POSITION_GROUPS, index=POSITION_GROUPS.index("FWD"), horizontal=True,
                     key="explorer_group")
    infos = logic.cluster_overview(data, group)

    c1, c2, c3 = st.columns(3)
    c1.metric("Players", f"{logic.group_size(data, group):,}", border=True)
    c2.metric("Clusters", len(infos), border=True)
    c3.metric("Variance shown on the map", f"{data.pca_variance[group][:2].sum():.0%}", border=True,
              help="Share of the statistical variation captured by the two map axes.")

    st.subheader("Cluster sizes")
    st.plotly_chart(
        charts.cluster_size_chart([i.key for i in infos], [i.n_players for i in infos], [i.cluster_id for i in infos]),
        width="stretch",
    )

    st.subheader("Cluster profiles")
    z, raw = logic.profile_matrix(data, group)
    st.plotly_chart(charts.profile_heatmap(z, raw, group), width="stretch")
    st.caption(
        "Each cell is a **standard deviation from the position average** (a z-score, not a percentile): blue = "
        "above average, red = below. Hover a cell for the exact value and the cluster's average real value."
    )

    st.subheader("Explore a cluster")
    labels = {i.key: f"{i.key} — {i.n_players:,} players ({i.share:.0%})" for i in infos}
    focus_key = st.selectbox("Cluster", options=list(labels), format_func=labels.get, key="explorer_cluster")
    info = next(i for i in infos if i.key == focus_key)
    with st.container(border=True):
        st.markdown(
            components.style_identity_html(info.key, info.n_players, info.share, info.group,
                                            logic.style_category_highlights(info)),
            unsafe_allow_html=True,
        )
        st.caption(logic.cluster_sentence(info))
        _trait_columns(info)

        st.markdown("**Players in this cluster**")
        f1, f2 = st.columns(2)
        league = f1.selectbox("League", [logic.ALL_LEAGUES, *data.leagues], key="explorer_league")
        min_minutes = f2.slider("Minimum minutes", 0, 3000, 900, step=50, key="explorer_minutes")
        members = logic.cluster_members(data, group, info.cluster_id, league, min_minutes)
        st.caption(f"{len(members)} shown")
        if members.empty:
            st.caption("No players match the filters.")
        else:
            st.dataframe(members, hide_index=True, width="stretch",
                         column_config={"Minutes": st.column_config.NumberColumn(format="%d")})

    st.subheader("Player style map")
    st.caption(
        f"Every {group} in the dataset, projected onto the first two principal components of the standardised "
        f"statistics ({data.pca_variance[group][:2].sum():.0%} of the variance); the selected cluster above is "
        "highlighted. Nearby players generally have more similar statistical profiles."
    )
    st.markdown(components.map_legend_html(), unsafe_allow_html=True)
    st.plotly_chart(
        charts.pca_chart(logic.pca_frame(data, group), data.pca_variance[group], group, focus_cluster=info.cluster_id),
        width="stretch",
    )
    with st.expander("How to read this map"):
        st.markdown(
            "- **PCA** reduces the model's many standardised statistics to two dimensions so the whole position "
            "group can be shown on one map.\n"
            "- Nearby points generally have more similar statistical profiles.\n"
            "- **Distance on this map is an approximation** - it is not the cosine similarity score used on the "
            "Player scouting page.\n"
            "- Saved PCA coordinates from the pipeline; nothing is recomputed. Hover a point for name, club, "
            "league and cluster."
        )
    st.divider()
    limitations()


# ---------------------------------------------------------- player database
def _database_filters(data: DashboardData) -> tuple[str, str, str, int, str]:
    c1, c2 = st.columns(2)
    league = c1.selectbox("League", [logic.ALL_LEAGUES, *data.leagues], key="db_league")
    position = c2.selectbox("Position", [logic.ALL_POSITIONS, *POSITION_GROUPS], key="db_position")

    c3, c4 = st.columns(2)
    cluster_options = [logic.DATABASE_ALL_CLUSTERS, *logic.database_clusters(data, position)]
    if st.session_state.get("db_cluster") not in cluster_options:
        st.session_state.pop("db_cluster", None)
    cluster = c3.selectbox("Cluster", cluster_options, key="db_cluster")
    min_minutes = c4.slider("Minimum minutes", 0, 3000, 900, step=50, key="db_min_minutes")

    search = st.text_input("Search", key="db_search", placeholder="Search player…")
    return league, position, cluster, min_minutes, search


def _database_row_actions(key_prefix: str, row: dict) -> None:
    pid = int(row["player_id"])
    b1, b2 = st.columns(2)
    b1.button("View Profile", key=f"{key_prefix}_view_{pid}", width="stretch", on_click=_goto_profile, args=(pid,))
    b2.button(_shortlist_label(pid), key=f"{key_prefix}_short_{pid}", width="stretch",
              on_click=_toggle_shortlist, args=(pid,))


def _database_cards(data: DashboardData, table, limit: int = 60) -> None:
    """The same filtered rows as the table, as browsing cards - for narrow screens or a quick scan.

    Capped (unlike the table) because each card is a handful of real Streamlit widgets; showing
    every row this way for an unfiltered database of ~2,000 players would be impractically slow.
    """
    shown = table.head(limit)
    if len(table) > limit:
        st.caption(f"Showing the first {limit} of {len(table):,} — narrow the filters to see fewer.")
    records = shown.to_dict("records")
    for start in range(0, len(records), 3):
        cols = st.columns(3)
        for col, row in zip(cols, records[start:start + 3]):
            with col, st.container(border=True):
                st.markdown(components.database_card_html(row, data.photos.get(int(row["player_id"]))),
                            unsafe_allow_html=True)
                _database_row_actions("dbcard", row)


def _database_table(data: DashboardData, table) -> None:
    """The full filtered table; selecting a row reveals its View Profile / Shortlist actions."""
    event = st.dataframe(
        table, hide_index=True, width="stretch", key="db_table",
        column_order=logic.DATABASE_COLUMNS[1:],
        column_config={"Minutes": st.column_config.NumberColumn(format="%d")},
        on_select="rerun", selection_mode="single-row",
    )
    rows = event.selection.rows if event is not None and event.selection else []
    if rows:
        row = table.iloc[rows[0]].to_dict()
        with st.container(border=True):
            st.markdown(f"**{row['Player']}** — {row['Club']} ({row['League']}) · {row['Position']} · {row['Cluster']}")
            _database_row_actions("dbsel", row)


def player_database_page(data: DashboardData) -> None:
    page_header("Player database")
    st.markdown("Browse players across Europe's top five leagues.")
    st.markdown(
        components.landing_stats_html(len(data.players), len(data.leagues), int(data.players["pos_group"].nunique())),
        unsafe_allow_html=True,
    )
    league, position, cluster, min_minutes, search = _database_filters(data)
    table = logic.database_table(data, league, position, cluster, min_minutes, search)
    if table.empty:
        st.info("No players match these filters. Try widening the league, position or cluster filter, "
                "or lowering the minimum minutes.")
        st.divider()
        limitations()
        return

    st.caption(f"{len(table):,} of {len(data.players):,} players")
    card_view = st.toggle("Compact cards (for narrow screens)", key="db_card_view")
    if card_view:
        _database_cards(data, table)
    else:
        _database_table(data, table)
    st.divider()
    limitations()


# -------------------------------------------------------------------- shortlist
def _shortlist_compare(data: DashboardData, ids: list[int]) -> None:
    st.subheader("Compare selected players")
    labels = {pid: f"{data.assignments.loc[pid, 'player_name']} ({data.assignments.loc[pid, 'position_group']})"
              for pid in ids}
    chosen = st.multiselect("Choose 2-3 players to compare", options=ids, format_func=labels.get,
                             key="shortlist_compare", max_selections=3)
    if len(chosen) < 2:
        st.caption("Choose at least two shortlisted players to compare.")
        return
    for a, b in itertools.combinations(chosen, 2):
        group_a = data.assignments.loc[a, "position_group"]
        group_b = data.assignments.loc[b, "position_group"]
        st.markdown(f"#### {labels[a]} vs {labels[b]}")
        if group_a != group_b:
            st.info(
                f"{data.assignments.loc[a, 'player_name']} ({group_a}) and {data.assignments.loc[b, 'player_name']} "
                f"({group_b}) are in different position groups. Detailed statistical comparison is restricted to "
                "players in the same group, because each group is modelled on its own statistics."
            )
            continue
        try:
            _comparison_block(data, a, b)
        except (ValueError, PlayerNotFoundError) as exc:
            st.warning(str(exc))
        st.divider()


def shortlist_page(data: DashboardData) -> None:
    page_header("My scouting shortlist")
    ids = _shortlist_ids()
    if not ids:
        st.markdown("No players shortlisted yet.")
        st.caption("Add players from Similar players or the Player database while scouting.")
        st.divider()
        limitations()
        return

    frame = logic.shortlist_frame(data, ids)
    st.caption(f"{len(frame)} player{'' if len(frame) == 1 else 's'}")
    for row in frame.to_dict("records"):
        pid = int(row["player_id"])
        with st.container(border=True):
            st.markdown(components.database_card_html(row, data.photos.get(pid)), unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            b1.button("View Profile", key=f"sl_view_{pid}", width="stretch", on_click=_goto_profile, args=(pid,))
            b2.button("Remove", key=f"sl_remove_{pid}", width="stretch", on_click=_remove_from_shortlist, args=(pid,))

    st.divider()
    _shortlist_compare(data, [int(pid) for pid in frame["player_id"]])
    st.divider()
    limitations()


# -------------------------------------------------------------------- page
def scouting_page(data: DashboardData) -> None:
    f = sidebar(data)
    if f.player_id is None:
        landing(data)
        return
    if st.session_state.get("_active_player") != f.player_id:
        # The active player changed (sidebar pick, a "View Profile" click, the landing search or a
        # shareable link) - drop any comparison chosen for the previous player rather than carry it
        # over, since it may no longer even be in the new player's position group.
        st.session_state.pop("compare_id", None)
        st.session_state["_active_player"] = f.player_id
    o = player_overview(data, f.player_id)
    st.query_params["player_id"] = str(o["player_id"])
    st.button(_shortlist_label(o["player_id"]), key=f"hero_short_{o['player_id']}",
              on_click=_toggle_shortlist, args=(o["player_id"],))
    st.write("")
    player_statistics(data, o)
    st.divider()
    table = similar_players(data, o, f)
    st.divider()
    player_comparison(data, o, table)
    st.divider()
    playing_style(data, o)
    st.divider()
    cluster_map(data, o, table)
    st.divider()
    limitations()
