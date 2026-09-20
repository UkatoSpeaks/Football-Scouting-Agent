"""Dashboard logic with no Streamlit dependency (so it can be unit-tested).

Everything here reads the saved pipeline outputs through ``DashboardData``.
Similarity is delegated to ``SimilarityEngine.find_similar_players``; cluster
descriptions are derived from ``cluster_profiles.csv`` by ``describe_cluster``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from scouting.clustering import describe_cluster
from scouting.dashboard.data import DashboardData
from scouting.labels import (
    CATEGORY_ORDER, FEATURE_CATEGORIES, RADAR_CATEGORIES, RADAR_CATEGORY_ORDER, display_name, label, sentence_case,
)
from scouting.similarity import DEFAULT_MIN_CANDIDATE_MINUTES

ALL_LEAGUES = "All Leagues"
ALL_POSITIONS = "All"


def _clean(value):
    """NaN/None -> None so the UI can show a dash for missing metadata."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return value


# ------------------------------------------------------------------- picker
def picker_labels(data: DashboardData, position: str = ALL_POSITIONS) -> dict[int, str]:
    """player_id -> 'Name — Club (League)', sorted by name; optionally one position group."""
    p = data.players
    if position != ALL_POSITIONS:
        p = p[p["pos_group"] == position]
    p = p.sort_values(["player", "squad"], kind="stable")
    return {int(pid): f"{r.player} — {r.squads} ({r.leagues})" for pid, r in p.iterrows()}


# ----------------------------------------------------------------- overview
def overview(data: DashboardData, player_id: int) -> dict:
    """Identity facts for one player; optional metadata may be None."""
    r = data.players.loc[player_id]
    a = data.assignments.loc[player_id]
    return {
        "player_id": int(player_id),
        "name": r["player"],
        "club": r["squads"],
        "league": r["leagues"],
        "position": r["pos_raw"],
        "position_group": r["pos_group"],
        "minutes": int(r["minutes"]),
        "matches": int(r["matches"]) if _clean(r.get("matches")) is not None else None,
        "age": int(r["age"]) if _clean(r.get("age")) is not None else None,
        "nation": _clean(r.get("nation")),
        "season": r["season"],
        "cluster_id": int(a["cluster_id"]),
        "cluster_key": a["cluster_key"],
    }


# ------------------------------------------------------------------ clusters
@dataclass
class ClusterInfo:
    key: str
    group: str
    cluster_id: int
    n_players: int
    share: float                              # of the position group
    high: list[tuple[str, float]]             # (feature label, mean z), strongest first
    low: list[tuple[str, float]]


def cluster_info(data: DashboardData, group: str, cluster_id: int, threshold: float = 0.5, top: int = 5) -> ClusterInfo:
    prof = data.profiles[(data.profiles["position_group"] == group) & (data.profiles["cluster_id"] == cluster_id)]
    if prof.empty:
        raise KeyError(f"no profile for {group} cluster {cluster_id}")
    traits = describe_cluster(prof, threshold=threshold, top=top)
    group_size = int(data.profiles[data.profiles["position_group"] == group].drop_duplicates("cluster_id")["n_players"].sum())
    n = int(prof["n_players"].iloc[0])
    return ClusterInfo(f"{group}-{cluster_id}", group, int(cluster_id), n, n / group_size, traits["high"], traits["low"])


def cluster_sentence(info: ClusterInfo) -> str:
    """Plain-language summary built only from the cluster's high/low features."""
    parts = []
    if info.high:
        parts.append("Higher than the position average: " + ", ".join(n for n, _ in info.high) + ".")
    if info.low:
        parts.append("Lower than the position average: " + ", ".join(n for n, _ in info.low) + ".")
    if not parts:
        parts.append("No feature is far from the position average (all within 0.5 standard deviations).")
    return " ".join(parts)


def trait_lines(traits: list[tuple[str, float]]) -> list[str]:
    """['Key passes (+1.6)', ...] for bullet lists."""
    return [f"{sentence_case(n)} ({z:+.1f})" for n, z in traits]


def _feature_for_label(data: DashboardData, group: str) -> dict[str, str]:
    """Reverse map from a cluster trait's display label back to its raw feature key.

    Scoped to one position group's own feature list, where labels are unique (this is the
    same ``label()`` used by ``describe_cluster`` to build ``ClusterInfo.high``/``low``).
    """
    return {label(f): f for f in data.config.modeling.position_feature_lists[group]}


def style_traits(data: DashboardData, player_id: int, info: ClusterInfo) -> dict[str, list[tuple[str, float]]]:
    """The player's own percentile on each feature that makes their cluster distinctive.

    ``info.high``/``info.low`` (the cluster's mean z-score) say which features define the
    *cluster*; this looks up the *player's own* percentile on those same features - the same
    numbers already shown in Player statistics - so "Playing style" says something about this
    player, not only the average cluster member. Presentation only: it does not change which
    features count as high/low, the clustering, or any underlying data.
    """
    by_label = _feature_for_label(data, info.group)
    pct = data.percentiles[info.group].loc[player_id]

    def rows(traits: list[tuple[str, float]]) -> list[tuple[str, float]]:
        return [(sentence_case(name), float(pct[by_label[name]])) for name, _ in traits if name in by_label]

    return {"high": rows(info.high), "low": rows(info.low)}


def style_category_highlights(info: ClusterInfo) -> list[str]:
    """Broad statistical categories the cluster's *high* traits fall into, strongest first.

    Reuses the comparison radar's category regrouping (``labels.RADAR_CATEGORIES``) - the same
    features, named at a coarser grain - so this never invents a new reading of the cluster; it
    only restates, at a glance, which of its already-identified high traits sit together.
    """
    by_label = {label(f): f for f in RADAR_CATEGORIES}
    seen: list[str] = []
    for name, _ in info.high:
        f = by_label.get(name)
        category = RADAR_CATEGORIES.get(f) if f else None
        if category and category not in seen:
            seen.append(category)
    return seen


# ----------------------------------------------------------- similar players
SIMILAR_COLUMNS = ["Rank", "Player", "Club", "League", "Position", "Minutes", "Cosine similarity", "Cluster"]


def similar_table(
    data: DashboardData,
    player_id: int,
    n: int = 10,
    min_minutes: float | None = DEFAULT_MIN_CANDIDATE_MINUTES,
    league: str = ALL_LEAGUES,
) -> pd.DataFrame:
    """Closest statistical matches (cosine similarity) within the player's position group.

    ``league`` restricts the candidates; the target may come from any league.
    Raises PlayerNotFoundError for a player_id outside the modeled pool.
    """
    leagues = None if league in (None, ALL_LEAGUES) else league
    found = data.engine.find_similar_players(player_id, n=n, min_candidate_minutes=min_minutes, leagues=leagues)
    out = pd.DataFrame({
        "Rank": found["rank"],
        "Player": found["player_name"],
        "Club": found["club"],
        "League": found["league"],
        "Position": found["position"],
        "Minutes": found["minutes"].astype(int),
        "Cosine similarity": found["similarity"],
        "Cluster": data.assignments.loc[found["player_id"], "cluster_key"].to_numpy(),
    })
    out.insert(0, "player_id", found["player_id"].to_numpy())
    return out.reset_index(drop=True)


# --------------------------------------------------------------- statistics
def format_value(feature: str, value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if feature.endswith(("_pct", "_share")):
        return f"{value:.1f}%"
    if feature == "npxg_per_shot":
        return f"{value:.3f}"
    if feature == "pass_prog_dist_p90":
        return f"{value:.0f}"
    return f"{value:.2f}"


def ordinal(n: float) -> str:
    n = int(round(n))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def player_stats(data: DashboardData, player_id: int) -> list[tuple[str, pd.DataFrame]]:
    """The player's position-specific model features, grouped by category.

    Only the features used for the player's own position group are returned, so
    a goalkeeper never shows shooting stats and a forward never shows save %.
    Each row carries the real (un-standardised) value and its percentile among
    the position group's modelled players.
    """
    group = data.players.loc[player_id, "pos_group"]
    feats = data.config.modeling.position_feature_lists[group]
    values = data.players.loc[player_id, feats]
    pct = data.percentiles[group].loc[player_id]
    rows = pd.DataFrame({
        "feature": feats,
        "name": [display_name(f) for f in feats],
        "category": [FEATURE_CATEGORIES[f] for f in feats],
        "value": values.to_numpy(dtype="float64"),
        "percentile": pct.to_numpy(dtype="float64"),
    })
    rows["text"] = [format_value(f, v) for f, v in zip(rows["feature"], rows["value"])]
    return [(c, rows[rows["category"] == c].reset_index(drop=True)) for c in CATEGORY_ORDER if (rows["category"] == c).any()]


def season_totals(data: DashboardData, player_id: int) -> dict[str, int]:
    r = data.players.loc[player_id]
    return {k: int(r[k]) for k in ("matches", "starts", "minutes", "goals", "assists") if _clean(r.get(k)) is not None}


def group_size(data: DashboardData, group: str) -> int:
    return len(data.engine.matrices[group])


def radar_scores(data: DashboardData, player_id: int) -> list[tuple[str, float]]:
    """Percentile-based category scores for the comparison radar chart (presentation only).

    Each of the player's position-group features is regrouped into a broader category
    (``scouting.labels.RADAR_CATEGORIES``) and the category's score is the mean of the
    player's percentiles (0-100, the same ones shown in "Player statistics") across the
    features assigned to it. Nothing here changes the similarity features, the scaler or
    the clustering; a category only appears if the group's own feature list uses it.
    """
    group = data.players.loc[player_id, "pos_group"]
    feats = data.config.modeling.position_feature_lists[group]
    pct = data.percentiles[group].loc[player_id]
    by_category: dict[str, list[float]] = {}
    for f in feats:
        category = RADAR_CATEGORIES.get(f)
        if category is not None:
            by_category.setdefault(category, []).append(float(pct[f]))
    return [(c, float(np.mean(by_category[c]))) for c in RADAR_CATEGORY_ORDER if c in by_category]


# --------------------------------------------------------------- comparison
def comparison_table(data: DashboardData, player_a: int, player_b: int) -> pd.DataFrame:
    """Side-by-side model features for two players of the same position group.

    Built on ``SimilarityEngine.explain_match`` (z-scores and raw values), so it
    raises ValueError for players of different groups and PlayerNotFoundError
    for ids outside the modelled pool. Rows follow the dashboard's category order.
    z_* are standard deviations from the position average (after log1p on the
    skewed features), which makes different statistics comparable on one axis.
    """
    group = data.engine.group_of(player_a)
    feats = data.config.modeling.position_feature_lists[group]
    ex = data.engine.explain_match(player_a, player_b, top=len(feats)).set_index("feature").loc[feats]
    out = pd.DataFrame({
        "feature": feats,
        "name": [display_name(f) for f in feats],
        "category": [FEATURE_CATEGORIES[f] for f in feats],
        "value_a": ex["raw_query"].to_numpy(),
        "value_b": ex["raw_match"].to_numpy(),
        "z_a": ex["z_query"].to_numpy(),
        "z_b": ex["z_match"].to_numpy(),
    })
    out["text_a"] = [format_value(f, v) for f, v in zip(out["feature"], out["value_a"])]
    out["text_b"] = [format_value(f, v) for f, v in zip(out["feature"], out["value_b"])]
    out["z_diff"] = out["z_b"] - out["z_a"]
    order = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    out = out.assign(_o=out["category"].map(order)).sort_values("_o", kind="stable").drop(columns="_o")
    return out.reset_index(drop=True)


def biggest_differences(cmp: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    return cmp.reindex(cmp["z_diff"].abs().sort_values(ascending=False).index).head(n)


# ------------------------------------------------------------ league mix / PCA
def league_counts(table: pd.DataFrame, leagues: list[str]) -> pd.DataFrame:
    """Number of similar players per league (all leagues listed, zeros included)."""
    counts = table["League"].value_counts() if not table.empty else pd.Series(dtype=int)
    return pd.DataFrame({"League": leagues, "Players": [int(counts.get(lg, 0)) for lg in leagues]})


def main_league(data: DashboardData, player_id: int) -> str:
    """League with the most minutes (players who moved mid-season list several in ``leagues``)."""
    return data.engine.players.loc[player_id, "league"]


def pca_frame(data: DashboardData, group: str) -> pd.DataFrame:
    """Saved PCA coordinates of one position group plus club/league/minutes for hover text."""
    coords = data.pca_coords[group]
    info = data.assignments.loc[coords["player_id"], ["club", "league", "minutes", "cluster_key"]]
    frame = coords.reset_index(drop=True).copy()
    for col in info.columns:
        frame[col] = info[col].to_numpy()
    return frame


# --------------------------------------------------------- cluster explorer
def cluster_overview(data: DashboardData, group: str) -> list[ClusterInfo]:
    """Every cluster of a position group, largest first (ids already follow that order)."""
    ids = sorted(data.profiles.loc[data.profiles["position_group"] == group, "cluster_id"].unique())
    return [cluster_info(data, group, int(c)) for c in ids]


def profile_matrix(data: DashboardData, group: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(mean z, mean raw value) with one row per feature and one column per cluster key.

    Rows follow the dashboard's category order so related statistics sit together.
    """
    prof = data.profiles[data.profiles["position_group"] == group].copy()
    prof["key"] = [f"{group}-{c}" for c in prof["cluster_id"]]
    feats = data.config.modeling.position_feature_lists[group]
    order = sorted(feats, key=lambda f: (CATEGORY_ORDER.index(FEATURE_CATEGORIES[f]), feats.index(f)))
    z = prof.pivot(index="feature", columns="key", values="mean_z").loc[order]
    raw = prof.pivot(index="feature", columns="key", values="mean_raw").loc[order]
    z.index = raw.index = [display_name(f) for f in order]
    return z, raw


def cluster_members(
    data: DashboardData, group: str, cluster_id: int, league: str = ALL_LEAGUES, min_minutes: int = 0
) -> pd.DataFrame:
    """Players of one cluster (from cluster_assignments.csv), most minutes first."""
    a = data.assignments
    m = a[(a["position_group"] == group) & (a["cluster_id"] == cluster_id) & (a["minutes"] >= min_minutes)]
    if league not in (None, ALL_LEAGUES):
        m = m[m["league"] == league]
    m = m.sort_values(["minutes", "player_name"], ascending=[False, True], kind="stable")
    return m[["player_name", "club", "league", "position", "minutes"]].rename(columns={
        "player_name": "Player", "club": "Club", "league": "League", "position": "Position", "minutes": "Minutes",
    }).reset_index(drop=True)


# ------------------------------------------------------------- profile hero
def cluster_headline(info: ClusterInfo, n: int = 3) -> str:
    """One line for the profile header, built only from the cluster's strongest traits."""
    if info.high:
        return "Higher than the position average: " + ", ".join(name for name, _ in info.high[:n])
    if info.low:
        return "Lower than the position average: " + ", ".join(name for name, _ in info.low[:n])
    return "Close to the position average on every statistic"


# -------------------------------------------------------------- player database
DATABASE_COLUMNS = ["player_id", "Player", "Club", "League", "Position", "Minutes", "Cluster"]
DATABASE_ALL_CLUSTERS = "All Clusters"


def _database_columns(a: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "player_id": a.index.to_numpy(),
        "Player": a["player_name"].to_numpy(),
        "Club": a["club"].to_numpy(),
        "League": a["league"].to_numpy(),
        "Position": a["position_group"].to_numpy(),
        "Minutes": a["minutes"].to_numpy(),
        "Cluster": a["cluster_key"].to_numpy(),
    })


def database_clusters(data: DashboardData, position: str = ALL_POSITIONS) -> list[str]:
    """Cluster keys available under the current position filter, grouped and ordered (GK-0, GK-1, DEF-0, ...)."""
    a = data.assignments
    if position != ALL_POSITIONS:
        a = a[a["position_group"] == position]
    keys = a[["position_group", "cluster_id"]].drop_duplicates().sort_values(["position_group", "cluster_id"])
    return [f"{r.position_group}-{r.cluster_id}" for r in keys.itertuples()]


def database_table(
    data: DashboardData,
    league: str = ALL_LEAGUES,
    position: str = ALL_POSITIONS,
    cluster: str = DATABASE_ALL_CLUSTERS,
    min_minutes: int = 0,
    search: str = "",
) -> pd.DataFrame:
    """The modelled player pool for browsing: Player/Club/League/Position/Minutes/Cluster.

    Filtered by league, position group, cluster and a minimum-minutes threshold, and matched
    by a case-insensitive substring search on the player's name. Sorted by minutes, most first.
    Reads only the already-saved ``cluster_assignments.csv`` (via ``data.assignments``); it does
    not recompute anything.
    """
    a = data.assignments
    mask = a["minutes"] >= min_minutes
    if league != ALL_LEAGUES:
        mask &= a["league"] == league
    if position != ALL_POSITIONS:
        mask &= a["position_group"] == position
    if cluster != DATABASE_ALL_CLUSTERS:
        mask &= a["cluster_key"] == cluster
    if search:
        mask &= a["player_name"].str.contains(re.escape(search), case=False, na=False)
    out = a[mask].sort_values("minutes", ascending=False)
    return _database_columns(out).reset_index(drop=True)


# ------------------------------------------------------------------- shortlist
def shortlist_frame(data: DashboardData, player_ids: list[int]) -> pd.DataFrame:
    """Shortlisted players (Player/Club/League/Position/Minutes/Cluster), in the order added.

    Ids no longer in the modelled pool (e.g. after the data changes) are silently dropped;
    duplicates are collapsed to their first occurrence. This never reads or writes session
    state - the caller owns the shortlist itself.
    """
    seen: list[int] = []
    for pid in player_ids:
        if pid in data.assignments.index and pid not in seen:
            seen.append(pid)
    return _database_columns(data.assignments.loc[seen]).reset_index(drop=True) if seen else _database_columns(data.assignments.loc[[]])


def hero_chips(data: DashboardData, o: dict) -> list[tuple[str, str]]:
    """Key facts for the profile header; anything missing is left out (never shown as 'None')."""
    chips = [("Minutes", f"{o['minutes']:,}")]
    if o.get("matches") is not None:
        chips.append(("Matches", str(o["matches"])))
    if o.get("age") is not None:
        chips.append(("Age", str(o["age"])))
    if o.get("nation"):
        chips.append(("Nation", str(o["nation"])))
    if o["position_group"] != "GK":
        totals = season_totals(data, o["player_id"])
        for key, name in (("goals", "Goals"), ("assists", "Assists")):
            if key in totals:
                chips.append((name, str(totals[key])))
    return chips
