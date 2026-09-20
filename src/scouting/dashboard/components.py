"""HTML fragments for the dashboard (no Streamlit dependency, so they can be unit-tested).

Every piece of player-derived text is HTML-escaped here; callers pass raw values.
"""
from __future__ import annotations

import html

from scouting.dashboard.photos import avatar_html


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


def sidebar_label(text: str) -> str:
    return f'<div class="sb-label">{_e(text)}</div>'


def landing_stats_html(n_players: int, n_leagues: int, n_groups: int) -> str:
    items = [(f"{n_players:,}", "Players"), (f"{n_leagues}", "Leagues"), (f"{n_groups}", "Position groups")]
    body = "".join(f'<div class="sc-stat"><b>{_e(v)}</b><span>{_e(k)}</span></div>' for v, k in items)
    return f'<div class="sc-stats">{body}</div>'


def stat_card_html(name: str, value_text: str, percentile: float) -> str:
    """One statistic card: name, the real value (prominent) and a percentile bar (secondary)."""
    pct = max(0.0, min(100.0, float(percentile)))
    return (
        '<div class="sc-stat-card">'
        f'<div class="sc-stat-name">{_e(name)}</div>'
        f'<div class="sc-stat-value">{_e(value_text)}</div>'
        f'<div class="sc-bar"><div class="sc-bar-fill" style="width:{pct:.1f}%"></div></div>'
        f'<div class="sc-stat-pct">{_e(_ordinal(pct))} percentile</div>'
        "</div>"
    )


def _ordinal(n: float) -> str:
    n = int(round(n))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def similar_card_html(row: dict, photo_url: str | None = None) -> str:
    """A scouting card for one similar-player result: photo, identity, similarity and cluster.

    ``row`` has the same keys as one record of ``logic.similar_table`` (``Player``, ``Club``,
    ``League``, ``Position``, ``Minutes``, ``Cosine similarity``, ``Cluster``).
    """
    minutes_text = f"{int(row['Minutes']):,}"
    return (
        '<div class="sc-sim-card">'
        '<div class="sc-sim-top">'
        f'{avatar_html(row["Player"], photo_url, size=64)}'
        '<div class="sc-sim-score">'
        f'<b>{row["Cosine similarity"]:.3f}</b><span>similarity</span>'
        "</div></div>"
        f'<div class="sc-sim-name">{_e(row["Player"])}</div>'
        f'<div class="sc-sim-club">{_e(row["Club"])}</div>'
        f'<div class="sc-sim-meta">{_e(row["League"])} · {_e(row["Position"])}</div>'
        f'<div class="sc-sim-meta">{_e(minutes_text)} minutes</div>'
        f'<span class="sc-badge-light">{_e(row["Cluster"])}</span>'
        "</div>"
    )


def database_card_html(row: dict, photo_url: str | None = None) -> str:
    """A compact browsing card for the Player Database / Shortlist (no similarity score).

    Reuses the similar-player card's CSS so browsing, discovery and comparison all share one
    visual language. ``row`` needs ``Player``, ``Club``, ``League``, ``Position``, ``Minutes``
    and ``Cluster`` (the same columns as ``logic.database_table``/``logic.shortlist_frame``).
    """
    minutes_text = f"{int(row['Minutes']):,}"
    return (
        '<div class="sc-sim-card">'
        f'{avatar_html(row["Player"], photo_url, size=64)}'
        f'<div class="sc-sim-name">{_e(row["Player"])}</div>'
        f'<div class="sc-sim-club">{_e(row["Club"])}</div>'
        f'<div class="sc-sim-meta">{_e(row["League"])} · {_e(row["Position"])}</div>'
        f'<div class="sc-sim-meta">{_e(minutes_text)} minutes</div>'
        f'<span class="sc-badge-light">{_e(row["Cluster"])}</span>'
        "</div>"
    )


def comparison_header_html(
    name_a: str, club_a: str, league_a: str, photo_a: str | None,
    name_b: str, club_b: str, league_b: str, photo_b: str | None,
) -> str:
    """The 'A vs B' header above a player comparison: both photos, identities and a divider."""
    def side(name, club, league, photo):
        return (
            '<div class="sc-vs-player">'
            f'{avatar_html(name, photo, size=88)}'
            f'<div class="sc-vs-name">{_e(name)}</div>'
            f'<div class="sc-vs-meta">{_e(club)}</div>'
            f'<div class="sc-vs-meta">{_e(league)}</div>'
            "</div>"
        )
    return (
        '<div class="sc-vs">'
        f'{side(name_a, club_a, league_a, photo_a)}'
        '<div class="sc-vs-mid">VS</div>'
        f'{side(name_b, club_b, league_b, photo_b)}'
        "</div>"
    )


def style_identity_html(cluster_key: str, n_players: int, share: float, group: str, categories: list[str]) -> str:
    """Playing-style identity block: cluster badge, size, and its high-trait categories.

    ``categories`` are broad statistical groupings (``logic.style_category_highlights``), not
    an invented reading of the cluster - if there are none, this says so plainly rather than
    guessing.
    """
    cats_line = (f"Higher than position average: {_e(', '.join(categories))}" if categories
                 else "Close to the position average on every statistical category")
    return (
        '<div class="sc-style-id">'
        f'<span class="sc-badge-light sc-badge-lg">{_e(cluster_key)}</span>'
        f'<div class="sc-style-size">{n_players:,} players in this cluster ({share:.0%} of {_e(group)}s)</div>'
        f'<div class="sc-style-cats">{cats_line}</div>'
        "</div>"
    )


def trait_percentile_rows_html(rows: list[tuple[str, float]]) -> str:
    """A divider-ruled list of (statistic name, this player's percentile on it)."""
    if not rows:
        return '<div class="sc-trait-list sc-trait-empty">None within 0.5 SD of the position average</div>'
    body = "".join(
        f'<div class="sc-trait-row"><span class="sc-trait-name">{_e(name)}</span>'
        f'<span class="sc-trait-pct">{_e(_ordinal(pct))}</span></div>'
        for name, pct in rows
    )
    return f'<div class="sc-trait-list">{body}</div>'


def style_profile_bars_html(scores: list[tuple[str, float]]) -> str:
    """Compact percentile bars, one row per statistical category (reuses the stat-card bar style)."""
    rows = "".join(
        '<div class="sc-profile-row">'
        f'<span class="sc-profile-label">{_e(name)}</span>'
        f'<div class="sc-bar sc-profile-bar"><div class="sc-bar-fill" '
        f'style="width:{max(0.0, min(100.0, val)):.1f}%"></div></div>'
        f'<span class="sc-profile-pct">{_e(_ordinal(val))}</span>'
        "</div>"
        for name, val in scores
    )
    return f'<div class="sc-profile-bars">{rows}</div>'


def map_legend_html() -> str:
    """A small, explicit legend for the PCA style map's marker shapes (colors are per-cluster)."""
    items = [("sc-legend-dot", "Cluster members"), ("sc-legend-star", "Selected player"), ("sc-legend-ring", "Similar players")]
    body = "".join(f'<span class="sc-legend-item"><span class="{cls}"></span>{_e(text)}</span>' for cls, text in items)
    return f'<div class="sc-legend">{body}</div>'


def profile_hero_html(o: dict, headline: str, chips: list[tuple[str, str]], photo_url: str | None = None) -> str:
    """The player profile header: photo/avatar, identity, cluster badge and key facts.

    ``o`` is ``logic.overview``; optional values that are None are simply left out.
    """
    position = f"{o['position_group']} · {o['position']}" if o.get("position") else o["position_group"]
    style = f'<span class="sc-badge">{_e(o["cluster_key"])}</span>'
    if headline:
        style += f'<span class="sc-headline">{_e(headline)}</span>'
    chips_html = "".join(f'<div class="sc-chip"><b>{_e(v)}</b><span>{_e(k)}</span></div>' for k, v in chips)
    return (
        '<div class="sc-hero">'
        f'{avatar_html(o["name"], photo_url, size=112)}'
        '<div class="sc-hero-main">'
        f'<div class="sc-name" role="heading" aria-level="1">{_e(o["name"])}</div>'
        f'<div class="sc-club">{_e(o["club"])}</div>'
        f'<div class="sc-sub">{_e(o["league"])} · {_e(position)}</div>'
        f'<div class="sc-style">{style}</div>'
        "</div>"
        f'<div class="sc-chips">{chips_html}</div>'
        "</div>"
    )
