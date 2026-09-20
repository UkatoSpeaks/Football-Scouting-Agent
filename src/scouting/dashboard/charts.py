"""Plotly figures for the dashboard. No Streamlit dependency; each function returns a Figure."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from scouting.palette import DIVERGING, GRID, INK, INK_2, MUTED, SERIES_1, SERIES_2, SURFACE, cluster_color

TRANSPARENT = "rgba(0,0,0,0)"
FONT = {"family": "system-ui, -apple-system, 'Segoe UI', sans-serif", "color": INK_2, "size": 12}


def _base(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height, paper_bgcolor=TRANSPARENT, plot_bgcolor=TRANSPARENT, font=FONT,
        margin={"l": 8, "r": 8, "t": 36, "b": 8},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "left", "x": 0, "font": {"color": INK}},
        hoverlabel={"font": {"family": FONT["family"]}},
    )
    return fig


def comparison_chart(cmp: pd.DataFrame, name_a: str, name_b: str, group: str) -> go.Figure:
    """Grouped horizontal bars of z-scores: one pair of bars per feature.

    Bars show how far each player is from the position-group average in standard
    deviations (0 = average), so features with different units share one axis.
    Hovering shows the real per-90 / percentage values.
    """
    labels = cmp["name"].tolist()
    fig = go.Figure()
    for col, text_col, player, color in [("z_a", "text_a", name_a, SERIES_1), ("z_b", "text_b", name_b, SERIES_2)]:
        fig.add_bar(
            y=labels, x=cmp[col], orientation="h", name=player, marker={"color": color},
            customdata=list(zip(cmp[text_col], cmp["category"])),
            hovertemplate="<b>%{y}</b><br>" + player + ": %{customdata[0]}"
            "<br>%{x:+.2f} standard deviations from the " + group + " average<extra></extra>",
        )
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont={"color": INK})
    fig.update_xaxes(title=f"Standard deviations from the {group} average (0 = average)",
                     gridcolor=GRID, zeroline=False)
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
    return _base(fig, height=max(320, 30 * len(labels) + 90))


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def radar_chart(categories: list[str], values_a: list[float], name_a: str, values_b: list[float], name_b: str) -> go.Figure:
    """Two-player radar of category percentiles (0-100), closed into a loop.

    ``values_a``/``values_b`` are ``logic.radar_scores`` output for each player, already
    limited to the categories both players' position group actually uses.
    """
    cats = [*categories, categories[0]]
    a = [*values_a, values_a[0]]
    b = [*values_b, values_b[0]]
    fig = go.Figure()
    for cat_values, name, color in [(a, name_a, SERIES_1), (b, name_b, SERIES_2)]:
        fig.add_scatterpolar(
            r=cat_values, theta=cats, name=name, line={"color": color, "width": 2},
            fill="toself", fillcolor=_rgba(color, 0.22),
            hovertemplate="<b>%{theta}</b><br>" + name + ": %{r:.0f}th percentile<extra></extra>",
        )
    fig.update_layout(
        polar={
            "bgcolor": TRANSPARENT,
            "radialaxis": {"range": [0, 100], "gridcolor": GRID, "tickfont": {"color": MUTED, "size": 10}},
            "angularaxis": {"gridcolor": GRID, "tickfont": {"color": INK, "size": 12}},
        },
    )
    fig = _base(fig, height=440)
    # The generic margin in _base is tuned for cartesian charts; a polar chart's angular labels
    # (e.g. a category sitting at the very bottom, 270 degrees) need real top/bottom clearance or
    # they clip against the figure edge.
    fig.update_layout(margin={"l": 60, "r": 60, "t": 50, "b": 50})
    return fig


def league_chart(counts: pd.DataFrame, highlight: str | None = None) -> go.Figure:
    """Horizontal bars: how many of the similar players play in each league."""
    colors = [SERIES_1 if lg == highlight else "#86b6ef" for lg in counts["League"]] if highlight else SERIES_1
    fig = go.Figure(go.Bar(
        y=counts["League"], x=counts["Players"], orientation="h", marker={"color": colors},
        text=counts["Players"], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x} players<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont={"color": INK}, automargin=True)
    fig.update_xaxes(visible=False, range=[0, max(1, counts["Players"].max()) * 1.2])
    fig.update_layout(showlegend=False, bargap=0.35)
    return _base(fig, height=40 * len(counts) + 40)


_HOVER = (
    "<b>%{customdata[0]}</b>%{customdata[4]}<br>%{customdata[1]} · %{customdata[2]}"
    "<br>Cluster %{customdata[3]}<extra></extra>"
)


def _custom(frame: pd.DataFrame, note: str = "") -> list:
    return list(zip(frame["player_name"], frame["club"], frame["league"], frame["cluster_key"], [note] * len(frame)))


def pca_chart(
    frame: pd.DataFrame,
    explained,
    group: str,
    selected_id: int | None = None,
    similar_ids=(),
    height: int = 560,
    focus_cluster: int | None = None,
) -> go.Figure:
    """Saved PCA coordinates of one position group, colored by cluster.

    Hover shows name, club, league and cluster. The selected player is a large
    star; similar players (optional) get an open ring; ``focus_cluster`` fades the other clusters. Nothing is recomputed:
    the coordinates come from the PCA fitted in the pipeline.
    """
    fig = go.Figure()
    for c in sorted(frame["cluster_id"].unique()):
        sub = frame[frame["cluster_id"] == c]
        fig.add_scatter(
            x=sub["pc1"], y=sub["pc2"], mode="markers", name=f"{group}-{c} (n={len(sub)})",
            marker={"size": 8, "color": cluster_color(int(c)), "line": {"width": 0.6, "color": SURFACE},
                    "opacity": 0.85 if focus_cluster in (None, c) else 0.12},
            customdata=_custom(sub), hovertemplate=_HOVER,
        )
    similar = frame[frame["player_id"].isin(list(similar_ids))]
    if not similar.empty:
        fig.add_scatter(
            x=similar["pc1"], y=similar["pc2"], mode="markers", name="Similar players",
            marker={"size": 15, "symbol": "circle-open", "color": INK, "line": {"width": 1.8, "color": INK}},
            customdata=_custom(similar, " (similar)"), hovertemplate=_HOVER,
        )
    if selected_id is not None:
        me = frame[frame["player_id"] == selected_id]
        if not me.empty:
            r = me.iloc[0]
            fig.add_scatter(
                x=[r["pc1"]], y=[r["pc2"]], mode="markers+text", name="Selected player",
                text=[r["player_name"]], textposition="top center", textfont={"color": INK, "size": 13},
                marker={"size": 20, "symbol": "star", "color": cluster_color(int(r["cluster_id"])),
                        "line": {"width": 2, "color": INK}},
                customdata=_custom(me, " (selected)"), hovertemplate=_HOVER,
            )
    fig.update_xaxes(title=f"PC1 ({explained[0]:.0%} of variance)", gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title=f"PC2 ({explained[1]:.0%} of variance)", gridcolor=GRID, zeroline=False)
    return _base(fig, height=height)


def cluster_size_chart(keys: list[str], sizes: list[int], cluster_ids: list[int]) -> go.Figure:
    """One bar per cluster, in the cluster's map color."""
    fig = go.Figure(go.Bar(
        x=keys, y=sizes, marker={"color": [cluster_color(c) for c in cluster_ids]},
        text=sizes, textposition="outside", cliponaxis=False,
        hovertemplate="%{x}: %{y} players<extra></extra>",
    ))
    fig.update_yaxes(visible=False, range=[0, max(sizes) * 1.18])
    fig.update_xaxes(tickfont={"color": INK})
    fig.update_layout(showlegend=False, bargap=0.4)
    return _base(fig, height=260)


def profile_heatmap(z: pd.DataFrame, raw: pd.DataFrame, group: str = "", limit: float = 2.0) -> go.Figure:
    """Cluster profiles: mean standardised statistic per cluster (blue above average, red below).

    Every cell is a **standard deviation from the position-group average** (a z-score, not a
    percentile): the colour scale is symmetric around 0 and clipped at +/-``limit`` so one
    extreme cluster does not wash out the others; cell text and hover show the exact z-score
    plus the cluster's average real value for that statistic.
    """
    hover = [[f"{feat}<br>{key}: {z.loc[feat, key]:+.2f} SD from the position average<br>mean value {raw.loc[feat, key]:.2f}"
              for key in z.columns] for feat in z.index]
    fig = go.Figure(go.Heatmap(
        z=z.to_numpy(), x=list(z.columns), y=list(z.index), zmin=-limit, zmax=limit, zmid=0,
        colorscale=[[0, DIVERGING["low"]], [0.5, DIVERGING["mid"]], [1, DIVERGING["high"]]],
        text=[[f"{round(v, 1) + 0.0:+.1f}" for v in row] for row in z.to_numpy()], texttemplate="%{text}",  # +0.0 not -0.0
        hovertext=hover, hoverinfo="text",
        colorbar={"title": "SD from<br>position average", "thickness": 12, "len": 0.6},
        xgap=2, ygap=2,
    ))
    fig.update_yaxes(title="Statistic", autorange="reversed", tickfont={"color": INK}, showgrid=False, automargin=True)
    fig.update_xaxes(title=f"{group} cluster" if group else "Cluster", side="top", tickfont={"color": INK}, showgrid=False)
    fig.update_layout(title={"text": f"{group} cluster profiles (standard deviations from the position average)" if group
                              else "Cluster profiles (standard deviations from the position average)",
                              "font": {"color": INK, "size": 13}, "x": 0, "xanchor": "left"})
    fig = _base(fig, height=max(400, 26 * len(z.index) + 130))
    fig.update_layout(margin={"t": 90})          # room for the chart title above the top-side x-axis labels
    return fig
