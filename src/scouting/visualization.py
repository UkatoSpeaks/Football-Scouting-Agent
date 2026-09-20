"""Matplotlib figures for the clustering step (elbow plots and PCA scatter plots).

Colors follow the dataviz reference palette (light surface). Cluster colors use
the slot order blue, aqua, yellow, green, violet, red, which was validated with
``--pairs all`` (scatter rule) for every prefix of size 2-6. More than six
clusters are drawn as small multiples (one highlighted cluster per panel)
because no larger set passes the all-pairs floors. Ink colors are text tokens;
clusters are also identified by direct labels at their centres, never by color
alone.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scouting.palette import CLUSTER_COLORS, GRID, INK, INK_2, NEUTRAL, SERIES_1, SURFACE  # noqa: E402

MAX_SINGLE_PANEL_CLUSTERS = len(CLUSTER_COLORS)

# Well-known players marked on the PCA plots when they are in the group.
LABEL_PLAYERS = [
    "Bukayo Saka", "Erling Haaland", "Declan Rice", "Virgil van Dijk", "Alisson",
    "Mohamed Salah", "Lamine Yamal", "Jude Bellingham", "Trent Alexander-Arnold", "Ederson",
]


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_elbow(table: pd.DataFrame, group: str, path: Path, chosen_k: int | None = None) -> None:
    """Inertia and silhouette against K for one position group."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150, facecolor=SURFACE)
    panels = [
        ("inertia", f"{group}: inertia (lower = tighter clusters)"),
        ("silhouette", f"{group}: silhouette (higher = better separated)"),
    ]
    for ax, (col, title) in zip(axes, panels):
        _style(ax)
        ax.plot(table["k"], table[col], color=SERIES_1, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("number of clusters K", color=INK_2, fontsize=9)
        ax.set_xticks(table["k"])
        if chosen_k is not None and chosen_k in set(table["k"]):
            ax.axvline(chosen_k, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
            y = float(table.loc[table["k"] == chosen_k, col].iloc[0])
            ax.annotate(f"chosen K = {chosen_k}", (chosen_k, y), xytext=(8, 10), textcoords="offset points",
                        color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _annotate(ax, coords: pd.DataFrame, names: list[str], color_of) -> None:
    """Mark and name players. Call after the figure layout is final (needs pixel positions).

    Each label goes to the candidate offset that is farthest from labels already
    placed, and stays inside the axes.
    """
    scale = ax.figure.dpi / 72
    box = ax.get_window_extent()
    candidates = [
        offset
        for r in (24, 42, 60)
        for offset in [(r, r), (-r, r), (r, -r), (-r, -r), (1.5 * r, 0), (-1.5 * r, 0), (0, 1.3 * r), (0, -1.3 * r)]
    ]
    placed: list[np.ndarray] = []
    for name in names:
        row = coords[coords["player_name"] == name]
        if len(row) != 1:
            continue
        x, y = float(row["pc1"].iloc[0]), float(row["pc2"].iloc[0])
        ax.scatter([x], [y], s=70, color=color_of(int(row["cluster_id"].iloc[0])), edgecolor=INK,
                   linewidth=1.5, zorder=5)
        p = np.array(ax.transData.transform((x, y)))

        text_width = 5.0 * len(name) * scale  # rough pixel width of the label at 8.5pt

        def room(o):
            q = p + np.array(o) * scale
            left = q[0] - (text_width if o[0] < 0 else 0)
            right = q[0] + (text_width if o[0] > 0 else 0)
            return box.x0 + 10 <= left and right <= box.x1 - 10 and box.y0 + 15 <= q[1] <= box.y1 - 15

        def clearance(o):
            q = p + np.array(o) * scale
            return min([np.linalg.norm(q - other) for other in placed] + [1e9])

        options = [o for o in candidates if room(o)] or candidates
        dx, dy = max(options, key=lambda o: (round(clearance(o) / 25), -abs(o[0]) - abs(o[1])))
        placed.append(p + np.array((dx, dy)) * scale)
        ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8.5, color=INK,
                    ha="left" if dx > 0 else ("right" if dx < 0 else "center"), zorder=6,
                    arrowprops={"arrowstyle": "-", "color": INK_2, "linewidth": 0.8},
                    bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none", "alpha": 0.85})


def plot_pca(coords: pd.DataFrame, group: str, explained: np.ndarray, path: Path) -> None:
    """Players on the first two principal components, colored by cluster."""
    k = int(coords["cluster_id"].nunique())
    names = [n for n in LABEL_PLAYERS if (coords["player_name"] == n).sum() == 1]
    sizes = coords["cluster_id"].value_counts().sort_index()
    xlabel = f"PC1 ({explained[0]:.0%} of variance)"
    ylabel = f"PC2 ({explained[1]:.0%} of variance)"
    title = f"{group}: players on the first two principal components ({explained.sum():.0%} of variance)"

    if k <= MAX_SINGLE_PANEL_CLUSTERS:
        fig, ax = plt.subplots(figsize=(9, 6.5), dpi=150, facecolor=SURFACE)
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        for c in range(k):
            sub = coords[coords["cluster_id"] == c]
            ax.scatter(sub["pc1"], sub["pc2"], s=26, color=CLUSTER_COLORS[c], edgecolor=SURFACE, linewidth=0.6,
                       alpha=0.9, label=f"Cluster {c} (n={sizes[c]})", zorder=3)
            ax.text(sub["pc1"].median(), sub["pc2"].median(), str(c), fontsize=13, fontweight="bold",
                    color=INK, ha="center", va="center", zorder=4,
                    bbox={"boxstyle": "circle,pad=0.25", "fc": SURFACE, "ec": CLUSTER_COLORS[c], "lw": 2})
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="best")
    else:
        cols = 3
        rows = int(np.ceil(k / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), dpi=150, facecolor=SURFACE,
                                 sharex=True, sharey=True, squeeze=False)
        for c, ax in zip(range(rows * cols), axes.ravel()):
            _style(ax)
            if c >= k:
                ax.set_visible(False)
                continue
            ax.scatter(coords["pc1"], coords["pc2"], s=8, color=NEUTRAL, edgecolor="none", zorder=2)
            sub = coords[coords["cluster_id"] == c]
            ax.scatter(sub["pc1"], sub["pc2"], s=18, color=SERIES_1, edgecolor=SURFACE, linewidth=0.5, zorder=3)
            ax.set_title(f"Cluster {c} (n={sizes[c]})", loc="left", fontsize=10, color=INK)
        fig.supxlabel(xlabel, color=INK_2, fontsize=9)
        fig.supylabel(ylabel, color=INK_2, fontsize=9)
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.canvas.draw()  # final layout, so label placement can use pixel positions
    if k <= MAX_SINGLE_PANEL_CLUSTERS:
        _annotate(ax, coords, names, lambda c: CLUSTER_COLORS[c])
    else:
        for c, panel in zip(range(k), axes.ravel()):
            sub = coords[coords["cluster_id"] == c]
            _annotate(panel, sub, [n for n in names if (sub["player_name"] == n).any()], lambda _c: SERIES_1)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
