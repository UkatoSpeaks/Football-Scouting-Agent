"""Chart colors shared by the report figures and the dashboard (light surface).

Values come from the dataviz reference palette. ``CLUSTER_COLORS`` is the slot order
blue, aqua, yellow, green, violet, red, validated with ``--pairs all`` for every
prefix of size 2-6 (scatter rule). ``SERIES`` is the blue/orange pair used when
exactly two entities are compared. ``DIVERGING`` is blue <-> red with a neutral
midpoint, for signed values such as standardised scores.
"""
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e6e5e1"
NEUTRAL = "#c9c8c2"

SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
CLUSTER_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
DIVERGING = {"high": "#2a78d6", "mid": "#f0efec", "low": "#e34948"}


def cluster_color(cluster_id: int) -> str:
    """Validated color for a cluster; clusters past the sixth fall back to neutral gray."""
    return CLUSTER_COLORS[cluster_id] if cluster_id < len(CLUSTER_COLORS) else NEUTRAL
