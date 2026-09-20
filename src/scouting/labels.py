"""Human-readable names for model features (used in cluster descriptions and plots)."""
from __future__ import annotations

FEATURE_LABELS = {
    # goalkeeper
    "gk_save_pct": "save %",
    "gk_cross_stop_pct": "cross stopping",
    "gk_def_actions_outside_pen_p90": "sweeper actions",
    "pass_prog_dist_p90": "progressive pass distance",
    # shared passing / possession
    "passes_attempted_p90": "passing volume",
    "pass_completion_pct": "pass completion",
    "prog_passes_p90": "progressive passing",
    "prog_pass_share": "progressive-pass share",
    "prog_carries_p90": "progressive carrying",
    "prog_passes_received_p90": "progressive passes received",
    "key_passes_p90": "key passes",
    "touches_att_third_p90": "attacking-third touches",
    "touches_att_pen_p90": "penalty-area touches",
    "crosses_p90": "crossing",
    "takeons_won_p90": "take-ons won",
    "takeons_attempted_p90": "take-ons attempted",
    "takeon_success_pct": "take-on success",
    # attacking output
    "goals_p90": "goals",
    "assists_p90": "assists",
    "npxg_p90": "npxG",
    "xag_p90": "xAG",
    "shots_p90": "shooting volume",
    "shot_accuracy_pct": "shot accuracy",
    "npxg_per_shot": "shot quality",
    "gca_p90": "goal-creating actions",
    "fouls_drawn_p90": "fouls drawn",
    # defending
    "tackles_p90": "tackles",
    "tackle_success_pct": "tackle success",
    "high_tackle_share": "high-up tackling",
    "interceptions_p90": "interceptions",
    "blocks_p90": "blocks",
    "clearances_p90": "clearances",
    "recoveries_p90": "ball recoveries",
    "aerials_won_p90": "aerial duels won",
    "aerial_win_pct": "aerial win %",
    "challenge_success_pct": "1v1 challenge success",
    "fouls_committed_p90": "fouls committed",
}


def label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


# --------------------------------------------------------------- dashboard grouping
CATEGORY_ORDER = [
    "Attacking",
    "Chance creation",
    "Passing",
    "Carrying & take-ons",
    "Defending",
    "Goalkeeping",
]

FEATURE_CATEGORIES = {
    # attacking
    "goals_p90": "Attacking",
    "npxg_p90": "Attacking",
    "shots_p90": "Attacking",
    "shot_accuracy_pct": "Attacking",
    "npxg_per_shot": "Attacking",
    "touches_att_pen_p90": "Attacking",
    # chance creation
    "assists_p90": "Chance creation",
    "xag_p90": "Chance creation",
    "key_passes_p90": "Chance creation",
    "gca_p90": "Chance creation",
    # passing
    "passes_attempted_p90": "Passing",
    "pass_completion_pct": "Passing",
    "prog_passes_p90": "Passing",
    "prog_pass_share": "Passing",
    "crosses_p90": "Passing",
    "pass_prog_dist_p90": "Passing",
    # carrying, take-ons and receiving in advanced areas
    "prog_carries_p90": "Carrying & take-ons",
    "takeons_attempted_p90": "Carrying & take-ons",
    "takeons_won_p90": "Carrying & take-ons",
    "takeon_success_pct": "Carrying & take-ons",
    "prog_passes_received_p90": "Carrying & take-ons",
    "touches_att_third_p90": "Carrying & take-ons",
    "fouls_drawn_p90": "Carrying & take-ons",
    # defending
    "tackles_p90": "Defending",
    "tackle_success_pct": "Defending",
    "high_tackle_share": "Defending",
    "interceptions_p90": "Defending",
    "blocks_p90": "Defending",
    "clearances_p90": "Defending",
    "recoveries_p90": "Defending",
    "aerials_won_p90": "Defending",
    "aerial_win_pct": "Defending",
    "challenge_success_pct": "Defending",
    "fouls_committed_p90": "Defending",
    # goalkeeping
    "gk_save_pct": "Goalkeeping",
    "gk_cross_stop_pct": "Goalkeeping",
    "gk_def_actions_outside_pen_p90": "Goalkeeping",
}


def unit(feature: str) -> str:
    """Short unit text: per-90 rates, percentages, or a named ratio."""
    if feature.endswith("_p90"):
        return "yds/90" if feature == "pass_prog_dist_p90" else "per 90"
    if feature.endswith(("_pct", "_share")):
        return "%"
    if feature == "npxg_per_shot":
        return "npxG per shot"
    return ""


def sentence_case(text: str) -> str:
    """Capitalise the first letter, except for football acronyms that have their own casing."""
    return text if text.startswith(("npxG", "xAG")) else text[:1].upper() + text[1:]


def display_name(feature: str) -> str:
    """Readable name with its unit, e.g. 'Key passes (per 90)'."""
    name = sentence_case(label(feature))
    u = unit(feature)
    return f"{name} ({u})" if u and "%" not in name else name


# --------------------------------------------------- dashboard radar chart (Phase 2)
# A second, coarser regrouping of the same model features, used only to keep the player
# comparison's radar chart readable (5-6 axes instead of 16-20). It is presentation only:
# it never feeds the similarity engine, the StandardScaler or the K-Means clustering, and
# every value plotted on the radar is a percentile already computed for the "Player
# statistics" tabs (see ``dashboard.logic.radar_scores``). A category is only shown for a
# position group if that group's feature list actually uses one of its features, which is
# why goalkeepers (six features) get their own three keeper-specific categories instead of
# "Shooting" or "Defending".
RADAR_CATEGORY_ORDER = [
    "Shooting", "Chance creation", "Passing", "Progression", "Carrying & take-ons", "Defending",
    "Shot stopping", "Cross claiming", "Sweeping",
]

RADAR_CATEGORIES = {
    # shooting
    "goals_p90": "Shooting", "npxg_p90": "Shooting", "shots_p90": "Shooting",
    "shot_accuracy_pct": "Shooting", "npxg_per_shot": "Shooting", "touches_att_pen_p90": "Shooting",
    # chance creation
    "assists_p90": "Chance creation", "xag_p90": "Chance creation",
    "key_passes_p90": "Chance creation", "gca_p90": "Chance creation",
    # passing / distribution
    "passes_attempted_p90": "Passing", "pass_completion_pct": "Passing",
    "crosses_p90": "Passing", "pass_prog_dist_p90": "Passing",
    # progression
    "prog_passes_p90": "Progression", "prog_pass_share": "Progression",
    "prog_carries_p90": "Progression", "prog_passes_received_p90": "Progression",
    "touches_att_third_p90": "Progression",
    # carrying & take-ons
    "takeons_attempted_p90": "Carrying & take-ons", "takeons_won_p90": "Carrying & take-ons",
    "takeon_success_pct": "Carrying & take-ons", "fouls_drawn_p90": "Carrying & take-ons",
    # defending
    "tackles_p90": "Defending", "tackle_success_pct": "Defending", "high_tackle_share": "Defending",
    "interceptions_p90": "Defending", "blocks_p90": "Defending", "clearances_p90": "Defending",
    "recoveries_p90": "Defending", "aerials_won_p90": "Defending", "aerial_win_pct": "Defending",
    "challenge_success_pct": "Defending", "fouls_committed_p90": "Defending",
    # goalkeeping
    "gk_save_pct": "Shot stopping", "gk_cross_stop_pct": "Cross claiming",
    "gk_def_actions_outside_pen_p90": "Sweeping",
}
