"""Column knowledge for the merged FBref player table.

The raw file is several FBref tables joined side by side. Where two tables share
a column name, later tables get a ``_stats_<table>`` suffix, so the SAME short
name can mean DIFFERENT things (e.g. ``Blocks`` = passes blocked by an opponent,
``Blocks_stats_defense`` = the player's own blocks). Every column we keep is
therefore renamed explicitly, and every dropped duplicate is verified equal.
"""
from __future__ import annotations

META_COLUMNS = {
    "Player": "player",
    "Nation": "nation",
    "Pos": "pos_raw",
    "Squad": "squad",
    "Comp": "comp_raw",
    "Age": "age",
    "Born": "born",
}

# Additive season totals: safe to sum when a player appears for two clubs.
# Rates/percentages (Cmp%, Tkl%, Sh/90, Dist, ...) are NOT kept; they are
# recomputed from these totals in feature engineering.
COUNT_COLUMNS = {
    # playing time
    "MP": "matches",
    "Starts": "starts",
    "Min": "minutes",
    # standard
    "Gls": "goals",
    "Ast": "assists",
    "G-PK": "np_goals",
    "PK": "pens_scored",
    "PKatt": "pens_attempted",
    "CrdY": "yellow_cards",
    "CrdR": "red_cards",
    "xG": "xg",
    "npxG": "npxg",
    "xAG": "xag",
    "PrgC": "prog_carries",
    "PrgP": "prog_passes",
    "PrgR": "prog_passes_received",
    # shooting
    "Sh": "shots",
    "SoT": "shots_on_target",
    # passing
    "Cmp": "passes_completed",
    "Att": "passes_attempted",
    "TotDist": "pass_total_dist",
    "PrgDist": "pass_prog_dist",
    "xA": "xa",
    "KP": "key_passes",
    "1/3": "passes_final_third",
    "PPA": "passes_penalty_area",
    "CrsPA": "crosses_penalty_area",
    # passing types
    "TB": "through_balls",
    "Sw": "switches",
    "Crs": "crosses",
    "TI": "throw_ins",
    "CK": "corner_kicks",
    "Blocks": "passes_blocked_by_opp",  # NOT the player's own blocks
    # goal/shot creation
    "SCA": "sca",
    "PassLive": "sca_pass_live",
    "PassDead": "sca_pass_dead",
    "TO": "sca_take_on",
    "Sh_stats_gca": "sca_shot",
    "Fld": "sca_fouled",  # fouls drawn that led to a shot, NOT all fouls drawn
    "Def": "sca_defensive",
    "GCA": "gca",
    # defense
    "Tkl": "tackles",
    "TklW": "tackles_won",
    "Def 3rd": "tackles_def_third",
    "Mid 3rd": "tackles_mid_third",
    "Att 3rd": "tackles_att_third",
    "Att_stats_defense": "challenges_attempted",  # dribblers contested
    "Lost": "challenges_lost",
    "Blocks_stats_defense": "blocks",
    "Sh_stats_defense": "blocks_shots",
    "Pass": "blocks_passes",
    "Int": "interceptions",
    "Tkl+Int": "tackles_plus_interceptions",
    "Clr": "clearances",
    "Err": "errors_leading_to_shot",
    # possession
    "Touches": "touches",
    "Def Pen": "touches_def_pen",
    "Def 3rd_stats_possession": "touches_def_third",
    "Mid 3rd_stats_possession": "touches_mid_third",
    "Att 3rd_stats_possession": "touches_att_third",
    "Att Pen": "touches_att_pen",
    "Att_stats_possession": "takeons_attempted",
    "Succ": "takeons_won",
    "Tkld": "takeons_tackled",
    "Carries": "carries",
    "TotDist_stats_possession": "carry_total_dist",
    "PrgDist_stats_possession": "carry_prog_dist",
    "1/3_stats_possession": "carries_final_third",
    "CPA": "carries_penalty_area",
    "Mis": "miscontrols",
    "Dis": "dispossessed",
    "Rec": "passes_received",
    # misc
    "Fls": "fouls_committed",
    "Fld_stats_misc": "fouls_drawn",
    "Off_stats_misc": "offsides",  # bare "Off" is passes that ended offside
    "2CrdY": "second_yellows",
    "PKwon": "pens_won",
    "PKcon": "pens_conceded",
    "OG": "own_goals",
    "Recov": "recoveries",
    "Won": "aerials_won",
    "Lost_stats_misc": "aerials_lost",
    # goalkeeper (NaN for outfield players)
    "GA": "gk_goals_against",
    "SoTA": "gk_shots_on_target_against",
    "Saves": "gk_saves",
    "CS": "gk_clean_sheets",
    "PKA": "gk_pens_against",
    "PKsv": "gk_pens_saved",
    "PSxG": "gk_psxg",
    "Opp": "gk_crosses_faced",
    "Stp": "gk_crosses_stopped",
    "#OPA": "gk_def_actions_outside_pen",
}

# (kept column, duplicate column). The duplicate is dropped only after
# verifying that both hold identical values in every row.
REDUNDANT_PAIRS = [
    ("Min", "Min_stats_playing_time"),
    ("MP", "MP_stats_playing_time"),
    ("Starts", "Starts_stats_playing_time"),
    ("Gls", "Gls_stats_shooting"),
    ("xG", "xG_stats_shooting"),
    ("npxG", "npxG_stats_shooting"),
    ("PK", "PK_stats_shooting"),
    ("PKatt", "PKatt_stats_shooting"),
    ("Ast", "Ast_stats_passing"),
    ("xAG", "xAG_stats_passing"),
    ("PrgP", "PrgP_stats_passing"),
    ("Att", "Att_stats_passing_types"),
    ("Cmp", "Cmp_stats_passing_types"),
    ("PrgC", "PrgC_stats_possession"),
    ("PrgR", "PrgR_stats_possession"),
    ("CrdY", "CrdY_stats_misc"),
    ("CrdR", "CrdR_stats_misc"),
    ("Crs", "Crs_stats_misc"),
    ("Int", "Int_stats_misc"),
    ("TklW", "TklW_stats_misc"),
    ("GA", "GA_stats_keeper_adv"),
    ("PKA", "PKA_stats_keeper_adv"),
    ("Age", "Age_stats_shooting"),
    ("Age", "Age_stats_passing"),
    ("Age", "Age_stats_defense"),
    ("Age", "Age_stats_possession"),
    ("Age", "Age_stats_misc"),
    ("Born", "Born_stats_shooting"),
    ("Pos", "Pos_stats_shooting"),
]

POSITION_GROUPS = {"GK": "GK", "DF": "DEF", "MF": "MID", "FW": "FWD"}
