import numpy as np
import pandas as pd
import pytest

from scouting.clean.data_preprocessing import (
    add_position_group,
    merge_multi_team_players,
    parse_labels,
    verify_redundant_columns,
)


def _row(player, born, squad, minutes, goals, league="Premier League", pos="FW", age=25.0, gk_saves=np.nan):
    return {
        "player": player, "nation": "ENG", "pos_raw": pos, "squad": squad,
        "league_code": "eng", "league": league, "season": "2024-2025",
        "born": born, "age": age, "minutes": minutes, "goals": goals, "gk_saves": gk_saves,
    }


def _merge(rows):
    # only COUNT_COLUMNS present in the frame are aggregated
    return merge_multi_team_players(pd.DataFrame(rows))


def test_transfer_rows_are_summed_and_identity_comes_from_most_minutes():
    out = _merge([
        _row("A", 2000, "Club1", 300, 1),
        _row("A", 2000, "Club2", 900, 4, age=26.0),
        _row("B", 1999, "Club3", 500, 2),
    ])
    a = out[out.player == "A"].iloc[0]
    assert len(out) == 2
    assert a.minutes == 1200 and a.goals == 5
    assert a.squad == "Club2" and a.squads == "Club2 / Club1" and a.n_teams == 2
    assert a.age == 26.0


def test_same_name_different_birth_year_is_not_merged():
    out = _merge([_row("Ali", 1990, "X", 900, 1), _row("Ali", 2001, "Y", 900, 2)])
    assert len(out) == 2


def test_missing_birth_year_is_never_merged():
    out = _merge([_row("C", np.nan, "X", 900, 1), _row("C", np.nan, "Y", 800, 2)])
    assert len(out) == 2


def test_all_nan_goalkeeper_columns_stay_nan_after_merge():
    out = _merge([_row("D", 2000, "X", 900, 1), _row("D", 2000, "Y", 100, 0)])
    assert out.gk_saves.isna().all()


def test_parse_labels_and_position_group():
    df = pd.DataFrame({"comp_raw": ["eng Premier League"], "nation": ["us USA"], "pos_raw": ["FW,MF"]})
    out = add_position_group(parse_labels(df, "2024-2025"))
    assert out.league.iloc[0] == "Premier League" and out.league_code.iloc[0] == "eng"
    assert out.nation.iloc[0] == "USA"
    assert out.pos_group.iloc[0] == "FWD" and out.pos_secondary.iloc[0] == "MID"


def test_unknown_position_raises():
    with pytest.raises(ValueError):
        add_position_group(pd.DataFrame({"pos_raw": ["XX"]}))


def test_verify_redundant_columns_detects_mismatch():
    from scouting.clean.columns import REDUNDANT_PAIRS

    data = {}
    for kept, dup in REDUNDANT_PAIRS:
        data[kept] = [1.0, 2.0]
        data[dup] = [1.0, 2.0]
    verify_redundant_columns(pd.DataFrame(data))
    data["Min_stats_playing_time"] = [1.0, 99.0]
    with pytest.raises(ValueError):
        verify_redundant_columns(pd.DataFrame(data))
