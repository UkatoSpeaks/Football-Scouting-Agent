import numpy as np
import pandas as pd
import pytest

from scouting.features.engineering import (
    add_derived_counts,
    add_per90,
    add_ratios,
    filter_min_minutes,
)


def test_filter_keeps_at_or_above_threshold_and_adds_nineties():
    df = pd.DataFrame({"minutes": [449, 450, 900]})
    out = filter_min_minutes(df, 450)
    assert out["minutes"].tolist() == [450, 900]
    assert out["nineties"].tolist() == [5.0, 10.0]


def test_per90_divides_by_nineties():
    df = pd.DataFrame({"goals": [10.0], "nineties": [20.0]})
    assert add_per90(df, ["goals"])["goals_p90"].iloc[0] == pytest.approx(0.5)


def test_per90_unknown_stat_raises():
    with pytest.raises(KeyError):
        add_per90(pd.DataFrame({"nineties": [5.0]}), ["nope"])


def test_ratio_zero_denominator_gives_group_average_not_nan_or_inf():
    df = pd.DataFrame({
        "pos_group": ["FWD", "FWD", "FWD"],
        "takeons_won": [30, 0, 10],
        "takeons_attempted": [60, 0, 40],
    })
    out = add_ratios(df, {"takeon_success_pct": ["takeons_won", "takeons_attempted"]}, prior_strength=10)
    prior = 40 / 100  # pooled group rate
    assert out["takeon_success_pct"].iloc[1] == pytest.approx(100 * prior)
    assert np.isfinite(out["takeon_success_pct"]).all()


def test_ratio_small_sample_is_shrunk_toward_group_rate():
    df = pd.DataFrame({
        "pos_group": ["MID"] * 3,
        "takeons_won": [1, 40, 40],
        "takeons_attempted": [1, 100, 100],
    })
    out = add_ratios(df, {"takeon_success_pct": ["takeons_won", "takeons_attempted"]}, prior_strength=10)
    assert out["takeon_success_pct"].iloc[0] < 60  # 1-for-1 no longer reads 100%
    assert out["takeon_success_pct"].iloc[1] == pytest.approx(40, abs=1.5)


def test_ratio_priors_are_computed_per_position_group():
    df = pd.DataFrame({
        "pos_group": ["DEF", "DEF", "FWD", "FWD"],
        "num": [9, 9, 1, 1],
        "den": [10, 10, 10, 10],
    })
    out = add_ratios(df, {"x_share": ["num", "den"]}, prior_strength=10)
    assert out["x_share"].iloc[0] == pytest.approx(90)
    assert out["x_share"].iloc[2] == pytest.approx(10)


def test_ratio_stays_nan_where_group_has_no_denominator():
    df = pd.DataFrame({
        "pos_group": ["GK", "DEF"],
        "gk_saves": [60.0, np.nan],
        "gk_shots_on_target_against": [90.0, np.nan],
    })
    out = add_ratios(df, {"gk_save_pct": ["gk_saves", "gk_shots_on_target_against"]}, prior_strength=10)
    assert np.isfinite(out["gk_save_pct"].iloc[0])
    assert np.isnan(out["gk_save_pct"].iloc[1])


def test_derived_counts_and_invalid_challenges():
    df = pd.DataFrame({
        "aerials_won": [3], "aerials_lost": [2], "challenges_attempted": [10], "challenges_lost": [4],
        "tackles_mid_third": [5], "tackles_att_third": [1],
    })
    out = add_derived_counts(df)
    assert out[["aerials_contested", "challenges_won", "tackles_mid_att_third"]].iloc[0].tolist() == [5, 6, 6]
    df["challenges_lost"] = 11
    with pytest.raises(ValueError):
        add_derived_counts(df)
