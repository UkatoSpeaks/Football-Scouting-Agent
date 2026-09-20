"""Clean season totals -> per-90 stats and efficiency ratios for players with enough minutes.

Raw totals stay in the clean table; this module produces the *model* features:
``<stat>_p90`` (rate per 90 minutes) and ratio features (``*_pct``, ``*_share``,
``npxg_per_shot``). Position-specific feature lists come from config.yaml.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scouting.config import Config, load_config
from scouting.utils.logging import get_logger

log = get_logger(__name__)

IDENTITY_COLUMNS = [
    "player_id", "player", "squad", "squads", "league", "leagues", "season",
    "nation", "born", "age", "pos_raw", "pos_group", "pos_secondary", "n_teams",
]


def load_clean(config: Config) -> pd.DataFrame:
    df = pd.read_csv(config.interim_dir / "players_clean.csv")
    df.insert(0, "player_id", df.index)  # stable row id of the clean table
    return df


def add_derived_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Counts that only exist as combinations of clean columns (used as ratio parts)."""
    df = df.copy()
    df["aerials_contested"] = df["aerials_won"] + df["aerials_lost"]
    df["challenges_won"] = df["challenges_attempted"] - df["challenges_lost"]
    df["tackles_mid_att_third"] = df["tackles_mid_third"] + df["tackles_att_third"]
    if (df["challenges_won"] < 0).any():
        raise ValueError("challenges_lost exceeds challenges_attempted for some players")
    return df


def filter_min_minutes(df: pd.DataFrame, min_minutes: int) -> pd.DataFrame:
    kept = df[df["minutes"] >= min_minutes].copy()
    log.info("minutes >= %d: kept %d of %d players", min_minutes, len(kept), len(df))
    kept["nineties"] = kept["minutes"] / 90.0
    return kept


def add_per90(df: pd.DataFrame, stats: list[str]) -> pd.DataFrame:
    missing = [s for s in stats if s not in df.columns]
    if missing:
        raise KeyError(f"per90 stats not in clean data: {missing}")
    new = {f"{s}_p90": df[s] / df["nineties"] for s in stats}  # nineties >= min_minutes/90 > 0
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def add_ratios(df: pd.DataFrame, ratios: dict[str, list[str]], prior_strength: float) -> pd.DataFrame:
    """Add smoothed ratio features.

    value = (num + k * prior) / (den + k), where ``prior`` is the pooled rate
    (sum num / sum den) of the player's position group. With den == 0 the value
    is the group average instead of a division by zero, and tiny samples
    (1 of 1) are pulled toward the group average instead of reading 100%.
    Groups where the ratio does not exist (e.g. goalkeeper ratios for
    outfielders) stay NaN.
    """
    new: dict[str, pd.Series] = {}
    for name, (num, den) in ratios.items():
        for col in (num, den):
            if col not in df.columns:
                raise KeyError(f"ratio '{name}' needs column '{col}', not in data")
        scale = 100.0 if name.endswith(("_pct", "_share")) else 1.0
        values = pd.Series(np.nan, index=df.index, dtype="float64")
        for _, group in df.groupby("pos_group"):
            den_total = group[den].sum()
            if den_total <= 0:
                continue
            prior = group[num].sum() / den_total
            values.loc[group.index] = (group[num] + prior_strength * prior) / (group[den] + prior_strength)
        new[name] = values * scale
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def build_feature_table(clean: pd.DataFrame, config: Config) -> pd.DataFrame:
    fe = config.feature_engineering
    df = add_derived_counts(clean)
    df = filter_min_minutes(df, fe.min_minutes)
    df = add_per90(df, fe.per90_stats)
    df = add_ratios(df, fe.ratios, fe.ratio_prior_strength)

    engineered = [f"{s}_p90" for s in fe.per90_stats] + list(fe.ratios)
    columns = IDENTITY_COLUMNS + ["nineties"] + fe.keep_as_is_stats + engineered
    table = df[list(dict.fromkeys(columns))].reset_index(drop=True)
    validate_feature_sets(table, config)
    return table


def skewed_features(table: pd.DataFrame, pos_group: str, features: list[str], threshold: float) -> list[str]:
    """Features that justify log1p within one position group.

    A feature qualifies when it is non-negative, its skewness is above
    ``threshold``, and log1p actually reduces that skewness.
    """
    sub = table.loc[table["pos_group"] == pos_group, features]
    chosen = []
    for f in features:
        s = sub[f]
        if s.min() < 0:
            continue
        skew = s.skew()
        if skew > threshold and np.log1p(s).skew() < skew:
            chosen.append(f)
    return chosen


def model_features(config: Config, pos_group: str) -> list[str]:
    return list(config.modeling.position_feature_lists[pos_group])


def validate_feature_sets(table: pd.DataFrame, config: Config) -> None:
    """Every listed feature must exist and be complete for its own position group."""
    for group, features in config.modeling.position_feature_lists.items():
        missing = [f for f in features if f not in table.columns]
        if missing:
            raise KeyError(f"{group}: features not in feature table: {missing}")
        subset = table.loc[table["pos_group"] == group, features]
        bad = subset.columns[subset.isna().any()].tolist()
        if bad:
            raise ValueError(f"{group}: NaN in features {bad}")
        dupes = [f for f in set(features) if features.count(f) > 1]
        if dupes:
            raise ValueError(f"{group}: duplicate features {dupes}")


def main() -> None:
    config = load_config()
    table = build_feature_table(load_clean(config), config)
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    out = config.processed_dir / "players_features.csv"
    table.to_csv(out, index=False)
    log.info("wrote %s (%d players x %d columns)", out, *table.shape)


if __name__ == "__main__":
    main()
