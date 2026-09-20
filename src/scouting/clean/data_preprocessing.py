"""Raw merged FBref table -> one clean row per player.

Steps: verify redundant columns -> keep/rename curated additive columns ->
parse league/nation/position -> merge players who appear for several teams.
"""
from __future__ import annotations

import pandas as pd

from scouting.clean.columns import (
    COUNT_COLUMNS,
    META_COLUMNS,
    POSITION_GROUPS,
    REDUNDANT_PAIRS,
)
from scouting.clean.download import download_raw
from scouting.config import Config, load_config
from scouting.utils.logging import get_logger

log = get_logger(__name__)


def load_raw(config: Config) -> pd.DataFrame:
    return pd.read_csv(download_raw(config))


def verify_redundant_columns(df: pd.DataFrame) -> None:
    """Fail loudly if a column we drop as a duplicate is not actually identical."""
    mismatched = []
    for kept, dup in REDUNDANT_PAIRS:
        both_null = df[kept].isna() & df[dup].isna()
        if not (both_null | (df[kept] == df[dup])).all():
            mismatched.append((kept, dup))
    if mismatched:
        raise ValueError(f"redundant columns differ, refusing to drop: {mismatched}")
    log.info("verified %d redundant column pairs are identical", len(REDUNDANT_PAIRS))


def select_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    wanted = {**META_COLUMNS, **COUNT_COLUMNS}
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        raise KeyError(f"expected columns not in raw data: {missing}")
    return df[list(wanted)].rename(columns=wanted).copy()


def parse_labels(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Split 'eng Premier League' / 'us USA' prefixes and add the position group."""
    df = df.copy()
    df["league_code"] = df["comp_raw"].str.split(" ", n=1).str[0]
    df["league"] = df["comp_raw"].str.split(" ", n=1).str[1]
    df["nation"] = df["nation"].str.split(" ", n=1).str[1]
    df["season"] = season
    return df.drop(columns="comp_raw")


def merge_multi_team_players(df: pd.DataFrame) -> pd.DataFrame:
    """Combine rows of players who appear for more than one team in the season.

    Rows are matched on player + birth year (names alone collide). Totals are
    summed; identity fields (club, league, position) come from the row with the
    most minutes. Players with no birth year are never merged.
    """
    df = df.sort_values("minutes", ascending=False, kind="stable").reset_index(drop=True)
    born = df["born"].astype("Int64").astype("string")
    unmergeable = pd.Series("row" + df.index.astype(str), index=df.index, dtype="string")
    key = df["player"] + "|" + born.fillna(unmergeable)

    count_cols = [c for c in COUNT_COLUMNS.values() if c in df.columns]
    agg: dict[str, object] = {c: "first" for c in ["player", "nation", "pos_raw", "squad", "league_code", "league", "season", "born"]}
    agg["age"] = "max"
    agg.update({c: (lambda s: s.sum(min_count=1)) for c in count_cols})

    grouped = df.groupby(key, sort=False)
    merged = grouped.agg(agg)
    merged["n_teams"] = grouped.size()
    merged["squads"] = grouped["squad"].agg(" / ".join)
    merged["leagues"] = grouped["league"].agg(lambda s: " / ".join(dict.fromkeys(s)))
    return merged.reset_index(drop=True)


def add_position_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tokens = df["pos_raw"].str.split(",")
    df["pos_group"] = tokens.str[0].map(POSITION_GROUPS)
    df["pos_secondary"] = tokens.str[1].map(POSITION_GROUPS)
    if df["pos_group"].isna().any():
        bad = df.loc[df["pos_group"].isna(), "pos_raw"].unique()
        raise ValueError(f"unmapped position labels: {bad}")
    return df


def build_clean_dataset(config: Config) -> pd.DataFrame:
    raw = load_raw(config)
    log.info("raw: %d rows x %d columns", *raw.shape)
    verify_redundant_columns(raw)
    df = select_and_rename(raw)
    df = parse_labels(df, config.season)
    df = merge_multi_team_players(df)
    df = add_position_group(df)
    log.info("clean: %d players x %d columns (%d multi-team rows merged)", *df.shape, len(raw) - len(df))
    return df


def main() -> None:
    config = load_config()
    df = build_clean_dataset(config)
    config.interim_dir.mkdir(parents=True, exist_ok=True)
    out = config.interim_dir / "players_clean.csv"
    df.to_csv(out, index=False)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
