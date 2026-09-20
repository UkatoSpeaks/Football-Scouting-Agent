"""Print the feature table for review: counts, skew before/after log1p,
redundancy on the transformed features, and raw -> log1p -> z-score examples.

Run:  python scripts/inspect_features.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scouting.config import load_config
from scouting.features.scaling import build_model_matrix, log1p_transform

EXAMPLES = ["Bukayo Saka", "Erling Haaland", "Declan Rice", "Virgil van Dijk", "Alisson"]
CORR_FLAG = 0.85
CORR_WATCH = 0.75


def main() -> None:
    config = load_config()
    table = pd.read_csv(config.processed_dir / "players_features.csv")
    lists = config.modeling.position_feature_lists
    pd.set_option("display.width", 220)

    print(f"players after {config.feature_engineering.min_minutes}-minute filter: {len(table)}")
    print(table["pos_group"].value_counts().to_string(), "\n")

    matrices = {}
    for group, features in lists.items():
        Z, _ = build_model_matrix(table, group, config)
        matrices[group] = Z
        X = log1p_transform(table, group, config)
        raw = table.loc[X.index, features]
        logged = config.modeling.log1p_features.get(group, [])

        print(f"===== {group}: {len(X)} players, {len(features)} features, {len(logged)} log1p =====")
        print("features:", features)
        if logged:
            rows = pd.DataFrame({"skew_before": raw[logged].skew(), "skew_after": X[logged].skew()})
            print(rows.round(2).to_string())
        print(f"z-score check: max |mean| = {Z[features].mean().abs().max():.1e}, "
              f"std range = {Z[features].std(ddof=0).min():.3f}-{Z[features].std(ddof=0).max():.3f}, "
              f"|z| max = {Z[features].abs().max().max():.1f}")

        corr = X.corr()
        pairs = [(a, b, round(corr.loc[a, b], 2)) for i, a in enumerate(features) for b in features[i + 1:]
                 if abs(corr.loc[a, b]) >= CORR_WATCH]
        print(f"correlated pairs on transformed features (|r| >= {CORR_WATCH}):", pairs or "none", "\n")

    print("===== example rows: raw -> log1p -> z-score =====")
    for name in EXAMPLES:
        row = table[table["player"] == name]
        if row.empty:
            continue
        r = row.iloc[0]
        group = r["pos_group"]
        features = lists[group]
        logged = set(config.modeling.log1p_features.get(group, []))
        Z = matrices[group]
        idx = row.index[0]
        X = log1p_transform(table, group, config)
        out = pd.DataFrame({
            "raw": table.loc[idx, features].astype(float),
            "log1p": [X.loc[idx, f] if f in logged else np.nan for f in features],
            "z": Z.loc[idx, features],
        })
        print(f"\n{r['player']} | {r['squad']} | {r['league']} | {r['pos_raw']} -> {group} | {r['minutes']:.0f} min")
        print(out.round(2).to_string())


if __name__ == "__main__":
    main()
