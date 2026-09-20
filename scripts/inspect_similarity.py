"""Diagnostics for the similarity engine (no results are adjusted).

Run:  python scripts/inspect_similarity.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scouting.similarity import SimilarityEngine

TEST_PLAYERS = ["Bukayo Saka", "Erling Haaland", "Declan Rice", "Virgil van Dijk", "Alisson"]
N = 10


def global_distribution(engine: SimilarityEngine) -> None:
    print("===== global cosine distribution per position group (all pairs) =====")
    rows = []
    for group, V in engine._values.items():
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        unit = V / np.where(norms == 0, 1, norms)
        S = unit @ unit.T
        iu = np.triu_indices(len(S), k=1)
        pairs = S[iu]
        np.fill_diagonal(S, -np.inf)
        nn1 = S.max(axis=1)
        nn10 = np.sort(S, axis=1)[:, -10:].mean(axis=1)
        q = np.quantile(pairs, [0.05, 0.25, 0.5, 0.75, 0.95])
        rows.append({
            "group": group, "players": len(V), "features": V.shape[1],
            "min": pairs.min(), "p05": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p95": q[4],
            "max": pairs.max(), "share>0.5": (pairs > 0.5).mean(),
            "nn1_median": np.median(nn1), "nn10_median": np.median(nn10),
        })
    print(pd.DataFrame(rows).round(2).to_string(index=False), "\n")


def report(engine: SimilarityEngine, name: str) -> None:
    pid = engine.find_id(name)
    q = engine.players.loc[pid]
    group = q["pos_group"]
    print("=" * 110)
    print(f"{name} (id {pid}) | {q['squad']} | {q['league']} | {q['pos_raw']} -> {group} | {q['minutes']:.0f} min")

    top = engine.find_similar_players(pid, n=N)
    show = top[["rank", "player_name", "club", "league", "position", "minutes", "similarity"]].copy()
    show["similarity"] = show["similarity"].round(3)
    print("\n-- top 10 by cosine similarity")
    print(show.to_string(index=False))

    scores = engine.scores(pid, "cosine")
    qs = scores.quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]).round(2)
    print(f"\n-- cosine distribution vs all {len(scores)} {group} players: "
          + ", ".join(f"{k:.0%}={v}" for k, v in qs.items())
          + f" | top-1 {top['similarity'].iloc[0]:.3f}, top-10 {top['similarity'].iloc[-1]:.3f}, "
            f"share of pool > 0.5: {(scores > 0.5).mean():.1%}")

    base = engine.players.loc[engine.players["pos_group"] == group, "league"].value_counts(normalize=True)
    lg = top["league"].value_counts()
    print("-- league mix of top 10 (pool share in brackets): "
          + ", ".join(f"{k} {v} ({base[k]:.0%})" for k, v in lg.items()))
    print("-- positions in top 10:", top["position"].value_counts().to_dict(),
          "| avg minutes:", round(top["minutes"].mean()))

    # do the matches share what defines the query player?
    feats = engine._features[group]
    Z = pd.DataFrame(engine._values[group], columns=feats, index=engine._ids[group])
    zq = Z.loc[pid]
    defining = zq.abs().sort_values(ascending=False).head(6).index
    z_top = Z.loc[top["player_id"]].mean()
    tab = pd.DataFrame({"query_z": zq[defining], "top10_mean_z": z_top[defining],
                        "pool_mean_z": 0.0}).round(2)
    print("\n-- query's most defining features vs the average of its top 10 (z-scores)")
    print(tab.to_string())

    # where the matches differ most from the query
    diffs = (Z.loc[top["player_id"]] - zq).abs().mean().sort_values(ascending=False).head(5).round(2)
    print("-- features where the top 10 differ most from the query (mean |z diff|):",
          diffs.to_dict())

    best = engine.explain_match(pid, int(top["player_id"].iloc[0]), top=6)
    print(f"\n-- biggest differences to closest match ({top['player_name'].iloc[0]}), raw per-90 units")
    print(best.round(2).to_string(index=False))

    cmp = engine.compare_metrics(pid, n=N)
    eu = cmp["top_euclidean"][["rank", "player_name", "club", "league", "distance"]].copy()
    eu["distance"] = eu["distance"].round(2)
    eu["cosine_rank"] = [cmp["cos_rank_of_euclidean_top"][i] for i in cmp["top_euclidean"]["player_id"]]
    print(f"\n-- euclidean top 10 (with each player's cosine rank) | overlap with cosine top 10: "
          f"{cmp['overlap']}/{N} | Spearman over all {len(scores)}: {cmp['spearman_all']:.2f}")
    print(eu.to_string(index=False))
    cos_only = engine.players.loc[cmp["only_cosine"], "player"].tolist()
    print("   only in cosine top 10:", cos_only)


def main() -> None:
    engine = SimilarityEngine.from_config()
    pd.set_option("display.width", 200)
    global_distribution(engine)
    for name in TEST_PLAYERS:
        report(engine, name)


if __name__ == "__main__":
    main()
