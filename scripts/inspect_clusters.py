"""Print the clustering results for review: evaluation at the chosen K, cluster sizes,
data-derived cluster descriptions, representative players, league mix, and where the
test players (and a wider set of well-known midfielders) landed.

Run:  python scripts/inspect_clusters.py
"""
from __future__ import annotations

import joblib
import pandas as pd

from scouting.clustering import describe_cluster, feature_columns, load_matrices, load_players, representatives
from scouting.config import load_config
from scouting.labels import label

TEST_PLAYERS = ["Bukayo Saka", "Erling Haaland", "Declan Rice", "Virgil van Dijk", "Alisson"]
KNOWN_MIDFIELDERS = [
    "Declan Rice", "Jude Bellingham", "Martín Zubimendi", "Joshua Kimmich", "Casemiro", "Moisés Caicedo",
    "Bruno Guimarães", "Frenkie de Jong", "Pedri", "Martin Ødegaard", "Kevin De Bruyne", "Sandro Tonali",
    "Granit Xhaka", "Luka Modrić", "Nicolò Barella", "Federico Valverde", "Enzo Fernández", "Florian Wirtz",
]


def describe(prof: pd.DataFrame, top: int = 5) -> str:
    d = describe_cluster(prof, threshold=0.5, top=top)
    hi = ", ".join(f"{n} {v:+.1f}" for n, v in d["high"]) or "(none above +0.5)"
    lo = ", ".join(f"{n} {v:+.1f}" for n, v in d["low"]) or "(none below -0.5)"
    return f"HIGH: {hi}\n      LOW : {lo}"


def main() -> None:
    config = load_config()
    pd.set_option("display.width", 220)
    matrices, players = load_matrices(config), load_players(config)
    models = joblib.load(config.models_dir / "kmeans.joblib")
    assign = pd.read_csv(config.processed_dir / "cluster_assignments.csv").set_index("player_id")
    profiles = pd.read_csv(config.processed_dir / "cluster_profiles.csv")
    evaluation = pd.read_csv(config.processed_dir / "kmeans_evaluation.csv")

    print("===== evaluation at the chosen K =====")
    rows = [evaluation[(evaluation.position_group == g) & (evaluation.k == k)] for g, k in config.modeling.chosen_k.items()]
    print(pd.concat(rows).round(3).to_string(index=False))

    print("\n===== clusters =====")
    for group, matrix in matrices.items():
        km = models[group]["model"]
        Z = matrix[feature_columns(matrix)].to_numpy()
        print(f"\n########## {group}: K={km.n_clusters}, sizes {pd.Series(km.labels_).value_counts().sort_index().tolist()}")
        for c in range(km.n_clusters):
            sub = assign[(assign.position_group == group) & (assign.cluster_id == c)]
            reps = players.loc[matrix["player_id"].iloc[representatives(Z, km, c, 5)], "player"].tolist()
            top_min = sub.sort_values("minutes", ascending=False).player_name.head(5).tolist()
            leagues = (sub.league.value_counts(normalize=True) * 100).round(0).astype(int).to_dict()
            print(f"\n  {group} cluster {c} (n={len(sub)})")
            print("      " + describe(profiles[(profiles.position_group == group) & (profiles.cluster_id == c)]))
            print(f"      closest to centre: {reps}")
            print(f"      most minutes     : {top_min}")
            print(f"      league mix %     : {leagues}")

    print("\n\n===== test players =====")
    for name in TEST_PLAYERS:
        pid = players.index[players.player == name][0]
        row = assign.loc[pid]
        group, c = row.position_group, int(row.cluster_id)
        prof = profiles[(profiles.position_group == group) & (profiles.cluster_id == c)]
        z = matrices[group].set_index("player_id").loc[pid]
        print(f"\n{name} | {row.club} | {row.league} | {row.position} -> {group} cluster {c} "
              f"(n={int(prof.n_players.iloc[0])})")
        print("      cluster: " + describe(prof))
        cmp = pd.DataFrame({
            "player_z": z,
            "cluster_mean_z": prof.set_index("feature")["mean_z"],
        }).assign(gap=lambda d: d.player_z - d.cluster_mean_z)
        cmp.index = [label(f) for f in cmp.index]
        strongest = cmp.reindex(cmp.player_z.abs().sort_values(ascending=False).index).head(6)
        print("      player's strongest traits vs his cluster mean:")
        print(strongest.round(2).to_string().replace("\n", "\n        "))

    print("\n\n===== where well-known midfielders landed (MID) =====")
    mid = assign[assign.position_group == "MID"]
    found = mid[mid.player_name.isin(KNOWN_MIDFIELDERS)].sort_values(["cluster_id", "player_name"])
    print(found[["player_name", "club", "position", "cluster_id"]].to_string(index=False))

    print("\n===== Rice: all MID z-scores next to each cluster's mean z =====")
    mm = matrices["MID"].set_index("player_id")
    pid = players.index[players.player == "Declan Rice"][0]
    table = pd.DataFrame({"Rice_z": mm.loc[pid]})
    for c in range(models["MID"]["model"].n_clusters):
        table[f"C{c}"] = profiles[(profiles.position_group == "MID") & (profiles.cluster_id == c)].set_index("feature")["mean_z"]
    table.index = [label(f) for f in table.index]
    print(table.round(2).to_string())
    centres = models["MID"]["model"].cluster_centers_
    dist = ((centres - mm.loc[pid].to_numpy()) ** 2).sum(axis=1) ** 0.5
    print("distance to each cluster centre:", dict(enumerate(dist.round(2))))


if __name__ == "__main__":
    main()
