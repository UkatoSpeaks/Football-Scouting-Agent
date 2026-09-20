"""Player similarity within position groups.

Players are compared only inside their own position group (GK, DEF, MID, FWD),
using that group's standardized feature matrix (see scouting.features.scaling).
Cosine similarity is the primary metric; Euclidean distance is a secondary
diagnostic.

What the scores mean
--------------------
The matrix is z-scored per group, so each player is a vector of "how far above or
below the position average" on every feature.

* cosine similarity compares the *direction* of two such vectors: do the players
  lean the same way relative to the average? It ignores how extreme they are.
  Range -1..1. It is a similarity score, NOT a probability.
* Euclidean distance compares the *size of the overall gap* in z-score space, so
  it also reacts to intensity (a milder and a more extreme version of the same
  profile are close in cosine but further apart in Euclidean distance).
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from scouting.config import Config, load_config

METRICS = ("cosine", "euclidean")
IDENTITY_COLUMNS = ["player", "squad", "league", "pos_raw", "pos_group", "minutes"]
DEFAULT_MIN_CANDIDATE_MINUTES = 900


def _check_min_minutes(value: float | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) or value < 0:
        raise ValueError("min_candidate_minutes must be a non-negative number or None")
    return float(value)


class PlayerNotFoundError(KeyError):
    """The player_id is not in the modeled pool (players need enough minutes)."""


@dataclass
class SimilarityEngine:
    """Holds the per-group matrices and player identity table.

    matrices: pos_group -> DataFrame with a ``player_id`` column plus feature columns
    players:  DataFrame indexed by player_id with IDENTITY_COLUMNS
    raw:      optional DataFrame indexed by player_id with the un-standardized
              features, used only to show real per-90 values in explanations
    """

    matrices: dict[str, pd.DataFrame]
    players: pd.DataFrame
    raw: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        self._features = {g: [c for c in m.columns if c != "player_id"] for g, m in self.matrices.items()}
        self._values = {g: m[self._features[g]].to_numpy(dtype="float64") for g, m in self.matrices.items()}
        self._ids = {g: m["player_id"].to_numpy() for g, m in self.matrices.items()}
        self._row = {
            g: {int(pid): i for i, pid in enumerate(ids)} for g, ids in self._ids.items()
        }
        self._group_of = {pid: g for g, rows in self._row.items() for pid in rows}

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_config(cls, config: Config | None = None) -> "SimilarityEngine":
        config = config or load_config()
        features = pd.read_csv(config.processed_dir / "players_features.csv")
        matrices = {
            g: pd.read_csv(config.processed_dir / f"model_matrix_{g}.csv")
            for g in config.modeling.position_feature_lists
        }
        engine = cls(
            matrices=matrices,
            players=features.set_index("player_id")[IDENTITY_COLUMNS],
            raw=features.set_index("player_id"),
        )
        engine.check_scalers(joblib.load(config.models_dir / "scalers.joblib"))
        return engine

    def check_scalers(self, artifacts: dict[str, dict]) -> None:
        """Fail if the saved scalers do not belong to these matrices (stale files)."""
        for group, art in artifacts.items():
            if art["features"] != self._features[group]:
                raise ValueError(
                    f"{group}: saved scaler features {art['features']} do not match the matrix "
                    f"columns {self._features[group]}; rerun scouting.features.scaling"
                )
            n = len(self._values[group])
            if art["scaler"].n_samples_seen_ != n:
                raise ValueError(f"{group}: scaler was fitted on a different player pool ({n} rows now)")
            if not np.allclose(self._values[group].mean(axis=0), 0, atol=1e-8):
                raise ValueError(f"{group}: matrix is not standardized")

    # ------------------------------------------------------------------ lookups
    def group_of(self, player_id: int) -> str:
        try:
            return self._group_of[int(player_id)]
        except (KeyError, TypeError, ValueError):
            raise PlayerNotFoundError(
                f"player_id {player_id!r} is not in the modeled pool "
                f"(unknown id, or below the minimum-minutes threshold)"
            ) from None

    def find_id(self, name: str) -> int:
        hits = self.players.index[self.players["player"] == name]
        if len(hits) != 1:
            raise PlayerNotFoundError(f"{len(hits)} players named {name!r}; use player_id")
        return int(hits[0])

    # ------------------------------------------------------------------- scores
    def _resolve_group(self, player_id: int, position_group: str | None) -> str:
        group = self.group_of(player_id)
        if position_group is not None and position_group != group:
            raise ValueError(
                f"player {player_id} is in group {group}, not {position_group}; "
                "players are only compared within their own position group"
            )
        return group

    def scores(
        self,
        player_id: int,
        metric: str = "cosine",
        min_candidate_minutes: float | None = 0,
        leagues: str | Iterable[str] | None = None,
    ) -> pd.Series:
        """Score of every other player in the same group (index = player_id).

        Candidates with fewer than ``min_candidate_minutes`` minutes are left out
        (the query player is never filtered by this; ``None`` or 0 keeps everyone).
        ``leagues`` (a name or several) keeps only candidates whose main league is
        one of them; the query player may come from any league.
        """
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}")
        min_candidate_minutes = _check_min_minutes(min_candidate_minutes)
        wanted = self._check_leagues(leagues)
        group = self.group_of(player_id)
        V, ids = self._values[group], self._ids[group]
        q = V[self._row[group][int(player_id)]]

        if metric == "cosine":
            q_norm = np.linalg.norm(q)
            if q_norm == 0:
                raise ValueError(f"player {player_id} has a zero feature vector; cosine is undefined")
            norms = np.linalg.norm(V, axis=1)
            dots = V @ q
            out = np.divide(dots, norms * q_norm, out=np.zeros(len(V)), where=norms > 0)
            out = np.clip(out, -1.0, 1.0)  # remove float noise like 1.0000000000000002
        else:
            out = np.linalg.norm(V - q, axis=1)

        series = pd.Series(out, index=ids, name=metric).drop(index=int(player_id))
        if min_candidate_minutes > 0:
            series = series[self.players.loc[series.index, "minutes"].to_numpy() >= min_candidate_minutes]
        if wanted is not None:
            series = series[self.players.loc[series.index, "league"].isin(wanted).to_numpy()]
        return series

    def _check_leagues(self, leagues: str | Iterable[str] | None) -> list[str] | None:
        if leagues is None:
            return None
        wanted = [leagues] if isinstance(leagues, str) else list(leagues)
        unknown = sorted(set(wanted) - set(self.players["league"].unique()))
        if unknown:
            raise ValueError(f"unknown league(s) {unknown}; known: {sorted(self.players['league'].unique())}")
        return wanted

    def find_similar_players(
        self,
        player_id: int,
        position_group: str | None = None,
        n: int = 10,
        metric: str = "cosine",
        min_candidate_minutes: float | None = DEFAULT_MIN_CANDIDATE_MINUTES,
        leagues: str | Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Top-n most similar players inside the query player's position group.

        The query player only needs to be in the modeled pool (the feature
        pipeline's minimum-minutes threshold); candidates must have at least
        ``min_candidate_minutes`` minutes (default 900; ``None`` or 0 disables it)
        and, if ``leagues`` is given, play in one of those leagues.

        Returns columns: rank, player_id, player_name, league, club, position,
        position_group, minutes (of the candidate) and ``similarity`` (cosine,
        higher = closer) or ``distance`` (euclidean, lower = closer). Fewer than
        n rows are returned when fewer eligible candidates exist.
        """
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("n must be a positive integer")
        self._resolve_group(player_id, position_group)
        scores = self.scores(player_id, metric, min_candidate_minutes, leagues)

        ascending = metric == "euclidean"
        # stable sort so ties are broken by player_id order (deterministic)
        scores = scores.sort_index().sort_values(ascending=ascending, kind="stable").head(n)

        info = self.players.loc[scores.index]
        result = pd.DataFrame({
            "rank": np.arange(1, len(scores) + 1),
            "player_id": scores.index.to_numpy(),
            "player_name": info["player"].to_numpy(),
            "league": info["league"].to_numpy(),
            "club": info["squad"].to_numpy(),
            "position": info["pos_raw"].to_numpy(),
            "position_group": info["pos_group"].to_numpy(),
            "minutes": info["minutes"].to_numpy(),
            ("distance" if metric == "euclidean" else "similarity"): scores.to_numpy(),
        })
        return result

    # -------------------------------------------------------------- diagnostics
    def explain_match(self, player_id: int, match_id: int, top: int = 6) -> pd.DataFrame:
        """Features where two players differ most (z-score units, plus raw per-90 values)."""
        group = self.group_of(player_id)
        if self.group_of(match_id) != group:
            raise ValueError("players are in different position groups")
        feats = self._features[group]
        zq = self._values[group][self._row[group][int(player_id)]]
        zm = self._values[group][self._row[group][int(match_id)]]
        out = pd.DataFrame({"feature": feats, "z_query": zq, "z_match": zm, "z_diff": zm - zq})
        if self.raw is not None:
            out["raw_query"] = self.raw.loc[int(player_id), feats].to_numpy(dtype="float64")
            out["raw_match"] = self.raw.loc[int(match_id), feats].to_numpy(dtype="float64")
        return out.reindex(out["z_diff"].abs().sort_values(ascending=False).index).head(top).reset_index(drop=True)

    def compare_metrics(
        self, player_id: int, n: int = 10, min_candidate_minutes: float | None = DEFAULT_MIN_CANDIDATE_MINUTES
    ) -> dict:
        """Cosine vs Euclidean rankings for one player, without picking a winner."""
        cos = self.scores(player_id, "cosine", min_candidate_minutes)
        euc = self.scores(player_id, "euclidean", min_candidate_minutes)
        cos_rank = cos.rank(ascending=False, method="first")
        euc_rank = euc.rank(ascending=True, method="first")
        top_cos = set(cos_rank.nsmallest(n).index)
        top_euc = set(euc_rank.nsmallest(n).index)
        return {
            "top_cosine": self.find_similar_players(player_id, n=n, metric="cosine", min_candidate_minutes=min_candidate_minutes),
            "top_euclidean": self.find_similar_players(player_id, n=n, metric="euclidean", min_candidate_minutes=min_candidate_minutes),
            "overlap": len(top_cos & top_euc),
            "spearman_all": float(cos_rank.corr(euc_rank, method="pearson")),  # Spearman = Pearson on ranks
            "only_cosine": sorted(top_cos - top_euc),
            "only_euclidean": sorted(top_euc - top_cos),
            "cos_rank_of_euclidean_top": cos_rank.loc[list(top_euc)].astype(int).to_dict(),
            "euc_rank_of_cosine_top": euc_rank.loc[list(top_cos)].astype(int).to_dict(),
        }


_default_engine: SimilarityEngine | None = None


def find_similar_players(
    player_id: int,
    position_group: str | None = None,
    n: int = 10,
    metric: str = "cosine",
    min_candidate_minutes: float | None = DEFAULT_MIN_CANDIDATE_MINUTES,
    leagues: str | Iterable[str] | None = None,
) -> pd.DataFrame:
    """Module-level convenience wrapper that lazily loads the engine from config."""
    global _default_engine
    if _default_engine is None:
        _default_engine = SimilarityEngine.from_config()
    return _default_engine.find_similar_players(
        player_id, position_group=position_group, n=n, metric=metric,
        min_candidate_minutes=min_candidate_minutes, leagues=leagues,
    )
