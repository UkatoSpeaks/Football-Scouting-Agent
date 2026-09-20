"""Player photos and avatars.

Photos are optional presentation metadata. They come from ``data/metadata/player_photos.csv``
(columns ``player_id, player_name, photo_url`` and an optional ``photo_credit``), keyed by
``player_id`` (never by name). Nothing in the project's statistics data contains photo URLs,
so URLs are never guessed: a player without an entry gets a consistent initials avatar.

Because ``player_id`` is the row id of the cleaned table, every entry also carries the
player's name and is dropped if it no longer matches (for example after the data changes).
"""
from __future__ import annotations

import html
import re
from pathlib import Path
from urllib.parse import quote, urlparse

import pandas as pd

PHOTO_COLUMNS = ["player_id", "player_name", "photo_url"]
AVATAR_BG = "#16263d"
AVATAR_FG = "#3ddc97"


def initials(name: str | None) -> str:
    """'Vinícius Júnior' -> 'VJ'; a single name such as 'Alisson' -> 'AL'; unknown -> '?'."""
    parts = re.findall(r"[^\W\d_]+", name or "", flags=re.UNICODE)
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def _valid_url(url: object) -> bool:
    if not isinstance(url, str) or not url.strip() or len(url) > 2000:
        return False
    parsed = urlparse(url.strip())
    return parsed.scheme in ("http", "https") and bool(parsed.netloc) and not any(c in url for c in '"\'<>\\ \n\r\t')


def load_photo_map(players: pd.DataFrame, path: Path | None) -> tuple[dict[int, str], list[str]]:
    """Return ({player_id: photo_url}, problems).

    ``players`` is indexed by player_id and has a ``player`` column. Rows with an unknown id,
    a name that does not match that id, or an unsafe/invalid URL are skipped and described
    in ``problems``; a missing file simply means "no photos".
    """
    if path is None or not Path(path).exists():
        return {}, []
    try:
        frame = pd.read_csv(path, dtype={"photo_url": "string", "player_name": "string"})
    except Exception as exc:  # unreadable/empty file must not break the app
        return {}, [f"could not read {path}: {exc}"]
    missing = [c for c in PHOTO_COLUMNS if c not in frame.columns]
    if missing:
        return {}, [f"{path} is missing columns {missing}"]

    photos: dict[int, str] = {}
    problems: list[str] = []
    for row in frame.itertuples(index=False):
        try:
            pid = int(row.player_id)
        except (TypeError, ValueError):
            problems.append(f"invalid player_id {row.player_id!r}")
            continue
        if pid not in players.index:
            problems.append(f"player_id {pid} is not in the modelled pool")
        elif str(row.player_name).strip() != players.loc[pid, "player"]:
            problems.append(f"player_id {pid}: name {row.player_name!r} does not match {players.loc[pid, 'player']!r}")
        elif not _valid_url(row.photo_url):
            problems.append(f"player_id {pid}: invalid photo_url")
        else:
            photos[pid] = str(row.photo_url).strip()
    return photos, problems


def _placeholder_uri(name: str | None) -> str:
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
        f"<rect width='100' height='100' fill='{AVATAR_BG}'/>"
        f"<text x='50' y='50' fill='{AVATAR_FG}' font-family='Arial,Helvetica,sans-serif' font-size='38' "
        f"font-weight='700' text-anchor='middle' dominant-baseline='central'>{html.escape(initials(name))}</text></svg>"
    )
    return "data:image/svg+xml;utf8," + quote(svg, safe="")


def avatar_html(name: str | None, photo_url: str | None = None, size: int = 96, css_class: str = "") -> str:
    """A square avatar of a fixed size.

    The photo is the top background layer and the initials placeholder sits underneath, so a
    missing or broken image still shows the placeholder (no JavaScript needed). ``name`` is
    HTML-escaped; ``photo_url`` must already have passed ``load_photo_map`` validation.
    """
    layers = []
    if photo_url and _valid_url(photo_url):
        # single quotes: the whole value sits inside a double-quoted style="..." attribute
        layers.append(f"url('{quote(photo_url.strip(), safe=':/?&=%#@!$*+,;~-._()')}')")
    layers.append(f"url('{_placeholder_uri(name)}')")
    label = html.escape(name or "Player", quote=True)
    return (
        f'<div class="sc-avatar {css_class}" role="img" aria-label="{label}" '
        f'style="width:{int(size)}px;height:{int(size)}px;background-image:{",".join(layers)};"></div>'
    )
