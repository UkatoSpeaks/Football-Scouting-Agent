"""Phase 1 UI tests: photos/avatars, profile hero, landing search, sidebar sections and reset."""
import re
from pathlib import Path

import pandas as pd
import pytest

from scouting.config import load_config
from scouting.dashboard import components, logic
from scouting.dashboard.photos import avatar_html, initials, load_photo_map

CONFIG = load_config()
READY = (CONFIG.models_dir / "kmeans.joblib").exists() and (CONFIG.processed_dir / "cluster_assignments.csv").exists()
real = pytest.mark.skipif(not READY, reason="run the pipeline first")
APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _players():
    return pd.DataFrame({"player": ["Bukayo Saka", "Alisson", "Vinícius Júnior"]}, index=pd.Index([10, 20, 30], name="player_id"))


def _csv(tmp_path, rows, header="player_id,player_name,photo_url\n"):
    path = tmp_path / "photos.csv"
    path.write_text(header + "".join(rows), encoding="utf-8")
    return path


# --------------------------------------------------------------------- initials
@pytest.mark.parametrize("name, expected", [
    ("Bukayo Saka", "BS"), ("Vinícius Júnior", "VJ"), ("Alisson", "AL"), ("Kylian Mbappé", "KM"),
    ("Trent Alexander-Arnold", "TA"), ("", "?"), (None, "?"), ("123", "?"),
])
def test_initials(name, expected):
    assert initials(name) == expected


# ------------------------------------------------------------------- photo map
def test_photo_map_accepts_valid_rows_keyed_by_player_id(tmp_path):
    path = _csv(tmp_path, ["10,Bukayo Saka,https://img.example.com/saka.jpg\n", "20,Alisson,http://img.example.com/a.png\n"])
    photos, problems = load_photo_map(_players(), path)
    assert photos == {10: "https://img.example.com/saka.jpg", 20: "http://img.example.com/a.png"} and problems == []


def test_photo_map_rejects_stale_ids_unknown_ids_and_unsafe_urls(tmp_path):
    path = _csv(tmp_path, [
        "10,Someone Else,https://img.example.com/x.jpg\n",              # id no longer belongs to that name
        "999,Bukayo Saka,https://img.example.com/x.jpg\n",              # unknown id
        "20,Alisson,javascript:alert(1)\n",                             # not http(s)
        '30,Vinícius Júnior,"https://img.example.com/a b.jpg"\n',       # whitespace
        '30,Vinícius Júnior,"https://x.com/a.jpg\'onload=1"\n',         # quote characters
        "abc,Bukayo Saka,https://img.example.com/x.jpg\n",              # id not a number
        "20,Alisson,\n",                                                # empty url
    ])
    photos, problems = load_photo_map(_players(), path)
    assert photos == {} and len(problems) == 7


def test_photo_map_missing_or_broken_files_mean_no_photos(tmp_path):
    assert load_photo_map(_players(), None) == ({}, [])
    assert load_photo_map(_players(), tmp_path / "nope.csv") == ({}, [])
    photos, problems = load_photo_map(_players(), _csv(tmp_path, [], header="a,b\n"))
    assert photos == {} and "missing columns" in problems[0]
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    assert load_photo_map(_players(), empty)[0] == {}


def test_repository_photo_template_is_valid_and_has_no_invented_urls():
    path = CONFIG.metadata_dir / "player_photos.csv"
    assert path.exists()
    frame = pd.read_csv(path)
    assert list(frame.columns)[:3] == ["player_id", "player_name", "photo_url"]
    if len(frame):    # any entry that is present must be a real http(s) URL
        assert frame["photo_url"].str.startswith(("http://", "https://")).all()


# ---------------------------------------------------------------------- avatar
def _attrs(html: str) -> dict:
    """Parse the outermost tag the way a browser does (so quoting mistakes show up)."""
    from html.parser import HTMLParser

    class First(HTMLParser):
        attrs = None

        def handle_starttag(self, tag, attrs):
            if self.attrs is None:
                self.attrs = dict(attrs)

    parser = First()
    parser.feed(html)
    return parser.attrs


def test_avatar_photo_layer_sits_above_the_placeholder():
    html = avatar_html("Bukayo Saka", "https://img.example.com/saka.jpg", size=96)
    attrs = _attrs(html)
    style = attrs["style"]                                    # would be truncated if the quoting were wrong
    assert style.startswith("width:96px;height:96px;background-image:url('https://img.example.com/saka.jpg')")
    assert style.index("saka.jpg") < style.index("data:image/svg+xml") and style.endswith(";")
    assert attrs["aria-label"] == "Bukayo Saka" and attrs["role"] == "img"


def test_avatar_without_photo_or_with_bad_url_is_the_placeholder_only():
    for url in (None, "", "javascript:alert(1)", 'https://x.com/"onerror=1', "https://x.com/'onerror=1"):
        style = _attrs(avatar_html("Alisson", url))["style"]
        assert style.count("url(") == 1 and "data:image/svg+xml" in style
        assert style.startswith("width:96px;height:96px;background-image:url('data:image/svg+xml") and style.endswith("');")
    assert "%3EAL%3C" in avatar_html("Alisson").upper()      # the initials are inside the SVG


def test_avatar_escapes_the_name_and_cannot_break_out_of_the_style_attribute():
    html = avatar_html('<script>alert(1)</script> "x"', "https://img.example.com/a.jpg")
    assert "<script>" not in html
    attrs = _attrs(html)
    assert attrs["aria-label"] == '<script>alert(1)</script> "x"'          # escaped in markup, intact once parsed
    assert set(attrs) == {"class", "role", "aria-label", "style"}          # nothing injected as an extra attribute
    assert "<" not in attrs["style"] and '"' not in attrs["style"]


# ------------------------------------------------------------------ hero HTML
def _overview(**over):
    o = {"player_id": 1, "name": "Test Player", "club": "Test FC", "league": "Premier League", "position": "FW,MF",
         "position_group": "FWD", "minutes": 1729, "matches": 25, "age": 22, "nation": "ENG", "season": "2024-2025",
         "cluster_id": 4, "cluster_key": "FWD-4"}
    return {**o, **over}


def test_hero_shows_identity_cluster_and_chips():
    html = components.profile_hero_html(_overview(), "Higher than the position average: xAG", [("Minutes", "1,729"), ("Age", "22")])
    for text in ("Test Player", "Test FC", "Premier League", "FWD · FW,MF", ">FWD-4<", "Higher than the position average: xAG",
                 "<b>1,729</b><span>Minutes</span>", "<b>22</b><span>Age</span>"):
        assert text in html


def test_hero_escapes_every_player_derived_value():
    html = components.profile_hero_html(
        _overview(name="<b>X</b>", club="A&B <i>", league="<img src=x>", position="<u>"), "<script>1</script>", [("<k>", "<v>")])
    assert "<b>X</b>" not in html and "<script>" not in html and "<img src=x>" not in html and "<k>" not in html
    assert "&lt;b&gt;X&lt;/b&gt;" in html and "A&amp;B" in html


def test_hero_without_a_headline_or_position_still_renders():
    html = components.profile_hero_html(_overview(position=None), "", [])
    assert "FWD" in html and "None" not in html and "sc-headline" not in html


def test_landing_stats_html_uses_the_given_numbers():
    html = components.landing_stats_html(1959, 5, 4)
    assert "<b>1,959</b><span>Players</span>" in html and "<b>5</b><span>Leagues</span>" in html
    assert "<b>4</b><span>Position groups</span>" in html


# -------------------------------------------------------------- logic helpers
def _info(high, low):
    return logic.ClusterInfo("FWD-4", "FWD", 4, 30, 0.07, high, low)


def test_cluster_headline_is_built_from_the_traits():
    assert logic.cluster_headline(_info([("xAG", 2.1), ("key passes", 1.6), ("crossing", 1.0), ("shots", 0.9)], [])) == \
        "Higher than the position average: xAG, key passes, crossing"
    assert logic.cluster_headline(_info([], [("passing volume", -0.6)])) == "Lower than the position average: passing volume"
    assert "Close to the position average" in logic.cluster_headline(_info([], []))


@real
def test_hero_chips_are_position_aware_and_never_contain_none(data):
    saka = logic.overview(data, data.engine.find_id("Bukayo Saka"))
    chips = dict(logic.hero_chips(data, saka))
    assert chips["Minutes"] == "1,729" and {"Goals", "Assists", "Age", "Matches", "Nation"} <= set(chips)
    alisson = logic.overview(data, data.engine.find_id("Alisson"))
    assert "Goals" not in dict(logic.hero_chips(data, alisson))         # goalkeepers: no goals/assists chips
    bare = dict(saka, matches=None, age=None, nation=None)
    assert [k for k, _ in logic.hero_chips(data, bare)] == ["Minutes", "Goals", "Assists"]


# ------------------------------------------------------------------- UI runs
@pytest.fixture(scope="module")
def data():
    from scouting.dashboard.views import get_data

    return get_data()


def _app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(APP, default_timeout=120)


def _hero(at):
    return " ".join(m.value for m in at.markdown if 'class="sc-hero"' in m.value)


def _similar_card_count(at) -> int:
    """Phase 2 replaced the similar-player dataframe with cards; count those instead."""
    return len([m for m in at.markdown if 'class="sc-sim-card"' in m.value])


@real
def test_sidebar_is_organised_into_sections_with_the_documented_defaults(data):
    at = _app().run()
    labels = [re.search(r">([A-Z]+)<", m.value).group(1) for m in at.sidebar.markdown if 'class="sb-label"' in m.value]
    assert labels == ["PLAYER", "PROFILE", "SCOUTING", "ACTIONS", "SHORTLIST"]
    assert at.sidebar.slider(key="min_minutes").value == 900 and at.sidebar.slider(key="n_similar").value == 10
    assert at.sidebar.selectbox(key="league").value == "All Leagues" and at.sidebar.selectbox(key="position").value == "All"
    assert [b.label for b in at.sidebar.button] == ["Reset filters", "View shortlist"]


@real
def test_landing_search_selects_the_player_and_opens_the_profile(data):
    at = _app().run()
    pid = data.engine.find_id("Erling Haaland")
    at.selectbox(key="landing_player").set_value(pid).run()
    assert not at.exception
    assert at.sidebar.selectbox(key="player_id").value == pid          # sidebar stays in sync
    assert "Erling Haaland" in _hero(at) and len(at.dataframe) >= 1
    assert len(at.title) == 0                                           # landing hero replaced by the profile


@real
def test_reset_filters_restores_defaults_but_keeps_the_player(data):
    at = _app().run()
    at.sidebar.selectbox(key="player_id").set_value(data.engine.find_id("Bukayo Saka")).run()
    at.sidebar.selectbox(key="league").set_value("La Liga").run()
    at.sidebar.selectbox(key="position").set_value("FWD").run()
    at.sidebar.slider(key="min_minutes").set_value(2000).run()
    at.sidebar.slider(key="n_similar").set_value(15).run()
    assert at.sidebar.selectbox(key="league").value == "La Liga" and _similar_card_count(at) == 15
    at.sidebar.button(key="reset_filters").click().run()
    assert not at.exception
    assert at.sidebar.selectbox(key="league").value == "All Leagues" and at.sidebar.selectbox(key="position").value == "All"
    assert at.sidebar.slider(key="min_minutes").value == 900 and at.sidebar.slider(key="n_similar").value == 10
    assert at.sidebar.selectbox(key="player_id").value == data.engine.find_id("Bukayo Saka")
    assert _similar_card_count(at) == 10


@real
def test_profile_uses_the_photo_when_one_exists_and_the_placeholder_otherwise(data):
    saka = data.engine.find_id("Bukayo Saka")
    url = "https://img.example.com/players/saka.jpg"
    at = _app().run()
    at.sidebar.selectbox(key="player_id").set_value(saka).run()
    without = _hero(at)
    assert "img.example.com" not in without and "data:image/svg+xml" in without            # placeholder avatar
    data.photos[saka] = url
    try:
        at = _app().run()
        at.sidebar.selectbox(key="player_id").set_value(saka).run()
        with_photo = _hero(at)
    finally:
        data.photos.pop(saka, None)
    assert url in with_photo and with_photo.index(url) < with_photo.index("data:image/svg+xml")


@real
def test_photos_are_keyed_by_player_id_not_by_name(data):
    """Two different players with the same name must not share a photo."""
    dup = data.players[data.players["player"].duplicated(keep=False)]
    if dup.empty:
        pytest.skip("no duplicate names among modelled players")
    name = dup["player"].iloc[0]
    ids = [int(i) for i in dup.index[dup["player"] == name]]
    data.photos[ids[0]] = "https://img.example.com/first.jpg"
    try:
        at = _app().run()
        at.sidebar.selectbox(key="player_id").set_value(ids[1]).run()
        assert "first.jpg" not in _hero(at)
    finally:
        data.photos.pop(ids[0], None)


@real
def test_goalkeeper_and_outfield_headers_differ_in_key_facts(data):
    at = _app().run()
    at.sidebar.selectbox(key="player_id").set_value(data.engine.find_id("Alisson")).run()
    gk = _hero(at)
    assert ">GK-" in gk and "<span>Goals</span>" not in gk and "<span>Minutes</span>" in gk
    at.sidebar.selectbox(key="player_id").set_value(data.engine.find_id("Bukayo Saka")).run()
    assert "<span>Goals</span>" in _hero(at)
