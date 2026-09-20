"""App-wide styling: one stylesheet injected once from ``app.py``.

Design tokens: dark navy / charcoal for the sidebar and hero panels, near-white cards, one
subtle green accent. Styling hooks are Streamlit's documented ``st-key-<key>`` container
classes and our own ``sc-*`` classes, so nothing depends on Streamlit's internal markup.
Chart colours live in ``scouting.palette`` and are unchanged.
"""
from __future__ import annotations

import streamlit as st

NAVY = "#0f1b2d"
NAVY_2 = "#16263d"
INK = "#0f1b2d"
MUTED = "#5b6b80"
LINE = "#e3e8ef"
PAGE = "#f4f6f9"
CARD = "#ffffff"
ACCENT = "#1f9d68"          # green on light surfaces
ACCENT_BRIGHT = "#3ddc97"   # green on navy

CSS = f"""
<style>
:root {{
  --sc-navy:{NAVY}; --sc-navy-2:{NAVY_2}; --sc-ink:{INK}; --sc-muted:{MUTED}; --sc-line:{LINE};
  --sc-card:{CARD}; --sc-accent:{ACCENT}; --sc-accent-bright:{ACCENT_BRIGHT};
}}

/* ---- page frame ---- */
[data-testid="stMainBlockContainer"] {{ padding-top: 4.6rem; max-width: 1180px; }}
h1, h2, h3, h4 {{ color: var(--sc-ink); letter-spacing: -0.01em; }}
h2 {{ font-weight: 800; }}
h3 {{ font-weight: 700; }}

/* ---- page kicker ---- */
.sc-kicker {{ font-size: 0.72rem; font-weight: 800; letter-spacing: 0.16em; color: var(--sc-accent); margin-bottom: 0.2rem; }}

/* ---- sidebar ---- */
.sb-label {{
  font-size: 0.68rem; font-weight: 800; letter-spacing: 0.14em; color: #7f93ad;
  margin: 1.3rem 0 0.15rem; padding-bottom: 0.3rem; border-bottom: 1px solid rgba(255,255,255,0.08);
}}
.sb-label:first-child {{ margin-top: 0.4rem; }}

/* ---- landing hero (a keyed container holding native widgets on navy) ---- */
.st-key-landing_hero {{
  background: var(--sc-navy); border-radius: 14px; padding: 2.4rem 2.6rem 2rem; margin-bottom: 1rem;
  border-left: 4px solid var(--sc-accent-bright);
}}
.st-key-landing_hero h1 {{ color: #ffffff; font-size: 2.5rem; font-weight: 800; padding: 0 0 0.3rem; line-height: 1.15; }}
.st-key-landing_hero p {{ color: #b9c6d8; font-size: 1.1rem; }}
.st-key-landing_hero [data-testid="stSelectbox"] {{ max-width: 560px; margin-top: 0.8rem; }}
.sc-hint {{ color: #8fa2bb; font-size: 0.85rem; margin-top: 0.35rem; }}
.sc-stats {{ display: flex; flex-wrap: wrap; gap: 1.2rem 3rem; margin-top: 1.8rem; }}
.sc-stat b {{ display: block; color: #ffffff; font-size: 2rem; font-weight: 800; line-height: 1.1; }}
.sc-stat span {{ color: #8fa2bb; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; }}

/* ---- player profile hero ---- */
.sc-hero {{
  display: flex; flex-wrap: wrap; align-items: center; gap: 1.6rem;
  background: var(--sc-navy); color: #ffffff; border-radius: 14px; padding: 1.5rem 1.8rem;
  border-left: 4px solid var(--sc-accent-bright); margin-bottom: 0.4rem;
}}
.sc-avatar {{
  flex: 0 0 auto; border-radius: 14px; background-color: {NAVY_2}; background-size: cover;
  background-position: center 20%; border: 2px solid rgba(61,220,151,0.45);
}}
.sc-hero-main {{ flex: 1 1 320px; min-width: 0; }}
.sc-name {{ font-size: 2.1rem; font-weight: 800; letter-spacing: 0.03em; text-transform: uppercase; line-height: 1.1; color: #ffffff; }}
.sc-club {{ font-size: 1.1rem; font-weight: 600; color: #e6ecf5; margin-top: 0.35rem; }}
.sc-sub {{ font-size: 0.95rem; color: #9fb0c6; margin-top: 0.1rem; }}
.sc-style {{ margin-top: 0.9rem; display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem 0.8rem; }}
.sc-badge {{
  display: inline-block; background: rgba(61,220,151,0.14); color: var(--sc-accent-bright);
  border: 1px solid rgba(61,220,151,0.5); border-radius: 6px; padding: 0.1rem 0.6rem;
  font-weight: 800; letter-spacing: 0.06em; font-size: 0.95rem;
}}
.sc-headline {{ color: #c6d2e2; font-size: 0.92rem; }}
.sc-chips {{ display: flex; flex-wrap: wrap; gap: 0.9rem 1.8rem; flex: 0 1 auto; }}
.sc-chip b {{ display: block; color: #ffffff; font-size: 1.35rem; font-weight: 800; line-height: 1.15; }}
.sc-chip span {{ color: #8fa2bb; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; }}

/* ---- generic cards / widgets ---- */
[data-testid="stMetric"] {{ background: var(--sc-card); }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 10px; border-color: var(--sc-line); }}
[data-testid="stDataFrame"] {{ border-radius: 10px; }}

/* ---- a badge for use on white/card surfaces (sc-badge is for the navy hero) ---- */
.sc-badge-light {{
  display: inline-block; background: rgba(31,157,104,0.1); color: var(--sc-accent);
  border: 1px solid rgba(31,157,104,0.35); border-radius: 6px; padding: 0.08rem 0.55rem;
  font-weight: 800; letter-spacing: 0.04em; font-size: 0.72rem; margin-top: 0.15rem;
}}

/* ---- statistic cards (Player statistics) ---- */
.sc-stat-card {{ display: flex; flex-direction: column; gap: 0.3rem; padding: 0.15rem 0; }}
.sc-stat-name {{ font-size: 0.78rem; color: var(--sc-muted); font-weight: 600; line-height: 1.25; }}
.sc-stat-value {{ font-size: 1.5rem; font-weight: 800; color: var(--sc-ink); line-height: 1.1; font-variant-numeric: tabular-nums; }}
.sc-bar {{ height: 6px; border-radius: 4px; background: var(--sc-line); overflow: hidden; }}
.sc-bar-fill {{ height: 100%; background: var(--sc-accent); border-radius: 4px; }}
.sc-stat-pct {{ font-size: 0.68rem; color: var(--sc-muted); font-weight: 700; }}

/* ---- similar-player scouting cards ---- */
.sc-sim-card {{ display: flex; flex-direction: column; gap: 0.3rem; }}
.sc-sim-top {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 0.6rem; }}
.sc-sim-score {{ text-align: right; flex: 0 0 auto; }}
.sc-sim-score b {{ display: block; font-size: 1.3rem; font-weight: 800; color: var(--sc-ink); font-variant-numeric: tabular-nums; }}
.sc-sim-score span {{ font-size: 0.62rem; color: var(--sc-muted); text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; }}
.sc-sim-name {{ font-size: 1.05rem; font-weight: 800; color: var(--sc-ink); margin-top: 0.15rem; }}
.sc-sim-club {{ font-size: 0.85rem; color: var(--sc-ink); font-weight: 600; }}
.sc-sim-meta {{ font-size: 0.78rem; color: var(--sc-muted); }}

/* ---- playing style identity (Phase 3) ---- */
.sc-badge-lg {{ font-size: 0.95rem; padding: 0.2rem 0.75rem; }}
.sc-style-id {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem 0.9rem; margin-bottom: 0.6rem; }}
.sc-style-size {{ font-size: 0.85rem; color: var(--sc-muted); font-weight: 600; }}
.sc-style-cats {{ flex-basis: 100%; font-size: 0.88rem; color: var(--sc-ink); }}

/* ---- trait list (Playing style high/low, percentile of the player) ---- */
.sc-trait-list {{ border-top: 1px solid var(--sc-line); margin-top: 0.3rem; }}
.sc-trait-row {{
  display: flex; justify-content: space-between; align-items: baseline; gap: 0.6rem;
  padding: 0.45rem 0; border-bottom: 1px solid var(--sc-line);
}}
.sc-trait-name {{ font-size: 0.86rem; color: var(--sc-ink); }}
.sc-trait-pct {{ font-size: 0.86rem; font-weight: 800; color: var(--sc-accent); font-variant-numeric: tabular-nums; flex: 0 0 auto; }}
.sc-trait-empty {{ font-size: 0.82rem; color: var(--sc-muted); padding: 0.5rem 0; border-top: none; }}

/* ---- compact single-player statistical profile bars ---- */
.sc-profile-bars {{ display: flex; flex-direction: column; gap: 0.5rem; margin-top: 0.4rem; }}
.sc-profile-row {{ display: flex; align-items: center; gap: 0.7rem; }}
.sc-profile-label {{ flex: 0 0 150px; font-size: 0.82rem; color: var(--sc-ink); font-weight: 600; }}
.sc-profile-bar {{ flex: 1 1 auto; }}
.sc-profile-pct {{ flex: 0 0 38px; text-align: right; font-size: 0.78rem; color: var(--sc-muted); font-variant-numeric: tabular-nums; }}

/* ---- player style map legend ---- */
.sc-legend {{ display: flex; flex-wrap: wrap; gap: 0.4rem 1.3rem; margin: 0.2rem 0 0.6rem; }}
.sc-legend-item {{ display: inline-flex; align-items: center; gap: 0.45rem; font-size: 0.8rem; color: var(--sc-muted); font-weight: 600; }}
.sc-legend-dot, .sc-legend-star, .sc-legend-ring {{ display: inline-block; width: 11px; height: 11px; flex: 0 0 auto; }}
.sc-legend-dot {{ border-radius: 50%; background: var(--sc-accent); }}
.sc-legend-star {{ background: var(--sc-ink); clip-path: polygon(50% 0%,61% 35%,98% 35%,68% 57%,79% 91%,50% 70%,21% 91%,32% 57%,2% 35%,39% 35%); }}
.sc-legend-ring {{ border-radius: 50%; border: 2px solid var(--sc-ink); }}

/* ---- comparison header ("A vs B") ---- */
.sc-vs {{ display: flex; align-items: center; justify-content: center; gap: 1.6rem; padding: 0.6rem 0 1rem; flex-wrap: wrap; }}
.sc-vs-player {{ display: flex; flex-direction: column; align-items: center; text-align: center; gap: 0.25rem; flex: 0 1 180px; }}
.sc-vs-name {{ font-size: 1.05rem; font-weight: 800; color: var(--sc-ink); margin-top: 0.3rem; }}
.sc-vs-meta {{ font-size: 0.8rem; color: var(--sc-muted); }}
.sc-vs-mid {{ font-size: 0.85rem; font-weight: 800; letter-spacing: 0.1em; color: var(--sc-accent); flex: 0 0 auto; }}

@media (max-width: 640px) {{
  [data-testid="stMainBlockContainer"] {{ padding-top: 4rem; }}
  .st-key-landing_hero {{ padding: 1.5rem 1.2rem; }}
  .st-key-landing_hero h1 {{ font-size: 1.8rem; }}
  .sc-hero {{ padding: 1.1rem 1.1rem; gap: 1rem; }}
  .sc-name {{ font-size: 1.5rem; }}
  .sc-stats {{ gap: 0.8rem 1.5rem; }}
  .sc-stat b {{ font-size: 1.4rem; }}
  .sc-stat span {{ font-size: 0.6rem; letter-spacing: 0.08em; }}
}}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
