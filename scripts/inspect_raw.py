"""Fetch one FBref page and dump its raw structure for manual inspection.

Run before writing any table-parsing/cleaning code, since exact table ids,
header labels, and HTML quirks must be confirmed against real data.
"""
from __future__ import annotations

import re
import sys
from io import StringIO

import pandas as pd

from scouting.config import load_config
from scouting.scrape.fbref_client import fetch

STANDARD_STATS_URL = "https://fbref.com/en/comps/9/stats/Premier-League-Stats"


def strip_comments(html: str) -> str:
    return re.sub(r"<!--|-->", "", html)


def main() -> None:
    config = load_config()
    html = fetch(STANDARD_STATS_URL, config)
    print(f"raw HTML length: {len(html)}")

    table_ids = sorted(set(re.findall(r'<table[^>]*id="([^"]+)"', html)))
    print(f"table ids found (raw, no comment-stripping): {table_ids}")

    uncommented = strip_comments(html)
    table_ids_uncommented = sorted(set(re.findall(r'<table[^>]*id="([^"]+)"', uncommented)))
    print(f"table ids found (after stripping comments): {table_ids_uncommented}")

    target_id = "stats_standard"
    if target_id not in table_ids_uncommented:
        print(f"WARNING: expected table id '{target_id}' not found; inspect table_ids list above.")
        sys.exit(1)

    tables = pd.read_html(StringIO(uncommented), attrs={"id": target_id})
    df = tables[0]
    print(f"\nparsed shape: {df.shape}")
    print(f"column type: {type(df.columns)}")
    print("\ncolumns:")
    for col in df.columns:
        print(f"  {col!r}")
    print("\nhead(15):")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.head(15))


if __name__ == "__main__":
    main()
