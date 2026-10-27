"""
mdcspc.dates

One shared, UK-friendly rule for turning user-supplied date values into
timestamps, so input data, phase config, target config and outputs all agree.

Rule:
- ISO-looking values (YYYY-MM-DD, optionally with a time) are parsed as ISO.
- Everything else is parsed day-first (dd/mm/yyyy).

Both halves matter: pandas' default is month-first, and applying
dayfirst=True to ISO strings swaps day and month on pandas 3.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

ISO_DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?$"


def parse_uk_dates(values: Any) -> pd.Series:
    """
    Parse date values using the ISO-or-day-first rule, normalised to midnight.

    Accepts a Series or list-like of strings, dates or timestamps. Missing or
    blank values become NaT. Unparseable values raise ValueError.
    """
    series = values if isinstance(values, pd.Series) else pd.Series(values)

    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.normalize()

    text = series.astype(str).str.strip()
    missing = series.isna() | text.eq("")
    iso_mask = ~missing & text.str.match(ISO_DATE_PATTERN)
    other_mask = ~missing & ~iso_mask

    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if iso_mask.any():
        parsed.loc[iso_mask] = pd.to_datetime(text.loc[iso_mask], format="ISO8601")
    if other_mask.any():
        parsed.loc[other_mask] = pd.to_datetime(text.loc[other_mask], dayfirst=True)

    return parsed.dt.normalize()
