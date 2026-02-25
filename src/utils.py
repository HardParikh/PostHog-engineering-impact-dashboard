from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Optional
import math
import pandas as pd
import numpy as np


KNOWN_BOTS = {
    "github-actions",
    "github-actions[bot]",
    "posthog-bot",
    "dependabot[bot]",
    "renovate[bot]",
    "codecov[bot]",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_github_dt(value: Optional[str]) -> Optional[datetime]:
    """Parse GitHub GraphQL datetime string like '2026-02-24T12:34:56Z'."""
    if value is None or value == "":
        return None
    try:
        # GitHub uses Z suffix
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def is_bot_login(login: Optional[str]) -> bool:
    if not login:
        return True
    l = login.strip().lower()
    return ("bot" in l) or (l in KNOWN_BOTS)


def year_week_key(dt: pd.Timestamp) -> str:
    if pd.isna(dt):
        return ""
    iso = dt.isocalendar()
    return f"{int(iso.year)}-W{int(iso.week):02d}"


def safe_log1p(x: float) -> float:
    try:
        return math.log1p(max(float(x), 0.0))
    except Exception:
        return 0.0


def winsorize_series(s: pd.Series, lower_q: float = 0.05, upper_q: float = 0.95) -> pd.Series:
    """Clip outliers for more stable scoring."""
    if s.empty:
        return s
    s_num = pd.to_numeric(s, errors="coerce")
    lo = s_num.quantile(lower_q)
    hi = s_num.quantile(upper_q)
    return s_num.clip(lower=lo, upper=hi)


def percentile_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """
    Percentile rank in [0,1]. If all values identical, return 0.5.
    NaNs become 0.
    """
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    if len(s) == 0:
        return s

    if s.nunique(dropna=True) <= 1:
        return pd.Series([0.5] * len(s), index=s.index, dtype=float)

    # rank pct in (0,1]; shift to [0,1]
    pct = s.rank(method="average", pct=True)
    pct = (pct - pct.min()) / (pct.max() - pct.min() + 1e-9)

    if not higher_is_better:
        pct = 1 - pct

    return pct.clip(0, 1)


def coalesce_str(val: Optional[str], default: str = "") -> str:
    if val is None:
        return default
    return str(val)


def ensure_datetime_utc_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df