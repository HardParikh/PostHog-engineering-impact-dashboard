from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Dict
import re

import pandas as pd
import numpy as np

from src.utils import (
    utc_now,
    is_bot_login,
    ensure_datetime_utc_col,
    safe_log1p,
    year_week_key,
    winsorize_series,
)


@dataclass
class TransformedData:
    prs_all: pd.DataFrame
    prs_merged_window: pd.DataFrame
    reviews_window: pd.DataFrame
    reviewer_author_pairs: pd.DataFrame
    pr_reviewer_pairs: pd.DataFrame
    authored_pr_review_flags: pd.DataFrame
    cutoff_ts: pd.Timestamp
    latest_ts: pd.Timestamp


DOMAIN_RULES = {
    "frontend": [r"\bfrontend\b", r"\bui\b", r"\bux\b", r"\breact\b", r"\bstorybook\b", r"\bdesign\b", r"\bchart\b", r"\bsql-editor\b"],
    "backend": [r"\bbackend\b", r"\bapi\b", r"\bserver\b", r"\bnodejs\b", r"\bplugin-server\b", r"\bpython\b"],
    "data": [r"\bdata\b", r"\bdagster\b", r"\bwarehouse\b", r"\bfunnel\b", r"\bllm analytics\b", r"\bmodeling\b", r"\bsql\b"],
    "infra_devex": [r"\bci\b", r"\bdocker\b", r"\bdeps\b", r"\bdevenv\b", r"\binfra\b", r"\bterraform\b", r"\bflox\b", r"\boxlint\b"],
    "docs_product_ops": [r"\bdocs\b", r"\bchangelog\b", r"\breadme\b", r"\bcontributing\b", r"\bworkflow\b", r"\bsurveys\b", r"\bfeature flags?\b"],
    "security": [r"\bsecurity\b", r"\bsemgrep\b", r"\bauth\b", r"\btoken\b", r"\bpermission\b"],
}


def _normalize_text(*parts: List[str]) -> str:
    txt = " ".join([str(p) for p in parts if p is not None])
    return txt.lower()


def infer_domains_from_pr(pr_title: str, labels: str) -> List[str]:
    text = _normalize_text(pr_title, labels.replace("|", " ") if isinstance(labels, str) else "")
    matched = []
    for domain, patterns in DOMAIN_RULES.items():
        if any(re.search(p, text) for p in patterns):
            matched.append(domain)
    if not matched:
        matched.append("other")
    return matched


def load_raw_data(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    prs = pd.read_csv(f"{data_dir}/prs_raw.csv")
    reviews = pd.read_csv(f"{data_dir}/reviews_raw.csv")
    return prs, reviews


def _prepare_prs(prs: pd.DataFrame) -> pd.DataFrame:
    prs = prs.copy()

    for c in ["created_at", "updated_at", "closed_at", "merged_at"]:
        prs = ensure_datetime_utc_col(prs, c)

    for c in ["pr_number", "additions", "deletions", "changed_files", "review_count_reported"]:
        if c in prs.columns:
            prs[c] = pd.to_numeric(prs[c], errors="coerce")

    prs["author_login"] = prs["author_login"].astype(str).replace({"nan": None})
    prs["is_bot_author"] = prs["author_login"].apply(is_bot_login)

    prs["pr_size"] = prs["additions"].fillna(0) + prs["deletions"].fillna(0)
    prs["pr_volume_points_raw"] = prs["pr_size"].apply(safe_log1p)
    prs["pr_volume_points"] = prs["pr_volume_points_raw"].clip(upper=8.0)

    prs["cycle_hours"] = (
        (prs["merged_at"] - prs["created_at"]).dt.total_seconds() / 3600.0
    )
    prs.loc[prs["cycle_hours"] < 0, "cycle_hours"] = np.nan

    prs["merged"] = prs["merged"].astype(str).str.lower().isin(["true", "1", "yes"])
    prs["created_week"] = prs["created_at"].apply(lambda x: year_week_key(x) if pd.notna(x) else "")
    prs["merged_week"] = prs["merged_at"].apply(lambda x: year_week_key(x) if pd.notna(x) else "")

    # Domain inference (multi-label)
    prs["domains"] = prs.apply(
        lambda r: infer_domains_from_pr(r.get("pr_title", ""), r.get("labels", "")),
        axis=1
    )
    prs["domain_primary"] = prs["domains"].apply(lambda x: x[0] if isinstance(x, list) and len(x) else "other")
    prs["domain_count_pr"] = prs["domains"].apply(lambda x: len(set(x)) if isinstance(x, list) else 1)

    # Risk/execution hygiene proxies
    prs["is_large_pr"] = prs["pr_size"].fillna(0) >= 1500
    prs["is_very_large_pr"] = prs["pr_size"].fillna(0) >= 5000

    return prs


def _prepare_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    reviews = reviews.copy()

    for c in ["review_submitted_at", "pr_merged_at", "pr_created_at"]:
        reviews = ensure_datetime_utc_col(reviews, c)

    if "pr_number" in reviews.columns:
        reviews["pr_number"] = pd.to_numeric(reviews["pr_number"], errors="coerce")

    for c in ["reviewer_login", "pr_author_login", "review_state"]:
        if c in reviews.columns:
            reviews[c] = reviews[c].astype(str).replace({"nan": None})

    reviews["is_bot_reviewer"] = reviews["reviewer_login"].apply(is_bot_login)
    reviews["is_bot_pr_author"] = reviews["pr_author_login"].apply(is_bot_login)

    reviews = reviews[reviews["review_submitted_at"].notna()].copy()
    reviews = reviews[reviews["reviewer_login"] != reviews["pr_author_login"]].copy()

    reviews["review_week"] = reviews["review_submitted_at"].apply(
        lambda x: year_week_key(x) if pd.notna(x) else ""
    )
    reviews["pr_merged"] = reviews["pr_merged"].astype(str).str.lower().isin(["true", "1", "yes"])
    reviews["is_approval"] = reviews["review_state"].astype(str).str.upper().eq("APPROVED")
    reviews["is_changes_requested"] = reviews["review_state"].astype(str).str.upper().eq("CHANGES_REQUESTED")

    return reviews


def _add_momentum_period_flags(prs_merged_window: pd.DataFrame, reviews_window: pd.DataFrame, latest_ts: pd.Timestamp):
    """
    Last 30 days vs prior period inside selected window.
    """
    recent_cutoff = latest_ts - pd.Timedelta(days=30)

    prs_merged_window["is_recent_30d"] = prs_merged_window["merged_at"] >= recent_cutoff
    reviews_window["is_recent_30d"] = reviews_window["review_submitted_at"] >= recent_cutoff

    return prs_merged_window, reviews_window


def transform_data(days: int = 90, data_dir: str = "data") -> TransformedData:
    prs_raw, reviews_raw = load_raw_data(data_dir=data_dir)

    prs = _prepare_prs(prs_raw)
    reviews = _prepare_reviews(reviews_raw)

    latest_ts = pd.Timestamp(utc_now())
    cutoff = latest_ts - pd.Timedelta(days=days)

    prs_merged_window = prs[
        (prs["merged"] == True)
        & (prs["merged_at"].notna())
        & (prs["merged_at"] >= cutoff)
        & (~prs["is_bot_author"])
    ].copy()

    prs_merged_window["cycle_hours_winsor"] = winsorize_series(
        prs_merged_window["cycle_hours"], lower_q=0.05, upper_q=0.95
    )

    reviews_window = reviews[
        (reviews["review_submitted_at"] >= cutoff)
        & (~reviews["is_bot_reviewer"])
        & (~reviews["is_bot_pr_author"])
    ].copy()

    prs_merged_window, reviews_window = _add_momentum_period_flags(prs_merged_window, reviews_window, latest_ts)

    reviewer_author_pairs = reviews_window[
        reviews_window["reviewer_login"].notna() & reviews_window["pr_author_login"].notna()
    ][["reviewer_login", "pr_author_login"]].drop_duplicates()

    pr_reviewer_pairs = reviews_window[
        reviews_window["pr_author_login"].notna() & reviews_window["reviewer_login"].notna()
    ][["pr_author_login", "reviewer_login"]].drop_duplicates()

    # PR-level review coverage flags for authored PRs
    reviewed_prs = (
        reviews_window[["pr_number"]]
        .dropna()
        .drop_duplicates()
        .assign(has_review_in_window=True)
    )

    authored_pr_review_flags = (
        prs_merged_window[["pr_number", "author_login", "merged_at", "domain_primary", "is_recent_30d"]]
        .drop_duplicates()
        .merge(reviewed_prs, on="pr_number", how="left")
    )
    authored_pr_review_flags["has_review_in_window"] = authored_pr_review_flags["has_review_in_window"].fillna(False)

    return TransformedData(
        prs_all=prs,
        prs_merged_window=prs_merged_window,
        reviews_window=reviews_window,
        reviewer_author_pairs=reviewer_author_pairs,
        pr_reviewer_pairs=pr_reviewer_pairs,
        authored_pr_review_flags=authored_pr_review_flags,
        cutoff_ts=cutoff,
        latest_ts=latest_ts,
    )


if __name__ == "__main__":
    td = transform_data(days=90, data_dir="data")
    print("Merged PRs in window:", len(td.prs_merged_window))
    print("Reviews in window:", len(td.reviews_window))
    print("Cutoff:", td.cutoff_ts)