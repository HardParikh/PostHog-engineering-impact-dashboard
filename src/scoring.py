from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import numpy as np

from src.transform import TransformedData
from src.utils import percentile_rank


WEIGHTS = {
    # Shipping depth (30%)
    "norm_merged_pr_count": 0.14,
    "norm_merged_pr_volume_points": 0.12,
    "norm_domain_coverage_count": 0.04,

    # Review leverage (25%)
    "norm_reviews_submitted": 0.10,
    "norm_approvals_given": 0.07,
    "norm_unique_authors_reviewed": 0.08,

    # Execution quality / flow (20%)
    "norm_cycle_time_inverse": 0.08,
    "norm_authored_pr_review_coverage_pct": 0.06,
    "norm_large_pr_share_inverse": 0.06,

    # Consistency + collaboration (15%)
    "norm_weeks_active": 0.08,
    "norm_unique_collaborators": 0.07,

    # Momentum (10%)
    "norm_momentum_score": 0.10,
}


@dataclass
class ScoringOutput:
    engineer_metrics: pd.DataFrame
    top5: pd.DataFrame
    weekly_activity: pd.DataFrame


def _authored_metrics(td: TransformedData) -> pd.DataFrame:
    prs = td.prs_merged_window
    if prs.empty:
        return pd.DataFrame(columns=[
            "engineer", "merged_pr_count", "merged_pr_volume_points", "median_pr_cycle_hours",
            "avg_pr_size", "large_pr_share", "very_large_pr_count", "weeks_active_authoring",
            "domain_coverage_count", "domains_touched",
        ])

    g = prs.groupby("author_login", dropna=True)

    authored = g.agg(
        merged_pr_count=("pr_number", "count"),
        merged_pr_volume_points=("pr_volume_points", "sum"),
        median_pr_cycle_hours=("cycle_hours_winsor", "median"),
        avg_pr_size=("pr_size", "mean"),
        large_pr_share=("is_large_pr", "mean"),
        very_large_pr_count=("is_very_large_pr", "sum"),
    ).reset_index().rename(columns={"author_login": "engineer"})

    author_weeks = (
        prs[["author_login", "merged_week"]]
        .dropna()
        .drop_duplicates()
        .groupby("author_login")
        .size()
        .reset_index(name="weeks_active_authoring")
        .rename(columns={"author_login": "engineer"})
    )
    authored = authored.merge(author_weeks, on="engineer", how="left")

    # Domain coverage count (unique primary domains + multi-tag flatten)
    domain_flat = prs[["author_login", "domains"]].explode("domains").dropna()
    domain_cov = (
        domain_flat.drop_duplicates()
        .groupby("author_login")
        .size()
        .reset_index(name="domain_coverage_count")
        .rename(columns={"author_login": "engineer"})
    )
    domains_touched = (
        domain_flat.drop_duplicates()
        .groupby("author_login")["domains"]
        .apply(lambda s: ", ".join(sorted(set([str(x) for x in s]))))
        .reset_index(name="domains_touched")
        .rename(columns={"author_login": "engineer"})
    )

    authored = authored.merge(domain_cov, on="engineer", how="left")
    authored = authored.merge(domains_touched, on="engineer", how="left")

    # Authored PR review coverage
    cov_df = td.authored_pr_review_flags.copy()
    if not cov_df.empty:
        coverage = (
            cov_df.groupby("author_login")
            .agg(
                authored_prs_in_window=("pr_number", "count"),
                authored_prs_reviewed=("has_review_in_window", "sum"),
            )
            .reset_index()
            .rename(columns={"author_login": "engineer"})
        )
        coverage["authored_pr_review_coverage_pct"] = np.where(
            coverage["authored_prs_in_window"] > 0,
            (coverage["authored_prs_reviewed"] / coverage["authored_prs_in_window"]) * 100.0,
            0.0,
        )
        authored = authored.merge(coverage, on="engineer", how="left")
    else:
        authored["authored_prs_in_window"] = 0
        authored["authored_prs_reviewed"] = 0
        authored["authored_pr_review_coverage_pct"] = 0.0

    for c in ["weeks_active_authoring", "domain_coverage_count", "authored_prs_in_window", "authored_prs_reviewed", "authored_pr_review_coverage_pct"]:
        if c in authored.columns:
            authored[c] = pd.to_numeric(authored[c], errors="coerce").fillna(0)

    authored["large_pr_share"] = pd.to_numeric(authored["large_pr_share"], errors="coerce").fillna(0)
    authored["domains_touched"] = authored["domains_touched"].fillna("")

    return authored


def _review_metrics(td: TransformedData) -> pd.DataFrame:
    reviews = td.reviews_window
    if reviews.empty:
        return pd.DataFrame(columns=[
            "engineer", "reviews_submitted", "approvals_given", "changes_requested_count",
            "reviews_on_merged_prs", "unique_authors_reviewed", "weeks_active_reviewing"
        ])

    g = reviews.groupby("reviewer_login", dropna=True)

    review_metrics = g.agg(
        reviews_submitted=("pr_number", "count"),
        approvals_given=("is_approval", "sum"),
        changes_requested_count=("is_changes_requested", "sum"),
        reviews_on_merged_prs=("pr_merged", "sum"),
    ).reset_index().rename(columns={"reviewer_login": "engineer"})

    unique_authors = (
        reviews[["reviewer_login", "pr_author_login"]]
        .dropna()
        .drop_duplicates()
        .groupby("reviewer_login")
        .size()
        .reset_index(name="unique_authors_reviewed")
        .rename(columns={"reviewer_login": "engineer"})
    )

    review_weeks = (
        reviews[["reviewer_login", "review_week"]]
        .dropna()
        .drop_duplicates()
        .groupby("reviewer_login")
        .size()
        .reset_index(name="weeks_active_reviewing")
        .rename(columns={"reviewer_login": "engineer"})
    )

    review_metrics = review_metrics.merge(unique_authors, on="engineer", how="left")
    review_metrics = review_metrics.merge(review_weeks, on="engineer", how="left")

    for c in ["unique_authors_reviewed", "weeks_active_reviewing", "changes_requested_count"]:
        review_metrics[c] = pd.to_numeric(review_metrics[c], errors="coerce").fillna(0)

    return review_metrics


def _collab_breadth(td: TransformedData) -> pd.DataFrame:
    pairs1 = td.reviewer_author_pairs.rename(columns={"reviewer_login": "engineer", "pr_author_login": "collaborator"})
    pairs2 = td.pr_reviewer_pairs.rename(columns={"pr_author_login": "engineer", "reviewer_login": "collaborator"})
    pairs = pd.concat([pairs1, pairs2], ignore_index=True).dropna().drop_duplicates()

    if pairs.empty:
        return pd.DataFrame(columns=["engineer", "unique_collaborators"])

    breadth = pairs.groupby("engineer").size().reset_index(name="unique_collaborators")
    return breadth


def _weekly_activity(td: TransformedData) -> pd.DataFrame:
    authored_weekly = (
        td.prs_merged_window.groupby(["author_login", "merged_week"])
        .agg(
            merged_pr_count=("pr_number", "count"),
            merged_pr_volume_points=("pr_volume_points", "sum"),
        )
        .reset_index()
        .rename(columns={"author_login": "engineer", "merged_week": "week"})
    )

    review_weekly = (
        td.reviews_window.groupby(["reviewer_login", "review_week"])
        .agg(
            reviews_submitted=("pr_number", "count"),
            approvals_given=("is_approval", "sum"),
        )
        .reset_index()
        .rename(columns={"reviewer_login": "engineer", "review_week": "week"})
    )

    weekly = authored_weekly.merge(review_weekly, on=["engineer", "week"], how="outer")
    for c in ["merged_pr_count", "merged_pr_volume_points", "reviews_submitted", "approvals_given"]:
        weekly[c] = pd.to_numeric(weekly.get(c, 0), errors="coerce").fillna(0)

    weekly["weekly_activity_points"] = (
        weekly["merged_pr_count"] * 2.0
        + weekly["merged_pr_volume_points"] * 0.6
        + weekly["reviews_submitted"] * 0.5
        + weekly["approvals_given"] * 0.5
    )
    return weekly.sort_values(["engineer", "week"])


def _momentum_metrics(td: TransformedData) -> pd.DataFrame:
    prs = td.prs_merged_window
    rv = td.reviews_window

    # authored points
    authored_points = (
        prs.assign(period=lambda d: np.where(d["is_recent_30d"], "recent", "prior"))
        .groupby(["author_login", "period"])
        .agg(pr_points=("pr_volume_points", "sum"), merged_pr_count=("pr_number", "count"))
        .reset_index()
        .rename(columns={"author_login": "engineer"})
    )

    review_points = (
        rv.assign(period=lambda d: np.where(d["is_recent_30d"], "recent", "prior"))
        .groupby(["reviewer_login", "period"])
        .agg(review_points=("pr_number", "count"), approvals=("is_approval", "sum"))
        .reset_index()
        .rename(columns={"reviewer_login": "engineer"})
    )

    # Pivot and merge
    ap = authored_points.pivot(index="engineer", columns="period", values=["pr_points", "merged_pr_count"]).fillna(0)
    rp = review_points.pivot(index="engineer", columns="period", values=["review_points", "approvals"]).fillna(0)

    def flatten_cols(df):
        df = df.copy()
        df.columns = [f"{a}_{b}" for a, b in df.columns]
        return df.reset_index()

    ap = flatten_cols(ap) if not ap.empty else pd.DataFrame(columns=["engineer"])
    rp = flatten_cols(rp) if not rp.empty else pd.DataFrame(columns=["engineer"])

    m = ap.merge(rp, on="engineer", how="outer").fillna(0)

    # Composite recent/prior activity points
    m["activity_recent"] = (
        m.get("pr_points_recent", 0) * 1.0
        + m.get("merged_pr_count_recent", 0) * 1.5
        + m.get("review_points_recent", 0) * 0.5
        + m.get("approvals_recent", 0) * 0.5
    )
    m["activity_prior"] = (
        m.get("pr_points_prior", 0) * 1.0
        + m.get("merged_pr_count_prior", 0) * 1.5
        + m.get("review_points_prior", 0) * 0.5
        + m.get("approvals_prior", 0) * 0.5
    )

    m["momentum_score"] = (m["activity_recent"] - m["activity_prior"]) / (m["activity_prior"] + 5.0)
    m["momentum_score"] = m["momentum_score"].clip(-1.0, 1.0)

    return m[["engineer", "activity_recent", "activity_prior", "momentum_score"]]


def _add_explanations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    reasons = []

    for _, row in df.iterrows():
        phrases = []

        # Core strengths
        if row.get("shipping_score", 0) >= 0.8:
            phrases.append("top-tier shipping impact")
        if row.get("review_leverage_score", 0) >= 0.8:
            phrases.append("high review leverage")
        if row.get("flow_score", 0) >= 0.75:
            phrases.append("strong execution flow")
        if row.get("norm_weeks_active", 0) >= 0.75:
            phrases.append("consistent activity")
        if row.get("norm_unique_collaborators", 0) >= 0.75:
            phrases.append("broad collaboration")

        # Supporting context
        if row.get("authored_pr_review_coverage_pct", 0) >= 75 and row.get("merged_pr_count", 0) >= 5:
            phrases.append("high PR review coverage")
        if row.get("domain_coverage_count", 0) >= 3:
            phrases.append("multi-domain contributions")
        if row.get("momentum_score", 0) > 0.15:
            phrases.append("positive recent momentum")

        if not phrases:
            phrases.append("balanced contribution profile")

        reasons.append(", ".join(phrases[:3]))

    df["why"] = reasons
    return df


def compute_scores(td: TransformedData) -> ScoringOutput:
    authored = _authored_metrics(td)
    review = _review_metrics(td)
    breadth = _collab_breadth(td)
    weekly = _weekly_activity(td)
    momentum = _momentum_metrics(td)

    # union of engineers
    engineers = pd.DataFrame({
        "engineer": pd.unique(pd.concat([
            authored.get("engineer", pd.Series(dtype=str)),
            review.get("engineer", pd.Series(dtype=str)),
            breadth.get("engineer", pd.Series(dtype=str)),
            momentum.get("engineer", pd.Series(dtype=str)),
        ], ignore_index=True))
    })

    metrics = engineers.merge(authored, on="engineer", how="left")
    metrics = metrics.merge(review, on="engineer", how="left")
    metrics = metrics.merge(breadth, on="engineer", how="left")
    metrics = metrics.merge(momentum, on="engineer", how="left")

    fill_zero_cols = [
        "merged_pr_count", "merged_pr_volume_points", "median_pr_cycle_hours", "avg_pr_size",
        "large_pr_share", "very_large_pr_count", "weeks_active_authoring", "domain_coverage_count",
        "authored_prs_in_window", "authored_prs_reviewed", "authored_pr_review_coverage_pct",
        "reviews_submitted", "approvals_given", "changes_requested_count", "reviews_on_merged_prs",
        "unique_authors_reviewed", "weeks_active_reviewing", "unique_collaborators",
        "activity_recent", "activity_prior", "momentum_score"
    ]
    for c in fill_zero_cols:
        if c not in metrics.columns:
            metrics[c] = 0
        metrics[c] = pd.to_numeric(metrics[c], errors="coerce").fillna(0)

    if "domains_touched" not in metrics.columns:
        metrics["domains_touched"] = ""
    metrics["domains_touched"] = metrics["domains_touched"].fillna("")

    metrics["weeks_active"] = metrics[["weeks_active_authoring", "weeks_active_reviewing"]].max(axis=1)

    # normalized metrics
    metrics["norm_merged_pr_count"] = percentile_rank(metrics["merged_pr_count"], True)
    metrics["norm_merged_pr_volume_points"] = percentile_rank(metrics["merged_pr_volume_points"], True)
    metrics["norm_domain_coverage_count"] = percentile_rank(metrics["domain_coverage_count"], True)

    metrics["norm_reviews_submitted"] = percentile_rank(metrics["reviews_submitted"], True)
    metrics["norm_approvals_given"] = percentile_rank(metrics["approvals_given"], True)
    metrics["norm_unique_authors_reviewed"] = percentile_rank(metrics["unique_authors_reviewed"], True)

    cycle = metrics["median_pr_cycle_hours"].replace(0, np.nan)
    cycle_filled = cycle.fillna(cycle.median() if cycle.notna().any() else 0)
    metrics["norm_cycle_time_inverse"] = percentile_rank(cycle_filled, higher_is_better=False)

    metrics["norm_authored_pr_review_coverage_pct"] = percentile_rank(metrics["authored_pr_review_coverage_pct"], True)
    metrics["norm_large_pr_share_inverse"] = percentile_rank(metrics["large_pr_share"], higher_is_better=False)

    metrics["norm_weeks_active"] = percentile_rank(metrics["weeks_active"], True)
    metrics["norm_unique_collaborators"] = percentile_rank(metrics["unique_collaborators"], True)

    metrics["norm_momentum_score"] = percentile_rank(metrics["momentum_score"], True)

    # composite subscores for story / plots
    metrics["shipping_score"] = (
        0.45 * metrics["norm_merged_pr_count"] +
        0.40 * metrics["norm_merged_pr_volume_points"] +
        0.15 * metrics["norm_domain_coverage_count"]
    )

    metrics["review_leverage_score"] = (
        0.40 * metrics["norm_reviews_submitted"] +
        0.25 * metrics["norm_approvals_given"] +
        0.35 * metrics["norm_unique_authors_reviewed"]
    )

    metrics["flow_score"] = (
        0.40 * metrics["norm_cycle_time_inverse"] +
        0.35 * metrics["norm_authored_pr_review_coverage_pct"] +
        0.25 * metrics["norm_large_pr_share_inverse"]
    )

    metrics["consistency_collab_score"] = (
        0.55 * metrics["norm_weeks_active"] +
        0.45 * metrics["norm_unique_collaborators"]
    )

    # balanced profile bonus (small): rewards engineers contributing to both shipping and reviews
    metrics["balanced_profile_bonus"] = np.minimum(metrics["shipping_score"], metrics["review_leverage_score"]) * 0.05

    # weighted score
    metrics["impact_score"] = 0.0
    for col, w in WEIGHTS.items():
        metrics["impact_score"] += metrics[col] * w
    metrics["impact_score"] += metrics["balanced_profile_bonus"]

    # cap after bonus
    metrics["impact_score"] = metrics["impact_score"].clip(0, 1.2)

    # Raw score (may exceed 100 after bonus)
    metrics["impact_score_raw_100"] = (metrics["impact_score"] * 100).round(1)

    # Display score normalized to 0-100 for leaderboard readability
    min_s = metrics["impact_score"].min()
    max_s = metrics["impact_score"].max()

    if pd.notna(min_s) and pd.notna(max_s) and max_s > min_s:
        metrics["impact_score_100"] = ((metrics["impact_score"] - min_s) / (max_s - min_s) * 100).round(1)
    else:
        metrics["impact_score_100"] = 100.0

    # useful ratio metrics
    metrics["review_to_ship_ratio"] = np.where(
        metrics["merged_pr_count"] > 0,
        metrics["reviews_submitted"] / metrics["merged_pr_count"],
        np.nan,
    )

    # archetype (storytelling filter)
    def archetype(row):
        ship = row["shipping_score"]
        rev = row["review_leverage_score"]
        if ship >= 0.65 and rev >= 0.65:
            return "Balanced"
        if ship > rev + 0.10:
            return "Shipper"
        if rev > ship + 0.10:
            return "Reviewer"
        return "Mixed"

    metrics["contributor_archetype"] = metrics.apply(archetype, axis=1)

    metrics["activity_points"] = (
        metrics["merged_pr_count"] * 2.0
        + metrics["reviews_submitted"] * 0.6
        + metrics["weeks_active"] * 0.5
    )

    metrics = metrics[metrics["activity_points"] > 0].copy()
    metrics = metrics.sort_values(["impact_score_100", "activity_points"], ascending=[False, False]).reset_index(drop=True)
    metrics["rank"] = range(1, len(metrics) + 1)

    metrics = _add_explanations(metrics)

    top5 = metrics.head(5).copy()

    return ScoringOutput(engineer_metrics=metrics, top5=top5, weekly_activity=weekly)


if __name__ == "__main__":
    from src.transform import transform_data
    td = transform_data(days=90, data_dir="data")
    out = compute_scores(td)
    print(out.top5[["rank", "engineer", "impact_score_100", "contributor_archetype", "why"]])