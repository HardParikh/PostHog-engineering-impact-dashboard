from __future__ import annotations

import os
from typing import Optional
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

def _safe(v, default=0):
    try:
        if pd.isna(v):
            return default
        return v
    except Exception:
        return default


def generate_rule_based_engineer_summary(row: pd.Series) -> str:
    """Fallback summary when no LLM key is available."""
    parts = []

    impact = _safe(row.get("impact_score_100"), 0)
    merged = int(_safe(row.get("merged_pr_count"), 0))
    reviews = int(_safe(row.get("reviews_submitted"), 0))
    approvals = int(_safe(row.get("approvals_given"), 0))
    weeks = int(_safe(row.get("weeks_active"), 0))
    collabs = int(_safe(row.get("unique_collaborators"), 0))
    cycle = _safe(row.get("median_pr_cycle_hours"), None)
    domains = int(_safe(row.get("domain_coverage_count"), 0))
    review_cov = _safe(row.get("authored_pr_review_coverage_pct"), None)
    momentum = _safe(row.get("momentum_score"), 0)

    # Main story
    if merged >= 15 and reviews >= 20:
        parts.append("This engineer combines strong shipping output with high review leverage, indicating both direct delivery and multiplier impact on teammates.")
    elif merged >= 15:
        parts.append("This engineer ranks highly primarily through shipping impact, with a sustained volume of merged work in the selected window.")
    elif reviews >= 25:
        parts.append("This engineer ranks highly through review leverage, helping unblock and improve work across many pull requests.")
    else:
        parts.append("This engineer shows balanced contribution across authored work and collaboration signals.")

    # Execution quality proxies
    if cycle is not None and not pd.isna(cycle):
        if cycle < 48:
            parts.append("PR turnaround is relatively fast, which suggests efficient execution and lower work-in-progress drag.")
        elif cycle > 200:
            parts.append("PR turnaround is slower than peers, which may reflect larger scope work or longer review cycles.")

    if review_cov is not None and not pd.isna(review_cov):
        if review_cov >= 80:
            parts.append("Most authored PRs received review activity, which improves reliability of delivery and collaboration quality.")
        elif review_cov < 40 and merged > 5:
            parts.append("A lower share of authored PRs show captured reviews; this can happen in low-friction changes but is worth monitoring.")

    # Breadth / consistency
    if weeks >= 8:
        parts.append("Contribution is consistent across many weeks rather than concentrated in short bursts.")
    if collabs >= 10:
        parts.append("Collaboration breadth is high, indicating influence across multiple contributors.")
    if domains >= 3:
        parts.append("Work spans multiple product or engineering domains, increasing organizational reach.")

    # Momentum
    if momentum > 0.15:
        parts.append("Recent momentum is positive, with stronger activity in the latest 30 days compared with the earlier part of the window.")
    elif momentum < -0.15:
        parts.append("Recent momentum is softer relative to the earlier period, which may reflect rotation, project phase changes, or time off.")

    return " ".join(parts[:5])


def generate_rule_based_exec_summary(metrics_df: pd.DataFrame, top_n: int = 5) -> str:
    if metrics_df.empty:
        return "No engineers match the current filters."

    top = metrics_df.head(top_n).copy()

    avg_shipping = top["shipping_score"].mean() if "shipping_score" in top else 0
    avg_review = top["review_leverage_score"].mean() if "review_leverage_score" in top else 0
    avg_consistency = top["norm_weeks_active"].mean() if "norm_weeks_active" in top else 0

    dominant = "balanced"
    if avg_shipping > avg_review + 0.08:
        dominant = "shipping-heavy"
    elif avg_review > avg_shipping + 0.08:
        dominant = "review-leverage-heavy"

    text = [
        f"Top {top_n} contributors in the current view show a {dominant} impact pattern."
    ]

    # Consistency
    if avg_consistency >= 0.7:
        text.append("At a group level, the top contributors are consistently active across the full period rather than relying on short bursts.")

    # Domain breadth signal
    if "domain_coverage_count" in top.columns:
        mean_domains = top["domain_coverage_count"].mean()
        if mean_domains >= 3:
            text.append("Several top contributors show multi-domain reach, which can be especially valuable for cross-cutting initiatives.")

    # Review coverage signal
    if "authored_pr_review_coverage_pct" in top.columns:
        med_cov = top["authored_pr_review_coverage_pct"].median()
        if med_cov >= 70:
            text.append("Review coverage is strong across top contributors, supporting healthy collaboration and quality checks.")
        elif med_cov < 40:
            text.append("Review coverage appears low for some top contributors; this may indicate low-friction changes or under-captured review workflows.")

    # Momentum signal
    if "momentum_score" in top.columns:
        pos = (top["momentum_score"] > 0.1).sum()
        neg = (top["momentum_score"] < -0.1).sum()
        if pos > neg:
            text.append("More top contributors are accelerating in recent weeks than slowing down.")
        elif neg > pos:
            text.append("Several top contributors show lower recent momentum, which may reflect project transitions or release timing.")

    return " ".join(text)


def maybe_openai_engineer_summary(row: pd.Series) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return "[LLM ERROR] No API key found in OPENAI_API_KEY"

    try:
        from openai import OpenAI  # type: ignore
        import httpx

        client = OpenAI(
            api_key=api_key,
            http_client=httpx.Client(verify=False, timeout=30.0)  # TEMP SSL BYPASS
        )

        payload = {
            "engineer": str(row.get("engineer")),
            "impact_score_100": float(_safe(row.get("impact_score_100"), 0)),
            "merged_pr_count": int(_safe(row.get("merged_pr_count"), 0)),
            "merged_pr_volume_points": float(_safe(row.get("merged_pr_volume_points"), 0)),
            "reviews_submitted": int(_safe(row.get("reviews_submitted"), 0)),
            "approvals_given": int(_safe(row.get("approvals_given"), 0)),
            "unique_authors_reviewed": int(_safe(row.get("unique_authors_reviewed"), 0)),
            "median_pr_cycle_hours": float(_safe(row.get("median_pr_cycle_hours"), 0)),
            "weeks_active": int(_safe(row.get("weeks_active"), 0)),
            "unique_collaborators": int(_safe(row.get("unique_collaborators"), 0)),
            "domain_coverage_count": int(_safe(row.get("domain_coverage_count"), 0)),
            "authored_pr_review_coverage_pct": float(_safe(row.get("authored_pr_review_coverage_pct"), 0)),
            "momentum_score": float(_safe(row.get("momentum_score"), 0)),
            "why": str(row.get("why", "")),
        }

        prompt = f"""
You are helping an engineering leader understand contributor impact at a glance.
Write a concise, evidence-based summary (4-6 sentences) of why this engineer ranks highly.
Use only the provided metrics. Do not overclaim code quality or leadership.
Highlight tradeoffs where relevant.

Metrics:
{payload}
"""

        print("DEBUG: Calling OpenAI with temporary SSL bypass")

        resp = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            max_output_tokens=220,
        )

        text = getattr(resp, "output_text", None)
        if text and isinstance(text, str) and text.strip():
            return text.strip()

        return "[LLM ERROR] OpenAI response returned empty output_text"

    except Exception as e:
        return f"[LLM ERROR] {type(e).__name__}: {e}"