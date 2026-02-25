from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.github_fetch import load_config, fetch_prs_and_reviews, save_raw_data
from src.transform import transform_data
from src.scoring import compute_scores, WEIGHTS
from src.ai_insights import (
    generate_rule_based_engineer_summary,
    generate_rule_based_exec_summary,
    maybe_openai_engineer_summary,
)

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="PostHog Engineering Impact Dashboard",
    page_icon="📈",
    layout="wide",
)

# ============================================================
# Session state defaults (preserve drilldown selection)
# ============================================================
if "selected_engineer" not in st.session_state:
    st.session_state.selected_engineer = None

if "use_llm_ai" not in st.session_state:
    st.session_state.use_llm_ai = False

# Sidebar filter defaults
_defaults = {
    "f_name_query": "",
    "f_role": "All",
    "f_min_merged": 0,
    "f_min_reviews": 0,
    "f_min_weeks": 0,
    "f_min_score": 0.0,
    "f_momentum": "All",
    "f_domain": "All",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Optional non-widget state
if "selected_engineer" not in st.session_state:
    st.session_state.selected_engineer = None

if "last_filters_signature" not in st.session_state:
    st.session_state.last_filters_signature = None


# ============================================================
# Cached compute
# ============================================================
@st.cache_data(show_spinner=False)
def load_transformed_scored(days: int, data_dir: str = "data"):
    td = transform_data(days=days, data_dir=data_dir)
    scoring = compute_scores(td)
    return td, scoring


def raw_data_exists(data_dir: str = "data") -> bool:
    return os.path.exists(f"{data_dir}/prs_raw.csv") and os.path.exists(f"{data_dir}/reviews_raw.csv")


def refresh_data_from_github(max_pages: int = 35):
    cfg = load_config()
    prs_df, reviews_df = fetch_prs_and_reviews(
        owner=cfg["owner"],
        repo=cfg["repo"],
        token=cfg["token"],
        max_pages=max_pages,
    )
    save_raw_data(prs_df, reviews_df, out_dir="data")
    st.cache_data.clear()


# ============================================================
# UI helper functions
# ============================================================
def metric_card(label: str, value, help_text: str = ""):
    st.metric(label=label, value=value, help=help_text)


def format_hours(v):
    try:
        if pd.isna(v):
            return "—"
        return f"{float(v):.1f}h"
    except Exception:
        return "—"


def format_pct(v):
    try:
        if pd.isna(v):
            return "—"
        return f"{float(v):.0f}%"
    except Exception:
        return "—"


def get_engineer_options(metrics_df: pd.DataFrame):
    return metrics_df["engineer"].dropna().astype(str).tolist()


def preserve_selection(metrics_df: pd.DataFrame):
    options = get_engineer_options(metrics_df)
    if not options:
        st.session_state.selected_engineer = None
        return None

    current = st.session_state.selected_engineer
    if current in options:
        return current

    fallback = options[0]
    st.session_state.selected_engineer = fallback
    return fallback


def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def apply_filters(metrics_df: pd.DataFrame):
    df = metrics_df.copy()

    name_query = st.session_state.get("f_name_query", "").strip().lower()
    role_filter = st.session_state.get("f_role", "All")
    min_merged = st.session_state.get("f_min_merged", 0)
    min_reviews = st.session_state.get("f_min_reviews", 0)
    min_weeks = st.session_state.get("f_min_weeks", 0)
    domain_filter = st.session_state.get("f_domain", "All")
    momentum_filter = st.session_state.get("f_momentum", "All")
    min_score = st.session_state.get("f_min_score", 0.0)

    if name_query:
        df = df[df["engineer"].astype(str).str.lower().str.contains(name_query, na=False)]

    df = df[df["merged_pr_count"] >= min_merged]
    df = df[df["reviews_submitted"] >= min_reviews]
    df = df[df["weeks_active"] >= min_weeks]
    df = df[df["impact_score_100"] >= min_score]

    if role_filter != "All":
        df = df[df["contributor_archetype"] == role_filter]

    if momentum_filter != "All":
        if momentum_filter == "Accelerating":
            df = df[df["momentum_score"] > 0.10]
        elif momentum_filter == "Stable":
            df = df[(df["momentum_score"] >= -0.10) & (df["momentum_score"] <= 0.10)]
        elif momentum_filter == "Slowing":
            df = df[df["momentum_score"] < -0.10]

    if domain_filter != "All":
        df = df[df["domains_touched"].astype(str).str.contains(domain_filter, case=False, na=False)]

    # Guard against scores >100 if scoring bonus/weights push beyond 100
    # (we preserve raw score for ranking, but show clipped version in UI if desired)
    df = df.sort_values(["impact_score_100", "activity_points"], ascending=[False, False]).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def build_component_breakdown(selected_row: pd.Series) -> pd.DataFrame:
    component_map = [
        ("Merged PR count", "norm_merged_pr_count", WEIGHTS["norm_merged_pr_count"]),
        ("PR volume points", "norm_merged_pr_volume_points", WEIGHTS["norm_merged_pr_volume_points"]),
        ("Domain coverage", "norm_domain_coverage_count", WEIGHTS["norm_domain_coverage_count"]),
        ("Reviews submitted", "norm_reviews_submitted", WEIGHTS["norm_reviews_submitted"]),
        ("Approvals given", "norm_approvals_given", WEIGHTS["norm_approvals_given"]),
        ("Unique authors reviewed", "norm_unique_authors_reviewed", WEIGHTS["norm_unique_authors_reviewed"]),
        ("Cycle time inverse", "norm_cycle_time_inverse", WEIGHTS["norm_cycle_time_inverse"]),
        ("PR review coverage", "norm_authored_pr_review_coverage_pct", WEIGHTS["norm_authored_pr_review_coverage_pct"]),
        ("Smaller PR discipline", "norm_large_pr_share_inverse", WEIGHTS["norm_large_pr_share_inverse"]),
        ("Weeks active", "norm_weeks_active", WEIGHTS["norm_weeks_active"]),
        ("Unique collaborators", "norm_unique_collaborators", WEIGHTS["norm_unique_collaborators"]),
        ("Momentum", "norm_momentum_score", WEIGHTS["norm_momentum_score"]),
    ]

    rows = []
    for label, col, weight in component_map:
        score = float(selected_row.get(col, 0) or 0)
        rows.append(
            {
                "component": label,
                "normalized_score": score,
                "weight": weight,
                "weighted_contribution": score * weight * 100,
            }
        )

    bonus = float(selected_row.get("balanced_profile_bonus", 0) or 0)
    rows.append(
        {
            "component": "Balanced profile bonus",
            "normalized_score": (bonus / 0.05) if bonus else 0,
            "weight": 0.05,
            "weighted_contribution": bonus * 100,
        }
    )

    out = pd.DataFrame(rows).sort_values("weighted_contribution", ascending=False)
    out["weighted_contribution"] = out["weighted_contribution"].round(2)
    out["normalized_score"] = out["normalized_score"].round(3)
    out["weight"] = out["weight"].round(3)
    return out


def render_top_table(df: pd.DataFrame, top_n: int):
    cols = [
        "rank",
        "engineer",
        "impact_score_100",
        "contributor_archetype",
        "merged_pr_count",
        "reviews_submitted",
        "weeks_active",
        "domain_coverage_count",
        "momentum_score",
        "why",
    ]
    out = df.head(top_n)[cols].copy()

    out = out.rename(
        columns={
            "impact_score_100": "impact_score",
            "contributor_archetype": "archetype",
            "merged_pr_count": "merged_prs",
            "reviews_submitted": "reviews",
            "domain_coverage_count": "domains",
            "momentum_score": "momentum",
        }
    )
    out["impact_score"] = pd.to_numeric(out["impact_score"], errors="coerce").round(1)
    out["momentum"] = pd.to_numeric(out["momentum"], errors="coerce").round(2)

    st.dataframe(out, width="stretch", hide_index=True)


def build_clean_impact_map(plot_df: pd.DataFrame, top_n: int):
    """
    Cleaner impact map:
    - shows all contributors lightly
    - highlights top N and selected engineer
    - fewer labels
    - consistent sizing and better readability
    """
    if plot_df.empty:
        return go.Figure()

    plot_df = plot_df.copy()
    plot_df["impact_score_display"] = pd.to_numeric(plot_df["impact_score_100"], errors="coerce").fillna(0).clip(0, 100)
    plot_df["weeks_active_safe"] = pd.to_numeric(plot_df["weeks_active"], errors="coerce").fillna(0).clip(lower=0)
    plot_df["reviews_submitted"] = pd.to_numeric(plot_df["reviews_submitted"], errors="coerce").fillna(0)
    plot_df["merged_pr_count"] = pd.to_numeric(plot_df["merged_pr_count"], errors="coerce").fillna(0)

    top_names = plot_df.head(top_n)["engineer"].tolist()
    selected_name = st.session_state.get("selected_engineer")

    plot_df["highlight_group"] = "Others"
    plot_df.loc[plot_df["engineer"].isin(top_names), "highlight_group"] = "Top N"
    if selected_name in plot_df["engineer"].values:
        plot_df.loc[plot_df["engineer"] == selected_name, "highlight_group"] = "Selected"

    # Base scatter with controlled bubble size
    fig = px.scatter(
        plot_df,
        x="shipping_score",
        y="review_leverage_score",
        size="weeks_active_safe",
        color="impact_score_display",
        symbol="contributor_archetype",
        hover_name="engineer",
        hover_data={
            "impact_score_display": ":.1f",
            "merged_pr_count": True,
            "reviews_submitted": True,
            "approvals_given": True,
            "authored_pr_review_coverage_pct": ":.0f",
            "domain_coverage_count": True,
            "momentum_score": ":.2f",
            "weeks_active_safe": True,
            "shipping_score": ":.2f",
            "review_leverage_score": ":.2f",
        },
        labels={
            "shipping_score": "Shipping score",
            "review_leverage_score": "Review leverage score",
            "impact_score_display": "Impact score",
            "weeks_active_safe": "Weeks active",
        },
        title="Shipping vs review leverage (bubble size = weeks active)",
    )

    fig.update_traces(
        marker=dict(
            sizemode="area",
            line=dict(width=0.8, color="rgba(255,255,255,0.65)"),
            opacity=0.78,
        ),
        selector=dict(mode="markers"),
    )

    # Highlight outlines for top N and selected
    top_df = plot_df[plot_df["highlight_group"] == "Top N"].copy()
    if not top_df.empty:
        fig.add_trace(
            go.Scatter(
                x=top_df["shipping_score"],
                y=top_df["review_leverage_score"],
                mode="markers",
                name="Top N highlight",
                marker=dict(
                    size=14,
                    symbol="circle-open",
                    line=dict(width=2),
                    color="rgba(0,0,0,0)"
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    sel_df = plot_df[plot_df["highlight_group"] == "Selected"].copy()
    if not sel_df.empty:
        fig.add_trace(
            go.Scatter(
                x=sel_df["shipping_score"],
                y=sel_df["review_leverage_score"],
                mode="markers+text",
                text=sel_df["engineer"],
                textposition="top center",
                name="Selected engineer",
                marker=dict(
                    size=18,
                    symbol="diamond-open",
                    line=dict(width=2.5),
                    color="rgba(0,0,0,0)"
                ),
                textfont=dict(size=11),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Label only top 5 to avoid clutter
    label_df = plot_df.head(min(5, len(plot_df))).copy()
    if not label_df.empty:
        fig.add_trace(
            go.Scatter(
                x=label_df["shipping_score"],
                y=label_df["review_leverage_score"],
                mode="text",
                text=label_df["engineer"],
                textposition="top center",
                textfont=dict(size=10),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=42, b=10),
        coloraxis_showscale=False,
        legend_title_text="",
    )

    fig.update_xaxes(range=[-0.03, 1.03], gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(range=[-0.03, 1.03], gridcolor="rgba(128,128,128,0.15)")
    return fig


# ============================================================
# Header
# ============================================================
st.title("📈 PostHog Engineering Impact Dashboard")
st.caption(
    "Explainable engineering impact using shipped PRs, review leverage, execution-flow proxies, consistency, collaboration, and momentum."
)

with st.expander("What this dashboard is trying to answer"):
    st.markdown(
        """
**Primary question:** Who is creating the most impact in the PostHog repo, and why?

This dashboard is designed for an engineering leader who needs:
- a quick top-contributor view
- evidence behind the ranking (not vanity metrics)
- drilldowns into each engineer’s contribution pattern
- validation tables to sanity check the model against raw GitHub activity
"""
    )

# ============================================================
# Sidebar controls
# ============================================================
with st.sidebar:
    st.header("Controls")

    st.caption("Use these filters to focus the leaderboard on the team slice or contribution pattern you care about.")

    days = st.slider(
        "Lookback window (days)",
        min_value=30,
        max_value=180,
        value=90,
        step=15,
        help="How far back to analyze GitHub PR and review metadata.",
    )
    top_n = st.slider(
        "Leaderboard size",
        5,
        25,
        10,
        1,
        help="How many engineers to include in the visible leaderboard and top-N highlights.",
    )
    min_activity_points = st.slider(
        "Min activity points",
        0,
        30,
        1,
        1,
        help="Removes extremely low-activity contributors to reduce noise.",
    )

    st.divider()
    st.subheader("Filters")
    st.text_input(
        "Search engineer",
        key="f_name_query",
        placeholder="Type the name...",
        help="Filter by engineer login / name text.",
    )
    st.selectbox(
        "Contributor archetype",
        ["All", "Balanced", "Shipper", "Reviewer", "Mixed"],
        key="f_role",
        help="Profile grouping based on contribution pattern (shipping-heavy, review-heavy, balanced, etc.).",
    )
    st.slider(
        "Min merged PRs",
        0,
        30,
        0,
        1,
        key="f_min_merged",
        help="Keep contributors with at least this many merged PRs.",
    )
    st.slider(
        "Min reviews",
        0,
        60,
        0,
        1,
        key="f_min_reviews",
        help="Keep contributors with at least this many review events.",
    )
    st.slider(
        "Min weeks active",
        0,
        20,
        0,
        1,
        key="f_min_weeks",
        help="Keep contributors active in at least this many weeks within the selected window.",
    )
    st.slider(
        "Min impact score",
        0.0,
        100.0,
        0.0,
        1.0,
        key="f_min_score",
        help="Filter out lower-ranked contributors and focus on the strongest impact signals.",
    )
    st.selectbox(
        "Momentum",
        ["All", "Accelerating", "Stable", "Slowing"],
        key="f_momentum",
        help="Momentum compares recent activity vs earlier period in the selected window.",
    )

    domain_options = ["All", "frontend", "backend", "data", "infra_devex", "docs_product_ops", "security", "other"]
    st.selectbox(
        "Domain touched",
        domain_options,
        key="f_domain",
        help="Heuristic domain inferred from PR labels/titles/paths. Useful for directional exploration.",
    )

    st.divider()
    st.subheader("AI")
    st.toggle(
        "Use LLM summaries",
        key="use_llm_ai",
        value=False,
        help="If OPENAI_API_KEY is available, selected engineer narrative can use LLM summarization. Otherwise falls back to a deterministic local summary.",
    )

    st.divider()
    st.subheader("Data")
    if st.button("Refresh raw data from GitHub API", help="Fetch latest PR and review metadata from the GitHub GraphQL API and rebuild cached data."):
        with st.spinner("Fetching from GitHub..."):
            try:
                refresh_data_from_github(max_pages=40)
                st.success("Raw data refreshed.")
            except Exception as e:
                st.error(f"Refresh failed: {e}")


# ============================================================
# Data check
# ============================================================
if not raw_data_exists():
    st.warning("No raw data files found in ./data. Use the sidebar to refresh from GitHub API first.")
    st.stop()

# ============================================================
# Load & score
# ============================================================
with st.spinner("Loading and scoring data..."):
    td, scoring = load_transformed_scored(days=days, data_dir="data")

metrics_df = scoring.engineer_metrics.copy()
metrics_df = metrics_df[metrics_df["activity_points"] >= min_activity_points].copy()

filtered_df = apply_filters(metrics_df)
selected_engineer = preserve_selection(filtered_df)

if filtered_df.empty:
    st.info("No engineers match the current filters. Try lowering thresholds or widening the lookback window.")
    st.stop()

# ============================================================
# Top summary row
# ============================================================
st.markdown("### Executive snapshot")
st.caption("A fast summary of the currently filtered view. Useful for understanding scope before drilling into individuals.")

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card(
        "Engineers",
        int(filtered_df["engineer"].nunique()),
        "Number of engineers currently included after filters.",
    )
with c2:
    metric_card(
        "Merged PRs",
        int(len(td.prs_merged_window)),
        "Total merged PRs in the selected time window (window-level metric, not filter-limited by engineer row selection).",
    )
with c3:
    metric_card(
        "Reviews",
        int(len(td.reviews_window)),
        "Total review events in the selected time window.",
    )
with c4:
    metric_card(
        "Top score",
        f"{pd.to_numeric(filtered_df['impact_score_100'], errors='coerce').max():.1f}",
        "Highest impact score among contributors in the filtered view.",
    )

c5, c6, c7, c8 = st.columns(4)
with c5:
    metric_card(
        "Median score",
        f"{pd.to_numeric(filtered_df['impact_score_100'], errors='coerce').median():.1f}",
        "Median impact score in the filtered cohort.",
    )
with c6:
    metric_card(
        "Window start",
        td.cutoff_ts.strftime("%Y-%m-%d"),
        "Lower bound of the analysis window.",
    )
with c7:
    metric_card(
        "Window end",
        td.latest_ts.strftime("%Y-%m-%d"),
        "Most recent timestamp present in the dataset.",
    )
with c8:
    metric_card(
        "Top archetype",
        filtered_df["contributor_archetype"].mode().iloc[0] if not filtered_df.empty else "—",
        "Most common contribution pattern in the filtered cohort.",
    )

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3 = st.tabs(
    [
        "🏆 Leaderboard & Drilldown",
        "🧠 AI Insights & Story",
        "✅ Validation & Methods",
    ]
)

# ============================================================
# TAB 1 — Leaderboard & Drilldown
# ============================================================
with tab1:
    st.caption(
        "This section ranks contributors and explains *why* they rank there. Use it to identify top impact and understand contribution shape (shipper/reviewer/balanced)."
    )

    left, right = st.columns([1.15, 1.55])

    with left:
        st.subheader("Impact leaderboard")
        st.caption("Top contributors in the filtered view. The `why` field is a quick, explainable rationale generated from the scoring features.")
        render_top_table(filtered_df, top_n=top_n)

        options = get_engineer_options(filtered_df)
        if st.session_state.selected_engineer not in options:
            st.session_state.selected_engineer = options[0]

        st.selectbox(
            "Select engineer for drilldown",
            options=options,
            key="selected_engineer",
            help="Choose a contributor to inspect their score composition, activity trend, and comparison to peers.",
        )

        compare_to_top5 = st.toggle(
            "Compare selected engineer to Top 5 averages",
            value=True,
            help="Shows whether the selected engineer is stronger/weaker than the top cohort across key score dimensions.",
        )

    with right:
        row = filtered_df.loc[filtered_df["engineer"] == st.session_state.selected_engineer].iloc[0]

        st.subheader(f"Engineer profile: {row['engineer']}")
        st.caption("This panel summarizes the selected engineer’s contribution pattern and score drivers.")

        # Smaller, cleaner profile metrics (10 metrics across two rows)
        r1, r2, r3, r4, r5 = st.columns(5)
        with r1:
            metric_card("Impact", f"{float(row['impact_score_100']):.1f}", "Composite impact score used for ranking.")
        with r2:
            metric_card("Archetype", row["contributor_archetype"], "Contribution profile classification.")
        with r3:
            metric_card("Merged PRs", int(row["merged_pr_count"]), "Merged PRs authored in the window.")
        with r4:
            metric_card("Reviews", int(row["reviews_submitted"]), "Reviews submitted in the window.")
        with r5:
            metric_card("Weeks active", int(row["weeks_active"]), "Number of active weeks in the window.")

        r6, r7, r8, r9, r10 = st.columns(5)
        with r6:
            metric_card("Median cycle", format_hours(row["median_pr_cycle_hours"]), "Median PR cycle time (creation to merge proxy).")
        with r7:
            metric_card("Review coverage", format_pct(row["authored_pr_review_coverage_pct"]), "Percent of authored PRs that received review coverage.")
        with r8:
            metric_card("Collaborators", int(row["unique_collaborators"]), "Unique collaborators interacted with.")
        with r9:
            metric_card("Domains", int(row["domain_coverage_count"]), "Number of inferred domains touched.")
        with r10:
            metric_card("Momentum", f"{float(row['momentum_score']):.2f}", "Recent activity trend vs prior period.")

        st.markdown(f"**Why they rank:** {safe_str(row.get('why', '—'))}")
        if safe_str(row.get("domains_touched", "")):
            st.caption(f"Domains touched (inferred): {safe_str(row.get('domains_touched'))}")

        # Breakdown chart
        st.markdown("#### Score contribution breakdown")
        st.caption("Shows how each normalized feature contributes to the selected engineer’s final impact score.")
        breakdown_df = build_component_breakdown(row)

        fig_breakdown = px.bar(
            breakdown_df.sort_values("weighted_contribution", ascending=True),
            x="weighted_contribution",
            y="component",
            orientation="h",
            title="Weighted impact score contributions",
            hover_data={
                "normalized_score": ":.2f",
                "weight": ":.2f",
                "weighted_contribution": ":.2f",
            },
        )
        fig_breakdown.update_layout(height=420, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig_breakdown, width="stretch")

        if compare_to_top5:
            st.markdown("#### Selected vs Top 5 average profile")
            st.caption("Helps show *how* this engineer creates impact relative to the strongest cohort.")
            peer = filtered_df.head(min(5, len(filtered_df)))

            peer_avg = {
                "Shipping": float(peer["shipping_score"].mean()),
                "Review leverage": float(peer["review_leverage_score"].mean()),
                "Flow": float(peer["flow_score"].mean()),
                "Consistency+Collab": float(peer["consistency_collab_score"].mean()),
                "Momentum": float(peer["norm_momentum_score"].mean()),
            }
            selected_vals = {
                "Shipping": float(row["shipping_score"]),
                "Review leverage": float(row["review_leverage_score"]),
                "Flow": float(row["flow_score"]),
                "Consistency+Collab": float(row["consistency_collab_score"]),
                "Momentum": float(row["norm_momentum_score"]),
            }

            comp_df = pd.DataFrame(
                {
                    "dimension": list(peer_avg.keys()) * 2,
                    "score": list(selected_vals.values()) + list(peer_avg.values()),
                    "series": ["Selected"] * 5 + ["Top 5 avg"] * 5,
                }
            )
            fig_comp = px.bar(
                comp_df,
                x="dimension",
                y="score",
                color="series",
                barmode="group",
                title="Selected vs Top 5 average profile",
            )
            fig_comp.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), yaxis_range=[0, 1])
            st.plotly_chart(fig_comp, width="stretch")

    st.divider()

    c_left, c_right = st.columns([1, 1.1])

    with c_left:
        st.subheader("Weekly activity trend")
        st.caption("Shows consistency over time for the selected engineer. Useful for spotting bursts vs sustained contribution.")

        weekly = scoring.weekly_activity.copy()
        w = weekly[weekly["engineer"] == st.session_state.selected_engineer].copy()

        if w.empty:
            st.info("No weekly activity data for selected engineer.")
        else:
            w = w.sort_values("week")
            fig_week = go.Figure()
            fig_week.add_trace(
                go.Scatter(
                    x=w["week"],
                    y=w["weekly_activity_points"],
                    mode="lines+markers",
                    name="Weekly activity",
                )
            )
            fig_week.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title="ISO week",
                yaxis_title="Activity points",
                showlegend=False,
            )
            st.plotly_chart(fig_week, width="stretch")

    with c_right:
        st.subheader("Impact map")
        st.caption(
            "A portfolio view of contributors. X = shipping strength, Y = review leverage. Bubble size reflects consistency (weeks active)."
        )
        fig_scatter = build_clean_impact_map(filtered_df.copy(), top_n=top_n)
        st.plotly_chart(fig_scatter, width="stretch")

# ============================================================
# TAB 2 — AI Insights & Story
# ============================================================
with tab2:
    st.caption(
        "This section translates metrics into a leadership-friendly narrative. It helps answer: what pattern do we see in this cohort, and what makes the selected engineer stand out?"
    )

    exec_ai_source = "Local rules summarizer"
    if st.session_state.use_llm_ai and os.getenv("OPENAI_API_KEY", "").strip():
        exec_ai_source = "Local rules summarizer (engineer-level LLM summaries enabled separately)"

    st.caption(f"Executive summary source: **{exec_ai_source}**")
    st.subheader("Executive summary")
    st.caption("A quick narrative for the current filtered cohort (top contributors + contribution mix).")

    top_view = filtered_df.head(top_n).copy()
    exec_summary = generate_rule_based_exec_summary(top_view, top_n=min(top_n, 10))
    st.markdown(exec_summary)

    st.divider()

    left2, right2 = st.columns([1.15, 1.35])

    with left2:
        st.markdown("### Selected engineer narrative")
        st.caption("A plain-English explanation of why the selected engineer ranks highly, based only on the scoring metrics shown in this dashboard.")

        selected_row = filtered_df.loc[filtered_df["engineer"] == st.session_state.selected_engineer].iloc[0]
        narrative: Optional[str] = None
        ai_source = "Local rules fallback"

        if st.session_state.use_llm_ai:
            llm_narrative = maybe_openai_engineer_summary(selected_row)
            if llm_narrative:
                narrative = llm_narrative
                ai_source = "OpenAI LLM"
            else:
                narrative = generate_rule_based_engineer_summary(selected_row)
                ai_source = "Local rules fallback (no key / API unavailable)"
        else:
            narrative = generate_rule_based_engineer_summary(selected_row)
            ai_source = "Local rules fallback (LLM toggle off)"

        st.caption(f"AI source: **{ai_source}**")
        st.markdown(narrative)

        st.markdown("### Story signals")
        st.caption("High-signal facts that help explain the selected engineer’s impact profile at a glance.")

        story_df = pd.DataFrame(
            [
                {"Signal": "Impact pattern", "Value": str(selected_row["contributor_archetype"])},
                {"Signal": "Momentum", "Value": f"{float(selected_row['momentum_score']):.2f}"},
                {"Signal": "PR review coverage", "Value": format_pct(selected_row["authored_pr_review_coverage_pct"])},
                {
                    "Signal": "Large PR share",
                    "Value": format_pct(float(selected_row["large_pr_share"]) * 100 if pd.notna(selected_row["large_pr_share"]) else 0),
                },
                {"Signal": "Domains touched", "Value": str(int(selected_row["domain_coverage_count"]))},
                {
                    "Signal": "Review-to-ship ratio",
                    "Value": f"{float(selected_row['review_to_ship_ratio']):.2f}"
                    if pd.notna(selected_row["review_to_ship_ratio"])
                    else "—",
                },
            ],
            dtype="object",
        )
        story_df["Signal"] = story_df["Signal"].astype(str)
        story_df["Value"] = story_df["Value"].astype(str)

        st.dataframe(story_df, width="stretch", hide_index=True)

    with right2:
        st.markdown("### Portfolio of top contributors")
        st.caption("A compact comparison table for the top contributors across normalized score dimensions.")

        top = top_view.copy().head(min(10, len(top_view)))
        matrix_cols = [
            "engineer",
            "shipping_score",
            "review_leverage_score",
            "flow_score",
            "consistency_collab_score",
            "norm_momentum_score",
            "impact_score_100",
        ]
        m = top[matrix_cols].copy().rename(columns={"norm_momentum_score": "momentum_norm"})
        # Make table more readable
        for col in ["shipping_score", "review_leverage_score", "flow_score", "consistency_collab_score", "momentum_norm", "impact_score_100"]:
            m[col] = pd.to_numeric(m[col], errors="coerce").round(2)

        st.dataframe(m, width="stretch", hide_index=True)

        st.markdown("### Archetype distribution")
        st.caption("Shows how contribution styles are distributed in the current filtered cohort.")

        arch_counts = filtered_df["contributor_archetype"].value_counts().reset_index()
        arch_counts.columns = ["archetype", "count"]
        fig_arch = px.bar(
            arch_counts,
            x="archetype",
            y="count",
            title="Contributor archetype distribution (filtered view)",
        )
        fig_arch.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_arch, width="stretch")

# ============================================================
# TAB 3 — Validation & Methods
# ============================================================
with tab3:
    st.caption(
        "This section improves trust. It shows raw data slices and method notes so a reviewer can validate the ranking and understand tradeoffs."
    )

    st.subheader("Validation checks")
    st.caption("These tables help verify that the scoring results align with actual GitHub PR and review activity in the selected window.")

    v1, v2 = st.columns(2)

    with v1:
        st.markdown("### Top raw merged PR authors (window)")
        st.caption("Sanity check for shipping activity leaders using raw merged PR counts.")
        top_auth = (
            td.prs_merged_window.groupby("author_login")
            .size()
            .reset_index(name="merged_prs")
            .sort_values("merged_prs", ascending=False)
            .head(15)
        )
        st.dataframe(top_auth, width="stretch", hide_index=True)

        st.markdown("### Top reviewers (window)")
        st.caption("Sanity check for review leverage leaders using raw review event counts.")
        top_rev = (
            td.reviews_window.groupby("reviewer_login")
            .size()
            .reset_index(name="reviews")
            .sort_values("reviews", ascending=False)
            .head(15)
        )
        st.dataframe(top_rev, width="stretch", hide_index=True)

    with v2:
        st.markdown("### Sample merged PR records")
        st.caption("Example records used to derive shipping volume, cycle proxies, and inferred domain coverage.")
        sample_pr = td.prs_merged_window[
            [
                "pr_number",
                "author_login",
                "pr_title",
                "merged_at",
                "pr_size",
                "pr_volume_points",
                "domain_primary",
            ]
        ].sort_values("merged_at", ascending=False).head(20)
        st.dataframe(sample_pr, width="stretch", hide_index=True)

        st.markdown("### Sample reviews")
        st.caption("Example review records used to measure approvals, review volume, and cross-author support.")
        sample_rv = td.reviews_window[
            [
                "pr_number",
                "pr_author_login",
                "reviewer_login",
                "review_state",
                "review_submitted_at",
            ]
        ].sort_values("review_submitted_at", ascending=False).head(20)
        st.dataframe(sample_rv, width="stretch", hide_index=True)

    st.divider()

    st.subheader("Methodology")
    st.caption("This explains what the score means, what it intentionally does *not* measure, and how to interpret the results.")

    with st.expander("What 'impact' means here", expanded=True):
        st.markdown(
            """
Impact is modeled as a combination of:

- **Shipped work** (merged PR count + log-scaled volume)
- **Review leverage** (reviews, approvals, breadth of authors supported)
- **Execution quality proxies** (cycle time, review coverage on authored PRs, large-PR discipline)
- **Consistency and collaboration breadth**
- **Momentum** (recent 30d vs prior period)

This is designed to be **leader-useful** and **explainable**, not a perfect measure of engineering quality.
"""
        )

    with st.expander("Why this is more useful than commit counts"):
        st.markdown(
            """
Commits and lines of code are often noisy and easy to game. This dashboard instead prioritizes:

- work that **shipped**
- contribution to **team throughput** through reviews
- **sustained activity** over time
- collaboration across contributors
- healthier execution patterns (e.g., review coverage and PR size discipline)
"""
        )

    with st.expander("AI layer (how it adds value)"):
        st.markdown(
            """
The AI layer helps a busy leader understand the data faster:

- **Executive view summary** explains the top cohort and contribution mix
- **Selected engineer narrative** translates metric patterns into a concise explanation

If no LLM API key is configured, the dashboard uses a **deterministic local summarizer** for reproducibility.
"""
        )

    st.markdown("### Weighting (base weights)")
    st.caption("These are the base feature weights used by the scoring model before normalization and bonus logic.")
    weights_df = pd.DataFrame([{"metric": k, "weight": v} for k, v in WEIGHTS.items()]).sort_values("weight", ascending=False)
    st.dataframe(weights_df, width="stretch", hide_index=True)

    with st.expander("Known limitations"):
        st.markdown(
            """
- Uses metadata proxies and **cannot directly measure** code quality, architecture quality, mentoring depth, or incident response.
- Domain classification is **heuristic** (labels/title-based) and intended for directional insight.
- Nested pagination for very large PR review threads may not be fully expanded in this assignment version (pragmatic scope tradeoff).
"""
        )