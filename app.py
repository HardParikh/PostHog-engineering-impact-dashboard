from __future__ import annotations

import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.github_fetch import load_config, fetch_prs_and_reviews, save_raw_data
from src.transform import transform_data
from src.scoring import compute_scores, WEIGHTS
from src.ai_insights import (
    generate_rule_based_engineer_summary,
    generate_rule_based_exec_summary,
    maybe_openai_engineer_summary,
)

import os
import streamlit as st

# TEMP DEBUG (remove before final submission)
# if st.sidebar.checkbox("Debug: show API key status", value=False):
#     raw_key = os.getenv("OPENAI_API_KEY")
#     st.sidebar.write("OPENAI_API_KEY present:", bool(raw_key))
#     if raw_key:
#         st.sidebar.write("Key prefix:", raw_key[:7] + "..." if len(raw_key) > 7 else raw_key)
#         st.sidebar.write("Key length:", len(raw_key))
#     else:
#         st.sidebar.error("OPENAI_API_KEY not found in this Streamlit process")

st.set_page_config(
    page_title="PostHog Engineering Impact Dashboard",
    page_icon="📈",
    layout="wide",
)

# -----------------------------------
# Session state defaults (selection preservation)
# -----------------------------------
if "selected_engineer" not in st.session_state:
    st.session_state.selected_engineer = None

if "last_filters_signature" not in st.session_state:
    st.session_state.last_filters_signature = None

if "use_llm_ai" not in st.session_state:
    st.session_state.use_llm_ai = False


# -----------------------------------
# Cached compute
# -----------------------------------
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


# -----------------------------------
# UI helpers
# -----------------------------------
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

    # fallback to top-ranked available
    fallback = options[0]
    st.session_state.selected_engineer = fallback
    return fallback


def apply_filters(metrics_df: pd.DataFrame, scoring, td):
    df = metrics_df.copy()

    # Sidebar filters are read from session_state set by widgets
    name_query = st.session_state.get("f_name_query", "").strip().lower()
    role_filter = st.session_state.get("f_role", "All")
    min_merged = st.session_state.get("f_min_merged", 0)
    min_reviews = st.session_state.get("f_min_reviews", 0)
    min_weeks = st.session_state.get("f_min_weeks", 0)
    domain_filter = st.session_state.get("f_domain", "All")
    momentum_filter = st.session_state.get("f_momentum", "All")
    min_score = st.session_state.get("f_min_score", 0.0)

    if name_query:
        df = df[df["engineer"].str.lower().str.contains(name_query, na=False)]

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
        # filter based on domains_touched string from authored work
        df = df[df["domains_touched"].str.contains(domain_filter, case=False, na=False)]

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
        ("Collaborators", "norm_unique_collaborators", WEIGHTS["norm_unique_collaborators"]),
        ("Momentum", "norm_momentum_score", WEIGHTS["norm_momentum_score"]),
    ]

    rows = []
    for label, col, weight in component_map:
        score = float(selected_row.get(col, 0))
        rows.append({
            "component": label,
            "normalized_score": score,
            "weight": weight,
            "weighted_contribution": score * weight * 100,
        })

    # bonus shown separately
    rows.append({
        "component": "Balanced profile bonus",
        "normalized_score": float(selected_row.get("balanced_profile_bonus", 0)) / 0.05 if selected_row.get("balanced_profile_bonus", 0) else 0,
        "weight": 0.05,
        "weighted_contribution": float(selected_row.get("balanced_profile_bonus", 0)) * 100,
    })

    return pd.DataFrame(rows).sort_values("weighted_contribution", ascending=False)


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
    out = out.rename(columns={
        "impact_score_100": "impact_score",
        "contributor_archetype": "archetype",
        "merged_pr_count": "merged_prs",
        "reviews_submitted": "reviews",
        "domain_coverage_count": "domains",
    })
    out["momentum_score"] = out["momentum_score"].round(2)
    st.dataframe(out, width="stretch", hide_index=True)


# -----------------------------------
# Header
# -----------------------------------
st.title("📈 PostHog Engineering Impact Dashboard")
st.caption("Explainable engineering impact across shipping, review leverage, execution flow, consistency, collaboration, and momentum")

# -----------------------------------
# Sidebar
# -----------------------------------
with st.sidebar:
    st.header("Controls")

    # Data controls
    days = st.slider("Lookback window (days)", min_value=30, max_value=180, value=90, step=15)
    top_n = st.slider("Leaderboard size", 5, 25, 10, 1)
    min_activity_points = st.slider("Min activity points", 0, 30, 1, 1)

    st.divider()
    st.subheader("Filters")

    st.text_input("Search engineer", key="f_name_query", placeholder="e.g., andrew, yakko...")
    st.selectbox("Contributor archetype", ["All", "Balanced", "Shipper", "Reviewer", "Mixed"], key="f_role")
    st.slider("Min merged PRs", 0, 30, 0, 1, key="f_min_merged")
    st.slider("Min reviews", 0, 60, 0, 1, key="f_min_reviews")
    st.slider("Min weeks active", 0, 20, 0, 1, key="f_min_weeks")
    st.slider("Min impact score", 0.0, 100.0, 0.0, 1.0, key="f_min_score")
    st.selectbox("Momentum", ["All", "Accelerating", "Stable", "Slowing"], key="f_momentum")

    # Domain filter options (static union of inferred domains)
    domain_options = ["All", "frontend", "backend", "data", "infra_devex", "docs_product_ops", "security", "other"]
    st.selectbox("Domain touched", domain_options, key="f_domain")

    st.divider()
    st.subheader("AI")
    st.toggle("Use LLM summaries", key="use_llm_ai", value=False)

    st.divider()
    st.subheader("Data")
    if st.button("Refresh raw data from GitHub API"):
        with st.spinner("Fetching from GitHub..."):
            try:
                refresh_data_from_github(max_pages=40)
                st.success("Raw data refreshed.")
            except Exception as e:
                st.error(f"Refresh failed: {e}")

# -----------------------------------
# Data presence check
# -----------------------------------
if not raw_data_exists():
    st.warning("No raw data files found in ./data. Use the sidebar to refresh from GitHub API first.")
    st.stop()

# -----------------------------------
# Load/scoring
# -----------------------------------
with st.spinner("Loading and scoring data..."):
    td, scoring = load_transformed_scored(days=days, data_dir="data")

metrics_df = scoring.engineer_metrics.copy()
metrics_df = metrics_df[metrics_df["activity_points"] >= min_activity_points].copy()

filtered_df = apply_filters(metrics_df, scoring, td)

# Preserve selection across reruns/filter changes
selected_engineer = preserve_selection(filtered_df)

# If nothing selected due to filters, stop gracefully
if filtered_df.empty:
    st.info("No engineers match the current filters. Try lowering the thresholds.")
    st.stop()

# -----------------------------------
# Summary row
# -----------------------------------
# Row 1
c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Engineers", int(filtered_df["engineer"].nunique()), "Number of engineers in current filtered view")
with c2:
    metric_card("Merged PRs", int(len(td.prs_merged_window)), "Merged PRs in selected time window")
with c3:
    metric_card("Reviews", int(len(td.reviews_window)), "Reviews submitted in selected time window")
with c4:
    metric_card("Top score", f"{filtered_df['impact_score_100'].max():.1f}", "Top normalized impact score in filtered view")

# Row 2
c5, c6, c7, c8 = st.columns(4)
with c5:
    metric_card("Median score", f"{filtered_df['impact_score_100'].median():.1f}", "Median normalized impact score in filtered view")
with c6:
    metric_card("Window start", td.cutoff_ts.strftime("%Y-%m-%d"), "Start date for analysis window")
with c7:
    metric_card("Window end", td.latest_ts.strftime("%Y-%m-%d"), "Latest timestamp used")
with c8:
    metric_card("Top archetype", filtered_df["contributor_archetype"].mode().iloc[0] if not filtered_df.empty else "—", "Most common archetype in filtered view")

# -----------------------------------
# Tabs (professional layout)
# -----------------------------------
tab1, tab2, tab3 = st.tabs(["🏆 Leaderboard & Drilldown", "🧠 AI Insights & Story", "✅ Methods"])

# -----------------------------------
# TAB 1 — Leaderboard & Drilldown
# -----------------------------------
with tab1:
    left, right = st.columns([1.2, 1.5])

    with left:
        st.subheader("Impact leaderboard")
        render_top_table(filtered_df, top_n=top_n)

        # preserve engineer selection widget using session_state
        options = get_engineer_options(filtered_df)
        if st.session_state.selected_engineer not in options:
            st.session_state.selected_engineer = options[0]

        st.selectbox(
            "Select engineer",
            options=options,
            key="selected_engineer",
            help="Selection persists across filter changes when the engineer remains in the filtered set."
        )

        # Quick peer comparison toggle
        compare_to_top5 = st.toggle("Compare selected engineer to Top 5 averages", value=True)

    with right:
        row = filtered_df.loc[filtered_df["engineer"] == st.session_state.selected_engineer].iloc[0]

        st.subheader(f"Engineer profile: {row['engineer']}")

        r1, r2, r3, r4, r5 = st.columns(5)
        with r1:
            metric_card("Impact score", f"{row['impact_score_100']:.1f}")
        with r2:
            metric_card("Archetype", row["contributor_archetype"])
        with r3:
            metric_card("Merged PRs", int(row["merged_pr_count"]))
        with r4:
            metric_card("Reviews", int(row["reviews_submitted"]))
        with r5:
            metric_card("Weeks active", int(row["weeks_active"]))

        r6, r7, r8, r9, r10 = st.columns(5)
        with r6:
            metric_card("Median cycle", format_hours(row["median_pr_cycle_hours"]))
        with r7:
            metric_card("PR review coverage", format_pct(row["authored_pr_review_coverage_pct"]))
        with r8:
            metric_card("Collaborators", int(row["unique_collaborators"]))
        with r9:
            metric_card("Domains", int(row["domain_coverage_count"]))
        with r10:
            metric_card("Momentum", f"{row['momentum_score']:.2f}")

        st.markdown(f"**Why they rank:** {row['why']}")
        if row.get("domains_touched", ""):
            st.caption(f"Domains touched (inferred): {row['domains_touched']}")

        # Breakdown chart
        breakdown_df = build_component_breakdown(row)
        fig_breakdown = px.bar(
            breakdown_df.sort_values("weighted_contribution", ascending=True),
            x="weighted_contribution",
            y="component",
            orientation="h",
            title="Weighted impact score contributions",
            hover_data={"normalized_score":":.2f", "weight":":.2f", "weighted_contribution":":.2f"},
        )
        fig_breakdown.update_layout(height=420, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig_breakdown, width="stretch")

        # Peer comparison radar-ish alternative using bar for readability
        if compare_to_top5:
            peer = filtered_df.head(min(5, len(filtered_df)))
            peer_avg = {
                "Shipping": peer["shipping_score"].mean(),
                "Review leverage": peer["review_leverage_score"].mean(),
                "Flow": peer["flow_score"].mean(),
                "Consistency+Collab": peer["consistency_collab_score"].mean(),
                "Momentum": peer["norm_momentum_score"].mean(),
            }
            selected_vals = {
                "Shipping": row["shipping_score"],
                "Review leverage": row["review_leverage_score"],
                "Flow": row["flow_score"],
                "Consistency+Collab": row["consistency_collab_score"],
                "Momentum": row["norm_momentum_score"],
            }
            comp_df = pd.DataFrame({
                "dimension": list(peer_avg.keys()) * 2,
                "score": list(selected_vals.values()) + list(peer_avg.values()),
                "series": ["Selected"] * 5 + ["Top 5 avg"] * 5
            })
            fig_comp = px.bar(comp_df, x="dimension", y="score", color="series", barmode="group", title="Selected vs Top 5 average profile")
            fig_comp.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), yaxis_range=[0, 1])
            st.plotly_chart(fig_comp, use_container_width=True)

    st.divider()

    # Weekly trend + impact map
    c_left, c_right = st.columns([1, 1.1])

    with c_left:
        st.subheader("Weekly activity trend")
        weekly = scoring.weekly_activity.copy()
        w = weekly[weekly["engineer"] == st.session_state.selected_engineer].copy()
        if w.empty:
            st.info("No weekly activity data for selected engineer.")
        else:
            w = w.sort_values("week")
            fig_week = go.Figure()
            fig_week.add_trace(go.Scatter(
                x=w["week"], y=w["weekly_activity_points"],
                mode="lines+markers", name="Weekly activity"
            ))
            fig_week.update_layout(
                height=340, margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title="ISO week", yaxis_title="Activity points"
            )
            st.plotly_chart(fig_week, use_container_width=True)

    with c_right:
        st.subheader("Impact map")
        plot_df = filtered_df.copy()
        plot_df["is_top_n"] = plot_df["engineer"].isin(filtered_df.head(top_n)["engineer"])

        fig_scatter = px.scatter(
            plot_df,
            x="shipping_score",
            y="review_leverage_score",
            size="weeks_active",
            color="impact_score_100",
            symbol="contributor_archetype",
            hover_name="engineer",
            hover_data={
                "impact_score_100":":.1f",
                "merged_pr_count":True,
                "reviews_submitted":True,
                "approvals_given":True,
                "authored_pr_review_coverage_pct":":.0f",
                "domain_coverage_count":True,
                "momentum_score":":.2f",
                "is_top_n":True,
            },
            title="Shipping vs review leverage"
        )

        labels_df = plot_df[plot_df["is_top_n"]].head(top_n)
        fig_scatter.add_trace(go.Scatter(
            x=labels_df["shipping_score"],
            y=labels_df["review_leverage_score"],
            mode="text",
            text=labels_df["engineer"],
            textposition="top center",
            showlegend=False
        ))
        fig_scatter.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_scatter, use_container_width=True)

# -----------------------------------
# TAB 2 — AI Insights & Story
# -----------------------------------
with tab2:
    exec_ai_source = "Local rules summarizer"
    if st.session_state.use_llm_ai and os.getenv("OPENAI_API_KEY", "").strip():
        exec_ai_source = "Local rules summarizer (engineer-level LLM summaries enabled separately)"

    st.caption(f"Executive summary source: **{exec_ai_source}**")
    st.subheader("Executive summary")

    top_view = filtered_df.head(top_n).copy()

    # Executive summary
    exec_summary = generate_rule_based_exec_summary(top_view, top_n=min(top_n, 10))
    st.markdown(exec_summary)

    st.divider()

    left2, right2 = st.columns([1.15, 1.35])

    with left2:
        st.markdown("### Selected Engineer Narrative")
        selected_row = filtered_df.loc[filtered_df["engineer"] == st.session_state.selected_engineer].iloc[0]

        narrative = None
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
        story_df = pd.DataFrame([
            {"Signal": "Impact pattern", "Value": str(selected_row["contributor_archetype"])},
            {"Signal": "Momentum", "Value": f"{selected_row['momentum_score']:.2f}"},
            {"Signal": "PR review coverage", "Value": format_pct(selected_row["authored_pr_review_coverage_pct"])},
            {"Signal": "Large PR share", "Value": format_pct(float(selected_row['large_pr_share']) * 100 if pd.notna(selected_row['large_pr_share']) else 0)},
            {"Signal": "Domains touched", "Value": str(int(selected_row["domain_coverage_count"]))},
            {"Signal": "Review-to-ship ratio", "Value": f"{selected_row['review_to_ship_ratio']:.2f}" if pd.notna(selected_row['review_to_ship_ratio']) else "—"},
        ], dtype="object")

        story_df["Signal"] = story_df["Signal"].astype(str)
        story_df["Value"] = story_df["Value"].astype(str)
        st.dataframe(story_df, use_container_width=True, hide_index=True)

    with right2:
        st.markdown("### Portfolio of top contributors")
        top = top_view.copy().head(min(10, len(top_view)))

        # Heatmap-like matrix using normalized dimensions
        matrix_cols = [
            "engineer",
            "shipping_score",
            "review_leverage_score",
            "flow_score",
            "consistency_collab_score",
            "norm_momentum_score",
            "impact_score_100",
        ]
        m = top[matrix_cols].copy()
        m = m.rename(columns={
            "norm_momentum_score": "momentum_norm"
        })
        st.dataframe(m, use_container_width=True, hide_index=True)

        # Archetype distribution
        arch_counts = filtered_df["contributor_archetype"].value_counts().reset_index()
        arch_counts.columns = ["archetype", "count"]
        fig_arch = px.bar(arch_counts, x="archetype", y="count", title="Contributor archetype distribution (filtered view)")
        fig_arch.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_arch, use_container_width=True)

# -----------------------------------
# TAB 3 — Validation & Methods
# -----------------------------------
with tab3:
    st.subheader("Methodology")
    st.markdown("""
### What 'impact' means here
Impact is modeled as a combination of:
- Shipped work (merged PR count + log-scaled volume)
- Review leverage (reviews, approvals, breadth of authors supported)
- Execution quality proxies (cycle time, review coverage on authored PRs, large-PR discipline)
- Consistency and collaboration breadth
- Momentum (recent 30d vs prior period)

### Why this is more useful than commit counts
Commits and LoC are easy to game and vary by working style. This dashboard prioritizes:
- Work that shipped
- Contribution to team throughput via reviews
- Sustained activity
- Collaboration across contributors
- Healthy execution patterns

### AI layer
The dashboard includes AI-assisted summaries:
- Executive view summary for the filtered cohort
- Selected engineer narrative summary
If no LLM API key is configured, it uses a deterministic local summarizer to ensure reproducibility.
""")

    st.markdown("### Weighting (base weights)")
    weights_df = pd.DataFrame([{"metric": k, "weight": v} for k, v in WEIGHTS.items()]).sort_values("weight", ascending=False)
    st.dataframe(weights_df, use_container_width=True, hide_index=True)

    st.markdown("""
### Known limitations
- This uses metadata proxies and cannot directly measure code quality, architectural decisions, mentoring depth, or incident response.
- Domain classification is heuristic (labels/title-based), intended for directional insight.
- Nested pagination for >50 reviews on a PR is not expanded in this version.
""")