from __future__ import annotations

import os
import time
from typing import Dict, Any, List, Tuple, Optional

import requests
import pandas as pd
from dotenv import load_dotenv

from src.utils import parse_github_dt

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"


def load_config() -> Dict[str, str]:
    load_dotenv()
    token = os.getenv("GITHUB_TOKEN", "").strip()
    owner = os.getenv("GITHUB_OWNER", "PostHog").strip()
    repo = os.getenv("GITHUB_REPO", "posthog").strip()

    if not token:
        raise ValueError("Missing GITHUB_TOKEN in .env")
    return {"token": token, "owner": owner, "repo": repo}


def graphql_request(query: str, variables: dict, token: str) -> Dict[str, Any]:
    import os
    import urllib3

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # Temporary SSL bypass for environments with broken local CA chain.
    # Set GITHUB_SSL_VERIFY=false in .env to enable.
    ssl_verify_env = os.getenv("GITHUB_SSL_VERIFY", "true").strip().lower()
    verify_ssl = ssl_verify_env not in {"false", "0", "no"}

    if not verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    resp = requests.post(
        GITHUB_GRAPHQL_URL,
        json={"query": query, "variables": variables},
        headers=headers,
        timeout=60,
        verify=verify_ssl,  # <-- key workaround
    )

    if resp.status_code != 200:
        raise RuntimeError(f"GraphQL HTTP {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {data['errors']}")
    return data["data"]


PRS_QUERY = """
query RepoPRs($owner: String!, $repo: String!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    pullRequests(
      first: 50
      after: $cursor
      orderBy: {field: UPDATED_AT, direction: DESC}
      states: [OPEN, CLOSED, MERGED]
    ) {
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        number
        title
        url
        state
        createdAt
        updatedAt
        closedAt
        mergedAt
        merged
        additions
        deletions
        changedFiles
        author {
          login
        }
        labels(first: 20) {
          nodes {
            name
          }
        }
        reviews(first: 50) {
          totalCount
          nodes {
            author {
              login
            }
            state
            submittedAt
          }
        }
      }
    }
  }
}
"""


def fetch_prs_and_reviews(
    owner: str,
    repo: str,
    token: str,
    max_pages: int = 25,
    sleep_s: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch recent PRs (ordered by updatedAt desc) and nested reviews.
    We fetch a fixed number of pages for speed and filter to 90d later.
    """
    all_pr_rows: List[Dict[str, Any]] = []
    all_review_rows: List[Dict[str, Any]] = []

    cursor: Optional[str] = None
    page_count = 0

    while True:
        page_count += 1
        variables = {"owner": owner, "repo": repo, "cursor": cursor}
        data = graphql_request(PRS_QUERY, variables, token)

        pr_conn = data["repository"]["pullRequests"]
        nodes = pr_conn["nodes"]

        if not nodes:
            break

        for pr in nodes:
            author_login = None
            if pr.get("author"):
                author_login = pr["author"].get("login")

            labels = pr.get("labels", {}).get("nodes", []) or []
            label_names = [x["name"] for x in labels if x and x.get("name")]

            pr_row = {
                "pr_number": pr.get("number"),
                "pr_title": pr.get("title"),
                "pr_url": pr.get("url"),
                "state": pr.get("state"),
                "created_at": pr.get("createdAt"),
                "updated_at": pr.get("updatedAt"),
                "closed_at": pr.get("closedAt"),
                "merged_at": pr.get("mergedAt"),
                "merged": pr.get("merged"),
                "additions": pr.get("additions"),
                "deletions": pr.get("deletions"),
                "changed_files": pr.get("changedFiles"),
                "author_login": author_login,
                "labels": "|".join(label_names),
                "review_count_reported": pr.get("reviews", {}).get("totalCount", 0),
            }
            all_pr_rows.append(pr_row)

            review_nodes = pr.get("reviews", {}).get("nodes", []) or []
            for rv in review_nodes:
                reviewer_login = None
                if rv.get("author"):
                    reviewer_login = rv["author"].get("login")

                all_review_rows.append(
                    {
                        "pr_number": pr.get("number"),
                        "pr_author_login": author_login,
                        "reviewer_login": reviewer_login,
                        "review_state": rv.get("state"),
                        "review_submitted_at": rv.get("submittedAt"),
                        "pr_merged": pr.get("merged"),
                        "pr_merged_at": pr.get("mergedAt"),
                        "pr_created_at": pr.get("createdAt"),
                    }
                )

        print(f"Fetched page {page_count} with {len(nodes)} PRs (total PR rows: {len(all_pr_rows)})")

        page_info = pr_conn["pageInfo"]
        has_next = page_info["hasNextPage"]
        cursor = page_info["endCursor"]

        if not has_next:
            break
        if page_count >= max_pages:
            break

        time.sleep(sleep_s)

    prs_df = pd.DataFrame(all_pr_rows)
    reviews_df = pd.DataFrame(all_review_rows)

    return prs_df, reviews_df


def save_raw_data(prs_df: pd.DataFrame, reviews_df: pd.DataFrame, out_dir: str = "data") -> None:
    os.makedirs(out_dir, exist_ok=True)
    prs_path = os.path.join(out_dir, "prs_raw.csv")
    reviews_path = os.path.join(out_dir, "reviews_raw.csv")

    prs_df.to_csv(prs_path, index=False)
    reviews_df.to_csv(reviews_path, index=False)
    print(f"Saved {prs_path} ({len(prs_df)} rows)")
    print(f"Saved {reviews_path} ({len(reviews_df)} rows)")


if __name__ == "__main__":
    cfg = load_config()
    prs_df, reviews_df = fetch_prs_and_reviews(
        owner=cfg["owner"],
        repo=cfg["repo"],
        token=cfg["token"],
        max_pages=30,  # increase if needed
    )
    save_raw_data(prs_df, reviews_df)