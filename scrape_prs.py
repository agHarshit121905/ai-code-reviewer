"""
GitHub PR Review Scraper
=========================
Builds an eval dataset of (diff, human_review_comments) pairs from real
GitHub PRs. This dataset is later used to measure your AI code reviewer's
precision/recall against actual human reviewer feedback.

Usage:
    export GITHUB_TOKEN="github_pat_xxxxx"
    python scrape_prs.py

Output:
    data/raw_prs.jsonl  — one JSON object per line, one per PR
"""

import os
import json
import time
import requests
from datetime import datetime, timedelta

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError(
        "Set GITHUB_TOKEN as an environment variable before running this script."
    )

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

# Repos chosen for high-quality, substantive review comments.
# Mix of Python and C++ as requested.
REPOS = [
    "psf/requests",
    "django/django",              # swapped in for pallets/flask — returned 0/25 kept
    "nlohmann/json",
    "fmtlib/fmt",
    "grpc/grpc",   # swapped in for protobuf — protobuf returned 0/60, likely reviewed off-GitHub
]

# Tune these to control dataset size vs. API rate limit usage
PRS_PER_REPO = 60          # raised from 25 — only ~15-20% of merged PRs survive filters
MIN_REVIEW_COMMENTS = 2    # skip PRs with too little review signal
MAX_REVIEW_COMMENTS = 25   # skip PRs that are really one sprawling thread, not many distinct issues
MAX_DIFF_LINES = 800       # skip huge diffs (not useful for eval, too noisy)

OUTPUT_PATH = "data/raw_prs.jsonl"


def get(url, params=None):
    """GET with basic rate-limit handling."""
    resp = requests.get(url, headers=HEADERS, params=params)
    if resp.status_code == 403 and "rate limit" in resp.text.lower():
        reset_time = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
        wait = max(reset_time - time.time(), 5)
        print(f"  Rate limited. Sleeping {wait:.0f}s...")
        time.sleep(wait)
        return get(url, params)
    resp.raise_for_status()
    return resp


def fetch_merged_prs(repo, count):
    """Fetch recently merged PRs for a repo."""
    prs = []
    page = 1
    while len(prs) < count:
        url = f"https://api.github.com/repos/{repo}/pulls"
        params = {
            "state": "closed",
            "sort": "updated",
            "direction": "desc",
            "per_page": 50,
            "page": page,
        }
        resp = get(url, params)
        batch = resp.json()
        if not batch:
            break
        for pr in batch:
            if pr.get("merged_at"):
                prs.append(pr)
            if len(prs) >= count:
                break
        page += 1
        if page > 5:  # safety stop
            break
    return prs[:count]


def fetch_review_comments(repo, pr_number):
    """Fetch inline review comments (the actual code-level feedback we want)."""
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/comments"
    comments = []
    page = 1
    while True:
        resp = get(url, {"per_page": 100, "page": page})
        batch = resp.json()
        if not batch:
            break
        comments.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return comments


def fetch_pr_diff(repo, pr_number):
    """Fetch the raw diff for a PR."""
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    resp = requests.get(
        url,
        headers={**HEADERS, "Accept": "application/vnd.github.v3.diff"},
    )
    resp.raise_for_status()
    return resp.text


def scrape_repo(repo, out_f):
    print(f"\n=== {repo} ===")
    prs = fetch_merged_prs(repo, PRS_PER_REPO)
    print(f"Found {len(prs)} merged PRs")

    kept = 0
    for pr in prs:
        pr_number = pr["number"]
        try:
            comments = fetch_review_comments(repo, pr_number)
        except requests.HTTPError as e:
            print(f"  PR #{pr_number}: failed to fetch comments ({e})")
            continue

        if len(comments) < MIN_REVIEW_COMMENTS or len(comments) > MAX_REVIEW_COMMENTS:
            continue

        try:
            diff = fetch_pr_diff(repo, pr_number)
        except requests.HTTPError as e:
            print(f"  PR #{pr_number}: failed to fetch diff ({e})")
            continue

        if diff.count("\n") > MAX_DIFF_LINES:
            continue

        record = {
            "repo": repo,
            "pr_number": pr_number,
            "title": pr["title"],
            "url": pr["html_url"],
            "merged_at": pr["merged_at"],
            "diff": diff,
            "review_comments": [
                {
                    "path": c["path"],
                    "line": c.get("line") or c.get("original_line"),
                    "body": c["body"],
                    "author": c["user"]["login"],
                }
                for c in comments
            ],
        }
        out_f.write(json.dumps(record) + "\n")
        kept += 1
        print(f"  PR #{pr_number}: kept ({len(comments)} review comments)")

        time.sleep(0.3)  # be polite to the API

    print(f"Kept {kept}/{len(prs)} PRs from {repo}")


def main():
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_PATH, "w") as out_f:
        for repo in REPOS:
            scrape_repo(repo, out_f)
    print(f"\nDone. Dataset written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
