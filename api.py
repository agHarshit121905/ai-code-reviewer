"""
Code Reviewer API — Phase 6
=============================
FastAPI web service that wraps the two-layer review pipeline (static
analyzer + LLM reviewer) behind an HTTP endpoint. Takes a unified diff,
returns structured issues as JSON.

Reuses analyze_diff() and review_diff() unchanged — this is purely an
HTTP layer on top of the existing pipeline, not a reimplementation.

Endpoints:
  GET  /              — health check / basic info
  GET  /health        — health check for uptime monitoring
  POST /review        — submit a diff, get back issues
  POST /review/github — submit a GitHub PR URL, fetch its diff, review it

Run locally:
    pip install fastapi uvicorn
    uvicorn api:app --reload --port 8000
    # then open http://localhost:8000/docs for interactive API docs

Deploy (Render, etc):
    uvicorn api:app --host 0.0.0.0 --port $PORT
"""

import os
import re
import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from static_analyzer import analyze_diff
from llm_reviewer import review_diff

app = FastAPI(
    title="AI Code Reviewer",
    description="Two-layer code review: deterministic static analysis + LLM semantic review. "
                "Submit a unified diff or a GitHub PR URL, get back structured issues.",
    version="1.0.0",
)

# Allow browser-based clients (e.g. a simple demo frontend) to call the API.
# Wide-open CORS is fine for a portfolio demo; tighten allow_origins for prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models (Pydantic gives automatic validation + API docs)
# ---------------------------------------------------------------------------

class ReviewRequest(BaseModel):
    diff: str = Field(
        ...,
        description="A unified diff (git diff format) to review",
        json_schema_extra={"example": "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1,2 @@\n+def f(items=[]):\n+    pass\n"},
    )
    use_llm: bool = Field(
        True,
        description="If false, run only the fast/free static analysis layer (no LLM call).",
    )


class GitHubReviewRequest(BaseModel):
    pr_url: str = Field(
        ...,
        description="A GitHub pull request URL, e.g. https://github.com/psf/requests/pull/7502",
        json_schema_extra={"example": "https://github.com/psf/requests/pull/7502"},
    )
    use_llm: bool = Field(True, description="Include the LLM semantic review layer.")


class Issue(BaseModel):
    path: str
    line: int | None
    category: str
    severity: str
    body: str
    source: str
    suggested_fix: str = ""


class ReviewResponse(BaseModel):
    total_issues: int
    static_issue_count: int
    llm_issue_count: int
    issues: list[Issue]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "AI Code Reviewer",
        "version": "1.0.0",
        "endpoints": {
            "POST /review": "Review a raw unified diff",
            "POST /review/github": "Review a GitHub PR by URL",
            "GET /health": "Health check",
            "GET /docs": "Interactive API documentation",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


def _run_pipeline(diff: str, use_llm: bool) -> ReviewResponse:
    """Shared logic for both review endpoints."""
    if not diff.strip():
        raise HTTPException(status_code=400, detail="Empty diff provided.")

    static_issues = analyze_diff(diff)

    llm_issues = []
    if use_llm:
        try:
            llm_issues = review_diff(diff, static_issues=static_issues)
        except RuntimeError as e:
            # LLM layer failed (e.g. missing API key, quota) — degrade
            # gracefully to static-only rather than failing the whole request.
            raise HTTPException(
                status_code=502,
                detail=f"LLM review layer failed: {e}. Retry with use_llm=false for static-only review.",
            )

    all_issues = static_issues + llm_issues
    # Normalize: static issues have no suggested_fix field; add empty default
    for issue in all_issues:
        issue.setdefault("suggested_fix", "")

    return ReviewResponse(
        total_issues=len(all_issues),
        static_issue_count=len(static_issues),
        llm_issue_count=len(llm_issues),
        issues=all_issues,
    )


@app.post("/review", response_model=ReviewResponse)
def review(request: ReviewRequest):
    """Review a raw unified diff and return structured issues."""
    return _run_pipeline(request.diff, request.use_llm)


# GitHub PR URL pattern: https://github.com/{owner}/{repo}/pull/{number}
GITHUB_PR_RE = re.compile(r"github\.com/([^/]+)/([^/]+)/pull/(\d+)")


@app.post("/review/github", response_model=ReviewResponse)
def review_github(request: GitHubReviewRequest):
    """Fetch a GitHub PR's diff by URL and review it."""
    match = GITHUB_PR_RE.search(request.pr_url)
    if not match:
        raise HTTPException(
            status_code=400,
            detail="Invalid GitHub PR URL. Expected format: "
                   "https://github.com/owner/repo/pull/123",
        )

    owner, repo, pr_number = match.group(1), match.group(2), match.group(3)

    # Fetch the diff. A GITHUB_TOKEN is optional but avoids the low
    # unauthenticated rate limit; use it if present in the environment.
    headers = {"Accept": "application/vnd.github.v3.diff"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
    try:
        resp = requests.get(api_url, headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Failed to fetch PR diff from GitHub: {e}",
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Network error fetching PR: {e}")

    return _run_pipeline(resp.text, request.use_llm)
