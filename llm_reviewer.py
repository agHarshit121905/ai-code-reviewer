
"""
LLM Review Layer — Phase 3 (Groq version)
============================================
Takes a diff (already passed through static_analyzer.py) and sends it to
Groq (running Llama 3.3 70B) for SEMANTIC code review — logic bugs, missing
edge cases, security implications that depend on context, and design
concerns that static analysis structurally cannot catch.

Uses Groq's structured output (json_schema mode) to force valid JSON, so
the result is directly comparable to your eval set and reliably parseable
downstream (Phase 5 eval harness, Phase 6 API).

Why Groq instead of Gemini: Groq's free tier is far more generous for a
one-off batch job like this — 30 requests/minute and roughly 1,000
requests/day per the free tier, vs Gemini's punishing 20/day cap on
gemini-2.5-flash. No credit card required either way. Note: Groq's rate
limits apply at the organization level, not per key, so multiple keys do
NOT extend your quota the way they did for Gemini — you don't need them
here, since one key comfortably covers a 60-PR run in a single sitting.

Design notes (same as the Gemini/Anthropic versions):
  - Reuses parse_diff() from static_analyzer.py to give the model exact,
    pre-computed line numbers instead of making it count through raw
    diff +/- markers.
  - Static analysis findings are passed in as context so the model doesn't
    waste effort re-flagging unused imports / naming / etc.
  - Responses are cached by diff hash (data/llm_review_cache.jsonl) so
    re-running the eval harness doesn't re-spend free-tier quota on diffs
    you've already reviewed.
  - A fixed delay between calls keeps you comfortably under the 30 RPM
    limit instead of burning retries on 429 errors.

Setup:
    pip install groq
    Get a free key at https://console.groq.com/keys (no card needed)
    export GROQ_API_KEY="gsk_..."

Usage:
    from llm_reviewer import review_diff
    issues = review_diff(diff_text, static_issues=static_issues)
"""

import os
import json
import hashlib
import time

from static_analyzer import parse_diff

# IMPORTANT: strict json_schema mode (used below for guaranteed valid JSON)
# is only supported on openai/gpt-oss-20b and openai/gpt-oss-120b per Groq's
# docs — NOT on llama-3.3-70b-versatile, which was the original choice here
# and caused inconsistent 400 errors ("This model does not support response
# format json_schema") partway through a batch run. 120b is the stronger of
# the two supported models; 20b is available if you want higher throughput
# at slightly lower quality. Free tier rate limits are comparable to Llama.
MODEL = "openai/gpt-oss-120b"

CACHE_PATH = "data/llm_review_cache.jsonl"

# Free tier allows 30 requests/minute (one every 2 seconds). A slightly
# longer pause than the bare minimum keeps you safely under that even
# accounting for network jitter.
SECONDS_BETWEEN_CALLS = 2.5

VALID_CATEGORIES = [
    "bug", "security", "design", "performance", "style",
    "test-coverage", "documentation", "type-safety", "build-config",
    "cleanup", "other",
]

# Groq's structured-output schema (OpenAI-compatible JSON Schema format).
# strict:true requires every field to be in "required" and objects to set
# additionalProperties:false — the API enforces this at the token level,
# guaranteeing valid output that matches the schema exactly.
RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "code_review_issues",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File path from the diff, exactly as given",
                            },
                            "line": {
                                "type": "integer",
                                "description": "Line number in the NEW file, exactly as annotated in the input",
                            },
                            "category": {
                                "type": "string",
                                "enum": VALID_CATEGORIES,
                            },
                            "severity": {
                                "type": "string",
                                "enum": ["error", "warning", "info"],
                            },
                            "body": {
                                "type": "string",
                                "description": "Clear, specific explanation, phrased as a human reviewer would write a PR comment",
                            },
                            "suggested_fix": {
                                "type": "string",
                                "description": "Short suggestion for how to fix it, or empty string if not applicable",
                            },
                        },
                        "required": ["path", "line", "category", "severity", "body", "suggested_fix"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["issues"],
            "additionalProperties": False,
        },
    },
}

SYSTEM_PROMPT = """You are an expert code reviewer for Python and C++ codebases, reviewing a single pull request diff.

A static analysis pass has already run and caught: unused imports, missing docstrings, naming convention violations, mutable default arguments, bare except clauses, long functions, cyclomatic complexity, and some raw C++ patterns (malloc, NULL, using namespace std, etc). Do NOT repeat any issue already listed under "Static analysis already found".

Focus only on issues that require semantic understanding of the code's intent:
- Logic bugs and incorrect behavior
- Missing edge cases (empty inputs, None/null handling, off-by-one, concurrent access)
- Security implications that depend on context (injection, unsafe deserialization, secrets handling)
- API design concerns (breaking changes, inconsistent interfaces, unclear naming that static rules can't catch)
- Missing test coverage for the specific change being made
- Performance issues that require understanding what the code does (not just complexity metrics)

Rules:
- Only flag issues on lines that were actually ADDED in this diff (given with explicit line numbers below). Never flag unchanged or deleted code, and never invent a line number that wasn't given to you.
- Be specific: reference the actual variable/function names in the diff, not generic advice.
- Do not invent issues to pad the list. If the diff is clean, return an empty issues array.
- Do not comment on style/formatting/naming — that's static analysis's job.
- Write each issue's body the way a real human reviewer would phrase a PR comment: direct, concise, no preamble like "I noticed that...".
- Limit to the most important issues (max 8 per file) — prioritize correctness and security over minor nitpicks.
- If a field like suggested_fix doesn't apply, use an empty string rather than omitting it.
"""


def annotate_diff_with_lines(diff_text: str) -> str:
    """
    Reconstruct the diff into a line-numbered view per file, reusing the
    same parse_diff() logic from static_analyzer.py. This gives the model
    exact, reliable line numbers instead of asking it to count through
    raw +/- diff markers.
    """
    chunks = parse_diff(diff_text)
    sections = []
    for chunk in chunks:
        path = chunk["path"]
        lines = chunk["added_lines"]
        if not lines:
            continue
        body = "\n".join(f"  {lineno}: {content}" for lineno, content in lines)
        sections.append(f"File: {path}\n{body}")
    return "\n\n".join(sections)


def format_static_issues(static_issues: list) -> str:
    if not static_issues:
        return "(none)"
    lines = []
    for issue in static_issues:
        lines.append(f"- {issue['path']}:{issue['line']} [{issue['category']}] {issue['body']}")
    return "\n".join(lines)


def _diff_hash(diff_text: str) -> str:
    return hashlib.sha256(diff_text.encode("utf-8")).hexdigest()[:16]


def _load_cache() -> dict:
    if not os.path.exists(CACHE_PATH):
        return {}
    cache = {}
    with open(CACHE_PATH) as f:
        for line in f:
            record = json.loads(line)
            cache[record["hash"]] = record["issues"]
    return cache


def _append_cache(diff_hash: str, issues: list):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "a") as f:
        f.write(json.dumps({"hash": diff_hash, "issues": issues}) + "\n")


def review_diff(
    diff_text: str,
    static_issues: list = None,
    use_cache: bool = True,
    max_retries: int = 5,
) -> list:
    """
    Send a diff to Groq (Llama 3.3 70B) for semantic code review.

    Args:
        diff_text: raw unified diff string (same format used throughout the project)
        static_issues: issues already found by static_analyzer.analyze_diff(),
                       passed in so the model doesn't repeat them
        use_cache: if True, skip the API call entirely when this exact diff
                   has already been reviewed (keyed by content hash)
        max_retries: retry attempts on rate limit / transient errors, with
                     exponential backoff

    Returns:
        List of issue dicts: {path, line, category, severity, body,
        suggested_fix, source: "llm"} — same shape as static_analyzer output
        plus "source": "llm" so Phase 5 can tell the two layers apart.
    """
    from groq import Groq  # imported lazily so this module can be tested without the package installed

    static_issues = static_issues or []
    diff_hash = _diff_hash(diff_text)

    if use_cache:
        cache = _load_cache()
        if diff_hash in cache:
            return cache[diff_hash]

    annotated = annotate_diff_with_lines(diff_text)
    if not annotated.strip():
        return []

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys "
            "and set it with: os.environ['GROQ_API_KEY'] = 'gsk_...'"
        )
    client = Groq(api_key=api_key)

    user_message = (
        f"Static analysis already found:\n{format_static_issues(static_issues)}\n\n"
        f"Diff (added lines only, with line numbers in the NEW file):\n\n"
        f"{annotated}\n\n"
        f"Call report_issues with anything you find. If nothing else needs "
        f"flagging, return an empty issues array."
    )

    response = None
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_format=RESPONSE_SCHEMA,
                temperature=0,
            )
            break
        except Exception as e:
            last_error = e
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower():
                wait = 2 ** attempt * 5
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            elif "503" in err_str or "502" in err_str or "500" in err_str or "UNAVAILABLE" in err_str:
                wait = 2 ** attempt * 5
                print(f"  Server unavailable, waiting {wait}s and retrying...")
                time.sleep(wait)
            else:
                raise
    if response is None:
        # Do NOT return [] here — that would get cached as a false "0 issues
        # found" result below, indistinguishable from a genuinely clean diff.
        # Raise instead so the caller's loop can catch it per-PR, skip caching,
        # and retry just this PR later without corrupting results for the rest.
        raise RuntimeError(f"Max retries exceeded on Groq API: {last_error}")

    # Throttle regardless of whether we hit a retry, to stay under free-tier RPM
    time.sleep(SECONDS_BETWEEN_CALLS)

    try:
        parsed = json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, AttributeError, TypeError, IndexError):
        return []

    issues = []
    for issue in parsed.get("issues", []):
        issues.append({
            "path": issue["path"],
            "line": issue["line"],
            "category": issue["category"],
            "severity": issue["severity"],
            "body": issue["body"],
            "suggested_fix": issue.get("suggested_fix", ""),
            "source": "llm",
        })

    if use_cache:
        _append_cache(diff_hash, issues)

    return issues
