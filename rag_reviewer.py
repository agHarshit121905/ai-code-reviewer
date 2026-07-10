"""
RAG-Augmented Reviewer — Phase 4 (part 2 of 2)
=================================================
Extends the Phase 3 LLM reviewer with codebase context: for each file
changed in the diff, retrieves the most similar existing functions from a
pre-built index (codebase_indexer.py) and injects them into the prompt.

This lets the reviewer answer questions the diff alone cannot:
  - "Does this new function duplicate an existing utility?"
  - "Does this match the error-handling pattern used elsewhere?"
  - "Is there an existing helper this should call instead?"

Design notes:
  - Retrieval is per-changed-file: the added lines for each file form the
    query. exclude_path skips the changed file's own (pre-change) chunks,
    since retrieving the old version of what you're editing is noise.
  - Retrieved context is capped (MAX_CONTEXT_CHUNKS per file, each chunk
    truncated) to control prompt size — more context is not always better,
    and Groq free tier has token limits.
  - Falls back gracefully: if no index is provided, behaves identically to
    the plain Phase 3 reviewer. This makes A/B comparison in the eval
    harness trivial: same code path, index vs no index.

Usage:
    from codebase_indexer import load_index
    from rag_reviewer import review_diff_with_rag

    index, chunks = load_index("requests_index")
    issues = review_diff_with_rag(diff, static_issues=[], index=index, chunks=chunks)
"""

import json

from llm_reviewer import (
    review_diff,           # plain reviewer, reused as the fallback path
    annotate_diff_with_lines,
    format_static_issues,
    _diff_hash,
    _load_cache,
    _append_cache,
    MODEL,
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
    SECONDS_BETWEEN_CALLS,
)
from static_analyzer import parse_diff
from codebase_indexer import search

# How many similar existing functions to retrieve per changed file.
# More isn't better: each chunk costs prompt tokens and dilutes attention.
MAX_CONTEXT_CHUNKS = 3

# Truncate each retrieved chunk to this many characters in the prompt.
MAX_CHUNK_CHARS = 800

# Only include retrieved chunks at least this similar to the query.
# Below this, the "similar" function usually isn't related enough to help,
# and irrelevant context actively hurts (the model may hallucinate
# connections to it).
MIN_SIMILARITY = 0.35

# Separate cache namespace: a diff reviewed WITH context is a different
# review than the same diff without it, so they must not share cache keys.
RAG_CACHE_SUFFIX = "+rag"

RAG_SYSTEM_ADDENDUM = """

Additionally, you are given EXISTING CODE FROM ELSEWHERE IN THIS CODEBASE that is semantically similar to the changed code. Use it to check for:
- Duplication: does the new code reimplement something that already exists? If so, flag it and name the existing function.
- Consistency: does the new code deviate from patterns the codebase already uses (error handling, naming, argument conventions)?
Do not flag issues in the existing-code context itself — it is reference material, not part of the diff under review."""


def _build_rag_context(diff_text: str, index, chunks: list) -> str:
    """
    For each changed file in the diff, retrieve similar existing functions
    and format them as a context block for the prompt. Returns "" if
    nothing sufficiently similar was found.
    """
    file_chunks = parse_diff(diff_text)
    sections = []

    for fc in file_chunks:
        path = fc["path"]
        added_lines = fc["added_lines"]
        if not added_lines:
            continue

        query_text = "\n".join(content for _, content in added_lines)
        if len(query_text.strip()) < 40:
            continue  # too little changed code to form a meaningful query

        results = search(index, chunks, query_text,
                         k=MAX_CONTEXT_CHUNKS, exclude_path=path)
        relevant = [r for r in results if r["similarity"] >= MIN_SIMILARITY]
        if not relevant:
            continue

        block_lines = [f"Similar existing code (relevant to changes in {path}):"]
        for r in relevant:
            snippet = r["text"][:MAX_CHUNK_CHARS]
            block_lines.append(
                f"\n--- {r['path']}:{r['start_line']} `{r['name']}` "
                f"(similarity {r['similarity']}) ---\n{snippet}"
            )
        sections.append("\n".join(block_lines))

    return "\n\n".join(sections)


def review_diff_with_rag(
    diff_text: str,
    static_issues: list = None,
    index=None,
    chunks: list = None,
    use_cache: bool = True,
    max_retries: int = 5,
) -> list:
    """
    Review a diff with retrieved codebase context injected into the prompt.

    If index/chunks are None, falls back to the plain Phase 3 reviewer —
    identical behavior, which makes baseline-vs-RAG comparison a pure A/B.

    Returns the same issue schema as review_diff(), so the eval harness
    works on the output unchanged.
    """
    from groq import Groq
    import os
    import time

    if index is None or chunks is None:
        return review_diff(diff_text, static_issues=static_issues,
                           use_cache=use_cache, max_retries=max_retries)

    static_issues = static_issues or []
    diff_hash = _diff_hash(diff_text) + RAG_CACHE_SUFFIX

    if use_cache:
        cache = _load_cache()
        if diff_hash in cache:
            return cache[diff_hash]

    annotated = annotate_diff_with_lines(diff_text)
    if not annotated.strip():
        return []

    rag_context = _build_rag_context(diff_text, index, chunks)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys"
        )
    client = Groq(api_key=api_key)

    context_section = (
        f"Existing code from elsewhere in this codebase (reference only):\n\n"
        f"{rag_context}\n\n" if rag_context else ""
    )

    user_message = (
        f"Static analysis already found:\n{format_static_issues(static_issues)}\n\n"
        f"{context_section}"
        f"Diff (added lines only, with line numbers in the NEW file):\n\n"
        f"{annotated}\n\n"
        f"Report anything you find. If nothing needs flagging, return an "
        f"empty issues array."
    )

    system_prompt = SYSTEM_PROMPT + (RAG_SYSTEM_ADDENDUM if rag_context else "")

    response = None
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
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
            elif any(code in err_str for code in ("503", "502", "500", "UNAVAILABLE")):
                wait = 2 ** attempt * 5
                print(f"  Server unavailable, waiting {wait}s and retrying...")
                time.sleep(wait)
            else:
                raise
    if response is None:
        raise RuntimeError(f"Max retries exceeded on Groq API: {last_error}")

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
            "source": "llm+rag",
        })

    if use_cache:
        _append_cache(diff_hash, issues)

    return issues
