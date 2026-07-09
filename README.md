# AI Code Reviewer

A two-layer code review system that combines deterministic static analysis with LLM-based semantic review, evaluated against real human reviewer comments from 62 merged pull requests.

**Live API:** https://ai-code-reviewer-8pz8.onrender.com/docs
*(Free tier — first request may take 30–60s to wake the service.)*

---

## What this is

Most "AI code reviewer" projects are a thin wrapper around an LLM: send a diff, print whatever comes back. This one is built around a different question — **how do you know if the review is any good?**

That question drove three design decisions:

1. **A hybrid architecture.** Static analysis handles what's deterministic (unused imports, mutable default arguments, bare `except:` clauses, complexity thresholds). The LLM only sees what requires semantic understanding. This is cheaper, faster, and more reliable than sending everything to a model.

2. **An eval harness built against real data.** 132 ground-truth issues scraped from actual human review comments on merged PRs across 5 open-source repos (Python and C++), used to measure precision and recall.

3. **An honest accounting of what the eval can't measure.** The most interesting result in this project isn't the score — it's what investigating the score revealed.

---

## Architecture

```
diff
 │
 ├─► Static Analysis Layer  (free, ~0ms)
 │     • AST: naming conventions, mutable defaults, bare except,
 │            missing docstrings, function length
 │     • pyflakes: unused imports, undefined names
 │     • radon: cyclomatic complexity
 │     • regex: C++ patterns (malloc, NULL, using namespace std)
 │
 ├─► LLM Semantic Layer  (Groq / gpt-oss-120b)
 │     • Receives the diff with pre-computed line numbers
 │     • Receives static findings as context, told not to repeat them
 │     • Structured JSON output via strict schema enforcement
 │     • Focuses on: logic bugs, edge cases, security, design, test gaps
 │
 └─► Combined structured output (JSON)
```

**Why pass static findings to the LLM?** So it doesn't waste tokens re-flagging an unused import. The model is told explicitly what's already been caught and instructed to focus only on issues requiring semantic understanding.

**Why pre-compute line numbers?** LLMs are unreliable at counting through raw `+`/`-` diff markers. The diff parser (shared with the static layer) annotates each added line with its exact line number in the new file before the model sees it.

---

## Results

Evaluated against 132 human review comments from 62 merged PRs (`psf/requests`, `django/django`, `nlohmann/json`, `fmtlib/fmt`, `grpc/grpc`).

A prediction counts as a match if it lands on the same file within a line-number tolerance of the human comment.

| Metric | ±3 line tolerance | ±15 line tolerance |
|---|---|---|
| Matched | 19 | 27 |
| **Precision** | 23.2% | 32.9% |
| **Recall** | 14.4% | 20.5% |
| F1 | 0.178 | 0.252 |

Recall by category (at ±15):

| Category | Recall | n |
|---|---|---|
| performance | 100% | 1 |
| type-safety | 60% | 5 |
| documentation | 50% | 2 |
| design | 30% | 23 |
| test-coverage | 25% | 20 |
| security | 18% | 17 |
| style | 13% | 30 |
| bug | 11% | 18 |
| cleanup | 11% | 9 |
| build-config | 0% | 4 |
| other | 0% | 3 |

*(Rows sum to 27 matched of 132 total.)*

Precision by layer (at ±15): static analysis **40%** (2/5), LLM **32.5%** (25/77).

---

## The interesting part: what the eval doesn't measure

Recall of 14–21% looks bad in isolation. Investigating *why* turned out to be the most valuable part of the project.

I built a diagnostic tool (`miss_diagnostic.py`) that, for each "missed" ground-truth issue, prints what the reviewer actually said about that same PR and file. Three patterns emerged:

**1. Near-misses outside the matching window.** In several cases the reviewer identified the same defect but commented a few lines away — close enough to be obviously the same issue to a human reader, far enough to fail positional matching. Widening tolerance from ±3 to ±15 lines caught 8 more matches (a 42% increase), which is itself evidence that exact-line matching systematically undercounts.

**2. The reviewer finding *more serious* issues than the human comment it was scored against.** On one `fmtlib/fmt` PR, the human comment was a naming suggestion ("It's not parsing, maybe rename to convert?"). The reviewer instead flagged three separate defects on that same file: an unchecked null `stream` argument passed to `fmt::vprint`, an unvalidated `buffer` and `size` risking overflow, and a `catch(...)` block silently swallowing all exceptions. Scored as a miss. Arguably the more useful review.

**3. Ground truth containing things no automated reviewer should reproduce.** Clarifying questions ("Why the `Prefetch()` object here?"), stylistic preferences, and inline `\`\`\`suggestion` blocks reflecting one maintainer's taste. These are real human review comments, but they're not defects a reviewer could independently derive from the code.

I tested whether splitting categories into "objective" (bug, security, type-safety, test-coverage, performance, build-config) and "subjective" (style, design, cleanup, documentation) would isolate this. **It didn't** — recall was 21.5% vs 19.4%, essentially identical. The reviewer isn't selectively blind to subjective feedback; the matching methodology is the bottleneck.

**Conclusion:** using historical human PR comments as ground truth measures *agreement with one particular reviewer's phrasing and line placement*, not *whether the AI found something worth flagging*. A production eval would need semantic matching — does the predicted issue describe the same underlying defect? — rather than positional matching.

This is a limitation of the eval design, and I'd rather report it than quote the flattering number.

---

## Repository structure

```
scrape_prs.py         Phase 1a — scrape merged PRs + inline review comments from GitHub
clean_eval_data.py    Phase 1b — collapse comment threads, filter noise, categorize
static_analyzer.py    Phase 2  — AST + pyflakes + radon + C++ regex checks
llm_reviewer.py       Phase 3  — Groq API, structured output, response caching
eval_harness.py       Phase 5  — precision/recall, category + layer breakdowns
miss_diagnostic.py    Phase 5  — inspect missed issues against actual reviewer output
api.py                Phase 6  — FastAPI service
render.yaml           Phase 6  — deployment config

data/
  eval_issues.jsonl              132 labeled ground-truth issues
  full_review_results_groq.jsonl reviewer output across all 62 PRs
  eval_report.json               metrics at ±15 line tolerance
  eval_report_tol3.json          metrics at ±3 line tolerance
```

---

## Building the eval dataset

The ground truth came from scraping merged PRs and their inline review comments, then cleaning aggressively. Roughly 400 raw inline comments across 62 PRs were reduced to **132 distinct ground-truth issues** by dropping:

- Comments on `.rst`/`.md`/`.txt` files — prose feedback on documentation, not code review
- Noise: `LGTM`, `thanks`, `+1`, standalone acknowledgments, bare parentheticals
- Thread replies — a 3-comment discussion on one line is *one* issue, not three; only the first substantive comment per `(file, line)` is kept

Each surviving issue is then categorized by a rule-based classifier.

Categorization is heuristic (regex-based, first-match-wins). It's imperfect — a comment saying "this type check is wrong" matches `bug` before it reaches `type-safety`. I use category only for reporting breakdowns, never as a matching requirement, precisely because I don't trust it enough to gate on.

Exact per-stage counts for any given run are written to `data/eval_summary.json`.

---

## API

```bash
# Review a raw diff
curl -X POST https://ai-code-reviewer-8pz8.onrender.com/review \
  -H "Content-Type: application/json" \
  -d '{"diff": "<unified diff>", "use_llm": true}'

# Review a live GitHub PR by URL
curl -X POST https://ai-code-reviewer-8pz8.onrender.com/review/github \
  -H "Content-Type: application/json" \
  -d '{"pr_url": "https://github.com/psf/requests/pull/7502"}'
```

Set `"use_llm": false` for static-only review — free, instant, no API call. The service degrades gracefully to a `502` with an actionable message if the LLM layer is misconfigured, rather than failing the whole request.

Interactive docs: https://ai-code-reviewer-8pz8.onrender.com/docs

---

## Running locally

```bash
pip install -r requirements.txt

export GROQ_API_KEY="gsk_..."      # free key: https://console.groq.com/keys
export GITHUB_TOKEN="github_pat_..." # optional, raises GitHub API rate limit

uvicorn api:app --reload --port 8000
```

To reproduce the eval:

```bash
python scrape_prs.py        # ~5 min, needs GITHUB_TOKEN
python clean_eval_data.py
python eval_harness.py
python miss_diagnostic.py 10
```

LLM responses are cached by diff hash, so re-running the pipeline after a crash or code change doesn't re-spend API quota on diffs already reviewed.

---

## Known limitations

- **Small eval set.** 62 PRs, 132 issues, 5 repos. Enough to surface directional signal and methodology problems; not enough to make confident claims about absolute performance.
- **Positional matching.** As discussed above, the central weakness. Semantic matching is the obvious next step.
- **Heuristic categorization.** First-match-wins regex rules on both sides (ground-truth labeling and, implicitly, model output). Reasonable for slicing results; not a rigorous taxonomy.
- **C++ static analysis is regex-based.** No AST parsing for C++, so the static layer catches only surface patterns. Python gets real AST analysis.
- **Diff-fragment parsing.** Mid-function diffs (e.g. a single `elif` branch) can't be parsed as standalone Python. The AST layer skips these silently rather than reporting false syntax errors — correct behavior, but it means static coverage is lower on small diffs.

---

## What I'd do next

1. **Semantic matching in the eval harness.** Use embeddings or an LLM judge to determine whether a predicted issue and a ground-truth issue describe the same defect, independent of line placement. This directly addresses the core limitation.

2. **Codebase-aware context (RAG).** Embed the surrounding codebase at function granularity so the reviewer can answer "does this duplicate an existing utility?" or "does this violate the error-handling pattern used elsewhere in this module?" — questions that require context beyond the diff.

3. **A feedback loop.** Track which suggestions get accepted vs. dismissed in real use, and train a ranking model to surface high-confidence issues first.

4. **GitHub Action integration.** Auto-comment on PRs, turning this from an API into something that lives in a real dev workflow.
