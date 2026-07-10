# AI Code Reviewer

A two-layer code review system that combines deterministic static analysis with LLM-based semantic review, evaluated against real human reviewer comments from 62 merged pull requests.

**Live API:** https://ai-code-reviewer-8pz8.onrender.com/docs
*(Free tier — first request may take 30–60s to wake the service.)*

---

## What this is

Most "AI code reviewer" projects are a thin wrapper around an LLM: send a diff, print whatever comes back. This one is built around a different question — **how do you know if the review is any good?**

That question drove four design decisions:

1. **A hybrid architecture.** Static analysis handles what's deterministic (unused imports, mutable default arguments, bare `except:` clauses, complexity thresholds). The LLM only sees what requires semantic understanding.

2. **An eval harness built against real data.** 132 ground-truth issues scraped from actual human review comments on merged PRs across 5 open-source repos (Python and C++), used to measure precision and recall.

3. **Retrieval-augmented review, measured against the baseline.** The reviewer retrieves semantically similar functions from a FAISS index of the codebase and injects them into the prompt. This improved recall ~30% relative — and, more interestingly, improved it *only on the categories where codebase context should matter* (duplication, security patterns, architectural consistency) and not at all on categories where it shouldn't (logic bugs, style).

4. **An honest accounting of what the eval can't measure.** The most interesting result in this project isn't the score — it's what investigating the score revealed.

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
 ├─► Retrieval Layer  (optional, local embeddings)
 │     • Codebase pre-indexed at function granularity (AST for Python,
 │       brace-matching for C++)
 │     • all-MiniLM-L6-v2 embeddings, FAISS inner-product index
 │     • Per changed file: retrieve top-k similar existing functions,
 │       excluding the file being changed
 │
 ├─► LLM Semantic Layer  (Groq / gpt-oss-120b)
 │     • Receives the diff with pre-computed line numbers
 │     • Receives static findings as context, told not to repeat them
 │     • Receives retrieved codebase context, told to check for
 │       duplication and pattern inconsistency
 │     • Structured JSON output via strict schema enforcement
 │
 └─► Combined structured output (JSON)
```

**Why pass static findings to the LLM?** So it doesn't waste tokens re-flagging an unused import. The model is told explicitly what's already been caught and instructed to focus only on issues requiring semantic understanding.

**Why pre-compute line numbers?** LLMs are unreliable at counting through raw `+`/`-` diff markers. The diff parser (shared with the static layer) annotates each added line with its exact line number in the new file before the model sees it.

**Why function-level chunks rather than fixed-size windows?** A function is a semantically complete unit. A 512-token window can bisect a function, embedding a fragment whose meaning differs from the whole. Function chunks also map 1:1 onto the retrieval question ("does a similar function already exist?").

**Why local embeddings and FAISS rather than an API and a managed vector DB?** At this scale (a few thousand functions per repo, ~118k total) an 80MB model on CPU indexes a repo in under two minutes and a flat FAISS index searches it in microseconds. A managed service would add latency, cost, and an account dependency for zero benefit.

---

## Results

Evaluated against 132 human review comments from 62 merged PRs (`psf/requests`, `django/django`, `nlohmann/json`, `fmtlib/fmt`, `grpc/grpc`).

A prediction counts as a match if it lands on the same file within a line-number tolerance of the human comment.

### Baseline (diff-only review)

| Metric | ±3 line tolerance | ±15 line tolerance |
|---|---|---|
| Matched | 19 | 27 |
| **Precision** | 23.2% | 32.9% |
| **Recall** | 14.4% | 20.5% |
| F1 | 0.178 | 0.252 |

### With RAG (codebase context retrieved and injected)

The reviewer additionally receives the most semantically similar existing functions from the repo, retrieved from a FAISS index of ~118k function-level chunks across the five repos.

| Metric | ±3 tolerance | ±15 tolerance |
|---|---|---|
| Matched | 25 | 35 |
| **Precision** | 21.2% | 29.7% |
| **Recall** | **18.9%** | **26.5%** |
| **F1** | **0.200** | **0.280** |

**RAG improved recall ~30% relative at both tolerance levels** (14.4%→18.9% at ±3; 20.5%→26.5% at ±15), at a ~2–3 point precision cost, for a net F1 gain. It made 44% more predictions overall.

The improvement is understated: 4 of 62 PRs failed under RAG (token limits — see below), so RAG is scored against the same 132 ground-truth issues while having fewer PRs in which to find them.

### Where RAG helped, and where it didn't

Recall by category, baseline → RAG (at ±3 tolerance):

| Category | Baseline | RAG | |
|---|---|---|---|
| security | 11.8% | **23.5%** | ↑ doubled |
| cleanup | 11.1% | **22.2%** | ↑ doubled |
| build-config | 0% | **25.0%** | ↑ |
| design | 17.4% | **26.1%** | ↑ |
| bug | 5.6% | 5.6% | — unchanged |
| style | 13.3% | 13.3% | — unchanged |
| type-safety | 40.0% | 40.0% | — unchanged |

This is the result I care most about. **RAG improved exactly the categories that require knowing what else exists in the codebase — duplication, security patterns, architectural consistency — and did nothing for categories that only require reading the diff.** A logic bug is a logic bug regardless of what's in the rest of the repo; a duplicated utility function is only visible if you know the utility already exists.

That's the mechanism working as theorized, not a lucky aggregate number.

**A concrete example.** On `psf/requests#7502`, the diff added an inline `isinstance(fp, _SupportsRead) or hasattr(fp, "read")` check. The baseline reviewer found nothing. RAG retrieved `has_read()` from `src/requests/_types.py` (cosine similarity 0.787) — a function that performs exactly that check, including `__getattr__` proxy handling — and the reviewer flagged the duplication by name. It separately caught that `isinstance()` against a `typing.Protocol` fails at runtime unless the protocol is `@runtime_checkable`.

### The cost

RAG roughly tripled token consumption per review. This wasn't free:

- **1 PR failed with HTTP 413** — the retrieved context pushed the prompt to 15,906 tokens against an 8,000 TPM limit.
- **2 PRs failed schema validation** — under longer prompts the model twice produced output violating the strict JSON schema (once emitting a string where an object belonged, once inventing a category outside the enum). Strict-mode enforcement caught both rather than silently accepting malformed data.
- **1 PR failed on daily token quota** — the run consumed 199,211 of 200,000 daily tokens.

Baseline hit none of these. If you're deploying this, **RAG buys ~30% more recall for ~3× the token cost** — a tradeoff that's obviously worth it in some contexts and obviously not in others.

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
codebase_indexer.py   Phase 4  — function-level chunking, embeddings, FAISS index
rag_reviewer.py       Phase 4  — retrieval-augmented review
eval_harness.py       Phase 5  — precision/recall, category + layer breakdowns
miss_diagnostic.py    Phase 5  — inspect missed issues against actual reviewer output
api.py                Phase 6  — FastAPI service
render.yaml           Phase 6  — deployment config

data/
  eval_issues.jsonl                132 labeled ground-truth issues
  full_review_results_groq.jsonl   baseline reviewer output (62 PRs)
  full_review_results_rag.jsonl    RAG reviewer output (58 PRs)
  eval_report.json                 baseline metrics, ±15 tolerance
  eval_report_tol3.json            baseline metrics, ±3 tolerance
  eval_report_rag.json             RAG metrics, ±3 tolerance
  eval_report_rag_tol15.json       RAG metrics, ±15 tolerance
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
- **Per-category recall is not comparable across tolerance levels.** The matcher is greedy and one-to-one: it iterates ground-truth issues in order and claims the first unmatched prediction that fits. Widening the tolerance can therefore cause a prediction that previously matched issue A to instead be claimed by issue B, encountered earlier. Aggregate matched counts are correct and monotone (25 → 35), but a category's recall can appear to *decrease* as tolerance widens (e.g. `cleanup` 22.2% → 11.1%) purely from reassignment. A proper bipartite matching (Hungarian algorithm) would fix this; greedy was chosen for auditability and the difference in aggregate is negligible.
- **Heuristic categorization.** First-match-wins regex rules on both sides (ground-truth labeling and, implicitly, model output). Reasonable for slicing results; not a rigorous taxonomy.
- **C++ static analysis is regex-based.** No AST parsing for C++, so the static layer catches only surface patterns. Python gets real AST analysis. C++ function extraction for the index uses brace-matching, which will miss some template-heavy definitions.
- **Diff-fragment parsing.** Mid-function diffs (e.g. a single `elif` branch) can't be parsed as standalone Python. The AST layer skips these silently rather than reporting false syntax errors — correct behavior, but it means static coverage is lower on small diffs.
- **Temporal mismatch in the index.** Repos are indexed at current `HEAD`, but the eval PRs were merged earlier. The reviewer therefore sees a slightly newer codebase than the PR author did. A rigorous version would check out each repo at the PR's parent commit.
- **RAG failure modes are real.** 4 of 62 PRs failed under RAG (token limits, schema-validation failures under longer prompts) that succeeded in the baseline. Recall gains are reported on the 58 that completed.

---

## What I'd do next

1. **Semantic matching in the eval harness.** Use embeddings or an LLM judge to determine whether a predicted issue and a ground-truth issue describe the same defect, independent of line placement. This directly addresses the core limitation, and would let me measure RAG's contribution far more precisely than positional matching allows.

2. **Bipartite matching instead of greedy.** Fixes the order-dependent category attribution described above.

3. **Retrieval evaluation in its own right.** Right now I measure RAG end-to-end (did recall improve?) but never measure retrieval quality directly (was the retrieved context actually relevant?). Building a small labeled set of "for this diff, these are the functions that should be retrieved" would let me tune `MIN_SIMILARITY`, `k`, and the chunking strategy against a target rather than by intuition.

4. **Adaptive context budgeting.** RAG's failures were all token-limit related. Retrieved context should be truncated proportionally to diff size — a large diff should get less context, not the same fixed amount, so total prompt size stays bounded.

5. **A feedback loop.** Track which suggestions get accepted vs. dismissed in real use, and train a ranking model to surface high-confidence issues first.

6. **GitHub Action integration.** Auto-comment on PRs, turning this from an API into something that lives in a real dev workflow.
