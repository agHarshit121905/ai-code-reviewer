"""
Eval Dataset Cleaner
=====================
Turns raw_prs.jsonl (scraped GitHub PRs + raw inline comments) into a
structured eval set: one row per distinct human-flagged issue, with
noise filtered out and a rough category tag attached.

This is the ground truth your AI code reviewer will be measured against.

Input:  data/raw_prs.jsonl
Output: data/eval_issues.jsonl   (one row per distinct issue)
        data/eval_summary.json   (dataset stats for your README/eval report)

Usage:
    python clean_eval_data.py
"""

import json
import re
from collections import defaultdict, Counter

INPUT_PATH = "data/raw_prs.jsonl"
ISSUES_OUTPUT_PATH = "data/eval_issues.jsonl"
SUMMARY_OUTPUT_PATH = "data/eval_summary.json"

# ---------------------------------------------------------------------------
# Noise filtering
# ---------------------------------------------------------------------------
# Comments that are acks, agreements, or pure social niceties rather than
# a substantive code issue. Checked against the comment with whitespace
# stripped and lowercased. These are intentionally short/exact-ish phrases —
# better to under-filter (leave some noise in) than over-filter (accidentally
# drop a real issue that happens to start with "thanks").
NOISE_PATTERNS = [
    r"^(lgtm|looks good( to me)?|sounds good|sgtm)[\.\!]*$",
    r"^(thanks?( you)?|thx|ty)[\.\!,]*( .{0,30})?$",
    r"^\+1$",
    r"^(done|fixed|good call|good point|agreed?|makes sense)[\.\!]*$",
    r"^(sure|ok(ay)?|yep|yes|no problem|np)[\.\!,]*$",
    r"^(will (do|fix)|on it)[\.\!]*$",
    r"^(nit:?\s*)?(typo)[\.\!]*$",
    r"^good (catch|find|spot)[\.\!].*$",          # "Good catch! Thanks for the fix!"
    r"^(lgtm|looks good).{0,60}$",                # "Looks good to me, just one nit"
    r"^\(.*\)\s*[\.\!]*$",                        # standalone parenthetical: "(The reference key was incomplete.)"
    r"^this is safe to do\b",                     # confirms something is fine — not an issue being raised
]
NOISE_RE = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]

# Minimum length for a comment to be considered substantive enough to keep
MIN_COMMENT_LENGTH = 10  # characters — lowered from 15 to catch short but real issues like "Missing newline"

# File extensions that are pure prose/documentation, not code. Comments on
# these files are about wording/clarity, not code review — out of scope for
# an AI *code* reviewer, so we drop them entirely rather than miscategorize.
NON_CODE_EXTENSIONS = (
    ".rst", ".md", ".markdown", ".txt", ".adoc", ".rdoc",
)


def is_non_code_file(path: str) -> bool:
    return path.lower().endswith(NON_CODE_EXTENSIONS)


def is_noise(body: str) -> bool:
    text = body.strip()
    if len(text) < MIN_COMMENT_LENGTH:
        return True
    for pattern in NOISE_RE:
        if pattern.match(text):
            return True
    return False


# ---------------------------------------------------------------------------
# Categorization
# ---------------------------------------------------------------------------
# Rough keyword-based categorization. Not meant to be perfect — meant to be
# good enough to slice eval metrics by category and explain the slicing
# logic plainly in an interview. Order matters: first match wins, so put
# more specific/important categories first.
CATEGORY_RULES = [
    ("security", [
        r"\bsecurit(y|ies)\b", r"\bvulnerab", r"\bCVE\b", r"\binjection\b",
        r"\bsanitiz", r"\bescape\b.*\binput\b", r"\bsecret(e)?\b", r"\bcredential",
        r"\bexception safety\b", r"\bthrow\b.*\bsafe", r"\btoken\b.*\bpush\b",
        r"\bpush\b.*\btoken\b",
    ]),
    ("bug", [
        r"\bbug\b", r"\bcrash\b", r"\bwrong\b", r"\bincorrect\b", r"\bbroken\b",
        r"\bfails?\b", r"\bedge case\b", r"\bnull\b.*\b(check|pointer)\b",
        r"\brace condition\b", r"\bmemory leak\b", r"\bundefined behav",
        r"\boff.by.one\b", r"\boverflow\b", r"\bregression\b",
        r"\bbehaviou?r change\b", r"\bunexpected\b", r"\bunintention",
        r"\bwon'?t work\b", r"\bwill (not|break|fail)\b",
        r"\bnot defined\b", r"\bnever defined\b", r"\bnot ensure",
        r"\bwe lose\b", r"\bwe (already)? (have|had) cleared",
        r"\bwas removed\b", r"\bwhy.*removed\b", r"\bneeds to be moved\b",
        r"\bwas missed\b", r"\bthis was missed\b",
        r"\breference key\b", r"\bincomplete\b",
        r"\binitializ", r"\bassignment to null\b",
        r"\bi don'?t understand why\b",
        r"\bredundant\b",                          # "isinstance appears redundant"
    ]),
    ("test-coverage", [
        r"\btest(s|ing)?\b", r"\bcoverage\b", r"\bparametriz", r"\bassert",
        r"\bmock\b", r"\bfixture\b", r"\btyped_test\b",
    ]),
    ("design", [
        r"\bapi\b", r"\binterface\b", r"\barchitectur", r"\babstraction\b",
        r"\brefactor", r"\bduplicat", r"\bcoupling\b", r"\bbackward.compat",
        r"\bbreaking change\b", r"\bwhy (not|the|did|was)\b",
        r"\bintentional\b", r"\bfollow.up (ticket|issue|pr)\b",
        r"\bseparate ticket\b", r"\bconsider\b", r"\bwhat do you think\b",
        r"\bcould (we|you) (use|do|make|just)\b", r"\bany value in\b",
        r"\bcurious about\b", r"\blet'?s (keep|change|move|use|remove|add)\b",
        r"\bdo we (need|want|have)\b", r"\bis this (file|needed|safe|related)\b",
        r"\bnot related\b", r"\boverkill\b", r"\bunrelated\b",
        r"\balternative\b", r"\boverload\b",
        r"\bneeds? to be moved\b", r"\bshould be moved\b",
        r"\bnot sure (this|it|if)\b",
        r"\bconsistent with\b",
        r"\bto double.check\b", r"\bdouble.check my understanding\b",
    ]),
    ("performance", [
        r"\bperformance\b", r"\bslow\b", r"\befficient\b", r"\ballocat",
        r"\bcomplexity\b", r"\bO\(n", r"\boptimiz",
    ]),
    ("style", [
        r"\bnit\b", r"\bstyle\b", r"\bnaming\b", r"\bconvention\b",
        r"\bformat(ting)?\b", r"\bindent", r"\btypo\b",
        r"```suggestion", r"\breword", r"\brephrase",
        r"\brename\b", r"\w+\s*->\s*\w+",          # "buffer_ -> buffer" rename patterns
        r"\bplease use\b", r"\buse the .* macro\b",
        r"\bshould go away\b",
        r"\bwhitespace\b", r"\bautogenerat",
        r"\bplease (bump|add|also|show|include|remove)\b",
        r"\bmissing newline\b", r"\bnewline\b", r"\bno newline\b",
        r"\btrailing (comma|whitespace|newline)\b",
        r"\bspelling\b", r"\bgrammar\b",
        r"\bno longer correct\b", r"\boutdated comment\b",
        r"\bthis comment is (no longer|outdated|wrong|stale)\b",
        r"\bI think we can use\b",                  # "I think we can use the same macro"
        r"\bwe could (probably )?(just )?(use|check|remove|add)\b",
        r"\bI don'?t think (this|it|we|these) (is|are)? ?(needed|necessary)\b",
        r"\bI don'?t think we need\b",
        r"\b__cpp_\w+\b",                            # C++ feature test macros
        r"\b__GNUC__\b", r"\b__(clang|msvc|linux)__\b",
        r"\bI think we (can|should|could)\b",
    ]),
    ("documentation", [
        r"\bdocstring\b", r"\breadab", r"\bdeprecat",
        r"\bclarify\b", r"\bconfusing\b",
        r"\bRemovedIn\w+Warning\b",
        r"\bdeprecation\.txt\b",
        r"\badd a (brief )?comment\b",
        r"\bexplain(ing)? what\b",
        r"\boptionally.*comment\b",
    ]),
    ("type-safety", [
        r"\btype\b", r"\btyping\b", r"\bpyright\b", r"\bmypy\b",
        r"\bcast\b", r"\bannotat", r"\bconstexpr\b",
        r"\bC\+\+\d+\b",                           # "only in C++20", "constexpr in C++26"
        r"\b16.bit\b", r"\bliteral",
    ]),
    ("build-config", [
        r"\bshared lib\b", r"\bstatic lib\b", r"\bCMake\b", r"\bMakefile\b",
        r"\bdependenc(y|ies)\b", r"\blatest version\b", r"\binstalled (auto|with)\b",
        r"\blibc\+\+\b", r"\bCI\b", r"\bworkflow\b", r"\bpre.commit\b",
        r"\bcompil(e|er|ing)\b", r"\blink(er|ing)?\b",
        r"\bshared libs?\b", r"\bneeds? shared\b",
    ]),
    ("cleanup", [
        r"\bcan be removed\b", r"\bcan remove\b", r"\bremove (this|it|the)\b",
        r"\bnot needed\b", r"\bnot necessary\b", r"\bundeed\b",
        r"\bunnecessar\b", r"\bsimplif\b", r"\bclean.?up\b",
        r"\bthis line\b.*\bremov", r"\brevert\b",
        r"\balready included\b", r"\bduplicate include\b",
        r"\bnot suggesting\b", r"\bjust the implementation\b",
        r"\bnowhere (ensures?|guarantees?)\b",
        r"\bdon'?t ensure\b", r"\bnot ensure\b",  # "don't ensure symbol is defined"
        r"\bexisting config\b",                    # "this is existing config, let's not change it"
        r"\bI don'?t think we need these\b",
        r"\beffectively a superset\b",             # "already included, is a superset of"
    ]),
]
COMPILED_CATEGORY_RULES = [
    (label, [re.compile(p, re.IGNORECASE) for p in patterns])
    for label, patterns in CATEGORY_RULES
]


def categorize(body: str) -> str:
    for label, patterns in COMPILED_CATEGORY_RULES:
        if any(p.search(body) for p in patterns):
            return label
    return "other"


# ---------------------------------------------------------------------------
# Thread collapsing
# ---------------------------------------------------------------------------
def collapse_threads(review_comments: list[dict]) -> list[dict]:
    """
    Group comments by (path, line), then keep only the first substantive
    (non-noise) comment per location as the ground-truth issue. Later
    comments on the same location are treated as replies/resolution and
    dropped — they're conversation, not new issues.

    Comments are assumed to arrive in chronological order from the GitHub
    API (they do, by default).
    """
    by_location = defaultdict(list)
    for c in review_comments:
        if is_non_code_file(c["path"]):
            continue
        key = (c["path"], c["line"])
        by_location[key].append(c)

    issues = []
    for (path, line), thread in by_location.items():
        for c in thread:
            if not is_noise(c["body"]):
                issues.append({
                    "path": path,
                    "line": line,
                    "body": c["body"],
                    "author": c["author"],
                    "thread_length": len(thread),  # context: how much discussion this sparked
                    "category": categorize(c["body"]),
                })
                break  # only the first substantive comment per location
    return issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    with open(INPUT_PATH) as f:
        prs = [json.loads(line) for line in f]

    all_issues = []
    category_counts = Counter()
    repo_counts = Counter()
    dropped_noise_count = 0
    dropped_duplicate_thread_count = 0
    dropped_non_code_file_count = 0

    with open(ISSUES_OUTPUT_PATH, "w") as out_f:
        for pr in prs:
            raw_comments = pr["review_comments"]
            issues = collapse_threads(raw_comments)

            # tally what got dropped, for the summary report
            kept_bodies = {i["body"] for i in issues}
            for c in raw_comments:
                if c["body"] in kept_bodies:
                    continue
                if is_non_code_file(c["path"]):
                    dropped_non_code_file_count += 1
                elif is_noise(c["body"]):
                    dropped_noise_count += 1
                else:
                    dropped_duplicate_thread_count += 1

            for issue in issues:
                record = {
                    "repo": pr["repo"],
                    "pr_number": pr["pr_number"],
                    "pr_url": pr["url"],
                    "diff": pr["diff"],
                    **issue,
                }
                out_f.write(json.dumps(record) + "\n")
                all_issues.append(record)
                category_counts[issue["category"]] += 1
                repo_counts[pr["repo"]] += 1

    summary = {
        "total_prs": len(prs),
        "total_distinct_issues": len(all_issues),
        "avg_issues_per_pr": round(len(all_issues) / len(prs), 2) if prs else 0,
        "issues_by_category": dict(category_counts.most_common()),
        "issues_by_repo": dict(repo_counts),
        "raw_comments_total": sum(len(pr["review_comments"]) for pr in prs),
        "dropped_as_noise": dropped_noise_count,
        "dropped_as_duplicate_thread_reply": dropped_duplicate_thread_count,
        "dropped_as_non_code_file": dropped_non_code_file_count,
    }
    with open(SUMMARY_OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"PRs processed:              {summary['total_prs']}")
    print(f"Raw inline comments:        {summary['raw_comments_total']}")
    print(f"  dropped (non-code file):  {dropped_non_code_file_count}")
    print(f"  dropped (noise):          {dropped_noise_count}")
    print(f"  dropped (thread replies): {dropped_duplicate_thread_count}")
    print(f"Distinct ground-truth issues: {summary['total_distinct_issues']}")
    print(f"Avg issues/PR:               {summary['avg_issues_per_pr']}")
    print(f"\nBy category:")
    for cat, count in category_counts.most_common():
        print(f"  {cat:15s} {count}")
    print(f"\nBy repo:")
    for repo, count in repo_counts.items():
        print(f"  {repo:25s} {count}")
    print(f"\nWritten to {ISSUES_OUTPUT_PATH} and {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
