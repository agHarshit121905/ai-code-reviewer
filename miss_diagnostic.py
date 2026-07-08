
"""
Miss Diagnostic — spot-check tool for Phase 5
================================================
For a sample of ground-truth issues the eval harness marked as "missed,"
shows you exactly what the reviewer actually said about that same PR and
file — so you can tell by eye whether:

  (a) the reviewer said nothing relevant at all (a genuine capability gap), or
  (b) the reviewer found essentially the same issue but on a line just
      outside the matching tolerance (a methodology artifact), or
  (c) the reviewer found the same issue but phrased differently enough
      that it's hard to tell without reading both side by side

This doesn't change your reported numbers — it's purely to help you
understand what's driving the recall number before you write it up.

Usage:
    python miss_diagnostic.py [N]

    N = number of misses to sample (default 8)
"""

import json
import random
import sys

from eval_harness import (
    load_ground_truth,
    load_review_results,
    flatten_predictions,
    match_predictions_to_ground_truth,
)

GROUND_TRUTH_PATH = "data/eval_issues.jsonl"
REVIEW_RESULTS_PATH = "data/full_review_results_groq.jsonl"


def main():
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 8

    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)
    review_results = load_review_results(REVIEW_RESULTS_PATH)
    predictions = flatten_predictions(review_results)

    _, _, unmatched_ground_truth = match_predictions_to_ground_truth(
        predictions, ground_truth
    )

    if not unmatched_ground_truth:
        print("No missed issues found — nothing to inspect.")
        return

    # Build a quick lookup: (repo, pr_number) -> all predictions for that PR,
    # so we can show "here's everything the reviewer said about this PR"
    # regardless of which file/line, since sometimes a relevant comment
    # ends up on a nearby line or even a different (but related) file.
    preds_by_pr = {}
    for pred in predictions:
        key = (pred["repo"], pred["pr_number"])
        preds_by_pr.setdefault(key, []).append(pred)

    random.seed(42)  # reproducible sample across runs
    sample = random.sample(unmatched_ground_truth, min(sample_size, len(unmatched_ground_truth)))

    print(f"Inspecting {len(sample)} of {len(unmatched_ground_truth)} missed ground-truth issues\n")
    print("=" * 70)

    for i, gt in enumerate(sample, 1):
        key = (gt["repo"], gt["pr_number"])
        pr_predictions = preds_by_pr.get(key, [])

        print(f"\n[{i}] MISSED: {gt['repo']} #{gt['pr_number']}  {gt['path']}:{gt['line']}  [{gt['category']}]")
        print(f"    Human said: {gt['body'][:200]}")

        if not pr_predictions:
            print(f"    Reviewer said: (nothing at all for this PR)")
        else:
            # Show predictions on the SAME FILE first (most likely to be
            # a near-miss), then everything else in the PR for context.
            same_file = [p for p in pr_predictions if p["path"] == gt["path"]]
            other_file = [p for p in pr_predictions if p["path"] != gt["path"]]

            if same_file:
                print(f"    Reviewer said on the SAME FILE ({gt['path']}):")
                for p in same_file:
                    distance = abs(p["line"] - gt["line"]) if p.get("line") is not None else "?"
                    print(f"      line {p['line']} (Δ{distance}) [{p['category']}/{p['layer']}]: {p['body'][:150]}")
            else:
                print(f"    Reviewer said NOTHING on {gt['path']} — checked other files in this PR:")
                for p in other_file[:3]:
                    print(f"      {p['path']}:{p['line']} [{p['category']}/{p['layer']}]: {p['body'][:120]}")
                if not other_file:
                    print(f"      (reviewer flagged nothing anywhere in this PR)")

        print("-" * 70)


if __name__ == "__main__":
    main()
