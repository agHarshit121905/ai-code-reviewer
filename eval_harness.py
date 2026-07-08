
"""
Eval Harness — Phase 5
========================
Scores your AI code reviewer's output (data/full_review_results_groq.jsonl)
against the human-labeled ground truth (data/eval_issues.jsonl), producing
precision/recall/F1 — in aggregate and broken down by category and by
source (static analyzer vs LLM layer).

This is the number you quote in your README and defend in an interview.

Matching logic:
  - A predicted issue and a ground-truth issue are considered a MATCH if
    they're on the same (repo, pr_number, path) and within LINE_TOLERANCE
    lines of each other. Reviewers don't always comment on the exact line
    a diff tool reconstructs, so exact-line matching would undercount
    genuine hits.
  - Matching is one-to-one: each ground-truth issue can be matched by at
    most one prediction, and vice versa. This prevents one lucky
    prediction from "double-counting" against multiple ground-truth rows,
    and prevents one ground-truth issue from being claimed by several
    near-duplicate predictions.
  - Category is NOT required to match for a hit — only location. Category
    is reported separately as a breakdown, since your categorization on
    both sides is heuristic (regex-based for ground truth, model-predicted
    for reviewer output) and forcing exact category agreement would
    conflate two different questions: "did it find the issue" and
    "did it label the issue the same way a human might."

Metrics:
  - Precision = matched predictions / total predictions
    ("of what the reviewer flagged, how much was real")
  - Recall = matched ground-truth issues / total ground-truth issues
    ("of what humans actually flagged, how much did the reviewer catch")
  - F1 = harmonic mean of precision and recall

Usage:
    python eval_harness.py
"""

import json
from collections import defaultdict

GROUND_TRUTH_PATH = "data/eval_issues.jsonl"
REVIEW_RESULTS_PATH = "data/full_review_results_groq.jsonl"
REPORT_OUTPUT_PATH = "data/eval_report.json"

# Predictions within this many lines of a ground-truth issue (on the same
# file) count as matching it. Diff line-number reconstruction and natural
# variation in exactly which line a human comments on both justify some
# slack here — 0 would undercount genuine hits, too large would inflate
# recall by matching unrelated nearby issues.
LINE_TOLERANCE = 3


def load_ground_truth(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_review_results(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def flatten_predictions(review_results: list) -> list:
    """
    Combine static_issues + llm_issues into one flat list of predictions,
    each tagged with which PR it came from and which layer produced it.
    """
    predictions = []
    for review in review_results:
        key = (review["repo"], review["pr_number"])
        for issue in review.get("static_issues", []):
            predictions.append({**issue, "repo": key[0], "pr_number": key[1], "layer": "static"})
        for issue in review.get("llm_issues", []):
            predictions.append({**issue, "repo": key[0], "pr_number": key[1], "layer": "llm"})
    return predictions


def is_match(pred: dict, gt: dict) -> bool:
    """Same PR, same file, line within tolerance."""
    if pred["repo"] != gt["repo"] or pred["pr_number"] != gt["pr_number"]:
        return False
    if pred["path"] != gt["path"]:
        return False
    if pred["line"] is None or gt["line"] is None:
        return False
    return abs(pred["line"] - gt["line"]) <= LINE_TOLERANCE


def match_predictions_to_ground_truth(predictions: list, ground_truth: list) -> tuple:
    """
    One-to-one greedy matching, grouped by (repo, pr_number) for efficiency
    (no point comparing predictions against ground truth from a different PR).

    Returns:
        matched_pairs: list of (prediction, ground_truth) tuples that matched
        unmatched_predictions: predictions with no matching ground truth (false positives)
        unmatched_ground_truth: ground-truth issues no prediction caught (false negatives / misses)
    """
    # Group both sides by (repo, pr_number) so matching is only attempted
    # within the same PR — an issue in PR A can never match one in PR B.
    gt_by_pr = defaultdict(list)
    for i, gt in enumerate(ground_truth):
        gt_by_pr[(gt["repo"], gt["pr_number"])].append(i)

    pred_by_pr = defaultdict(list)
    for i, pred in enumerate(predictions):
        pred_by_pr[(pred["repo"], pred["pr_number"])].append(i)

    matched_gt_indices = set()
    matched_pred_indices = set()
    matched_pairs = []

    for pr_key in gt_by_pr:
        gt_indices = gt_by_pr[pr_key]
        pred_indices = pred_by_pr.get(pr_key, [])

        # Greedy matching: for each ground-truth issue (in order), find the
        # first unmatched prediction that matches it. Good enough here since
        # near-duplicate predictions on the same line are rare in practice;
        # a more rigorous approach would solve this as bipartite matching,
        # but greedy is simpler to explain and audit, and the difference in
        # practice is negligible at this dataset's scale.
        for gt_idx in gt_indices:
            gt = ground_truth[gt_idx]
            for pred_idx in pred_indices:
                if pred_idx in matched_pred_indices:
                    continue
                pred = predictions[pred_idx]
                if is_match(pred, gt):
                    matched_gt_indices.add(gt_idx)
                    matched_pred_indices.add(pred_idx)
                    matched_pairs.append((pred, gt))
                    break

    unmatched_predictions = [
        predictions[i] for i in range(len(predictions)) if i not in matched_pred_indices
    ]
    unmatched_ground_truth = [
        ground_truth[i] for i in range(len(ground_truth)) if i not in matched_gt_indices
    ]

    return matched_pairs, unmatched_predictions, unmatched_ground_truth


def compute_metrics(matched_count: int, total_predictions: int, total_ground_truth: int) -> dict:
    precision = matched_count / total_predictions if total_predictions else 0.0
    recall = matched_count / total_ground_truth if total_ground_truth else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "matched": matched_count,
        "total_predictions": total_predictions,
        "total_ground_truth": total_ground_truth,
    }


def breakdown_by_category(matched_pairs: list, unmatched_gt: list) -> dict:
    """
    Recall broken down by the GROUND TRUTH category (i.e. "of the bugs
    humans flagged, how many did we catch, split by what kind of issue
    they were"). This is the more interesting cut than predicted category,
    since it tells you where your reviewer is systematically weak.
    """
    caught_by_category = defaultdict(int)
    total_by_category = defaultdict(int)

    for _, gt in matched_pairs:
        caught_by_category[gt["category"]] += 1
        total_by_category[gt["category"]] += 1
    for gt in unmatched_gt:
        total_by_category[gt["category"]] += 1

    breakdown = {}
    for category in sorted(total_by_category.keys()):
        caught = caught_by_category.get(category, 0)
        total = total_by_category[category]
        breakdown[category] = {
            "recall": round(caught / total, 3) if total else 0.0,
            "caught": caught,
            "total": total,
        }
    return breakdown


# Categories where a correct answer follows from the code's actual behavior
# (correctness, security, missing tests, type errors) rather than from taste
# or team convention. "design" is a judgment call: kept in SUBJECTIVE because
# manual inspection of this dataset showed most design comments were
# preference questions ("why not use X instead") rather than objectively
# broken abstractions — a defensible call, not a certainty.
OBJECTIVE_CATEGORIES = {
    "bug", "security", "type-safety", "test-coverage", "performance", "build-config",
}
SUBJECTIVE_CATEGORIES = {
    "style", "design", "cleanup", "documentation", "other",
}


def breakdown_by_group(matched_pairs: list, unmatched_gt: list) -> dict:
    """
    Recall aggregated into two groups: OBJECTIVE (correctness-driven, a
    reasonable target for any automated reviewer) vs SUBJECTIVE (taste,
    wording, convention — arguably not a fair target for automated review).

    This answers the more useful question for judging the reviewer's actual
    capability: "of the issues that are actually checkable from the code,
    how many did it catch" — separate from stylistic feedback no automated
    tool would be expected to originate unprompted.
    """
    def group_of(category: str) -> str:
        if category in OBJECTIVE_CATEGORIES:
            return "objective"
        if category in SUBJECTIVE_CATEGORIES:
            return "subjective"
        return "unclassified"  # safety net if a new category appears later

    caught_by_group = defaultdict(int)
    total_by_group = defaultdict(int)

    for _, gt in matched_pairs:
        group = group_of(gt["category"])
        caught_by_group[group] += 1
        total_by_group[group] += 1
    for gt in unmatched_gt:
        group = group_of(gt["category"])
        total_by_group[group] += 1

    breakdown = {}
    for group in ["objective", "subjective", "unclassified"]:
        if group not in total_by_group:
            continue
        caught = caught_by_group.get(group, 0)
        total = total_by_group[group]
        breakdown[group] = {
            "recall": round(caught / total, 3) if total else 0.0,
            "caught": caught,
            "total": total,
        }
    return breakdown

def breakdown_by_layer(matched_pairs: list, unmatched_predictions: list) -> dict:
    """
    Precision broken down by which layer (static analyzer vs LLM) produced
    the prediction — tells you which layer is contributing more signal vs
    noise.
    """
    matched_by_layer = defaultdict(int)
    total_by_layer = defaultdict(int)

    for pred, _ in matched_pairs:
        matched_by_layer[pred["layer"]] += 1
        total_by_layer[pred["layer"]] += 1
    for pred in unmatched_predictions:
        total_by_layer[pred["layer"]] += 1

    breakdown = {}
    for layer in sorted(total_by_layer.keys()):
        matched = matched_by_layer.get(layer, 0)
        total = total_by_layer[layer]
        breakdown[layer] = {
            "precision": round(matched / total, 3) if total else 0.0,
            "matched": matched,
            "total": total,
        }
    return breakdown


def main():
    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)
    review_results = load_review_results(REVIEW_RESULTS_PATH)
    predictions = flatten_predictions(review_results)

    matched_pairs, unmatched_predictions, unmatched_ground_truth = (
        match_predictions_to_ground_truth(predictions, ground_truth)
    )

    overall = compute_metrics(
        matched_count=len(matched_pairs),
        total_predictions=len(predictions),
        total_ground_truth=len(ground_truth),
    )

    category_breakdown = breakdown_by_category(matched_pairs, unmatched_ground_truth)
    group_breakdown = breakdown_by_group(matched_pairs, unmatched_ground_truth)
    layer_breakdown = breakdown_by_layer(matched_pairs, unmatched_predictions)

    report = {
        "overall": overall,
        "by_category_recall": category_breakdown,
        "by_group_recall": group_breakdown,
        "by_layer_precision": layer_breakdown,
        "line_tolerance": LINE_TOLERANCE,
        "prs_evaluated": len(review_results),
        "objective_categories": sorted(OBJECTIVE_CATEGORIES),
        "subjective_categories": sorted(SUBJECTIVE_CATEGORIES),
    }

    with open(REPORT_OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    # --- Console report ---
    print("=" * 60)
    print("EVAL HARNESS REPORT")
    print("=" * 60)
    print(f"PRs evaluated: {report['prs_evaluated']}")
    print(f"Ground-truth issues: {overall['total_ground_truth']}")
    print(f"Reviewer predictions: {overall['total_predictions']}")
    print(f"Matched (line tolerance ±{LINE_TOLERANCE}): {overall['matched']}")
    print()
    print(f"Precision: {overall['precision']:.1%}  (of what the reviewer flagged, how much was real)")
    print(f"Recall:    {overall['recall']:.1%}  (of what humans flagged, how much the reviewer caught)")
    print(f"F1:        {overall['f1']:.3f}")
    print()
    print("Recall by OBJECTIVE vs SUBJECTIVE issue group:")
    print("  (objective = bug/security/type-safety/test-coverage/performance/build-config)")
    print("  (subjective = style/design/cleanup/documentation/other)")
    for group in ["objective", "subjective", "unclassified"]:
        if group not in group_breakdown:
            continue
        stats = group_breakdown[group]
        print(f"  {group:12s} {stats['recall']:.1%}  ({stats['caught']}/{stats['total']})")
    print()
    print("Recall by ground-truth category:")
    for cat, stats in sorted(category_breakdown.items(), key=lambda x: -x[1]["total"]):
        print(f"  {cat:15s} {stats['recall']:.1%}  ({stats['caught']}/{stats['total']})")
    print()
    print("Precision by prediction source:")
    for layer, stats in layer_breakdown.items():
        print(f"  {layer:10s} {stats['precision']:.1%}  ({stats['matched']}/{stats['total']})")
    print()
    print(f"Full report written to {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
