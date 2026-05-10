"""
Step 4a — Few-Shot ADOS-2 Scorer
══════════════════════════════════
Scores a session using static few-shot examples drawn from the training pairs.
No fine-tuning, no vector index — just the right examples in the prompt.

Example selection strategy:
  1. Match by module (M1 or M2) — always
  2. Match by classification if possible (Autism / Spectrum / Non-Spectrum)
  3. If not enough, fill with highest-scoring sessions from matching module
  Default K = 3 examples (sweet spot for 7B models)

Usage:
    # Score a single session
    python step4a_fewshot_scorer.py \
        --features   data/features/V2D02022_features.json \
        --pairs      data/training_pairs.json \
        --taxonomy   data/ados_taxonomy_combined.json \
        --module     M2 \
        --lang_type  under_5_years

    # Score all sessions in features dir (leave-one-out evaluation)
    python step4a_fewshot_scorer.py \
        --features_dir  data/features/ \
        --pairs         data/training_pairs.json \
        --taxonomy      data/ados_taxonomy_combined.json \
        --evaluate      # compares against ground truth in pairs
"""

import json
import argparse
from pathlib import Path
from scoring_utils import (
    call_ollama, load_taxonomy, get_algorithm_items,
    build_scoring_prompt, parse_scores, render_report, print_report
)


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE SELECTOR
# ─────────────────────────────────────────────────────────────────────────────

def select_examples(
    target_video_id: str,
    target_module:   str,
    all_pairs:       list[dict],
    k:               int = 3,
    target_class:    str = None,
) -> list[dict]:
    """
    Select K examples from training pairs.

    Priority:
      1. Same module, same classification
      2. Same module, any classification
      3. Fill remaining slots from same module by descending total score
    Never includes the target video itself (leave-one-out safe).
    """
    # Filter by module, exclude target
    pool = [
        p for p in all_pairs
        if p["module"] == target_module
        and p["id"] != target_video_id
    ]

    if not pool:
        return []

    selected = []

    # Priority 1 — same classification
    if target_class:
        matched = [p for p in pool if p["meta"]["classification"] == target_class]
        selected.extend(matched[:k])

    # Priority 2 — fill with highest total from same module
    remaining_ids = {p["id"] for p in selected}
    for p in sorted(pool, key=lambda x: x["meta"]["total"], reverse=True):
        if len(selected) >= k:
            break
        if p["id"] not in remaining_ids:
            selected.append(p)
            remaining_ids.add(p["id"])

    # Build example dicts for the prompt
    examples = []
    for p in selected[:k]:
        scores = json.loads(p["output"]) if isinstance(p["output"], str) else p["output"]
        examples.append({
            "narrative": p["input"].split("## Items to Score")[0].replace(
                "You are an expert ADOS-2 clinician. Score each item based on the observations below.\n\n"
                "## Behavioral Observations\n", ""
            ).strip(),
            "scores": scores,
        })

    return examples


# ─────────────────────────────────────────────────────────────────────────────
# SCORE ONE SESSION
# ─────────────────────────────────────────────────────────────────────────────

def score_session(
    features:    dict,
    all_pairs:   list[dict],
    taxonomy:    dict,
    lang_type:   str,
    k:           int = 3,
    verbose:     bool = True,
) -> dict:
    """Score one session using few-shot prompting."""
    video_id = features["video_id"]
    module   = features.get("module", "M1")
    narrative= features["narrative"]

    items    = get_algorithm_items(taxonomy, module)
    sa_items = [i["code"] for i in items if i["algorithm_domain"] == "SA"]
    crr_items= [i["code"] for i in items if i["algorithm_domain"] == "CRR"]
    all_codes= [i["code"] for i in items]

    # Select examples
    examples = select_examples(video_id, module, all_pairs, k=k)

    if verbose:
        print(f"\n  Session:   {video_id}  [{module}]")
        print(f"  Examples:  {[e.get('video_id', '?') for e in examples[:k]]}")
        ex_ids = [p["id"] for p in all_pairs if p["module"] == module and p["id"] != video_id][:k]
        print(f"  Using:     {ex_ids}")

    # Build prompt and call model
    prompt   = build_scoring_prompt(narrative, items, examples)
    response = call_ollama(prompt, temperature=0.05)

    # Parse scores
    scores = parse_scores(response, all_codes)

    # Build report
    report = render_report(
        video_id, module, scores,
        sa_items, crr_items, lang_type, "few-shot", taxonomy
    )

    if verbose:
        print_report(report)

    return report


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION — LEAVE-ONE-OUT
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    features_dir: str,
    all_pairs:    list[dict],
    taxonomy:     dict,
    lang_type:    str,
    k:            int = 3,
    output_dir:   str = "outputs/fewshot",
) -> None:
    """
    Leave-one-out evaluation across all features files.
    For each session: exclude it from examples, score it, compare to ground truth.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Build ground truth index from pairs
    gt_index = {p["id"]: json.loads(p["output"]) for p in all_pairs}
    gt_meta  = {p["id"]: p["meta"] for p in all_pairs}

    feat_files = sorted(Path(features_dir).glob("*_features.json"))
    results    = []

    print(f"\nLeave-one-out evaluation on {len(feat_files)} sessions (k={k})...\n")

    for fpath in feat_files:
        with open(fpath, encoding="utf-8") as f:
            features = json.load(f)

        vid = features["video_id"]

        # Only evaluate if we have ground truth
        if vid not in gt_index:
            print(f"  ⚠  {vid}: no ground truth — skipping")
            continue

        report = score_session(features, all_pairs, taxonomy, lang_type, k=k, verbose=True)

        # Compare with ground truth
        gt_scores = gt_index[vid]
        gt_total  = gt_meta[vid]["total"]
        gt_class  = gt_meta[vid]["classification"]

        pred_scores = report["item_scores_raw"]
        item_errors = {}
        exact_match = 0
        total_items = 0

        for code, gt_val in gt_scores.items():
            pred_val = pred_scores.get(code, 9)
            err = abs(gt_val - pred_val)
            item_errors[code] = err
            exact_match += (1 if err == 0 else 0)
            total_items += 1

        mae        = sum(item_errors.values()) / total_items if total_items > 0 else 0
        exact_pct  = exact_match / total_items if total_items > 0 else 0
        class_match= report["classification"] == gt_class

        result = {
            "video_id":         vid,
            "module":           report["module"],
            "pred_total":       report["total"],
            "gt_total":         gt_total,
            "total_error":      abs(report["total"] - gt_total),
            "pred_class":       report["classification"],
            "gt_class":         gt_class,
            "class_correct":    class_match,
            "item_mae":         round(mae, 3),
            "item_exact_pct":   round(exact_pct, 3),
            "item_errors":      item_errors,
        }
        results.append(result)

        print(f"  {vid}:  total {report['total']} (gt {gt_total})  "
              f"class {'✓' if class_match else '✗'} {report['classification']} vs {gt_class}  "
              f"MAE {mae:.2f}  exact {exact_pct:.0%}")

    # Summary
    if results:
        avg_mae       = sum(r["item_mae"] for r in results) / len(results)
        avg_exact     = sum(r["item_exact_pct"] for r in results) / len(results)
        class_acc     = sum(1 for r in results if r["class_correct"]) / len(results)
        avg_tot_err   = sum(r["total_error"] for r in results) / len(results)

        print(f"\n{'═'*50}")
        print(f"  FEW-SHOT EVALUATION SUMMARY  (k={k})")
        print(f"{'═'*50}")
        print(f"  Sessions evaluated:     {len(results)}")
        print(f"  Classification acc:     {class_acc:.0%}")
        print(f"  Total score MAE:        {avg_tot_err:.2f}")
        print(f"  Per-item MAE:           {avg_mae:.3f}")
        print(f"  Per-item exact match:   {avg_exact:.0%}")
        print(f"{'═'*50}\n")

        summary = {
            "method":         "few-shot",
            "k":              k,
            "n_evaluated":    len(results),
            "classification_accuracy": round(class_acc, 3),
            "total_score_mae":         round(avg_tot_err, 3),
            "per_item_mae":            round(avg_mae, 3),
            "per_item_exact_pct":      round(avg_exact, 3),
            "per_session":    results,
        }

        out_path = Path(output_dir) / "fewshot_evaluation.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"  Results saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",     default=None,
                        help="Single features JSON to score")
    parser.add_argument("--features_dir", default="data/features",
                        help="Directory of feature JSONs (for --evaluate)")
    parser.add_argument("--pairs",        default="data/training_pairs.json")
    parser.add_argument("--taxonomy",     default="data/ados_taxonomy_combined.json")
    parser.add_argument("--module",       default="M1", help="M1 or M2")
    parser.add_argument("--lang_type",    default="few_or_no_words")
    parser.add_argument("--k",            type=int, default=3,
                        help="Number of few-shot examples")
    parser.add_argument("--evaluate",     action="store_true",
                        help="Run leave-one-out evaluation on all sessions")
    parser.add_argument("--output_dir",   default="outputs/fewshot")
    args = parser.parse_args()

    with open(args.pairs, encoding="utf-8") as f:
        all_pairs = json.load(f)
    taxonomy = load_taxonomy(args.taxonomy)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.evaluate:
        evaluate(
            features_dir = args.features_dir,
            all_pairs    = all_pairs,
            taxonomy     = taxonomy,
            lang_type    = args.lang_type,
            k            = args.k,
            output_dir   = args.output_dir,
        )
    elif args.features:
        with open(args.features, encoding="utf-8") as f:
            features = json.load(f)
        report = score_session(features, all_pairs, taxonomy, args.lang_type, k=args.k)
        out = Path(args.output_dir) / f"{features['video_id']}_fewshot_report.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved → {out}")
    else:
        print("Provide --features <file> or --features_dir with --evaluate")