"""
Step 3c — Training Pair Builder
═════════════════════════════════
Joins behavioral features (from step3_extract_features.py) with
ground-truth ADOS scores (from step3b_parse_scores.py) to produce
instruction-tuning pairs for QLoRA fine-tuning.

Each pair is:
    INPUT:  behavioral narrative + taxonomy rubrics for each item
    OUTPUT: per-item scores with rationale (structured JSON)

The video ID in the annotation filename must match the patient ID
in the score sheet. Matching is done by substring (e.g., "02022"
matches "02-EY-02022").

Usage:
    python step3c_build_pairs.py --features_dir data/features/
                                  --scores_dir   data/scores/
                                  --taxonomy     data/ados_taxonomy_combined.json
                                  --output       data/training_pairs.json
"""

import json
import re
import argparse
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def extract_id_key(name: str) -> str:
    """
    Extract a normalised numeric key from a filename or patient ID.
    e.g. "V2D-02022" → "02022"
         "02-EY-02022" → "02022"
         "02022_features" → "02022"
    """
    nums = re.findall(r"\d{4,}", name)
    return nums[-1] if nums else name.lower()


def load_json(path: str) -> dict | list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_taxonomy_item(taxonomy: dict, module: str, item_code: str) -> dict | None:
    """Retrieve a single item from the taxonomy by module and code."""
    mod_data = taxonomy["modules"].get(module, {})
    for item in mod_data.get("items_flat", []):
        if item["code"] == item_code:
            return item
    return None


def build_rubric_text(item: dict) -> str:
    """Build a readable rubric string from an item dict."""
    scores = item.get("scores_en") or item.get("scores_fr", {})
    if not scores:
        return "No rubric available."
    return "\n".join(
        f"  Score {k}: {v}"
        for k in sorted(scores.keys(), key=lambda x: (x == "B", x))
        for v in [scores[k]]
    )


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert clinical psychologist trained in ADOS-2 assessment. "
    "You will be given behavioral observations from a clinical session and "
    "must score each ADOS-2 item according to the provided rubrics. "
    "Return only a valid JSON object with item codes as keys and integer scores as values."
)

def build_input_prompt(narrative: str, taxonomy: dict, module: str, items_to_score: list) -> str:
    """
    Build the full input prompt for an LLM training pair.
    Includes behavioral narrative + rubrics for each item to score.
    """
    lines = [
        "You are an expert ADOS-2 clinician. Score each item based on the observations below.",
        "",
        "## Behavioral Observations",
        narrative.strip(),
        "",
        "## Items to Score",
    ]

    for code in items_to_score:
        item = get_taxonomy_item(taxonomy, module, code)
        if item:
            lines.append(f"\n### {code} — {item['name_en']}")
            lines.append(f"Domain: {item['domain_name_en']}")
            lines.append("Rubric:")
            lines.append(build_rubric_text(item))

    lines += [
        "",
        "## Instructions",
        "Score each item 0, 1, 2, or 3 based on the observations.",
        "Use score 9 if there is no information available for that item.",
        "Return ONLY a JSON object, e.g.: {\"A2\": 2, \"A7\": 3, \"D1\": 1, ...}",
    ]

    return "\n".join(lines)


def build_output(item_scores: dict, items_to_score: list) -> str:
    """Build the expected output JSON string for the training pair."""
    output_scores = {
        code: item_scores.get(code, 9)
        for code in items_to_score
    }
    return json.dumps(output_scores, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# PAIR BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_pairs(
    features_dir: str,
    scores_dir:   str,
    taxonomy_path: str,
    output_path:  str,
) -> list[dict]:
    """
    Match feature files to score files, build instruction-tuning pairs.
    """
    features_path = Path(features_dir)
    scores_path   = Path(scores_dir)

    # Load taxonomy
    taxonomy = load_json(taxonomy_path)

    # Load all scores, index by normalized key
    all_scores_file = scores_path / "all_scores.json"
    all_scores = load_json(str(all_scores_file))
    score_index = {extract_id_key(r["patient_id"]): r for r in all_scores}

    # Load all features, index by normalized key
    feature_files = sorted(features_path.glob("*_features.json"))
    feature_index = {}
    for fp in feature_files:
        feat = load_json(str(fp))
        if isinstance(feat, list):
            for item in feat:
                feature_index[extract_id_key(item["video_id"])] = item
        else:
            feature_index[extract_id_key(feat["video_id"])] = feat

    print(f"Features: {len(feature_index)} | Scores: {len(score_index)}")

    # Match and build pairs
    pairs = []
    matched = 0
    unmatched_features = []

    for feat_key, feat in feature_index.items():
        # Try direct match, then partial match
        score_rec = score_index.get(feat_key)
        if score_rec is None:
            # Try partial match: find any score key that contains feat_key
            for sk, sv in score_index.items():
                if feat_key in sk or sk in feat_key:
                    score_rec = sv
                    break

        if score_rec is None:
            unmatched_features.append(feat["video_id"])
            continue

        module         = score_rec["module"]
        item_scores    = score_rec["item_scores"]
        sa_items       = score_rec["sa_items"]
        crr_items      = score_rec["crr_items"]
        items_to_score = sorted(set(sa_items + crr_items))

        # Build the training pair
        input_text  = build_input_prompt(feat["narrative"], taxonomy, module, items_to_score)
        output_text = build_output(item_scores, items_to_score)

        pair = {
            "id":             feat["video_id"],
            "patient_id":     score_rec["patient_id"],
            "module":         module,
            "system":         SYSTEM_PROMPT,
            "input":          input_text,
            "output":         output_text,
            # Metadata — not used in training, useful for analysis
            "meta": {
                "sa_total":       score_rec["sa_total"],
                "crr_total":      score_rec["crr_total"],
                "total":          score_rec["total"],
                "classification": score_rec["classification"],
                "session_duration_sec": feat["session_duration_sec"],
            }
        }

        # Also add a chat-format version (for models like Qwen2.5-Instruct)
        pair["chat_format"] = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": input_text},
            {"role": "assistant", "content": output_text},
        ]

        pairs.append(pair)
        matched += 1
        print(f"  ✓ {feat['video_id']} ↔ {score_rec['patient_id']} "
              f"[{module}] {score_rec['classification']}")

    if unmatched_features:
        print(f"\n  ⚠ No score match for: {unmatched_features}")

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    # Save chat-format only (for direct use with trl/SFTTrainer)
    chat_path = out.parent / "training_pairs_chat.json"
    chat_only = [p["chat_format"] for p in pairs]
    with open(chat_path, "w", encoding="utf-8") as f:
        json.dump(chat_only, f, indent=2, ensure_ascii=False)

    print(f"\n✅ {matched} training pairs built")
    print(f"   Full pairs:  {out}")
    print(f"   Chat format: {chat_path}")

    # Stats
    modules = {}
    classes = {}
    for p in pairs:
        modules[p["module"]] = modules.get(p["module"], 0) + 1
        c = p["meta"]["classification"]
        classes[c] = classes.get(c, 0) + 1

    print(f"\n   Module distribution: {modules}")
    print(f"   Classification distribution: {classes}")

    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir",  default="data/features")
    parser.add_argument("--scores_dir",    default="data/scores")
    parser.add_argument("--taxonomy",      default="data/ados_taxonomy_combined.json")
    parser.add_argument("--output",        default="data/training_pairs.json")
    args = parser.parse_args()

    build_pairs(
        features_dir  = args.features_dir,
        scores_dir    = args.scores_dir,
        taxonomy_path = args.taxonomy,
        output_path   = args.output,
    )
