"""
Shared Scoring Utilities
═════════════════════════
Common functions used by both step4a_fewshot_scorer.py and step4b_rag_scorer.py.
Keeps both approaches directly comparable by using identical scoring logic.
"""

import json
import re
import urllib.request
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"   # change to "mistral", "llama3.2", etc.


def call_ollama(prompt: str, temperature: float = 0.1, max_tokens: int = 1024) -> str:
    """Call local Ollama and return response text."""
    data = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL, data=data,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read()).get("response", "").strip()
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Cannot reach Ollama at {OLLAMA_URL}. Run: ollama serve\n{e}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAXONOMY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_taxonomy(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_algorithm_items(taxonomy: dict, module: str) -> list[dict]:
    """Return only the items that appear in the SA or CRR algorithm."""
    mod = taxonomy["modules"].get(module, {})
    return [i for i in mod.get("items_flat", []) if i.get("in_algorithm")]


def build_rubric_text(item: dict) -> str:
    scores = item.get("scores_en") or item.get("scores_fr", {})
    if not scores:
        return "No rubric available."
    return "\n".join(
        f"  Score {k}: {v}"
        for k in sorted(scores.keys(), key=lambda x: (x == "B", x))
        for v in [scores[k]]
    )


# ─────────────────────────────────────────────────────────────────────────────
# ALGORITHM: TOTALS + CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def algo_convert(raw: int) -> int:
    """Apply ADOS-2 algorithm score conversion: 3→2, 8/9→0, else unchanged."""
    if raw == 3:      return 2
    if raw in (8, 9): return 0
    return max(0, raw)


def compute_totals(scores: dict, sa_items: list, crr_items: list) -> tuple[int, int, int]:
    sa  = sum(algo_convert(scores.get(i, 9)) for i in sa_items)
    crr = sum(algo_convert(scores.get(i, 9)) for i in crr_items)
    return sa, crr, sa + crr


def classify(total: int, module: str, lang_type: str) -> tuple[str, int]:
    """
    Returns (classification, approx_comparison_score).
    lang_type — M1: 'few_or_no_words' | 'some_words'
                M2: 'under_5_years'   | '5_years_and_older'
    """
    cutoffs = {
        "M1": {
            "few_or_no_words": {"autism": 16, "spectrum": 11},
            "some_words":      {"autism": 12, "spectrum": 8},
        },
        "M2": {
            "under_5_years":     {"autism": 10, "spectrum": 7},
            "5_years_and_older": {"autism": 9,  "spectrum": 8},
        },
    }
    c = cutoffs.get(module, {}).get(lang_type, {"autism": 10, "spectrum": 7})

    if total >= c["autism"]:
        label, comp = "Autism", min(10, 7 + (total - c["autism"]) // 2)
    elif total >= c["spectrum"]:
        label, comp = "Autism Spectrum", 5
    else:
        label, comp = "Non-Spectrum", max(1, total // 3)

    return label, comp


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM = (
    "You are an expert ADOS-2 clinical psychologist. "
    "Score each item strictly according to the rubric. "
    "Return ONLY a valid JSON object mapping item codes to integer scores. "
    "Use 9 if there is no information for an item."
)


def build_scoring_prompt(
    narrative: str,
    items: list[dict],
    examples: list[dict],   # list of {"narrative": ..., "scores": {...}}
) -> str:
    """
    Build the full prompt with:
      - System instruction
      - K demonstration examples (few-shot or RAG-retrieved)
      - Rubrics for each item
      - The target narrative to score
    """
    lines = [f"[INST] {SYSTEM}\n"]

    # ── Few-shot examples ────────────────────────────────────────────────────
    if examples:
        lines.append("## Demonstration Examples\n")
        for i, ex in enumerate(examples, 1):
            lines.append(f"### Example {i}")
            lines.append(f"Observations:\n{ex['narrative'].strip()}\n")
            lines.append(f"Correct scores:\n{json.dumps(ex['scores'], indent=2)}\n")

    # ── Item rubrics ─────────────────────────────────────────────────────────
    lines.append("## Items to Score\n")
    for item in items:
        lines.append(f"### {item['code']} — {item['name_en']}")
        lines.append(f"Domain: {item['domain_name_en']} | Algorithm: {item['algorithm_domain']}")
        lines.append(build_rubric_text(item))
        lines.append("")

    # ── Target ───────────────────────────────────────────────────────────────
    lines.append("## Session to Score")
    lines.append(f"Observations:\n{narrative.strip()}\n")
    lines.append(
        "Return ONLY a JSON object with item codes as keys and integer scores as values. "
        "Example format: {\"A2\": 2, \"A7\": 3, \"D1\": 1}\n[/INST]"
    )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_scores(response: str, expected_items: list[str]) -> dict:
    """Extract item scores from LLM response. Robust to markdown fences."""
    # Strip markdown fences
    clean = re.sub(r"```(?:json)?|```", "", response).strip()

    # Find the JSON object
    match = re.search(r"\{[^{}]+\}", clean, re.DOTALL)
    if not match:
        # Fallback: score all as 9 (not observed)
        return {code: 9 for code in expected_items}

    try:
        parsed = json.loads(match.group())
        # Validate and clean
        scores = {}
        for code in expected_items:
            val = parsed.get(code, 9)
            try:
                scores[code] = int(float(str(val)))
            except (ValueError, TypeError):
                scores[code] = 9
        return scores
    except json.JSONDecodeError:
        return {code: 9 for code in expected_items}


# ─────────────────────────────────────────────────────────────────────────────
# REPORT RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def render_report(
    video_id:    str,
    module:      str,
    scores:      dict,
    sa_items:    list,
    crr_items:   list,
    lang_type:   str,
    method:      str,   # "few-shot" or "RAG"
    taxonomy:    dict,
) -> dict:
    """Compute totals, classify, return structured report dict."""
    sa, crr, total = compute_totals(scores, sa_items, crr_items)
    label, comp    = classify(total, module, lang_type)

    # Per-item details with algo conversion shown
    items_flat = {
        i["code"]: i
        for i in get_algorithm_items(taxonomy, module)
    }
    item_details = []
    for code in sorted(set(sa_items + crr_items)):
        raw  = scores.get(code, 9)
        algo = algo_convert(raw)
        item = items_flat.get(code, {})
        item_details.append({
            "code":             code,
            "name":             item.get("name_en", code),
            "algorithm_domain": item.get("algorithm_domain"),
            "raw_score":        raw,
            "algo_score":       algo,
        })

    return {
        "video_id":        video_id,
        "module":          module,
        "method":          method,
        "lang_type":       lang_type,
        "item_scores_raw": scores,
        "item_details":    item_details,
        "sa_total":        sa,
        "crr_total":       crr,
        "total":           total,
        "classification":  label,
        "comparison_score": comp,
    }


def print_report(report: dict) -> None:
    print(f"\n{'═'*55}")
    print(f"  ADOS-2 Report  [{report['method'].upper()}]  —  {report['video_id']}")
    print(f"{'═'*55}")
    print(f"  Module: {report['module']}  |  Lang type: {report['lang_type']}")
    print(f"\n  Item Scores:")
    print(f"  {'Code':<6} {'Name':<42} {'Dom':<4} {'Raw':<5} {'Algo'}")
    print(f"  {'-'*62}")
    for d in report["item_details"]:
        print(
            f"  {d['code']:<6} {d['name'][:40]:<42} "
            f"{d['algorithm_domain'] or '—':<4} {d['raw_score']:<5} {d['algo_score']}"
        )
    print(f"\n  SA Total:   {report['sa_total']}")
    print(f"  CRR Total:  {report['crr_total']}")
    print(f"  Total:      {report['total']}")
    print(f"\n  Classification:   {report['classification']}")
    print(f"  Comparison Score: ~{report['comparison_score']}/10")
    print(f"{'═'*55}\n")