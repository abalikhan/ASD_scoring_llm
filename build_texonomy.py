"""
ADOS-2 Taxonomy Builder — Step 1
═════════════════════════════════
Extracts ADOS-2 scoring items from French PDFs, translates them using
Helsinki-NLP/opus-mt-fr-en (dedicated translation model), and produces
an enriched English taxonomy JSON.

Requirements:
    pip install transformers sentencepiece torch pdfplumber

Usage:
    python build_taxonomy.py

Input:
    ADOS_2__M1.pdf    (French, Module 1)
    ADOS_2__M2.pdf    (French, Module 2)

Output:
    data/ados_taxonomy_m1.json
    data/ados_taxonomy_m2.json
    data/ados_taxonomy_combined.json
"""

import re
import json
from pathlib import Path

from fr_en_model import translate_fr_to_en
from pdf_extraction import extract_pdf_text, normalize_text
from item_parsers import split_into_item_blocks, extract_scores, extract_description
from taxonomy import ITEM_NAMES_EN, DOMAIN_META, BEHAVIORAL_INDICATORS, ACTIVITIES_M1, ACTIVITIES_M2, ALGORITHM_M1, ALGORITHM_M2, ITEM_NAMES_EN_M2
# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — update these paths to match your setup
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path

PDF_M1  = "/home/aali/ASD_score/ADOS_2_M1.pdf"
PDF_M2  = "/home/aali/ASD_score/ADOS_2_M2.pdf"
OUT_DIR = Path("./data")
OUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# FULL MODULE EXTRACTION + TRANSLATION + ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────

def build_module_taxonomy(
    pdf_path: str,
    module: str,
    activities: list[dict],
    algorithm: dict,
    item_name_overrides: dict = None,
) -> dict:
    """Full pipeline: extract → translate → enrich → structure."""

    print(f"\n{'═'*60}")
    print(f"  Module {module}  — {pdf_path}")
    print(f"{'═'*60}")

    # Extract text
    print("  [1/4] Extracting PDF text...")
    raw_text   = extract_pdf_text(pdf_path)
    clean_text = normalize_text(raw_text)

    # Parse items
    print("  [2/4] Parsing item blocks...")
    blocks = split_into_item_blocks(clean_text)
    print(f"        Found {len(blocks)} items: {[b['code'] for b in blocks]}")

    # Translate + build items
    print("  [3/4] Translating rubrics (opus-mt-fr-en)...")
    items = []
    for block in blocks:
        code = block["code"]

        # Extract French content
        scores_fr = extract_scores(block["raw_block"])
        desc_fr   = extract_description(block["raw_block"])

        # Translate to English
        scores_en = {}
        for score_key, score_text in scores_fr.items():
            scores_en[score_key] = translate_fr_to_en(score_text)

        desc_en = translate_fr_to_en(desc_fr) if desc_fr else ""

        # Official English name
        name_en = (item_name_overrides or {}).get(code) or ITEM_NAMES_EN.get(code, block["name_fr"])

        # Algorithm role
        in_sa  = code in algorithm["SA_items"]
        in_crr = code in algorithm["CRR_items"]

        # Behavioral indicators (enrichment)
        indicators = BEHAVIORAL_INDICATORS.get(code, {})

        item = {
            "code":               code,
            "domain":             block["domain"],
            "domain_name_en":     DOMAIN_META[block["domain"]]["name_en"],
            "name_fr":            block["name_fr"],
            "name_en":            name_en,
            "description_fr":     desc_fr,
            "description_en":     desc_en,
            "scores_fr":          scores_fr,
            "scores_en":          scores_en,
            "in_algorithm":       in_sa or in_crr,
            "algorithm_domain":   "SA" if in_sa else ("CRR" if in_crr else None),
            "behavioral_indicators": indicators,
        }
        items.append(item)

        algo_tag = f" [{item['algorithm_domain']}]" if item['in_algorithm'] else ""
        print(f"        {code:>4}  ✓  {len(scores_fr)} scores  {name_en[:40]}{algo_tag}")

    # Organise by domain
    print("  [4/4] Building taxonomy...")
    domains = {}
    for d_code, d_meta in DOMAIN_META.items():
        d_items = [i for i in items if i["domain"] == d_code]
        if d_items:
            domains[d_code] = {"name_en": d_meta["name_en"], "items": d_items}

    taxonomy = {
        "module":       module,
        "activities":   activities,
        "domains":      domains,
        "algorithm":    algorithm,
        "total_items":  len(items),
        "items_flat":   items,
    }

    return taxonomy


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Module 1
    tax_m1 = build_module_taxonomy(PDF_M1, "M1", ACTIVITIES_M1, ALGORITHM_M1)
    with open(OUT_DIR / "ados_taxonomy_m1.json", "w", encoding="utf-8") as f:
        json.dump(tax_m1, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Saved → data/ados_taxonomy_m1.json")

    # Module 2
    tax_m2 = build_module_taxonomy(PDF_M2, "M2", ACTIVITIES_M2, ALGORITHM_M2, ITEM_NAMES_EN_M2)
    with open(OUT_DIR / "ados_taxonomy_m2.json", "w", encoding="utf-8") as f:
        json.dump(tax_m2, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Saved → data/ados_taxonomy_m2.json")

    # Combined
    combined = {"modules": {"M1": tax_m1, "M2": tax_m2}}
    with open(OUT_DIR / "ados_taxonomy_combined.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Saved → data/ados_taxonomy_combined.json")

    # Summary
    print(f"\n{'═'*60}")
    for mod, tax in [("M1", tax_m1), ("M2", tax_m2)]:
        algo = [i["code"] for i in tax["items_flat"] if i["in_algorithm"]]
        enriched = [i["code"] for i in tax["items_flat"] if i["behavioral_indicators"]]
        print(f"  {mod}: {tax['total_items']} items | {len(algo)} in algorithm | {len(enriched)} enriched")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()