"""
Step 3b — Score Sheet Parser
══════════════════════════════
Parses the ADOS-2 severity/scoring spreadsheet (ODS format) into clean
per-video JSON files, one per patient, with item scores, domain totals,
and classification.

The ODS file has a multi-header structure:
    Row 0 → item codes (A1, A2, ... E3)
    Row 1+ → one patient per row
    Columns are grouped into sections:
        ADOS-2 Module 1 | ADOS-2 Module 2 | Scores (AS/CRR/TOTAL) |
        ADOS-1 Module 1 | ADOS-1 Module 2

We extract only ADOS-2 scores (the clinically current version).

Usage:
    python step3b_parse_scores.py --scores_file data/Severity_report_According_ADOS_2.ods
                                  --output_dir  data/scores/
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN STRUCTURE (derived from reading the ODS file)
# The spreadsheet has merged headers — we map by column index.
# ─────────────────────────────────────────────────────────────────────────────

# ADOS-2 Module 1: columns 2–34 (header row 0 gives item codes)
M1_START_COL = 2
M1_END_COL   = 34   # inclusive

# ADOS-2 Module 2: columns 35–62
M2_START_COL = 35
M2_END_COL   = 62   # inclusive

# Summary scores: cols 63, 64, 65 → AS, CRR, TOTAL
AS_COL    = 63
CRR_COL   = 64
TOTAL_COL = 65

# Patient ID columns
PATIENT_NUM_COL = 0
PATIENT_ID_COL  = 1

# Module 1 item codes (from row 0 of the spreadsheet)
M1_ITEMS = [
    "A1","A2","A3","A4","A5","A6","A7","A8",
    "B1","B2","B3","B4","B5","B6","B7","B8","B9",
    "B10","B11","B12","B13","B14","B15","B16",
    "C1","C2",
    "D1","D2","D3","D4",
    "E1","E2","E3",
]

# Module 2 item codes
M2_ITEMS = [
    "A1","A2","A3","A4","A5","A6","A7",
    "B1","B2","B3","B4","B5","B6","B7","B8","B9",
    "B10","B11","B12",
    "C1","C2",
    "D1","D2","D3","D4",
    "E1","E2","E3",
]

# Items that contribute to the SA (Social Affect) algorithm total
SA_ITEMS_M1  = {"A2","A7","A8","B1","B3","B4","B5","B9","B10","B11","B12"}
CRR_ITEMS_M1 = {"A3","A5","D1","D2","D4"}

SA_ITEMS_M2  = {"A6","A7","B1","B2","B3","B5","B6","B8","B11","B12"}
CRR_ITEMS_M2 = {"A4","D1","D2","D4"}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def clean_score(val) -> int | None:
    """
    Convert a raw cell value to an integer score or None.
    Handles: NaN, "-", "", numeric strings, floats.
    """
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    s = str(val).strip()
    if s in ("-", "", "nan", "NaN", "N/A"):
        return None
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def determine_module(row_scores_m1: dict, row_scores_m2: dict) -> str:
    """
    Determine which module was used based on which has non-null scores.
    Returns "M1", "M2", or "unknown".
    """
    m1_filled = sum(1 for v in row_scores_m1.values() if v is not None)
    m2_filled = sum(1 for v in row_scores_m2.values() if v is not None)

    if m1_filled > m2_filled:
        return "M1"
    elif m2_filled > m1_filled:
        return "M2"
    elif m1_filled == 0 and m2_filled == 0:
        return "unknown"
    return "M1"  # default


def classify_from_total(total: int, module: str, language_type: str = None) -> str:
    """Classify based on ADOS-2 algorithm cutoffs."""
    if module == "M1":
        lt = language_type or "few_or_no_words"
        if lt == "few_or_no_words":
            if total >= 16: return "Autism"
            if total >= 11: return "Autism Spectrum"
            return "Non-Spectrum"
        else:  # some_words
            if total >= 12: return "Autism"
            if total >= 8:  return "Autism Spectrum"
            return "Non-Spectrum"
    else:  # M2
        lt = language_type or "under_5_years"
        if lt == "under_5_years":
            if total >= 10: return "Autism"
            if total >= 7:  return "Autism Spectrum"
            return "Non-Spectrum"
        else:
            if total >= 9:  return "Autism"
            if total >= 8:  return "Autism Spectrum"
            return "Non-Spectrum"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_scores(ods_path: str, output_dir: str) -> list[dict]:
    """
    Parse the severity report ODS into per-patient score dicts.
    Saves one JSON per patient + a combined all_scores.json.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading {ods_path}...")
    # Read without header — we parse the header manually
    df = pd.read_excel(ods_path, engine="odf", header=None)
    print(f"  Shape: {df.shape}")

    # Row 0 is sub-header (item codes), rows 1+ are patients
    item_row     = df.iloc[0]
    patient_rows = df.iloc[1:]

    all_records = []

    for _, row in patient_rows.iterrows():
        patient_num = clean_score(row.iloc[PATIENT_NUM_COL])
        patient_id  = str(row.iloc[PATIENT_ID_COL]).strip()

        # Skip empty rows
        if patient_num is None and patient_id in ("nan", "", "NaN"):
            continue

        # ── Extract M1 scores ──────────────────────────────────────────────
        m1_scores = {}
        m1_cols = list(range(M1_START_COL, M1_END_COL + 1))
        for i, item_code in enumerate(M1_ITEMS):
            if i < len(m1_cols):
                m1_scores[item_code] = clean_score(row.iloc[m1_cols[i]])

        # ── Extract M2 scores ──────────────────────────────────────────────
        m2_scores = {}
        m2_cols = list(range(M2_START_COL, M2_END_COL + 1))
        for i, item_code in enumerate(M2_ITEMS):
            if i < len(m2_cols):
                m2_scores[item_code] = clean_score(row.iloc[m2_cols[i]])

        # ── Determine module ───────────────────────────────────────────────
        module = determine_module(m1_scores, m2_scores)
        scores = m1_scores if module == "M1" else m2_scores
        sa_items  = SA_ITEMS_M1  if module == "M1" else SA_ITEMS_M2
        crr_items = CRR_ITEMS_M1 if module == "M1" else CRR_ITEMS_M2

        # ── Read summary scores from spreadsheet ──────────────────────────
        # Apply ADOS algorithm score conversion: raw 3 → 2, raw 8/9 → 0
        def algo_score(raw):
            if raw is None: return 0
            if raw == 3:    return 2
            if raw in (8, 9): return 0
            return raw

        # Compute SA and CRR from item scores (more reliable than reading cells)
        sa_total  = sum(algo_score(scores.get(i)) for i in sa_items)
        crr_total = sum(algo_score(scores.get(i)) for i in crr_items)
        total     = sa_total + crr_total

        # Try reading totals from spreadsheet as fallback/cross-check
        sheet_as    = clean_score(row.iloc[AS_COL])
        sheet_crr   = clean_score(row.iloc[CRR_COL])
        sheet_total = clean_score(row.iloc[TOTAL_COL])

        # Use sheet values if available, else computed values
        final_sa    = sheet_as    if sheet_as    is not None else sa_total
        final_crr   = sheet_crr   if sheet_crr   is not None else crr_total
        final_total = sheet_total if sheet_total is not None else total

        classification = classify_from_total(final_total, module)

        # ── Build clean record ─────────────────────────────────────────────
        # Filter to only non-null scores for the active module
        active_scores = {k: v for k, v in scores.items() if v is not None}

        if not active_scores:
            print(f"  ⚠  {patient_id}: no scores found, skipping")
            continue

        record = {
            "patient_id":     patient_id,
            "patient_num":    patient_num,
            "module":         module,
            "item_scores":    active_scores,
            "sa_total":       final_sa,
            "crr_total":      final_crr,
            "total":          final_total,
            "classification": classification,
            "sa_items":       sorted(sa_items),
            "crr_items":      sorted(crr_items),
        }
        all_records.append(record)

        # Save individual patient JSON
        safe_id = patient_id.replace("/", "_").replace(" ", "_")
        out_file = out_path / f"{safe_id}_scores.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        score_str = " | ".join(f"{k}={v}" for k, v in list(active_scores.items())[:6])
        print(f"  ✓ {patient_id} [{module}] SA={final_sa} CRR={final_crr} "
              f"Total={final_total} → {classification}")
        print(f"    Scores: {score_str}...")

    # Save combined
    combined_path = out_path / "all_scores.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)

    print(f"\n✅ {len(all_records)} patients parsed → {out_path}")
    print(f"   M1: {sum(1 for r in all_records if r['module']=='M1')} | "
          f"M2: {sum(1 for r in all_records if r['module']=='M2')}")
    print(f"   Combined: {combined_path}")

    return all_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_file", required=True,
                        help="Path to the ODS severity report file")
    parser.add_argument("--output_dir",  default="data/scores",
                        help="Directory to save parsed score JSONs")
    args = parser.parse_args()
    parse_scores(args.scores_file, args.output_dir)
