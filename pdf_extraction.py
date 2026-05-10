import re
import subprocess

# ─────────────────────────────────────────────────────────────────────────────
# PDF TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf_text(pdf_path: str) -> str:
    """Extract text using pdftotext (poppler-utils). Install: apt install poppler-utils"""
    result = subprocess.run(
        ["pdftotext", "-layout", pdf_path, "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdftotext failed: {result.stderr}")
    return result.stdout


def normalize_text(text: str) -> str:
    """Fix OCR artifacts in ADOS-2 French PDFs."""
    # Form-feed chars glue to item headers — strip them
    text = text.replace("\f", "\n")

    lines = text.split("\n")
    fixed = []
    for line in lines:
        # Bold "B" renders as "8" in item headers: "81." → "B1."
        line = re.sub(r"^8([0-9]{1,2}[ab]?)\.\s", r"B\1. ", line)
        # M2 OCR: "42." → "A2.", "46." → "A6.", "47." → "A7."
        line = re.sub(r"^4([2679])\.\s", r"A\1. ", line)
        # Fix apostrophe corruption
        line = line.replace("I'", "l'").replace("I'", "l'")
        # Score code "8 =" → "B =" in rubric lines
        if re.match(r"^\s+8\s+=", line):
            line = re.sub(r"^(\s+)8(\s+=)", r"\1B\2", line)
        fixed.append(line)
    return "\n".join(fixed)

