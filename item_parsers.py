import re
import json
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# ITEM BLOCK PARSER
# ─────────────────────────────────────────────────────────────────────────────

ITEM_HEADER_RE = re.compile(r"^([A-E])([0-9]{1,2}[ab]?)\.\s+(.+)$", re.MULTILINE)
SCORE_LINE_RE  = re.compile(r"^\s{3,}([0-9B])\s*=\s*(.+)$", re.MULTILINE)

def split_into_item_blocks(text: str) -> list[dict]:
    """Split PDF text into one block per scoring item."""
    matches = list(ITEM_HEADER_RE.finditer(text))
    obs_pos = text.find("\nObservations\n")
    if obs_pos == -1:
        obs_pos = 0
    scoring_matches = [m for m in matches if m.start() > obs_pos]

    blocks = []
    for i, m in enumerate(scoring_matches):
        end = scoring_matches[i + 1].start() if i + 1 < len(scoring_matches) else len(text)
        blocks.append({
            "code":      f"{m.group(1)}{m.group(2)}",
            "domain":    m.group(1),
            "name_fr":   m.group(3).strip(),
            "raw_block": text[m.start():end].strip(),
        })
    return blocks


def extract_scores(raw_block: str) -> dict:
    """Extract score rubrics {score_code: description_text} from an item block."""
    scores = {}
    lines = raw_block.split("\n")
    current_score = None
    current_text = []

    for line in lines:
        score_match = SCORE_LINE_RE.match(line)
        if score_match:
            if current_score is not None:
                scores[current_score] = " ".join(current_text).strip()
            current_score = score_match.group(1)
            current_text = [score_match.group(2).strip()]
        elif current_score is not None:
            stripped = line.strip()
            if stripped and not ITEM_HEADER_RE.match(line) and len(stripped) > 3:
                if not stripped.startswith("ADOS-2"):
                    current_text.append(stripped)

    if current_score is not None:
        scores[current_score] = " ".join(current_text).strip()
    return scores


def extract_description(raw_block: str) -> str:
    """Extract the clinical description paragraph (before score rubrics)."""
    lines = raw_block.split("\n")
    desc = []
    for line in lines:
        if SCORE_LINE_RE.match(line):
            break
        stripped = line.strip()
        if stripped and not ITEM_HEADER_RE.match(line):
            if not re.match(r"^ADOS-2 Module|^\d+\s+ADOS-2", stripped):
                desc.append(stripped)
    return re.sub(r"\s+", " ", " ".join(desc)).strip()

