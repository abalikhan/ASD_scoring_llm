"""
Step 3 — Annotation Feature Extractor  (CSV version)
══════════════════════════════════════════════════════
Converts ELAN annotation CSV files into structured English behavioral
summaries for LLM input.

All French→English translation is driven by annotation_vocabulary.json.
Start and end times are preserved in every event for downstream use
(temporal alignment with video frames in the VLM pipeline).

CSV format (tab-separated, no header):
    CATEGORY  <empty>  start_sec  end_sec  duration_sec  [subcategory]

Output per video — all fields in English:
    video_id, module, activities_observed,
    session_duration_sec, category_stats (with events including start/end),
    narrative, ados_item_hints

Usage:
    python step3_extract_features.py
        --annotations_dir  data/annotations/
        --output_dir       data/features/
        --vocabulary       data/annotation_vocabulary.json
"""

import re
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# VOCABULARY LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_vocabulary(vocab_path: str) -> dict:
    """Load annotation_vocabulary.json. Raises clear error if missing."""
    path = Path(vocab_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Vocabulary file not found: {vocab_path}\n"
            "Make sure annotation_vocabulary.json is in your data directory."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def translate_category(cat: str, vocab: dict) -> str:
    """Translate a French category name to English."""
    return vocab["categories"].get(cat, {}).get("en", cat)


def translate_subcategory(sub: str, vocab: dict) -> str:
    """Translate a French subcategory to English. Handles compound values."""
    if not sub:
        return ""
    # Try exact match first
    if sub in vocab["subcategories"]:
        return vocab["subcategories"][sub]
    # Try each space-separated token (e.g. "Saute-place Saute-place")
    tokens = sub.split()
    translated = [vocab["subcategories"].get(t, t) for t in tokens]
    # Deduplicate while preserving order
    seen, unique = set(), []
    for t in translated:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return ", ".join(unique)


def translate_activity(act_fr: str, vocab: dict) -> str:
    """Translate a French activity code to English."""
    return vocab["activities"].get(act_fr, {}).get("en", act_fr)


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Event:
    """A single annotated behavioral event — fully translated to English."""
    category_fr:   str
    category_en:   str
    subcategory_fr: str
    subcategory_en: str
    start_sec:     float
    end_sec:       float
    duration_sec:  float
    ados_items:    list    # which ADOS items this event informs


@dataclass
class CategoryStats:
    category_fr:        str
    category_en:        str
    ados_items:         list
    count:              int   = 0
    total_duration_sec: float = 0.0
    mean_duration_sec:  float = 0.0
    max_duration_sec:   float = 0.0
    # Full event list with start/end times — for VLM temporal alignment
    events:             list  = field(default_factory=list)
    subcategories_fr:   list  = field(default_factory=list)
    subcategories_en:   list  = field(default_factory=list)
    subcategory_counts: dict  = field(default_factory=dict)


@dataclass
class BehavioralSummary:
    video_id:             str
    module:               str
    activities_observed:  list   # English activity names in session order
    session_duration_sec: float
    category_stats:       dict   # category_en → CategoryStats (as dict)
    narrative:            str    # English narrative for LLM input
    ados_item_hints:      dict   # item_code → list of English observation strings


# ─────────────────────────────────────────────────────────────────────────────
# CSV PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_csv(filepath: str, vocab: dict) -> tuple[str, list[Event]]:
    """
    Parse a CSV annotation file into a list of fully translated Events.
    Preserves start_sec and end_sec on every event.
    """
    video_id = Path(filepath).stem
    events   = []

    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = re.split(r"\t", line)
            parts = [p.strip() for p in parts]

            if len(parts) < 5:
                continue

            try:
                category_fr = parts[0]

                # Collect the 3 float values and any remaining text (subcategory)
                float_vals, sub_parts = [], []
                for p in parts[1:]:
                    if len(float_vals) < 3:
                        try:
                            float_vals.append(float(p))
                        except ValueError:
                            if p:
                                sub_parts.append(p)
                    else:
                        if p:
                            sub_parts.append(p)

                if len(float_vals) < 3:
                    continue

                start_sec    = float_vals[0]
                end_sec      = float_vals[1]
                duration_sec = float_vals[2]
                sub_fr       = " ".join(sub_parts).strip()

                # Normalise case variation before translation
                cat_key = category_fr
                if cat_key.lower() == "attention conjointe":
                    cat_key = "Attention Conjointe"

                cat_info   = vocab["categories"].get(cat_key, {})
                cat_en     = cat_info.get("en", cat_key)
                ados_items = cat_info.get("ados_items", [])
                sub_en     = translate_subcategory(sub_fr, vocab)

                events.append(Event(
                    category_fr    = category_fr,
                    category_en    = cat_en,
                    subcategory_fr = sub_fr,
                    subcategory_en = sub_en,
                    start_sec      = start_sec,
                    end_sec        = end_sec,
                    duration_sec   = duration_sec,
                    ados_items     = ados_items,
                ))

            except (ValueError, IndexError):
                continue

    return video_id, events


# ─────────────────────────────────────────────────────────────────────────────
# MODULE & ACTIVITY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_module_and_activities(events: list[Event], vocab: dict) -> tuple[str, list[dict]]:
    """
    Detect ADOS module (M1/M2) and extract ordered activities from ADOS Activity events.
    Returns (module_str, activities_list) where each activity has fr/en names + timing.
    """
    activity_events = [e for e in events if e.category_fr == "ADOS-Module"]

    if not activity_events:
        return "unknown", []

    m1_count = sum(1 for e in activity_events if e.subcategory_fr.startswith("M1"))
    m2_count = sum(1 for e in activity_events if e.subcategory_fr.startswith("M2"))
    module   = "M1" if m1_count >= m2_count else "M2"

    # Ordered unique activities preserving first occurrence
    seen, activities = set(), []
    for e in sorted(activity_events, key=lambda x: x.start_sec):
        name_fr = e.subcategory_fr.split()[0].strip() if e.subcategory_fr else ""
        if name_fr and name_fr not in seen:
            seen.add(name_fr)
            act_info = vocab["activities"].get(name_fr, {})
            activities.append({
                "name_fr":   name_fr,
                "name_en":   act_info.get("en", name_fr),
                "start_sec": e.start_sec,
                "end_sec":   e.end_sec,
            })

    return module, activities


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_category_stats(events: list[Event], vocab: dict) -> dict:
    """
    Aggregate events by English category name.
    Each CategoryStats includes the full event list with start/end times.
    Keyed by category_en for a fully English output.
    """
    behavioral = [e for e in events if e.category_fr != "ADOS-Module"]

    grouped = defaultdict(list)
    for e in behavioral:
        grouped[e.category_en].append(e)

    stats = {}
    for cat_en, cat_events in grouped.items():
        durations = [e.duration_sec for e in cat_events]

        sub_counts_en = defaultdict(int)
        sub_fr_set, sub_en_set = set(), set()
        for e in cat_events:
            if e.subcategory_en:
                sub_counts_en[e.subcategory_en] += 1
                sub_fr_set.add(e.subcategory_fr)
                sub_en_set.add(e.subcategory_en)

        # First event gives us the fr name and ados_items
        sample      = cat_events[0]
        cat_fr      = sample.category_fr
        ados_items  = sample.ados_items

        # Full event list — preserves start/end for VLM temporal alignment
        event_list = [
            {
                "start_sec":      e.start_sec,
                "end_sec":        e.end_sec,
                "duration_sec":   e.duration_sec,
                "subcategory_fr": e.subcategory_fr,
                "subcategory_en": e.subcategory_en,
            }
            for e in sorted(cat_events, key=lambda x: x.start_sec)
        ]

        stats[cat_en] = CategoryStats(
            category_fr        = cat_fr,
            category_en        = cat_en,
            ados_items         = ados_items,
            count              = len(cat_events),
            total_duration_sec = round(sum(durations), 3),
            mean_duration_sec  = round(sum(durations) / len(durations), 3),
            max_duration_sec   = round(max(durations), 3),
            events             = event_list,
            subcategories_fr   = sorted(sub_fr_set),
            subcategories_en   = sorted(sub_en_set),
            subcategory_counts = dict(sub_counts_en),
        )

    return stats


def build_ados_hints(stats: dict, vocab: dict) -> dict:
    """Build per-ADOS-item English observation hints from category stats."""
    hints       = defaultdict(list)
    stereo_cats = set(vocab["stereotypy_categories"])
    # Map fr→en for stereotypy check
    stereo_en   = {
        vocab["categories"].get(c, {}).get("en", c)
        for c in stereo_cats
    }

    for cat_en, s in stats.items():
        sub_en = s.subcategories_en

        if cat_en in stereo_en:
            desc = (
                f"{cat_en}: {s.count} episodes "
                f"(total {s.total_duration_sec:.1f}s, "
                f"avg {s.mean_duration_sec:.1f}s, "
                f"max {s.max_duration_sec:.1f}s)"
            )
            if sub_en:
                desc += f"; subtypes: {', '.join(sub_en)}"
            hints["D2"].append(desc)
            if cat_en in ("Stereotypies", "Gait Stereotypies"):
                hints["D4"].append(desc)

        elif cat_en == "Sensory Behaviors":
            desc = f"Sensory exploration: {s.count} episodes ({s.total_duration_sec:.1f}s)"
            if sub_en:
                desc += f"; focus: {', '.join(sub_en)}"
            hints["D1"].append(desc)

        elif cat_en == "Pointing":
            hints["A7"].append(
                f"{s.count} pointing gesture(s) ({s.total_duration_sec:.1f}s total)"
                if s.count > 0 else "No pointing gestures observed"
            )

        elif cat_en == "Unusual Gaze":
            hints["B1"].append(
                f"Unusual gaze: {s.count} episodes ({s.total_duration_sec:.1f}s)"
            )

        elif cat_en == "Response to Name":
            hints["B6"].append(f"Response to Name prompts: {s.count} instances")

        elif cat_en == "Joint Attention":
            hints["B11"].append(
                f"Joint Attention prompts: {s.count} ({s.total_duration_sec:.1f}s)"
            )

        elif cat_en == "Clinician Prompts":
            name_calls = s.subcategory_counts.get("Name Call", 0)
            tickles    = s.subcategory_counts.get("Tickling", 0)
            if name_calls:
                hints["B6"].append(f"Clinician name calls: {name_calls}")
            if tickles:
                hints["B11"].append(f"Clinician physical prompts (tickle): {tickles}")

    if "A7" not in hints:
        hints["A7"].append("No pointing behavior observed")
    if "B1" not in hints:
        hints["B1"].append("No unusual gaze episodes recorded")

    return dict(hints)


# ─────────────────────────────────────────────────────────────────────────────
# NARRATIVE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_narrative(
    video_id:    str,
    module:      str,
    activities:  list,
    duration_sec: float,
    stats:       dict,
    vocab:       dict,
) -> str:
    dur_min     = duration_sec / 60
    stereo_cats = set(vocab["stereotypy_categories"])
    stereo_en   = {vocab["categories"].get(c, {}).get("en", c) for c in stereo_cats}

    lines = [
        f"Behavioral observation summary for session {video_id}.",
        f"ADOS-2 Module: {module}. Session duration: {dur_min:.1f} minutes.",
    ]
    if activities:
        act_names = [a["name_en"] for a in activities]
        lines.append(f"Activities: {', '.join(act_names)}.")
    lines.append("")

    # Stereotypy / motor
    stereo = [c for c in stats if c in stereo_en]
    if stereo:
        lines.append("Repetitive and stereotyped motor behaviors:")
        for cat_en in stereo:
            s    = stats[cat_en]
            rate = s.count / dur_min if dur_min > 0 else 0
            line = (
                f"  - {cat_en}: {s.count} episodes "
                f"({s.total_duration_sec:.1f}s total, {rate:.1f}/min, "
                f"max {s.max_duration_sec:.1f}s)"
            )
            if s.subcategories_en:
                line += f". Subtypes: {', '.join(s.subcategories_en)}."
            lines.append(line)
    else:
        lines.append("No repetitive motor behaviors recorded.")
    lines.append("")

    # Sensory
    if "Sensory Behaviors" in stats:
        s    = stats["Sensory Behaviors"]
        line = (
            f"Unusual sensory behaviors: {s.count} episodes "
            f"({s.total_duration_sec:.1f}s total)"
        )
        if s.subcategories_en:
            line += f". Focus: {', '.join(s.subcategories_en)}."
        lines.append(line)
    elif "Hand Movements" in stats:
        s    = stats["Hand Movements"]
        line = (
            f"Hand/sensory movements: {s.count} episodes "
            f"({s.total_duration_sec:.1f}s total)"
        )
        if s.subcategories_en:
            line += f". Types: {', '.join(s.subcategories_en)}."
        lines.append(line)
    else:
        lines.append("No unusual sensory behaviors recorded.")
    lines.append("")

    # Communication
    if "Pointing" in stats:
        s = stats["Pointing"]
        lines.append(
            f"Pointing: {s.count} gesture(s) ({s.total_duration_sec:.1f}s)."
            if s.count > 0 else "Pointing: Not observed."
        )
    else:
        lines.append("Pointing: Not observed.")

    if "Unusual Gaze" in stats:
        s = stats["Unusual Gaze"]
        lines.append(f"Unusual gaze: {s.count} episodes ({s.total_duration_sec:.1f}s total).")
    lines.append("")

    # Prompts
    prompt_parts = []
    for cat_en in ["Response to Name", "Joint Attention", "Clinician Prompts"]:
        if cat_en in stats:
            s     = stats[cat_en]
            label = f"{cat_en} ({', '.join(s.subcategories_en)})" if s.subcategories_en else cat_en
            prompt_parts.append(f"{label}: {s.count}")
    if prompt_parts:
        lines.append("Clinician prompts — " + "; ".join(prompt_parts) + ".")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(csv_path: str, vocab: dict) -> BehavioralSummary:
    """Full pipeline: CSV + vocabulary → BehavioralSummary (all English)."""
    video_id, events = parse_csv(csv_path, vocab)
    if not events:
        raise ValueError(f"No events parsed from {csv_path}")

    module, activities   = detect_module_and_activities(events, vocab)
    session_duration     = round(max(e.end_sec for e in events), 1)
    stats                = compute_category_stats(events, vocab)
    hints                = build_ados_hints(stats, vocab)
    narrative            = build_narrative(
        video_id, module, activities, session_duration, stats, vocab
    )

    return BehavioralSummary(
        video_id             = video_id,
        module               = module,
        activities_observed  = activities,
        session_duration_sec = session_duration,
        category_stats       = {k: asdict(v) for k, v in stats.items()},
        narrative            = narrative,
        ados_item_hints      = hints,
    )


def run(annotations_dir: str, output_dir: str, vocab_path: str) -> None:
    ann_path = Path(annotations_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    vocab     = load_vocabulary(vocab_path)
    csv_files = sorted(ann_path.glob("*.csv"))

    if not csv_files:
        print(f"No .csv files found in {annotations_dir}")
        return

    print(f"Vocabulary loaded from {vocab_path}")
    print(f"Processing {len(csv_files)} CSV files...\n")
    summaries = []

    for fpath in csv_files:
        try:
            summary = extract_features(str(fpath), vocab)
            summaries.append(asdict(summary))

            out_file = out_path / f"{summary.video_id}_features.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(asdict(summary), f, indent=2, ensure_ascii=False)

            total_ev = sum(s["count"] for s in summary.category_stats.values())
            print(
                f"  ✓ {summary.video_id}  [{summary.module}]  "
                f"{summary.session_duration_sec/60:.1f}min  "
                f"{total_ev} events  "
                f"cats(en): {list(summary.category_stats.keys())}"
            )

        except Exception as e:
            print(f"  ✗ {fpath.name}: {e}")

    combined = out_path / "all_features.json"
    with open(combined, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    m1 = sum(1 for s in summaries if s["module"] == "M1")
    m2 = sum(1 for s in summaries if s["module"] == "M2")
    print(f"\n✅ {len(summaries)} files processed → {out_path}  (M1:{m1} | M2:{m2})")
    print(f"   All features combined: {combined}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_dir", default="data/annotations")
    parser.add_argument("--output_dir",      default="data/features")
    parser.add_argument("--vocabulary",      default="data/annotation_vocabulary.json")
    args = parser.parse_args()
    run(args.annotations_dir, args.output_dir, args.vocabulary)