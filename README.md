# ADOS-2 Behavioral Report Generator via LLM

Automated ADOS-2 scoring from clinical observations using fine-tuned open-source LLMs.  
Bridges autism assessment expertise with modern language model capabilities.

---

## Overview

This project builds an LLM-based pipeline that takes raw behavioral observations from  
ADOS-2 clinical sessions and produces structured, per-item ADOS-2 scores with clinical rationale.

**Architecture:**
```
Annotation file (.txt)  →  Feature extraction  →  Behavioral summary
Severity sheet (.ods)   →  Score parser        →  Ground-truth scores
                                                        ↓
                                    Taxonomy JSON + Training pairs
                                                        ↓
                                     QLoRA fine-tuned Qwen2.5-7B
                                                        ↓
                                      Per-item ADOS-2 score output
```

---

## Project Structure

```
ados-llm/
├── data/
│   ├── annotations/          # Raw .txt annotation files (one per video)
│   ├── features/             # Extracted behavioral features (step 3)
│   ├── scores/               # Parsed ADOS scores per patient (step 3b)
│   └── training_pairs.json   # Final training dataset (step 3c)
├── build_taxonomy.py         # Step 1+2: PDF extraction + translation
├── step3_extract_features.py # Step 3:  Annotation → behavioral summary
├── step3b_parse_scores.py    # Step 3b: ODS score sheet → per-patient JSON
├── step3c_build_pairs.py     # Step 3c: Join features + scores → training pairs
└── requirements.txt
```

---

## Steps

### Step 1 & 2 — Build ADOS-2 Taxonomy

Extracts all scoring items from French ADOS-2 Module 1 and 2 PDFs,  
translates with `Helsinki-NLP/opus-mt-fr-en`, builds enriched taxonomy JSON.

```bash
python build_taxonomy.py
# Output: data/ados_taxonomy_combined.json
```

---

### Step 3 — Feature Extraction

Converts ELAN annotation files into structured behavioral summaries.

```bash
python step3_extract_features.py \
    --annotations_dir data/annotations/ \
    --output_dir      data/features/
```

| Category     | Maps to ADOS items |
|--------------|--------------------|
| STEREOTYPIES | D2, D4             |
| Tete         | D2                 |
| SENSORIEL    | D1                 |
| POINTAGE     | A7                 |
| DEMARCHE     | D2                 |
| CLINICIEN    | B6, B11 (context)  |

---

### Step 3b — Score Sheet Parser

Parses the ODS severity report into per-patient score JSONs.

```bash
python step3b_parse_scores.py \
    --scores_file data/Severity_report_According_ADOS_2.ods \
    --output_dir  data/scores/
```

---

### Step 3c — Training Pair Builder

Joins features + scores into instruction-tuning pairs for QLoRA.

```bash
python step3c_build_pairs.py \
    --features_dir data/features/ \
    --scores_dir   data/scores/ \
    --taxonomy     data/ados_taxonomy_combined.json \
    --output       data/training_pairs.json
```

---

## Requirements

```bash
pip install transformers sentencepiece torch pandas odfpy numpy
apt install poppler-utils
```

---

## Next Steps

- Step 4: QLoRA fine-tuning of Qwen2.5-7B-Instruct
- Step 5: Evaluation on held-out sessions
- Step 6: Multimodal extension with video VLM

---

## Disclaimer

Research prototype only. All outputs must be validated by a qualified clinician.
