"""
Step 4b — RAG-Based ADOS-2 Scorer
════════════════════════════════════
Instead of fixed examples, retrieves the most behaviorally similar sessions
from the training set using FAISS vector search, then uses them as dynamic
few-shot examples in the same scoring prompt.

Why RAG beats static few-shot here:
  - A child with severe stereotypies gets examples from similarly severe sessions
  - A mild M2 case gets examples from mild M2 cases
  - With only 18 sessions, this matters a lot

Index: BAAI/bge-base-en-v1.5 embeddings of behavioral narratives → FAISS IndexFlatIP
       Build once, query at inference time.

Usage:
    # Build the FAISS index from training pairs
    python step4b_rag_scorer.py --build_index \ 
        --pairs    data/training_pairs.json \
        --index    data/rag_index/

    # Score a single session
    python step4b_rag_scorer.py \
        --features data/features/V2D02022_features.json \
        --index    data/rag_index/ \
        --taxonomy data/ados_taxonomy_combined.json

    # Leave-one-out evaluation (comparable to step4a)
    python step4b_rag_scorer.py --evaluate \
        --features_dir data/features/ \
        --index        data/rag_index/ \
        --taxonomy     data/ados_taxonomy_combined.json

Requirements:
    pip install faiss-cpu sentence-transformers
"""

import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from scoring_utils import (
    call_ollama, load_taxonomy, get_algorithm_items,
    build_scoring_prompt, parse_scores, render_report, print_report
)

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "Install required packages:\n"
        "  pip install faiss-cpu sentence-transformers"
    )


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING MODEL
# ─────────────────────────────────────────────────────────────────────────────

EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
# BGE retrieval prefix — improves recall significantly for this model
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_embed_model = None   # lazy loaded

def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        print(f"  Loading embedding model: {EMBED_MODEL_NAME}...")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def embed_texts(texts: list[str], is_query: bool = False) -> np.ndarray:
    """Embed a list of texts. Apply BGE query prefix for queries."""
    model = get_embed_model()
    if is_query:
        texts = [BGE_QUERY_PREFIX + t for t in texts]
    embs = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > 10,
    )
    return embs.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# INDEX BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_index(pairs_path: str, index_dir: str) -> None:
    """
    Embed all training pair narratives and build a FAISS IndexFlatIP index.
    Saves index + metadata to disk.
    """
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    with open(pairs_path, encoding="utf-8") as f:
        pairs = json.load(f)

    print(f"Building RAG index from {len(pairs)} training pairs...")

    # Extract narratives — the behavioral summary text is the first section of input
    records, narratives = [], []
    for p in pairs:
        # Pull the narrative from the input field (before "## Items to Score")
        raw_input = p["input"]
        if "## Items to Score" in raw_input:
            narrative = raw_input.split("## Items to Score")[0]
            narrative = narrative.replace(
                "You are an expert ADOS-2 clinician. Score each item based on "
                "the observations below.\n\n## Behavioral Observations\n", ""
            ).strip()
        else:
            narrative = raw_input[:1000]

        records.append({
            "id":             p["id"],
            "patient_id":     p.get("patient_id", ""),
            "module":         p["module"],
            "narrative":      narrative,
            "scores":         json.loads(p["output"]) if isinstance(p["output"], str) else p["output"],
            "classification": p["meta"]["classification"],
            "total":          p["meta"]["total"],
            "sa_total":       p["meta"]["sa_total"],
            "crr_total":      p["meta"]["crr_total"],
        })
        narratives.append(narrative)

    # Embed
    print(f"Embedding {len(narratives)} narratives...")
    embeddings = embed_texts(narratives, is_query=False)
    print(f"Embedding shape: {embeddings.shape}")

    # Build FAISS index — IndexFlatIP for exact cosine search
    # With 18 sessions, exact is always correct choice
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"Index built: {index.ntotal} vectors, dim={dim}")

    # Save
    faiss.write_index(index, str(Path(index_dir) / "rag.faiss"))
    with open(Path(index_dir) / "rag_records.pkl", "wb") as f:
        pickle.dump(records, f)
    np.save(Path(index_dir) / "rag_embeddings.npy", embeddings)

    config = {
        "embed_model": EMBED_MODEL_NAME,
        "n_records":   len(records),
        "dim":         dim,
        "index_type":  "IndexFlatIP",
    }
    with open(Path(index_dir) / "rag_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ RAG index saved → {index_dir}")
    print(f"   Records: {len(records)} | Dim: {dim}")


# ─────────────────────────────────────────────────────────────────────────────
# INDEX LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_index(index_dir: str) -> tuple:
    """Load FAISS index and metadata records from disk."""
    index = faiss.read_index(str(Path(index_dir) / "rag.faiss"))
    with open(Path(index_dir) / "rag_records.pkl", "rb") as f:
        records = pickle.load(f)
    print(f"RAG index loaded: {index.ntotal} vectors")
    return index, records


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(
    query_narrative: str,
    index:           faiss.Index,
    records:         list[dict],
    k:               int = 3,
    exclude_id:      str = None,
    module_filter:   str = None,
) -> list[dict]:
    """
    Retrieve top-k most similar sessions from the index.
    Excludes the target session itself (leave-one-out safe).
    Optionally filters by module.
    """
    # Embed query with BGE prefix
    q_emb = embed_texts([query_narrative], is_query=True)

    # Search — retrieve more than k to allow filtering
    search_k = min(len(records), k + 5)
    distances, indices = index.search(q_emb, search_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(records):
            continue
        rec = records[idx]
        if exclude_id and rec["id"] == exclude_id:
            continue
        if module_filter and rec["module"] != module_filter:
            continue
        results.append({**rec, "similarity": float(dist)})
        if len(results) >= k:
            break

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SCORE ONE SESSION
# ─────────────────────────────────────────────────────────────────────────────

def score_session(
    features:  dict,
    index:     faiss.Index,
    records:   list[dict],
    taxonomy:  dict,
    lang_type: str,
    k:         int = 3,
    verbose:   bool = True,
) -> dict:
    """Score one session using RAG-retrieved examples."""
    video_id  = features["video_id"]
    module    = features.get("module", "M1")
    narrative = features["narrative"]

    items     = get_algorithm_items(taxonomy, module)
    sa_items  = [i["code"] for i in items if i["algorithm_domain"] == "SA"]
    crr_items = [i["code"] for i in items if i["algorithm_domain"] == "CRR"]
    all_codes = [i["code"] for i in items]

    # Retrieve similar sessions
    retrieved = retrieve(
        narrative, index, records, k=k,
        exclude_id    = video_id,
        module_filter = module,
    )

    if verbose:
        print(f"\n  Session:   {video_id}  [{module}]")
        print(f"  Retrieved: {[(r['id'], f\"{r['similarity']:.3f}\") for r in retrieved]}")

    # Build examples for the prompt
    examples = [
        {"narrative": r["narrative"], "scores": r["scores"]}
        for r in retrieved
    ]

    # Build prompt and call model
    prompt   = build_scoring_prompt(narrative, items, examples)
    response = call_ollama(prompt, temperature=0.05)

    # Parse and report
    scores = parse_scores(response, all_codes)
    report = render_report(
        video_id, module, scores,
        sa_items, crr_items, lang_type, "RAG", taxonomy
    )
    report["retrieved_examples"] = [
        {"id": r["id"], "similarity": r["similarity"], "classification": r["classification"]}
        for r in retrieved
    ]

    if verbose:
        print_report(report)

    return report


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION — LEAVE-ONE-OUT
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    features_dir: str,
    index:        faiss.Index,
    records:      list[dict],
    taxonomy:     dict,
    lang_type:    str,
    k:            int = 3,
    output_dir:   str = "outputs/rag",
) -> None:
    """Leave-one-out evaluation — identical structure to step4a for easy comparison."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    gt_index = {r["id"]: r["scores"] for r in records}
    gt_meta  = {r["id"]: r for r in records}

    feat_files = sorted(Path(features_dir).glob("*_features.json"))
    results    = []

    print(f"\nRAG leave-one-out evaluation: {len(feat_files)} sessions (k={k})...\n")

    for fpath in feat_files:
        with open(fpath, encoding="utf-8") as f:
            features = json.load(f)

        vid = features["video_id"]
        if vid not in gt_index:
            print(f"  ⚠  {vid}: no ground truth — skipping")
            continue

        report = score_session(features, index, records, taxonomy, lang_type, k=k, verbose=True)

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

        mae       = sum(item_errors.values()) / total_items if total_items > 0 else 0
        exact_pct = exact_match / total_items if total_items > 0 else 0
        class_match = report["classification"] == gt_class

        result = {
            "video_id":       vid,
            "module":         report["module"],
            "pred_total":     report["total"],
            "gt_total":       gt_total,
            "total_error":    abs(report["total"] - gt_total),
            "pred_class":     report["classification"],
            "gt_class":       gt_class,
            "class_correct":  class_match,
            "item_mae":       round(mae, 3),
            "item_exact_pct": round(exact_pct, 3),
            "item_errors":    item_errors,
            "retrieved":      report.get("retrieved_examples", []),
        }
        results.append(result)

        print(f"  {vid}:  total {report['total']} (gt {gt_total})  "
              f"class {'✓' if class_match else '✗'} {report['classification']} vs {gt_class}  "
              f"MAE {mae:.2f}  exact {exact_pct:.0%}")

    if results:
        avg_mae     = sum(r["item_mae"] for r in results) / len(results)
        avg_exact   = sum(r["item_exact_pct"] for r in results) / len(results)
        class_acc   = sum(1 for r in results if r["class_correct"]) / len(results)
        avg_tot_err = sum(r["total_error"] for r in results) / len(results)

        print(f"\n{'═'*50}")
        print(f"  RAG EVALUATION SUMMARY  (k={k})")
        print(f"{'═'*50}")
        print(f"  Sessions evaluated:     {len(results)}")
        print(f"  Classification acc:     {class_acc:.0%}")
        print(f"  Total score MAE:        {avg_tot_err:.2f}")
        print(f"  Per-item MAE:           {avg_mae:.3f}")
        print(f"  Per-item exact match:   {avg_exact:.0%}")
        print(f"{'═'*50}\n")

        summary = {
            "method":                  "RAG",
            "k":                       k,
            "embed_model":             EMBED_MODEL_NAME,
            "n_evaluated":             len(results),
            "classification_accuracy": round(class_acc, 3),
            "total_score_mae":         round(avg_tot_err, 3),
            "per_item_mae":            round(avg_mae, 3),
            "per_item_exact_pct":      round(avg_exact, 3),
            "per_session":             results,
        }
        out_path = Path(output_dir) / "rag_evaluation.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"  Results saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build_index",  action="store_true")
    parser.add_argument("--evaluate",     action="store_true")
    parser.add_argument("--features",     default=None)
    parser.add_argument("--features_dir", default="data/features")
    parser.add_argument("--pairs",        default="data/training_pairs.json")
    parser.add_argument("--index",        default="data/rag_index")
    parser.add_argument("--taxonomy",     default="data/ados_taxonomy_combined.json")
    parser.add_argument("--lang_type",    default="few_or_no_words")
    parser.add_argument("--k",            type=int, default=3)
    parser.add_argument("--output_dir",   default="outputs/rag")
    args = parser.parse_args()

    if args.build_index:
        build_index(args.pairs, args.index)

    elif args.evaluate:
        index, records = load_index(args.index)
        taxonomy       = load_taxonomy(args.taxonomy)
        evaluate(args.features_dir, index, records, taxonomy,
                 args.lang_type, args.k, args.output_dir)

    elif args.features:
        index, records = load_index(args.index)
        taxonomy       = load_taxonomy(args.taxonomy)
        with open(args.features, encoding="utf-8") as f:
            features = json.load(f)
        report = score_session(features, index, records, taxonomy, args.lang_type, k=args.k)
        out = Path(args.output_dir) / f"{features['video_id']}_rag_report.json"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved → {out}")

    else:
        print("Use --build_index, --evaluate, or --features <file>")