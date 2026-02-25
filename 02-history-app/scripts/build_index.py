#!/usr/bin/env python3
# coding: utf-8
"""
build_index.py
- Reads JSONL blocks from data/processed/*.jsonl (output of ingest.py)
- Embeds each block text using SentenceTransformers
- Builds a FAISS index (cosine similarity via inner product on normalized vectors)
- Saves:
  * index/faiss.index
  * index/docs.jsonl  (metadata per vector, for citations/UI)

Install:
  python -m pip install sentence-transformers faiss-cpu numpy
Usage:
  python scripts/build_index.py --processed data/processed --outdir index --model intfloat/multilingual-e5-base
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("faiss not installed. Run: python -m pip install faiss-cpu") from e

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:
    raise RuntimeError("sentence-transformers not installed. Run: python -m pip install sentence-transformers") from e


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


def make_doc_record(obj: Dict) -> Dict:
    # obj is ProcessedBlock dict from ingest.py
    meta = obj.get("meta", {}) or {}
    loc = obj.get("loc", {}) or {}
    return {
        "source_id": obj.get("source_id", ""),
        "chunk_source": obj.get("chunk_source", ""),
        "loc": loc,
        "meta": meta,
        "text": obj.get("text", ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed", help="directory containing processed JSONL files")
    ap.add_argument("--outdir", default="index", help="output directory for index files")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base", help="SentenceTransformers model name")
    ap.add_argument("--pattern", default="*.jsonl", help="glob pattern for processed files")
    ap.add_argument("--max_docs", type=int, default=0, help="limit docs for quick test (0=all)")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    ensure_dir(args.outdir)

    files = sorted(glob.glob(os.path.join(args.processed, args.pattern)))
    if not files:
        raise FileNotFoundError(f"No processed JSONL files found in: {args.processed}")

    print(f"[1/4] Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    # Collect docs
    docs: List[Dict] = []
    texts: List[str] = []

    print(f"[2/4] Reading processed files: {len(files)}")
    for fp in files:
        for obj in iter_jsonl(fp):
            rec = make_doc_record(obj)
            t = (rec.get("text") or "").strip()
            if not t:
                continue

            docs.append(rec)
            texts.append(t)

            if args.max_docs and len(docs) >= args.max_docs:
                break
        if args.max_docs and len(docs) >= args.max_docs:
            break

    if not docs:
        raise RuntimeError("No documents found after reading processed JSONL.")

    print(f"Documents: {len(docs):,}")

    # Embed
    print("[3/4] Embedding texts...")
    # E5 models work best with "query: " / "passage: " prefixes, but we keep it simple for MVP.
    # If you want: texts = [f\"passage: {t}\" for t in texts]
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # makes cosine similarity = inner product
    ).astype("float32")

    dim = embeddings.shape[1]
    print(f"Embedding dim: {dim}")

    # Build FAISS index (inner product)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index size: {index.ntotal:,}")

    # Save
    index_path = os.path.join(args.outdir, "faiss.index")
    docs_path = os.path.join(args.outdir, "docs.jsonl")

    print("[4/4] Saving index and docs...")
    faiss.write_index(index, index_path)

    with open(docs_path, "w", encoding="utf-8") as f:
        for rec in docs:
            # store without full text if you want smaller files.
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\nDone.")
    print(f"- {index_path}")
    print(f"- {docs_path}")


if __name__ == "__main__":
    main()
