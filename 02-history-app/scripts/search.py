#!/usr/bin/env python3
# coding: utf-8
"""
search.py
- Loads FAISS index + docs.jsonl
- Embeds query
- Returns Top-k matches with citation-friendly metadata

Install:
  python -m pip install sentence-transformers faiss-cpu numpy
Usage:
  python scripts/search.py --indexdir index --query "이승만 건국" --k 5
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("faiss not installed. Run: python -m pip install faiss-cpu") from e

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:
    raise RuntimeError("sentence-transformers not installed. Run: python -m pip install sentence-transformers") from e


def load_docs(path: str) -> List[Dict]:
    docs: List[Dict] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except Exception:
                continue
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indexdir", default="index", help="directory containing faiss.index and docs.jsonl")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base", help="SentenceTransformers model name")
    ap.add_argument("--query", required=True, help="search query")
    ap.add_argument("--k", type=int, default=5, help="top-k results")
    args = ap.parse_args()

    index_path = os.path.join(args.indexdir, "faiss.index")
    docs_path = os.path.join(args.indexdir, "docs.jsonl")

    if not os.path.exists(index_path):
        raise FileNotFoundError(index_path)
    if not os.path.exists(docs_path):
        raise FileNotFoundError(docs_path)

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    print("Loading FAISS index...")
    index = faiss.read_index(index_path)

    print("Loading docs metadata...")
    docs = load_docs(docs_path)

    q = args.query.strip()
    if not q:
        raise ValueError("Empty query")

    # For E5 you can optionally use: q = f"query: {q}"
    q_emb = model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype("float32")

    scores, idxs = index.search(q_emb, args.k)
    scores = scores[0].tolist()
    idxs = idxs[0].tolist()

    print("\nTop results:")
    for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
        if i < 0 or i >= len(docs):
            continue
        d = docs[i]
        source_id = d.get("source_id", "")
        loc = d.get("loc", {})
        meta = d.get("meta", {}) or {}
        title = meta.get("title", "")
        org = meta.get("org", "")
        year = meta.get("year", "")

        cite = source_id
        if "page" in loc and loc["page"]:
            cite += f"#p{loc['page']}"
        elif "url" in loc and loc["url"]:
            cite += f"#url"

        snippet = (d.get("text") or "").strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."

        print(f"\n[{rank}] score={s:.4f}  cite={cite}")
        print(f"    {title} / {org} / {year}")
        print(f"    {snippet}")


if __name__ == "__main__":
    main()
