from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("faiss not installed. Run: python -m pip install faiss-cpu") from e

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:
    raise RuntimeError("sentence-transformers not installed. Run: python -m pip install sentence-transformers") from e


def load_docs_jsonl(path: str) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
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


def make_cite_id(doc: Dict[str, Any]) -> str:
    sid = doc.get("source_id", "")
    loc = doc.get("loc", {}) or {}
    if "page" in loc and loc.get("page"):
        return f"{sid}#p{loc['page']}"
    if "url" in loc and loc.get("url"):
        return f"{sid}#url"
    if "section" in loc and loc.get("section"):
        return f"{sid}#sec"
    return sid


import re

def make_snippet(text: str, n: int = 240) -> str:
    t = (text or "").strip()

    # 1) normalize whitespace
    t = t.replace("\r", "\n")
    t = t.replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()

    # 2) fix "single-syllable split" artifacts only:
    #    e.g., "만드 가" -> "만드가", "힘입 는" -> "힘입는"
    #    This keeps normal spacing, only joins when the right side is 1 Hangul syllable.
    t = re.sub(r"([가-힣]{2,})\s+([가-힣])\b", r"\1\2", t)

    # 3) optional: collapse multiple spaces again
    t = re.sub(r"\s+", " ", t).strip()

    if len(t) > n:
        t = t[:n] + "..."
    return t



def clean_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    return s


class FaissRAG:
    """
    Minimal RAG:
      - embed question
      - FAISS top-k search
      - return citation-rich response (LLM-free by default)
    """

    def __init__(
        self,
        index_dir: str = "index",
        model_name: str = "intfloat/multilingual-e5-base",
    ) -> None:
        index_path = os.path.join(index_dir, "faiss.index")
        docs_path = os.path.join(index_dir, "docs.jsonl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(index_path)
        if not os.path.exists(docs_path):
            raise FileNotFoundError(docs_path)

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        self.docs = load_docs_jsonl(docs_path)

        if self.index.ntotal != len(self.docs):
            # Not fatal, but suspicious
            print(f"[WARN] index.ntotal({self.index.ntotal}) != docs({len(self.docs)})")

    def search(self, question: str, k: int = 5) -> List[Tuple[int, float, Dict[str, Any]]]:
        q = question.strip()
        if not q:
            return []

        # For E5 you can optionally prefix: q = f"query: {q}"
        q_emb = self.model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        scores, idxs = self.index.search(q_emb, k)

        out: List[Tuple[int, float, Dict[str, Any]]] = []
        for i, s in zip(idxs[0].tolist(), scores[0].tolist()):
            if i < 0 or i >= len(self.docs):
                continue
            out.append((i, float(s), self.docs[i]))
        return out

    def answer_without_llm(self, question: str, hits: List[Tuple[int, float, Dict[str, Any]]]) -> str:
        """
        MVP: LLM 없이도 usable하게:
        - 상위 근거들의 스니펫을 '근거 요약' 형식으로 보여줌
        """
        if not hits:
            return "관련 근거를 찾지 못했습니다. 질문을 더 구체화해 주시면 다시 찾아볼게요."

        lines = []
        lines.append("검색된 근거(상위 결과)를 바탕으로 정리하면:")
        for rank, (_, score, d) in enumerate(hits, start=1):
            meta = d.get("meta", {}) or {}
            title = meta.get("title", "")
            org = meta.get("org", "")
            year = meta.get("year", "")
            cite = make_cite_id(d)
            snippet = make_snippet(d.get("text", ""))
            lines.append(f"\n{rank}. {snippet}\n   - 출처: {title} / {org} / {year} ({cite})  [score={score:.3f}]")

        lines.append("\n※ 위 내용은 ‘근거 스니펫’ 기반이며, 추후 LLM 요약/비교(관점 병렬)를 붙이면 더 읽기 좋은 답변으로 확장됩니다.")
        return "\n".join(lines)

    def build_citations(self, hits: List[Tuple[int, float, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        cites: List[Dict[str, Any]] = []
        for (_, score, d) in hits:
            meta = d.get("meta", {}) or {}
            loc = d.get("loc", {}) or {}

            cite_id = make_cite_id(d)
            cites.append(
                {
                    "id": cite_id,
                    "source_id": d.get("source_id", ""),
                    "title": meta.get("title", "") or "",
                    "org": meta.get("org", "") or "",
                    "year": meta.get("year", None),
                    "url": clean_str(meta.get("url", "")) or clean_str(loc.get("url", "")),
                    "loc": loc,
                    "snippet": make_snippet(d.get("text", "")),
                    "score": score,
                }
            )
        return cites

