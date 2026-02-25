from __future__ import annotations

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import QueryRequest, QueryResponse, Citation
from .rag import FaissRAG

INDEX_DIR = os.getenv("INDEX_DIR", "index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")

app = FastAPI(title="Modern Korean History RAG API", version="0.1.0")

# (개발용) CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = None


@app.on_event("startup")
def _startup() -> None:
    global rag
    rag = FaissRAG(index_dir=INDEX_DIR, model_name=EMBED_MODEL)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "index_dir": INDEX_DIR, "embed_model": EMBED_MODEL}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    global rag
    if rag is None:
        raise HTTPException(status_code=500, detail="RAG not initialized")

    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty question")

    hits = rag.search(q, k=req.k)
    answer = rag.answer_without_llm(q, hits)

    citations_raw = rag.build_citations(hits)
    citations = [
        Citation(
            id=c["id"],
            source_id=c["source_id"],
            title=c["title"],
            org=c["org"],
            year=c["year"],
            url=c["url"],
            loc=c["loc"],
            snippet=c["snippet"],
        )
        for c in citations_raw
    ]

    debug = {
        "k": req.k,
        "hits": [{"id": c["id"], "score": c.get("score")} for c in citations_raw],
    }

    return QueryResponse(question=q, answer=answer, citations=citations, debug=debug)
