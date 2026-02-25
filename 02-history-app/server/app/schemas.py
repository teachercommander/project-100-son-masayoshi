from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=20)
    # Optional: simple filters (not enforced in FAISS v1; reserved for future)
    reliability_tier: Optional[List[str]] = None
    source_type: Optional[List[str]] = None
    tags: Optional[List[str]] = None


class Citation(BaseModel):
    id: str
    source_id: str
    title: str = ""
    org: str = ""
    year: Optional[int] = None
    url: str = ""
    loc: Dict[str, Any] = {}
    snippet: str = ""


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation]
    debug: Dict[str, Any] = {}
