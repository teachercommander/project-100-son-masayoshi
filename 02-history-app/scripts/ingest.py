#!/usr/bin/env python3
# coding: utf-8
"""
ingest.py
- Reads sources.csv (or sources.xlsx sheet 'sources')
- Extracts text from:
  * local PDF (page-based)
  * local HTML (section-based)
  * local TXT/MD
  * local JSON/JSONL (record-based)
  * optionally fetches URL HTML (--fetch-web)
- Writes JSONL to data/processed/{source_id}.jsonl
Each JSONL line is a block with location metadata for citations.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

try:
    import pdfplumber  # type: ignore
except Exception:
    pdfplumber = None

try:
    import requests  # type: ignore
except Exception:
    requests = None

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None


@dataclass
class ProcessedBlock:
    chunk_source: str  # "page" | "web_section" | "text" | "json_record"
    source_id: str
    loc: Dict
    text: str
    meta: Dict


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def norm_space(s: str) -> str:
    s = s.replace("\u00a0", " ")

    # Fix hyphenation at line breaks: "민주-\n주의" -> "민주주의"
    s = re.sub(r"-\s*\n\s*([A-Za-z가-힣])", r"\1", s)

    # Remove line breaks that split words (Korean/English) like "만드\n가" -> "만드가"
    s = re.sub(r"([가-힣A-Za-z0-9])\s*\n\s*([가-힣A-Za-z0-9])", r"\1 \2", s)

    # Keep paragraph breaks, but normalize others
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]+", " ", s)

    return s.strip()


def is_na_like(v: Any) -> bool:
    if v is None:
        return True
    s = str(v).strip()
    return s == "" or s.lower() in {"nan", "none", "null"}


def safe_int(v: Any) -> Optional[int]:
    try:
        if pd is not None and pd.isna(v):
            return None
        if is_na_like(v):
            return None
        return int(float(str(v).strip()))
    except Exception:
        return None


def split_long_text(text: str, max_chars: int = 8000) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    parts: List[str] = []
    buf = ""
    for p in paras:
        if not buf:
            buf = p
        elif len(buf) + 2 + len(p) <= max_chars:
            buf += "\n\n" + p
        else:
            parts.append(buf)
            buf = p
    if buf:
        parts.append(buf)

    final: List[str] = []
    for part in parts:
        if len(part) <= max_chars:
            final.append(part)
        else:
            for i in range(0, len(part), max_chars):
                final.append(part[i : i + max_chars].strip())

    return [x for x in final if x]


def iter_json_records(path: str) -> Iterable[Dict[str, Any]]:
    if path.lower().endswith(".jsonl"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    yield item
        elif isinstance(obj, dict):
            # Try common list containers first.
            for k in ["results", "items", "documents", "data", "rows"]:
                v = obj.get(k)
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            yield item
                    return
            yield obj


def pick_json_text(record: Dict[str, Any], text_fields: List[str]) -> str:
    if not text_fields:
        text_fields = [
            "title",
            "subtitle",
            "summary",
            "abstract",
            "description",
            "content",
            "body",
            "text",
        ]

    pieces: List[str] = []
    for field in text_fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            pieces.append(value.strip())
        elif isinstance(value, list):
            joined = "\n".join(str(x).strip() for x in value if str(x).strip())
            if joined:
                pieces.append(joined)

    if not pieces:
        # fallback: stringify short scalar fields
        for k, v in record.items():
            if isinstance(v, (str, int, float)):
                s = str(v).strip()
                if s and len(s) <= 1200:
                    pieces.append(f"{k}: {s}")

    return norm_space("\n\n".join(pieces))


def normalize_row(row: Dict[str, Any], required: List[str], optional: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for col in required + optional:
        out[col] = row.get(col, "")
    return out


def load_sources_table(path: str) -> List[Dict[str, Any]]:
    required = ["source_id", "title", "org", "author", "year", "url", "file_path", "license", "tags"]
    optional = [
        "source_type",
        "genre",
        "reliability_tier",
        "stance_label",
        "anchor_weight",
        "use_as_anchor",
        "content_fields",
        "notes",
    ]

    rows: List[Dict[str, Any]] = []
    if path.lower().endswith(".csv"):
        with open(path, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                nr = normalize_row(row, required, optional)
                sid = str(nr.get("source_id", "")).strip()
                if sid:
                    nr["source_id"] = sid
                    rows.append(nr)
    elif path.lower().endswith((".xlsx", ".xlsm", ".xls")):
        if pd is None:
            raise RuntimeError("pandas is required to read Excel sources. Install: python -m pip install pandas")
        df = pd.read_excel(path, sheet_name="sources")
        for col in required + optional:
            if col not in df.columns:
                df[col] = ""
        for _, row in df.iterrows():
            nr = normalize_row(row.to_dict(), required, optional)
            sid = str(nr.get("source_id", "")).strip()
            if sid and sid.lower() != "nan":
                nr["source_id"] = sid
                rows.append(nr)
    else:
        raise ValueError("sources must be .csv or .xlsx")

    return rows


def meta_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    tags = [t.strip() for t in str(row.get("tags", "")).split(";") if t.strip()]
    return {
        "title": str(row.get("title", "")).strip(),
        "org": str(row.get("org", "")).strip(),
        "author": str(row.get("author", "")).strip(),
        "year": safe_int(row.get("year", "")),
        "url": str(row.get("url", "")).strip(),
        "license": str(row.get("license", "")).strip(),
        "tags": tags,
        "source_type": str(row.get("source_type", "")).strip(),
        "genre": str(row.get("genre", "")).strip(),
        "reliability_tier": str(row.get("reliability_tier", "")).strip(),
        "stance_label": str(row.get("stance_label", "")).strip(),
        "anchor_weight": row.get("anchor_weight", ""),
        "use_as_anchor": str(row.get("use_as_anchor", "")).strip(),
        "content_fields": str(row.get("content_fields", "")).strip(),
        "notes": str(row.get("notes", "")).strip(),
    }


def extract_pdf_pages(pdf_path: str) -> List[Tuple[int, str]]:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber not installed. Run: python -m pip install pdfplumber")
    pages: List[Tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text(layout=True) or ""
            txt = txt.replace("\r", "\n")
            txt = norm_space(txt)
            if txt:
                pages.append((i, txt))
    return pages


def fetch_url(url: str, timeout: int = 20, user_agent: str = "modern-khistory-rag/0.1") -> str:
    if requests is None:
        raise RuntimeError("requests not installed. Run: python -m pip install requests")
    headers = {"User-Agent": user_agent}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or r.encoding
    return r.text


def html_to_sections(html: str) -> List[Tuple[str, str]]:
    if BeautifulSoup is None:
        raise RuntimeError("bs4 not installed. Run: python -m pip install beautifulsoup4 lxml")

    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    body = soup.body or soup
    headings = body.find_all(["h1", "h2", "h3"])

    if not headings:
        text = norm_space(body.get_text("\n"))
        return [("body", text)] if text else []

    sections: List[Tuple[str, str]] = []
    for h in headings:
        sec_title = norm_space(h.get_text(" "))
        texts: List[str] = []
        for sib in h.next_siblings:
            if getattr(sib, "name", None) in ["h1", "h2", "h3"]:
                break
            if getattr(sib, "get_text", None):
                t = norm_space(sib.get_text("\n"))
                if t:
                    texts.append(t)
        sec_text = norm_space("\n\n".join(texts))
        if sec_text:
            sections.append((sec_title or "section", sec_text))

    if not sections:
        text = norm_space(body.get_text("\n"))
        return [("body", text)] if text else []

    return sections


def read_local_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return norm_space(f.read())


def read_local_html(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def ingest_one(row: Dict[str, Any], outdir: str, fetch_web: bool, sleep_sec: float) -> int:
    source_id = str(row["source_id"]).strip()
    meta = meta_from_row(row)

    url = str(row.get("url", "")).strip()
    file_path = str(row.get("file_path", "")).strip()

    out_path = os.path.join(outdir, f"{source_id}.jsonl")
    blocks: List[ProcessedBlock] = []

    is_pdf = file_path.lower().endswith(".pdf") if file_path else False
    is_html = file_path.lower().endswith((".html", ".htm")) if file_path else False
    is_text = file_path.lower().endswith((".txt", ".md")) if file_path else False
    is_json = file_path.lower().endswith((".json", ".jsonl")) if file_path else False

    if file_path and is_pdf:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[{source_id}] PDF not found: {file_path}")
        pages = extract_pdf_pages(file_path)
        for page_no, page_text in pages:
            for part_idx, part in enumerate(split_long_text(page_text), start=1):
                loc = {"page": page_no, "part": part_idx}
                blocks.append(ProcessedBlock("page", source_id, loc, part, meta))

    elif file_path and is_html:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[{source_id}] HTML not found: {file_path}")
        html = read_local_html(file_path)
        sections = html_to_sections(html)
        for sec_title, sec_text in sections:
            for part_idx, part in enumerate(split_long_text(sec_text), start=1):
                loc = {"section": sec_title, "part": part_idx}
                blocks.append(ProcessedBlock("web_section", source_id, loc, part, meta))

    elif file_path and is_text:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[{source_id}] Text file not found: {file_path}")
        text = read_local_text(file_path)
        for part_idx, part in enumerate(split_long_text(text), start=1):
            loc = {"part": part_idx}
            blocks.append(ProcessedBlock("text", source_id, loc, part, meta))

    elif file_path and is_json:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[{source_id}] JSON not found: {file_path}")

        content_fields = [x.strip() for x in str(row.get("content_fields", "")).split(";") if x.strip()]

        for rec_idx, rec in enumerate(iter_json_records(file_path), start=1):
            text = pick_json_text(rec, content_fields)
            if not text:
                continue

            for part_idx, part in enumerate(split_long_text(text), start=1):
                loc = {"record": rec_idx, "part": part_idx}
                blocks.append(ProcessedBlock("json_record", source_id, loc, part, meta))

    elif url and fetch_web:
        html = fetch_url(url)
        sections = html_to_sections(html)
        for sec_title, sec_text in sections:
            for part_idx, part in enumerate(split_long_text(sec_text), start=1):
                loc = {"url": url, "section": sec_title, "part": part_idx}
                blocks.append(ProcessedBlock("web_section", source_id, loc, part, meta))

        if sleep_sec > 0:
            time.sleep(sleep_sec)

    else:
        return 0

    ensure_dir(outdir)
    with open(out_path, "w", encoding="utf-8") as f:
        for b in blocks:
            if not b.text.strip():
                continue
            f.write(json.dumps(asdict(b), ensure_ascii=False) + "\n")

    return len(blocks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", required=True, help="path to sources.csv or sources.xlsx")
    parser.add_argument("--outdir", default="data/processed", help="output directory for JSONL")
    parser.add_argument("--fetch-web", action="store_true", help="fetch URL HTML if url is provided")
    parser.add_argument("--sleep", type=float, default=0.3, help="sleep between web requests (sec)")
    parser.add_argument("--limit", type=int, default=0, help="process only first N rows (0=all)")
    args = parser.parse_args()

    rows = load_sources_table(args.sources)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    ensure_dir(args.outdir)

    processed, skipped, failed = 0, 0, 0
    for row in tqdm(rows, total=len(rows), desc="Ingesting"):
        sid = str(row["source_id"]).strip()
        try:
            n = ingest_one(row, args.outdir, args.fetch_web, args.sleep)
            if n == 0:
                skipped += 1
            else:
                processed += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {sid}: {e}")

    print("\n--- ingest summary ---")
    print(f"processed: {processed}")
    print(f"skipped  : {skipped}")
    print(f"failed   : {failed}")
    print(f"outdir   : {args.outdir}")


if __name__ == "__main__":
    main()
