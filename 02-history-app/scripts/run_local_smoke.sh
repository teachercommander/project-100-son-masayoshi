#!/usr/bin/env bash
set -euo pipefail

# Quick local smoke test for the FastAPI RAG server.
# Usage:
#   bash scripts/run_local_smoke.sh
# Optional env:
#   PORT=8000 INDEX_DIR=index EMBED_MODEL=intfloat/multilingual-e5-base

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PORT="${PORT:-8000}"
INDEX_DIR="${INDEX_DIR:-index}"
EMBED_MODEL="${EMBED_MODEL:-intfloat/multilingual-e5-base}"

echo "[1/3] starting API server on port ${PORT}..."
INDEX_DIR="$INDEX_DIR" EMBED_MODEL="$EMBED_MODEL" \
python -m uvicorn server.app.main:app --host 127.0.0.1 --port "$PORT" >/tmp/history-app-api.log 2>&1 &
API_PID=$!

cleanup() {
  if kill -0 "$API_PID" 2>/dev/null; then
    kill "$API_PID" || true
    wait "$API_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for _ in {1..30}; do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "[ERROR] API did not become ready. Check /tmp/history-app-api.log"
  exit 1
fi

echo "[2/3] health check"
curl -fsS "http://127.0.0.1:${PORT}/health"
echo

echo "[3/3] query smoke test"
curl -fsS -X POST "http://127.0.0.1:${PORT}/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"이승만과 김구의 비교 관점 요약","k":3}'
echo

echo "done"
