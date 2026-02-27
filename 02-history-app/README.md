# 02-history-app: 한국 현대사 RAG 검색 앱 확장 가이드

이 앱은 `sources` 매니페스트에 데이터를 등록하고, `ingest.py`로 텍스트를 표준화한 뒤,
`build_index.py`로 벡터 인덱스를 만들어 검색합니다.

## 1) 어떤 자료를 넣을 수 있나

현재 파이프라인은 아래 입력을 지원합니다.

- PDF (`.pdf`): 페이지 단위
- HTML (`.html/.htm`): 섹션 단위
- 텍스트 (`.txt/.md`): 문서 단위
- JSON/JSONL (`.json/.jsonl`): **레코드 단위**
- URL(웹 크롤링): `--fetch-web` 옵션 사용 시 HTML 섹션 단위

JSON/JSONL 지원으로, 한국사 DB/국가기록원/논문 API 결과를 파일로 저장해 바로 색인할 수 있습니다.

## 2) sources 매니페스트 확장

`data/manifests/sources_utf8.csv`를 기준으로 자료를 등록하세요.

권장 컬럼:

- `source_id`: 고유 ID (예: `NAK-1948-0001`)
- `title`, `org`, `author`, `year`
- `url`: 원문 링크
- `file_path`: 로컬 파일 경로
- `source_type`: `primary|secondary|reference|media`
- `genre`: `government_record|archive_db|academic_paper|...`
- `reliability_tier`: `A|B|C`
- `stance_label`: `mixed|conservative|progressive|...`
- `anchor_weight`, `use_as_anchor`
- `content_fields`: JSON에서 본문으로 쓸 필드 목록 (`;` 구분, 예: `title;abstract;content`)
- `notes`: DOI/서지/수집 메모

## 3) 기관별 수집 전략 (실무형)

### 국회도서관 / 학술논문

- 원칙: 라이선스 확인 후 원문 PDF 또는 메타데이터+초록 사용
- 우선순위:
  1. 제목/초록/키워드(항상)
  2. 본문(권한이 허용될 때)
- JSONL 추천 스키마:

```json
{"id":"paper-1","title":"...","abstract":"...","content":"...","year":2018,"doi":"..."}
```

`sources`의 `content_fields`를 `title;abstract;content`로 지정하면 ingest 단계에서 자동 결합됩니다.

### 한국사 데이터베이스(db.history.go.kr)

- 문서/항목 단위로 저장 (사료 1건 = JSONL 1레코드 권장)
- 권장 필드: `title`, `period`, `source_type`, `content`, `url`
- 긴 사료는 ingest에서 자동 분할되므로 원문 텍스트 보존 중심으로 저장하세요.

### 국가기록원(archives.go.kr)

- PDF가 있으면 PDF 그대로 등록 (페이지 인용 유리)
- HTML/목록형 데이터는 JSONL로 정규화 저장
- 기록물 고유 식별자(관리번호)를 JSON 필드에 남기고 `notes`에도 함께 기입하면 추적이 쉬움


### 추가 대상: 역사 아카이브 웹사이트 / 군사편찬연구소

- `http://xn--zb0bnwy6egumoslu1g.com/`
  - 웹 문서형이면 `--fetch-web`로 수집 가능
  - 중요 문서는 HTML/PDF로 별도 저장 후 `file_path` 기반 수집 권장
- `https://www.imhc.mil.kr/` (국방부 군사편찬연구소)
  - 공개 군사사 자료(연구총서/사료집/연표 등)를 문서 단위로 등록
  - PDF 원문이 있으면 PDF 우선(페이지 인용 정확도 향상)

## 4) 폴더 구성 권장

```text
02-history-app/
  data/
    raw/
      nak/
      khdb/
      nlk/
      papers/
    processed/
  scripts/
```

기관별 하위 폴더를 두면 소스별 재수집/재인덱싱이 쉬워집니다.

## 5) 실행 순서

```bash
# 1) ingest
python scripts/ingest.py \
  --sources data/manifests/sources_utf8.csv \
  --outdir data/processed \
  --fetch-web

# 2) index build
python scripts/build_index.py \
  --processed data/processed \
  --outdir index \
  --model intfloat/multilingual-e5-base
```

## 6) 품질 개선 팁

- 질문 라우팅: `사실 질의`는 A티어 가중치, `관점 비교`는 B티어도 노출
- 중복 제거: 동일 기사/논문의 재배포본은 source_id 통합
- 최신성 관리: `year` + `notes`에 수집일 기록
- 출처 가시성: 답변에서 `title/org/year/url`을 항상 함께 표시

## 7) 운영 팁

- 데이터 추가 후에는 `data/processed/*.jsonl`과 `index/*`를 재생성하세요.
- JSONL로 수집 파이프라인을 먼저 고정하면, 이후 API 커넥터(국가기관/논문 DB)를 붙이기 쉽습니다.

## 8) 내가 직접 앱 테스트하기 (로컬)

아래 순서대로 하면, 본인 PC에서 바로 API 테스트를 할 수 있습니다.

```bash
cd 02-history-app

# (권장) 가상환경
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# API + 검색 인덱스 구동에 필요한 최소 패키지
python -m pip install -U pip
python -m pip install fastapi uvicorn pydantic sentence-transformers faiss-cpu numpy
```

기본 인덱스(`index/faiss.index`, `index/docs.jsonl`)가 이미 있으면 바로 서버를 띄울 수 있습니다.

```bash
# 방법 A: 수동 실행
INDEX_DIR=index EMBED_MODEL=intfloat/multilingual-e5-base \
python -m uvicorn server.app.main:app --host 0.0.0.0 --port 8000
```

다른 터미널에서:

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"해방정국 핵심 쟁점 요약", "k": 5}'
```

또는 원클릭 스모크 테스트:

```bash
# 방법 B: 자동 실행 스크립트(서버 실행 + health + query + 종료)
bash scripts/run_local_smoke.sh
```

### 인덱스를 새로 만들고 테스트하고 싶다면

```bash
# (선택) 수집 단계: 필요한 의존성 추가
python -m pip install tqdm requests beautifulsoup4 lxml pdfplumber pandas

python scripts/ingest.py \
  --sources data/manifests/sources_utf8.csv \
  --outdir data/processed \
  --fetch-web

python scripts/build_index.py \
  --processed data/processed \
  --outdir index \
  --model intfloat/multilingual-e5-base
```

