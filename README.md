# Fraud Investigation Assistant

A RAG-powered QA system that lets fraud analysts ask natural-language questions over a corpus of fraud investigation case reports. The system retrieves the most relevant case chunks from a local vector database and generates concise, source-cited answers using a free LLM via Groq.

Built with: **LangChain · ChromaDB · Groq (Llama 3.3 70B) · FastAPI · Streamlit · FlashRank**

---

## Evaluation results

Benchmarked using a custom LLM-as-judge evaluator (8 test questions, `llama-3.1-8b-instant` as judge):

| Metric | Baseline (similarity) | Reranked (FlashRank) | Improvement |
|---|---|---|---|
| Faithfulness | 62.5% | 75.0% | ↑ 12.5pp |
| Answer Relevancy | 56.2% | 81.2% | ↑ 25.0pp |
| Context Precision | 68.8% | 81.2% | ↑ 12.5pp |

FlashRank cross-encoder reranking fetches 3× candidate chunks and reranks them before passing to the LLM, significantly improving answer relevancy and context quality.

---

## Architecture

```
                        INGESTION PIPELINE
                        ──────────────────
  PDF / JSON / TXT reports
          │
          ▼
  LangChain document loaders
          │
          ▼
  RecursiveCharacterTextSplitter   (800 chars, 150 overlap)
          │
          ▼
  HuggingFace Embeddings           (all-MiniLM-L6-v2, local, free)
          │
          ▼
  ChromaDB vector store            (persisted to chroma_db/)


                        QUERY PIPELINE
                        ─────────────
  Analyst question
          │
          ▼
  FastAPI POST /query
          │
          ▼
  RunnableParallel (LCEL chain)
          ├── ChromaDB retriever   (top-8 chunks by similarity)
          │       │
          │       ▼
          │   FlashRank reranker   (reranks top-24 → best 8)
          └── question passthrough
          │
          ├── PromptTemplate | ChatGroq (Llama 3.3 70B) | StrOutputParser → answer
          └── context passthrough                                          → sources
          │
          ▼
  Streamlit UI                     (chat + sidebar with source chunks)


                        EVALUATION PIPELINE
                        ───────────────────
  8 test questions
          │
          ▼
  RAG pipeline (baseline or reranked)
          │
          ▼
  Custom LLM-as-judge (llama-3.1-8b-instant via Groq)
          │
          ▼
  Faithfulness / Answer Relevancy / Context Precision scores
          │
          ▼
  CSV results → results/
```

---

## Project structure

```
fraud-rag/
├── .env                          # Secret keys (never commit)
├── .env.example                  # Template — copy to .env and fill in keys
├── .gitignore
├── requirements.txt
├── app/
│   ├── main.py                   # FastAPI backend (/query, /ingest, /health)
│   └── ui.py                     # Streamlit chat interface
├── scripts/
│   └── ingest.py                 # Document loader + chunker + ChromaDB writer
├── data/
│   └── reports/
│       ├── sample_reports.json   # 5 synthetic fraud cases (seed data)
│       └── extended_reports.json # 10 additional synthetic cases
├── eval/
│   └── evaluate.py               # Custom LLM-as-judge evaluation + FlashRank benchmark
├── results/                      # Auto-created by evaluate.py — stores CSV scores
└── chroma_db/                    # Auto-created by ingest.py — persisted vector store
```

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/Mufaddal24/Fraud-Investigation-Assistant
cd Fraud-Investigation-Assistant
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Set up API keys

```bash
cp .env.example .env
```

Edit `.env`:
```
GROQ_API_KEY=gsk_...
```

Get a free Groq key at [console.groq.com](https://console.groq.com). No OpenAI key needed.

### 3. Ingest reports

```bash
# Ingest all reports in data/reports/ (sample + extended = 15 cases)
python scripts/ingest.py --data_dir data/reports

# Or generate + ingest the 5-case sample only
python scripts/ingest.py --sample
```

The embedding model (~80 MB) downloads automatically on first run and is cached locally.

### 4. Start the API server

```bash
uvicorn app.main:app --reload
```

API live at `http://127.0.0.1:8000`. Interactive docs at `http://127.0.0.1:8000/docs`.

### 5. Start the Streamlit UI (optional, separate terminal)

```bash
streamlit run app/ui.py
```

UI live at `http://localhost:8501`.

---

## Dataset

### Sample reports (`data/reports/sample_reports.json`) — 5 cases

| Case | Fraud type | Resolution |
|---|---|---|
| FR-2024-001 | card_not_present | fraud_confirmed |
| FR-2024-002 | account_takeover | fraud_confirmed |
| FR-2024-003 | friendly_fraud | fraud_not_confirmed |
| FR-2024-004 | synthetic_identity | fraud_confirmed |
| FR-2024-005 | internal_fraud | fraud_confirmed |

### Extended reports (`data/reports/extended_reports.json`) — 10 cases

| Case | Fraud type | Resolution |
|---|---|---|
| FR-2024-006 | card_not_present | fraud_confirmed |
| FR-2024-007 | account_takeover | fraud_confirmed |
| FR-2024-008 | synthetic_identity | fraud_confirmed |
| FR-2024-009 | friendly_fraud | fraud_not_confirmed |
| FR-2024-010 | internal_fraud | fraud_confirmed |
| FR-2024-011 | card_not_present | fraud_confirmed |
| FR-2024-012 | account_takeover | fraud_confirmed |
| FR-2024-013 | friendly_fraud | fraud_not_confirmed |
| FR-2024-014 | synthetic_identity | fraud_confirmed |
| FR-2024-015 | internal_fraud | fraud_confirmed |

All reports are synthetic and for demonstration purposes only.

---

## API reference

### `GET /health`

```json
{
  "status":     "ok",
  "llm":        "llama-3.3-70b-versatile",
  "embeddings": "sentence-transformers/all-MiniLM-L6-v2",
  "collection": "fraud_reports"
}
```

### `POST /query`

**Request:**
```json
{
  "question":   "What signals were common in account takeover cases?",
  "fraud_type": "account_takeover"
}
```

`fraud_type` is optional. Accepted values: `card_not_present`, `account_takeover`, `friendly_fraud`, `synthetic_identity`, `internal_fraud`. When provided, retrieval is filtered to matching chunks only.

**Response:**
```json
{
  "answer": "In cases FR-2024-002, FR-2024-007, and FR-2024-012, common signals include ...",
  "sources": [
    {
      "content":    "Automated login attempts from 15 distinct IPs ...",
      "case_id":    "FR-2024-002",
      "fraud_type": "account_takeover",
      "date":       "2024-04-01",
      "source":     "data/reports/sample_reports.json"
    }
  ]
}
```

### `POST /ingest`

Multipart file upload. Accepts `.pdf`, `.txt`, `.json`.

```bash
curl -X POST http://localhost:8000/ingest -F "file=@my_case_report.pdf"
```

**Response:**
```json
{
  "message":      "Successfully ingested 'my_case_report.pdf'",
  "chunks_added": 14
}
```

---

## Evaluation

Run the custom LLM-as-judge evaluator:

```bash
# Baseline only
python eval/evaluate.py

# Baseline vs FlashRank reranked (generates resume delta)
python eval/evaluate.py --rerank --output results/
```

The evaluator scores three metrics per question using `llama-3.1-8b-instant` as judge:

- **Faithfulness** — every claim in the answer is supported by the retrieved context
- **Answer Relevancy** — the answer directly addresses the question asked
- **Context Precision** — the retrieved chunks are relevant to the question

Results are saved as CSVs in `results/`. The `--rerank` flag also prints a delta table and a resume-ready bullet.

### Why a custom evaluator instead of RAGAS?

RAGAS internally requests `n>1` completions from the judge LLM, which Groq's API does not support (max `n=1`). Rather than work around a framework that was fighting the stack, a custom LLM-as-judge was implemented — giving full control over scoring prompts, no parallel job timeouts, and cleaner results.

---

## Key design decisions

| Decision | Rationale |
|---|---|
| Groq instead of OpenAI | Free tier, no credit card required, fast inference |
| `all-MiniLM-L6-v2` embeddings | Runs fully locally, no API key, ~80MB, strong performance |
| Manual FlashRank reranker | `ContextualCompressionRetriever` had import issues across LangChain versions; direct FlashRank call is more stable |
| Custom LLM-as-judge evaluator | RAGAS incompatible with Groq's `n=1` constraint; custom evaluator gives identical metrics with full control |
| LCEL `RunnableParallel` chain | Retriever runs exactly once; same docs used for LLM context and source attribution — no double retrieval |
| ChromaDB | Zero-config local vector store, no external service needed for development |

---