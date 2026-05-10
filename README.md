# Fraud Investigation Assistant — RAG API

A Retrieval-Augmented Generation (RAG) backend that lets fraud analysts ask natural-language questions over a corpus of fraud investigation case reports. The system retrieves the most relevant case chunks from a local vector database and generates concise, source-cited answers using a free LLM.

---

## Architecture overview

```
Analyst question
      │
      ▼
 FastAPI /query
      │
      ▼
 RunnableParallel (LCEL chain)
      ├── ChromaDB retriever      ← all-MiniLM-L6-v2 embeddings (local, free)
      │    (top-5 chunks)
      └── question passthrough
      │
      ├── PromptTemplate | ChatGroq (Llama 3.3 70B) | StrOutputParser → answer
      └── context passthrough                                          → sources
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

### 3. Ingest the sample data

```bash
# Generates sample_reports.json, then embeds and stores it in ChromaDB
python scripts/ingest.py --sample

# Or ingest your own reports
python scripts/ingest.py --data_dir data/reports
```

The embedding model (~80 MB) downloads automatically on first run.

### 4. Start the API server

```bash
uvicorn app.main:app --reload
```

The API is now live at `http://127.0.0.1:8000`. Interactive docs at `http://127.0.0.1:8000/docs`.

---

## Directory structure

```
fraud-rag/
├── .env                        # Secret keys (never commit this)
├── requirements.txt            # Dependency list
├── app/
│   └── main.py                 # FastAPI application — the main server
├── data/
│   └── reports/
│       └── sample_reports.json # 5 synthetic fraud case reports (seed data)
├── scripts/
│   └── ingest.py               # One-time ingestion CLI
└── chroma_db/                  # Vector store written at ingest time (auto-created)
```

---

## File reference

### `.env`

Stores secrets loaded at runtime by `python-dotenv`. The only required key is:

```
GROQ_API_KEY=gsk_...
```

Never commit this file. Add it to `.gitignore`.

---

### `requirements.txt`

Lists the Python packages the project depends on. Install with `pip install -r requirements.txt`.

Install with `pip install -r requirements.txt`.

---

### `app/main.py`

The FastAPI application. It is the only process that needs to be running during normal use. It starts the HTTP server, builds the LCEL RAG chain in memory at startup, and exposes three endpoints.

#### Imports and config block

Key third-party imports: `langchain_groq.ChatGroq`, `langchain_core.prompts.PromptTemplate`, `langchain_core.output_parsers.StrOutputParser`, `langchain_core.runnables.RunnablePassthrough`/`RunnableParallel`, `langchain_chroma.Chroma`, `langchain_huggingface.HuggingFaceEmbeddings`.

The config constants defined here are the single place to tune behaviour:

| Constant | Default | Purpose |
|---|---|---|
| `CHROMA_DIR` | `"chroma_db"` | Path to the persisted ChromaDB vector store |
| `COLLECTION_NAME` | `"fraud_reports"` | ChromaDB collection name (must match what `ingest.py` wrote) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace sentence-transformer used for both ingestion and retrieval |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq model used to generate answers |
| `TOP_K` | `5` | Number of document chunks retrieved per query |

#### `PROMPT` (module-level `PromptTemplate`)

A `langchain_core.prompts.PromptTemplate` with input variables `context` and `question`. The prompt instructs the LLM to:
- Answer only from the provided context.
- Always cite case IDs (e.g. `FR-2024-001`).
- Return a fixed fallback if the answer is not in the context.

#### `format_docs(docs)` helper

Concatenates a list of retrieved `Document` objects into a single context string, separating chunks with `---` dividers.

#### `class RAGPipeline`

A singleton class instantiated once at startup (`pipeline = RAGPipeline()`). Holds all expensive resources in memory so they are reused across requests.

**`__init__`**

Runs when the server starts. Builds all pipeline components and wires them into a single LCEL chain:

1. Loads the HuggingFace embedding model onto CPU. The same model must be used here and in `ingest.py`, otherwise retrieved chunks will be nonsense.
2. Opens the ChromaDB collection at `CHROMA_DIR`. If the directory does not yet exist (i.e. ingest has never been run), ChromaDB creates an empty collection — queries will return no results but won't crash.
3. Creates `self.retriever` from the vector store (`similarity` search, top-K results).
4. Validates `GROQ_API_KEY` is set; raises `RuntimeError` immediately if missing so the server fails fast rather than on first request.
5. Instantiates `ChatGroq` (temperature `0.1`, max tokens `512`).
6. Assembles the LCEL chain using `RunnableParallel`:

```
question
    │
RunnableParallel
    ├── retriever  → [doc1..doc5]   (retrieved once)
    └── passthrough → question
    │
    ├── PROMPT | llm | StrOutputParser  → "answer"
    └── passthrough                     → "context" (same docs)
```

**`query(question, fraud_type=None)`**

Called on every `POST /query` request. Executes the full RAG loop in a single `chain.invoke(question)` call — the retriever runs exactly once:

1. If `fraud_type` is provided, updates the retriever's `search_kwargs` with a ChromaDB metadata filter so only chunks with a matching `fraud_type` field are considered.
2. Calls `self.chain.invoke(question)` — returns `{"answer": str, "context": [docs]}`.
3. Formats source metadata for each retrieved chunk (content preview, `case_id`, `fraud_type`, `date`, source file) and returns them alongside the answer.

**`ingest_file(file_path, file_suffix)`**

Called on every `POST /ingest` request (file upload). Handles three formats:
- `.pdf` — loaded with `PyPDFLoader` (requires `pypdf`)
- `.txt` — loaded with `TextLoader`
- `.json` — parsed manually; each record in the JSON array becomes one `Document` with metadata extracted from keys `case_id`, `fraud_type`, and `date`

After loading, `RecursiveCharacterTextSplitter` splits documents into 800-character chunks (150-character overlap), and the chunks are added to the live ChromaDB collection. Returns the number of chunks added.

#### FastAPI app and middleware

Creates the `FastAPI` instance and attaches `CORSMiddleware` with `allow_origins=["*"]`. The wildcard is fine for local development — restrict this before any public deployment.

#### Pydantic schemas

Request and response shapes validated automatically by FastAPI:

| Schema | Direction | Fields |
|---|---|---|
| `QueryRequest` | request body for `/query` | `question` (str), `fraud_type` (optional str) |
| `QueryResponse` | response from `/query` | `answer` (str), `sources` (list of `SourceChunk`) |
| `SourceChunk` | nested in `QueryResponse` | `content`, `case_id`, `fraud_type`, `date`, `source` |
| `IngestResponse` | response from `/ingest` | `message` (str), `chunks_added` (int) |

#### Routes

**`GET /health`**  
Returns the running LLM model name, embedding model name, and collection name. Use this to confirm the server started correctly.

**`POST /query`**  
Accepts a `QueryRequest` body. Validates the question is non-empty, delegates to `pipeline.query()`, and returns a `QueryResponse`. Returns HTTP 400 for an empty question, HTTP 500 for any pipeline error (with the error message in the detail field).

**`POST /ingest`**  
Accepts a multipart file upload. Validates the file extension is `.pdf`, `.txt`, or `.json`. Writes the upload to a temporary directory, calls `pipeline.ingest_file()`, then cleans up the temp file. Returns an `IngestResponse` with the count of chunks added. Returns HTTP 400 for unsupported file types, HTTP 500 for processing errors.

---

### `scripts/ingest.py`

A standalone CLI script run once (or whenever you have new reports) to populate ChromaDB. It does not import anything from `app/` and has no dependency on the server being running.

#### Config block

Same `CHROMA_DIR`, `EMBEDDING_MODEL`, `CHUNK_SIZE`, and `CHUNK_OVERLAP` constants as `app/main.py`. If you change these values, change them in both files — they must match.

#### Loader functions

Three file-format-specific loaders, registered in `LOADER_MAP`:

| Function | Format | Library |
|---|---|---|
| `load_pdf(path)` | `.pdf` | `PyPDFLoader` from `langchain-community` |
| `load_txt(path)` | `.txt` | `TextLoader` from `langchain-community` |
| `load_json(path)` | `.json` | Custom — reads a JSON array, converts each record to a `Document` with `case_id`, `fraud_type`, and `date` metadata |

#### `load_documents(data_dir)`

Walks the given directory recursively, finds all files with extensions in `LOADER_MAP`, and loads them. Skips files that raise exceptions (prints a warning) so a single bad file doesn't abort the whole run. Returns a flat list of `Document` objects.

#### `chunk_documents(docs)`

Splits every document using `RecursiveCharacterTextSplitter`. The splitter tries to break on paragraph boundaries first (`\n\n`), then lines, sentences, words, and finally characters. Returns a flat list of chunk `Document` objects.

#### `build_vectorstore(chunks, collection)`

Loads the HuggingFace embedding model (downloads on first run, then cached), embeds all chunks, and writes them to ChromaDB at `CHROMA_DIR`. Uses `Chroma.from_documents` which creates the collection if it doesn't exist and overwrites if it does.

#### `SAMPLE_REPORTS` and `generate_sample_data()`

A hardcoded list of 5 synthetic fraud cases used for development and testing. `generate_sample_data()` serialises them to `data/reports/sample_reports.json`. Triggered by the `--sample` CLI flag. The five cases cover:

| Case | Fraud type | Outcome |
|---|---|---|
| FR-2024-001 | card_not_present | fraud confirmed |
| FR-2024-002 | account_takeover | fraud confirmed |
| FR-2024-003 | friendly_fraud | not confirmed |
| FR-2024-004 | synthetic_identity | fraud confirmed |
| FR-2024-005 | internal_fraud | fraud confirmed |

#### CLI entry point

Parses three arguments:

| Flag | Default | Effect |
|---|---|---|
| `--data_dir` | `data/reports` | Directory to scan for report files |
| `--collection` | `fraud_reports` | ChromaDB collection name |
| `--sample` | off | Generate `sample_reports.json` before ingesting |

---

### `data/reports/sample_reports.json`

The serialised form of `SAMPLE_REPORTS` from `ingest.py`. Each object has:

```json
{
  "case_id":         "FR-2024-001",
  "fraud_type":      "card_not_present",
  "date":            "2024-03-12",
  "summary":         "...",
  "details":         "...",
  "flagged_signals": ["geo_velocity", "ip_mismatch", "unusual_merchant_category"],
  "resolution":      "fraud_confirmed",
  "analyst":         "A. Patel"
}
```

All fields are preserved in the embedded text so the LLM can reason about them. `case_id`, `fraud_type`, and `date` are additionally stored as ChromaDB metadata so they can be filtered and surfaced in API responses without re-parsing the chunk text.

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
`fraud_type` is optional. When provided, retrieval is filtered to chunks whose metadata field matches.

**Response:**
```json
{
  "answer": "In case FR-2024-002, the signals were ...",
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
curl -X POST http://localhost:8000/ingest \
     -F "file=@my_case_report.pdf"
```

**Response:**
```json
{
  "message":      "Successfully ingested 'my_case_report.pdf'",
  "chunks_added": 14
}
```

