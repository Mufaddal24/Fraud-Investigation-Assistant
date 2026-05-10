"""
app/main.py
-----------
FastAPI backend for the Fraud Investigation Assistant.

LLM:        Groq (llama-3.3-70b-versatile) via langchain-groq
Embeddings: sentence-transformers/all-MiniLM-L6-v2 (local)

Setup:
    1. Get a free Groq key: https://console.groq.com
    2. Add to .env:  GROQ_API_KEY=gsk_...
    3. uvicorn app.main:app --reload
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()


# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_DIR      = "chroma_db"
COLLECTION_NAME = "fraud_reports"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL       = "llama-3.3-70b-versatile"
TOP_K           = 5


# ── Prompt (defined at module level so it's always available) ─────────────────

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a fraud investigation assistant for a financial company.
Use ONLY the context below to answer. If the answer is not present, say:
"I don't have enough information in the loaded reports."

Rules:
- Always reference case IDs (e.g. FR-2024-001) when citing specific cases.
- Be concise and factual.
- If multiple cases are relevant, summarise patterns across them.

Context:
{context}

Question: {question}

Answer:""",
)


# ── Helper ────────────────────────────────────────────────────────────────────

def format_docs(docs: list) -> str:
    """Concatenate retrieved doc chunks into a single context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ── RAG pipeline ──────────────────────────────────────────────────────────────

class RAGPipeline:
    def __init__(self):
        # 1. Embeddings (local, no API key needed)
        print("Loading embedding model (first run downloads ~80MB)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # 2. Vector store
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DIR,
        )

        # 3. Retriever (defined before chain so chain can reference it)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K},
        )

        # 4. LLM via Groq
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise RuntimeError(
                "GROQ_API_KEY not set.\n"
                "Add it to your .env file.\n"
                "Get a free key at: https://console.groq.com"
            )
        print(f"Connecting to Groq ({LLM_MODEL})...")
        self.llm = ChatGroq(
            model=LLM_MODEL,
            api_key=groq_key,
            temperature=0.1,
            max_tokens=512,
        )

        # 5. LCEL chain — retriever runs ONCE, docs flow to both LLM and caller
        #
        #   question
        #       │
        #   RunnableParallel
        #       ├── retriever  →  [doc1..doc5]  (retrieved once)
        #       └── passthrough → question
        #       │
        #       ├── PROMPT | LLM | parser  →  "answer"
        #       └── passthrough            →  "context" (same docs)
        #
        retrieval = RunnableParallel({
            "context":  self.retriever,
            "question": RunnablePassthrough(),
        })

        self.chain = retrieval | {
            "answer":  PROMPT | self.llm | StrOutputParser(),
            "context": lambda x: x["context"],   # pass docs through unchanged
        }

        print("Pipeline ready.")

    def query(self, question: str, fraud_type: Optional[str] = None) -> dict:
        # Apply optional metadata filter
        search_kwargs = {"k": TOP_K}
        if fraud_type:
            search_kwargs["filter"] = {"fraud_type": fraud_type}
        self.retriever.search_kwargs = search_kwargs

        # Single invoke — retriever runs exactly once
        result = self.chain.invoke(question)

        answer      = result["answer"]
        source_docs = result["context"]   # exact docs used to generate the answer

        sources = [
            {
                "content":    doc.page_content[:400],
                "case_id":    doc.metadata.get("case_id", "—"),
                "fraud_type": doc.metadata.get("fraud_type", "—"),
                "date":       doc.metadata.get("date", "—"),
                "source":     doc.metadata.get("source", "—"),
            }
            for doc in source_docs
        ]
        return {"answer": answer, "sources": sources}

    def ingest_file(self, file_path: str, file_suffix: str) -> int:
        from langchain_community.document_loaders import PyPDFLoader, TextLoader

        suffix = file_suffix.lower()
        if suffix == ".pdf":
            docs = PyPDFLoader(file_path).load()
        elif suffix == ".txt":
            docs = TextLoader(file_path, encoding="utf-8").load()
        elif suffix == ".json":
            with open(file_path) as f:
                data = json.load(f)
            records = data if isinstance(data, list) else [data]
            docs = [
                Document(
                    page_content=json.dumps(r, indent=2),
                    metadata={
                        "source":     file_path,
                        "case_id":    r.get("case_id", "unknown"),
                        "fraud_type": r.get("fraud_type", "unknown"),
                        "date":       r.get("date", "unknown"),
                    },
                )
                for r in records
            ]
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
        self.vectorstore.add_documents(chunks)
        return len(chunks)


# Initialise once at startup
pipeline = RAGPipeline()


# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Investigation Assistant",
    description="RAG-powered QA over fraud case reports",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:   str
    fraud_type: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "question":   "What signals triggered the account takeover case?",
                "fraud_type": None,
            }
        }
    }


class SourceChunk(BaseModel):
    content:    str
    case_id:    str
    fraud_type: str
    date:       str
    source:     str


class QueryResponse(BaseModel):
    answer:  str
    sources: list[SourceChunk]


class IngestResponse(BaseModel):
    message:      str
    chunks_added: int


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":     "ok",
        "llm":        LLM_MODEL,
        "embeddings": EMBEDDING_MODEL,
        "collection": COLLECTION_NAME,
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    try:
        result = pipeline.query(request.question, request.fraud_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceChunk(**s) for s in result["sources"]],
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".json"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Use PDF, TXT, or JSON.",
        )
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = os.path.join(tmp, file.filename)
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        try:
            n = pipeline.ingest_file(tmp_path, suffix)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return IngestResponse(
        message=f"Successfully ingested '{file.filename}'",
        chunks_added=n,
    )