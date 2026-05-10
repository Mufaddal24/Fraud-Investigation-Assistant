"""
scripts/ingest.py
-----------------
Loads fraud investigation reports (PDF / JSON / TXT),
chunks them, embeds them using a FREE HuggingFace model,
and stores them in ChromaDB.

Embedding model: all-MiniLM-L6-v2 (runs fully locally, ~80MB)

Usage:
    python scripts/ingest.py --sample          # generate + ingest sample data
    python scripts/ingest.py --data_dir data/reports

Requirements:
    pip install langchain langchain-community langchain-huggingface \
                chromadb sentence-transformers pypdf tqdm
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_DIR      = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # free, local, ~80MB
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 150


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_pdf(path: Path) -> list:
    return PyPDFLoader(str(path)).load()


def load_txt(path: Path) -> list:
    return TextLoader(str(path), encoding="utf-8").load()


def load_json(path: Path) -> list:
    from langchain_core.documents import Document

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    records = data if isinstance(data, list) else [data]
    docs = []
    for record in records:
        text = json.dumps(record, indent=2)
        metadata = {
            "source":     str(path),
            "case_id":    record.get("case_id", "unknown"),
            "fraud_type": record.get("fraud_type", "unknown"),
            "date":       record.get("date", "unknown"),
        }
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


LOADER_MAP = {".pdf": load_pdf, ".txt": load_txt, ".json": load_json}


# ── Core pipeline ─────────────────────────────────────────────────────────────

def load_documents(data_dir: str) -> list:
    docs = []
    paths = [p for p in Path(data_dir).rglob("*")
             if p.suffix.lower() in LOADER_MAP and p.is_file()]
    print(f"Found {len(paths)} file(s) in '{data_dir}'")

    for path in tqdm(paths, desc="Loading"):
        try:
            loaded = LOADER_MAP[path.suffix.lower()](path)
            for doc in loaded:
                doc.metadata.setdefault("source", str(path))
            docs.extend(loaded)
        except Exception as e:
            print(f"  [WARN] {path}: {e}")

    print(f"Loaded {len(docs)} page(s)")
    return docs


def chunk_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunk(s)")
    return chunks


def build_vectorstore(chunks: list, collection: str) -> Chroma:
    print(f"\nLoading embedding model (downloads once ~80MB on first run)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},          # swap to "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Embedding chunks and writing to ChromaDB (~30s on CPU)...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection,
        persist_directory=CHROMA_DIR,
    )
    print(f"Done. Stored {len(chunks)} chunk(s) → {CHROMA_DIR}/")
    return vectorstore


# ── Synthetic sample data ─────────────────────────────────────────────────────

SAMPLE_REPORTS = [
    {
        "case_id": "FR-2024-001", "fraud_type": "card_not_present",
        "date": "2024-03-12",
        "summary": "Unauthorised e-commerce transactions totalling $4,200 across 3 merchants.",
        "details": (
            "The customer reported 7 transactions they did not initiate, all placed within "
            "a 40-minute window from an IP address in Eastern Europe. Card details were likely "
            "obtained via a phishing email two days prior. Resolution: all transactions reversed, "
            "new card issued, customer enrolled in 2FA for online purchases."
        ),
        "flagged_signals": ["geo_velocity", "ip_mismatch", "unusual_merchant_category"],
        "resolution": "fraud_confirmed", "analyst": "A. Patel",
    },
    {
        "case_id": "FR-2024-002", "fraud_type": "account_takeover",
        "date": "2024-04-01",
        "summary": "Account credentials compromised via credential stuffing.",
        "details": (
            "Automated login attempts from 15 distinct IPs within 10 minutes succeeded using "
            "leaked credentials from a third-party breach. Attacker changed the registered email "
            "and initiated a $1,800 wire transfer. Resolution: transfer blocked, account locked, "
            "credentials reset."
        ),
        "flagged_signals": ["login_velocity", "email_change", "wire_transfer_new_beneficiary"],
        "resolution": "fraud_confirmed", "analyst": "B. Kumar",
    },
    {
        "case_id": "FR-2024-003", "fraud_type": "friendly_fraud",
        "date": "2024-04-18",
        "summary": "Customer disputed a $320 restaurant charge.",
        "details": (
            "Investigation found CCTV footage showing the cardholder present at the premises. "
            "Chip-and-PIN authentication confirmed. Dispute ruled in favour of the merchant. "
            "Resolution: chargeback denied, customer informed."
        ),
        "flagged_signals": ["chargeback_history", "merchant_dispute"],
        "resolution": "fraud_not_confirmed", "analyst": "C. Singh",
    },
    {
        "case_id": "FR-2024-004", "fraud_type": "synthetic_identity",
        "date": "2024-05-05",
        "summary": "Synthetic identity account; $6,500 charged and abandoned.",
        "details": (
            "Applicant combined a real SSN (belonging to a minor) with fabricated name/address. "
            "Passed KYC but flagged when address failed USPS validation two weeks post-opening. "
            "Spending in high-liquidation categories (electronics, gift cards). "
            "Resolution: account closed, $6,500 loss, SAR filed with FinCEN."
        ),
        "flagged_signals": ["address_invalid", "high_liquidation_categories", "thin_credit_file"],
        "resolution": "fraud_confirmed", "analyst": "D. Mehta",
    },
    {
        "case_id": "FR-2024-005", "fraud_type": "internal_fraud",
        "date": "2024-05-20",
        "summary": "Employee made unauthorised fee waivers totalling $8,900 over 6 months.",
        "details": (
            "Audit flag triggered on high fee-waiver volume from a single employee ID. "
            "47 waivers issued outside normal approval workflow, mostly to accounts sharing "
            "the employee's address. Resolution: employee terminated, case referred to law "
            "enforcement, controls gap remediated."
        ),
        "flagged_signals": ["fee_waiver_volume", "employee_linked_accounts", "audit_anomaly"],
        "resolution": "fraud_confirmed", "analyst": "E. Nair",
    },
]


def generate_sample_data(output_dir: str = "data/reports"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "sample_reports.json"
    with open(dest, "w") as f:
        json.dump(SAMPLE_REPORTS, f, indent=2)
    print(f"Sample data written → {dest}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data/reports")
    p.add_argument("--collection", default="fraud_reports")
    p.add_argument("--sample",     action="store_true", help="Generate sample data first")
    return p.parse_args()


def main():
    args = parse_args()
    if args.sample:
        generate_sample_data(args.data_dir)
    docs   = load_documents(args.data_dir)
    chunks = chunk_documents(docs)
    build_vectorstore(chunks, args.collection)
    print("\nIngestion complete! Run next:\n  uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
