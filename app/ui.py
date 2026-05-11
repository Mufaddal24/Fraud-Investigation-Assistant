"""
app/ui.py
---------
Streamlit chat interface for the Fraud Investigation Assistant.

Run with:
    streamlit run app/ui.py

Make sure the FastAPI server is running first:
    uvicorn app.main:app --reload
"""

import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"

FRAUD_TYPES = [
    "All types",
    "card_not_present",
    "account_takeover",
    "friendly_fraud",
    "synthetic_identity",
    "internal_fraud",
]

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fraud Investigation Assistant",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Fraud Investigation Assistant")
st.caption("Ask questions about fraud case reports. Powered by RAG + Llama 3.3 70B.")

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []   # [{role, content, sources}]

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    selected_type = st.selectbox(
        "Filter by fraud type",
        FRAUD_TYPES,
        help="Restrict retrieval to a specific fraud category",
    )
    fraud_type = None if selected_type == "All types" else selected_type

    st.divider()

    st.header("📤 Upload new report")
    uploaded_file = st.file_uploader(
        "Upload a fraud report (PDF, TXT, JSON)",
        type=["pdf", "txt", "json"],
    )
    if uploaded_file:
        if st.button("Ingest report"):
            with st.spinner("Ingesting..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/ingest",
                        files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)},
                        timeout=30,
                    )
                    if resp.ok:
                        data = resp.json()
                        st.success(f"✅ {data['message']} ({data['chunks_added']} chunks)")
                    else:
                        st.error(f"Error: {resp.json().get('detail', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach API. Is `uvicorn app.main:app --reload` running?")

    st.divider()

    st.header("🗂️ Retrieved sources")
    st.caption("Sources from the last query appear here.")

    if st.session_state.last_sources:
        for i, src in enumerate(st.session_state.last_sources, 1):
            with st.expander(f"📄 {src['case_id']} — {src['fraud_type']} ({src['date']})"):
                st.markdown(f"**Source file:** `{src['source']}`")
                st.markdown("**Chunk preview:**")
                st.code(src["content"], language=None)
    else:
        st.info("Ask a question to see retrieved sources.")

    st.divider()

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────

if question := st.chat_input("Ask about a fraud case..."):

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Call the FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating answer..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/query",
                    json={"question": question, "fraud_type": fraud_type},
                    timeout=30,
                )

                if resp.ok:
                    data = resp.json()
                    answer = data["answer"]
                    sources = data["sources"]

                    st.markdown(answer)

                    # Badge showing active filter
                    if fraud_type:
                        st.caption(f"🔎 Filtered to: `{fraud_type}`")

                    # Show how many sources were retrieved
                    st.caption(f"📚 {len(sources)} source chunk(s) retrieved — see sidebar for details")

                    # Store for sidebar
                    st.session_state.last_sources = sources
                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                else:
                    error = resp.json().get("detail", "Unknown error")
                    st.error(f"API error: {error}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot reach the FastAPI backend. "
                    "Make sure `uvicorn app.main:app --reload` is running in another terminal."
                )

    # Refresh sidebar sources without full rerun
    st.rerun()