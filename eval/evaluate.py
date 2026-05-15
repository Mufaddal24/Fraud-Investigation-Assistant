"""
eval/evaluate.py
----------------
Custom RAG evaluation pipeline using Groq as the judge LLM.
Measures the same three core metrics as RAGAS:

  • Faithfulness       — is the answer grounded in the retrieved context?
  • Answer Relevancy   — does the answer address the question?
  • Context Precision  — are the retrieved chunks relevant to the question?

No RAGAS dependency. Groq judges each metric on a 0.0–1.0 scale.

Usage:
    # Baseline only
    python eval/evaluate.py

    # Baseline + FlashRank reranked comparison (for resume delta)
    python eval/evaluate.py --rerank

    # Save CSVs
    python eval/evaluate.py --rerank --output results/

Requirements:
    pip install langchain-groq langchain-huggingface langchain-chroma flashrank pandas
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

load_dotenv()


# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_DIR      = "chroma_db"
COLLECTION_NAME = "fraud_reports"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PIPELINE_MODEL  = "llama-3.3-70b-versatile"   # model being evaluated
JUDGE_MODEL     = "llama-3.1-8b-instant"       # smaller model as judge (saves tokens)
TOP_K           = 8


# ── RAG prompt (same as production) ──────────────────────────────────────────

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


# ── Test cases ────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "question": "What signals triggered the account takeover case FR-2024-002?",
        "ground_truth": (
            "Case FR-2024-002 was triggered by login_velocity (15 IPs in 10 minutes), "
            "email_change, and wire_transfer_new_beneficiary signals."
        ),
    },
    {
        "question": "Which fraud types were confirmed across all cases?",
        "ground_truth": (
            "Confirmed fraud types include card_not_present, account_takeover, "
            "synthetic_identity, and internal_fraud. Friendly fraud was not confirmed."
        ),
    },
    {
        "question": "Were any SARs filed? Which cases?",
        "ground_truth": (
            "SARs were filed with FinCEN for case FR-2024-004 (synthetic identity) "
            "and FR-2024-008 (fabricated EIN business account)."
        ),
    },
    {
        "question": "Which case was ruled in favour of the merchant and why?",
        "ground_truth": (
            "Case FR-2024-003 was ruled in favour of the merchant. "
            "CCTV footage showed the cardholder present and chip-and-PIN confirmed the card was used."
        ),
    },
    {
        "question": "What are the common signals across internal fraud cases?",
        "ground_truth": (
            "Common signals across internal fraud cases (FR-2024-005, FR-2024-010, FR-2024-015) "
            "include employee-linked accounts, anomalous approval or transaction volumes, "
            "and activity outside normal approval workflows."
        ),
    },
    {
        "question": "Which account takeover cases involved wire transfers?",
        "ground_truth": (
            "FR-2024-002 involved a $1,800 wire transfer to a new beneficiary. "
            "FR-2024-007 involved two wire transfers totalling $22,000 to overseas accounts. "
            "FR-2024-012 involved a $67,000 payroll transfer redirection."
        ),
    },
    {
        "question": "How was the synthetic identity fraud in FR-2024-004 discovered?",
        "ground_truth": (
            "It was discovered when the applicant's address failed USPS validation "
            "two weeks after account opening. Spending in high-liquidation categories "
            "was an additional indicator."
        ),
    },
    {
        "question": "What remediation steps were taken in the internal fraud case FR-2024-005?",
        "ground_truth": (
            "The employee was terminated, the case was referred to law enforcement, "
            "and the controls gap allowing unsupervised fee waivers was remediated."
        ),
    },
]


# ── Judge prompts ─────────────────────────────────────────────────────────────
# Each prompt asks the judge to return a JSON with a score (0.0-1.0) and reason.

FAITHFULNESS_PROMPT = """You are an expert evaluator assessing RAG system outputs.

Question: {question}
Retrieved Context: {context}
Generated Answer: {answer}

Task: Score FAITHFULNESS — whether every claim in the answer is supported by the context.
- 1.0 = all claims in the answer are directly supported by the context
- 0.5 = some claims are supported, some are not
- 0.0 = the answer contains claims not found in the context (hallucination)

Respond ONLY with valid JSON, no other text:
{{"score": <float 0.0-1.0>, "reason": "<one sentence explanation>"}}"""

ANSWER_RELEVANCY_PROMPT = """You are an expert evaluator assessing RAG system outputs.

Question: {question}
Generated Answer: {answer}

Task: Score ANSWER RELEVANCY — whether the answer directly addresses the question asked.
- 1.0 = the answer fully and directly addresses the question
- 0.5 = the answer partially addresses the question
- 0.0 = the answer is off-topic or does not address the question

Respond ONLY with valid JSON, no other text:
{{"score": <float 0.0-1.0>, "reason": "<one sentence explanation>"}}"""

CONTEXT_PRECISION_PROMPT = """You are an expert evaluator assessing RAG system outputs.

Question: {question}
Retrieved Context Chunks:
{context}

Task: Score CONTEXT PRECISION — whether the retrieved chunks are relevant to the question.
- 1.0 = all retrieved chunks are highly relevant to answering the question
- 0.5 = some chunks are relevant, some are noise
- 0.0 = the retrieved chunks are not relevant to the question

Respond ONLY with valid JSON, no other text:
{{"score": <float 0.0-1.0>, "reason": "<one sentence explanation>"}}"""


# ── Judge ─────────────────────────────────────────────────────────────────────

def judge_score(client: Groq, prompt: str, retries: int = 3) -> tuple[float, str]:
    """
    Call the judge LLM and parse its JSON response.
    Returns (score, reason). On failure returns (0.0, "error").
    """
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()

            # Parse JSON response
            # Sometimes the model wraps in ```json ... ``` — strip it
            if "```" in raw:
                raw = raw.split("```")[1].replace("json", "").strip()

            data   = json.loads(raw)
            score  = float(data["score"])
            reason = data.get("reason", "")
            return max(0.0, min(1.0, score)), reason   # clamp to [0, 1]

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)   # brief pause before retry
                continue
            print(f"    [WARN] Judge call failed after {retries} attempts: {e}")
            return 0.0, "evaluation failed"

    return 0.0, "evaluation failed"


def evaluate_single(
    client: Groq,
    question: str,
    answer: str,
    contexts: list[str],
) -> dict:
    """Run all three metrics for a single question."""
    context_str = "\n\n---\n\n".join(contexts)

    faith_score, faith_reason = judge_score(
        client,
        FAITHFULNESS_PROMPT.format(
            question=question, context=context_str, answer=answer
        ),
    )
    time.sleep(1)   # avoid rate limiting between calls

    rel_score, rel_reason = judge_score(
        client,
        ANSWER_RELEVANCY_PROMPT.format(question=question, answer=answer),
    )
    time.sleep(1)

    prec_score, prec_reason = judge_score(
        client,
        CONTEXT_PRECISION_PROMPT.format(
            question=question, context=context_str
        ),
    )
    time.sleep(1)

    return {
        "faithfulness":              faith_score,
        "faithfulness_reason":       faith_reason,
        "answer_relevancy":          rel_score,
        "answer_relevancy_reason":   rel_reason,
        "context_precision":         prec_score,
        "context_precision_reason":  prec_reason,
    }


# ── RAG chain helpers ─────────────────────────────────────────────────────────

def build_chain(retriever, llm):
    """
    Builds chain compatible with both LangChain retrievers
    and our ManualReranker (which only has .invoke()).
    """
    def run(question: str) -> dict:
        docs    = retriever.invoke(question)
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        prompt  = PROMPT.format(context=context, question=question)
        answer  = llm.invoke(prompt).content
        return {"answer": answer, "context": docs}

    return run


def build_baseline_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )


def build_rerank_retriever(vectorstore):
    """
    Manual FlashRank reranker — no ContextualCompressionRetriever needed.
    Fetches TOP_K*3 candidates, reranks with FlashRank cross-encoder,
    keeps top TOP_K results.
    """
    from flashrank import Ranker, RerankRequest
    from langchain_core.documents import Document

    ranker = Ranker()  # loads default cross-encoder model

    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K * 3},
    )

    class ManualReranker:
        def invoke(self, query: str) -> list:
            # Step 1 — broad candidate set via similarity search
            candidates = base_retriever.invoke(query)

            # Step 2 — build FlashRank request
            passages = [
                {"id": i, "text": doc.page_content, "meta": doc.metadata}
                for i, doc in enumerate(candidates)
            ]
            request = RerankRequest(query=query, passages=passages)

            # Step 3 — rerank and keep top TOP_K
            results = ranker.rerank(request)[:TOP_K]

            # Step 4 — reconstruct LangChain Documents in reranked order
            return [
                Document(page_content=r["text"], metadata=r["meta"])
                for r in results
            ]

    return ManualReranker()


# ── Evaluation runner ─────────────────────────────────────────────────────────

def run_evaluation(chain, judge_client: Groq, label: str) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"  Evaluating: {label}")
    print(f"{'='*60}")

    rows = []
    for i, tc in enumerate(TEST_CASES, 1):
        q = tc["question"]
        print(f"  [{i}/{len(TEST_CASES)}] {q[:65]}...")

        # Run RAG pipeline
        result = chain(q)
        answer   = result["answer"]
        contexts = [doc.page_content for doc in result["context"]]

        # Score with judge
        scores = evaluate_single(judge_client, q, answer, contexts)

        row = {
            "question":      q,
            "answer":        answer,
            "ground_truth":  tc["ground_truth"],
            "label":         label,
            **scores,
        }
        rows.append(row)

        print(
            f"    faithfulness={scores['faithfulness']:.2f}  "
            f"relevancy={scores['answer_relevancy']:.2f}  "
            f"precision={scores['context_precision']:.2f}"
        )

    return pd.DataFrame(rows)


# ── Results ───────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, label: str):
    metrics = ["faithfulness", "answer_relevancy", "context_precision"]
    print(f"\n  ── {label} ──")
    for m in metrics:
        if m in df.columns:
            val = df[m].mean()
            print(f"    {m:<25} {val:.3f}  ({val*100:.1f}%)")


def print_delta(df_base: pd.DataFrame, df_rerank: pd.DataFrame):
    metrics = ["faithfulness", "answer_relevancy", "context_precision"]
    print("\n  ── Improvement from FlashRank reranking ──")
    for m in metrics:
        delta = df_rerank[m].mean() - df_base[m].mean()
        arrow = "↑" if delta >= 0 else "↓"
        print(f"    {m:<25} {arrow} {abs(delta)*100:.1f}pp")

    cp_base   = df_base["context_precision"].mean() * 100
    cp_rerank = df_rerank["context_precision"].mean() * 100
    print(f"\n  ── Resume bullet ──")
    print(f"    Improved context precision from {cp_base:.0f}% → {cp_rerank:.0f}%")
    print(f"    using FlashRank reranking (custom LLM-as-judge eval, {len(TEST_CASES)} test cases)")


def save_results(dfs: list, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")

    for df in dfs:
        label    = df["label"].iloc[0].lower().replace(" ", "_").replace("(", "").replace(")", "")
        csv_path = out / f"{label}_{ts}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved → {csv_path}")

    if len(dfs) > 1:
        pd.concat(dfs).to_csv(out / f"comparison_{ts}.csv", index=False)
        print(f"  Saved → {out}/comparison_{ts}.csv")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rerank", action="store_true", help="Also run FlashRank reranked retriever")
    p.add_argument("--output", default="results",   help="Directory to save CSV results")
    return p.parse_args()


def main():
    args = parse_args()

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("ERROR: GROQ_API_KEY not set in .env")
        sys.exit(1)

    # Shared components
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Connecting to ChromaDB...")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    print(f"Connecting to Groq pipeline model ({PIPELINE_MODEL})...")
    llm = ChatGroq(
        model=PIPELINE_MODEL,
        api_key=groq_key,
        temperature=0.1,
        max_tokens=512,
    )

    print(f"Judge model: {JUDGE_MODEL}")
    judge_client = Groq(api_key=groq_key)

    dfs = []

    # Baseline
    baseline_chain = build_chain(build_baseline_retriever(vectorstore), llm)
    df_base = run_evaluation(baseline_chain, judge_client, "Baseline (similarity)")
    print_summary(df_base, "Baseline (similarity)")
    dfs.append(df_base)

    # Reranked
    if args.rerank:
        rerank_chain = build_chain(build_rerank_retriever(vectorstore), llm)
        df_rerank = run_evaluation(rerank_chain, judge_client, "Reranked (FlashRank)")
        print_summary(df_rerank, "Reranked (FlashRank)")
        dfs.append(df_rerank)
        print_delta(df_base, df_rerank)

    save_results(dfs, args.output)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()