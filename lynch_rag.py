"""
lynch_rag.py
────────────
RAG backend for the Peter Lynch ChatBot.
Loads documents from data/lynch/, indexes them in ChromaDB,
and answers questions in Peter Lynch's voice via xAI (Grok).
"""

import os
import torch
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import CrossEncoder

load_dotenv()

# ── Trader persona ────────────────────────────────────────────────────────────
LYNCH_SYSTEM = (
    "You are Peter Lynch, legendary manager of Fidelity Magellan Fund "
    "from 1977 to 1990, achieving a 29.2% average annual return. "
    "Respond in first person using 'I', 'my', 'me'. "
    "Answer the user's CURRENT question directly and concisely. "
    "NEVER repeat yourself. "
    "Use the provided context as your PRIMARY source. "
    "You may use your general knowledge about Peter Lynch's well-known "
    "investment philosophy (PEG ratio, ten-baggers, invest in what you know) "
    "ONLY when the context does not contain a direct answer. "
    "If asked who you are, introduce yourself as Peter Lynch. "
    "Keep answers focused and grounded in Lynch's real investment thinking."
)

# ── Global state ──────────────────────────────────────────────────────────────
_vector_db:      Chroma | None         = None
_groq_client:    OpenAI | None         = None
_embedding_model: HuggingFaceEmbeddings | None = None
_reranker:       CrossEncoder | None   = None


# ─────────────────────────────────────────────────────────────────────────────
# Document loading
# ─────────────────────────────────────────────────────────────────────────────

# Accepted column names for question / answer / label — order = priority
_Q_COLS = ["question",  "questions",  "Question",  "Questions",  "q", "Q"]
_A_COLS = ["answer",    "answers",    "Answer",    "Answers",    "a", "A"]
_L_COLS = ["label",     "labels",     "Label",     "Labels",     "category", "Category"]


def _resolve_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name that exists in df, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _df_to_documents(df: pd.DataFrame, filename: str) -> list[Document]:
    """
    Converts a DataFrame (from CSV or Excel) to LangChain Documents.
    Auto-detects question / answer / label column names.
    Each row becomes one Document formatted as:
        Label: <label>
        Q: <question>
        A: <answer>
    """
    df.columns = df.columns.str.strip()

    q_col = _resolve_col(df, _Q_COLS)
    a_col = _resolve_col(df, _A_COLS)
    l_col = _resolve_col(df, _L_COLS)

    if q_col is None or a_col is None:
        raise ValueError(
            f"Could not find question/answer columns in {filename}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df.dropna(subset=[q_col, a_col])

    docs = []
    for _, row in df.iterrows():
        label   = row[l_col] if l_col else "General"
        content = (
            f"Label: {label}\n"
            f"Q: {row[q_col]}\n"
            f"A: {row[a_col]}"
        )
        docs.append(Document(
            page_content=content,
            metadata={
                "source": filename,
                "trader": "lynch",
                "label":  str(label),
            },
        ))
    return docs


def _load_documents(data_dir: str) -> list[Document]:
    """
    Loads all .xlsx, .csv, .pdf, and .txt files from data/lynch/.

    Spreadsheet column requirements (auto-detected, case-insensitive):
      • question  / questions
      • answer    / answers
      • label     / labels      (optional)
    """
    lynch_dir = os.path.join(data_dir, "lynch")
    documents: list[Document] = []

    if not os.path.exists(lynch_dir):
        print(f"⚠️  Directory not found: {lynch_dir}")
        return documents

    files = os.listdir(lynch_dir)
    if not files:
        print(f"⚠️  No files found in {lynch_dir}")
        return documents

    for filename in sorted(files):
        filepath = os.path.join(lynch_dir, filename)
        ext      = filename.lower().rsplit(".", 1)[-1]

        try:
            # ── Excel (.xlsx / .xls) ──────────────────────────────────────────
            if ext in ("xlsx", "xls"):
                xl = pd.ExcelFile(filepath)
                total = 0
                for sheet in xl.sheet_names:
                    df   = xl.parse(sheet)
                    docs = _df_to_documents(df, filename)
                    documents += docs
                    total     += len(docs)
                    print(f"   📊 {filename} [{sheet}] → {len(docs)} Q&A pairs")
                if len(xl.sheet_names) > 1:
                    print(f"      └─ Total: {total} Q&A pairs across {len(xl.sheet_names)} sheets")

            # ── CSV ───────────────────────────────────────────────────────────
            elif ext == "csv":
                df   = pd.read_csv(filepath)
                docs = _df_to_documents(df, filename)
                documents += docs
                print(f"   📄 {filename} → {len(docs)} Q&A pairs")

            # ── PDF ───────────────────────────────────────────────────────────
            elif ext == "pdf":
                loader   = PyPDFLoader(filepath)
                pdf_docs = loader.load()
                for doc in pdf_docs:
                    doc.metadata.update({"source": filename, "trader": "lynch"})
                documents += pdf_docs
                print(f"   📕 {filename} → {len(pdf_docs)} pages")

            # ── Plain text ────────────────────────────────────────────────────
            elif ext == "txt":
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                documents.append(
                    Document(
                        page_content=content,
                        metadata={"source": filename, "trader": "lynch"},
                    )
                )
                print(f"   📝 {filename} → loaded")

            else:
                print(f"   ⏭️  {filename} — skipped (unsupported type)")

        except Exception as exc:
            print(f"   ❌ Error loading {filename}: {exc}")

    return documents


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline initialisation
# ─────────────────────────────────────────────────────────────────────────────
def load_pipeline() -> None:
    """
    Initialises embedding model, reranker, vector DB, and Groq client.
    Safe to call multiple times — skips if already loaded.
    """
    global _vector_db, _groq_client, _embedding_model, _reranker

    if _groq_client is not None and _vector_db is not None:
        return  # Already loaded

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Embedding model (better semantic quality than MiniLM-L6)
    print("⏳ Loading embedding model…")
    _embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": device},
    )

    # Cross-encoder reranker
    print("⏳ Loading reranker…")
    _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    base_dir   = os.path.dirname(__file__)
    data_dir   = os.path.join(base_dir, "data")
    chroma_dir = os.path.join(base_dir, "chroma_lynch")

    if os.path.exists(chroma_dir):
        print("⚡ Loading vector DB from cache…")
        _vector_db = Chroma(
            persist_directory=chroma_dir,
            embedding_function=_embedding_model,
            collection_name="lynch",
        )
    else:
        print("🔄 Building vector DB…")
        documents = _load_documents(data_dir)
        if not documents:
            raise RuntimeError(
                "No documents found. Add files to data/lynch/ and restart."
            )
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=64
        ).split_documents(documents)
        _vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=_embedding_model,
            persist_directory=chroma_dir,
            collection_name="lynch",
        )
        print(f"✅ {len(chunks)} chunks indexed")

    # Load API key — works locally (.env) and on HuggingFace (Secrets)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. "
            "Add it to your .env file locally, or to Secrets on HuggingFace Spaces."
        )
    _groq_client = OpenAI(api_key=api_key)
    print("✅ Pipeline ready")


# ─────────────────────────────────────────────────────────────────────────────
# Query helpers
# ─────────────────────────────────────────────────────────────────────────────
def _expand_query(question: str) -> str:
    """
    Prepends 'Peter Lynch' to the question so vector search matches
    stored Q&A pairs that use his name instead of 'you'/'your'.
    e.g. 'where were you born?' → 'Peter Lynch where were you born?'
    """
    q = question.strip()
    if "peter lynch" not in q.lower():
        return f"Peter Lynch {q}"
    return q


def _rerank(query: str, chunks: list[Document], top_k: int = 4) -> list[Document]:
    """Reranks retrieved chunks with a cross-encoder for precision."""
    if not chunks or _reranker is None:
        return chunks[:top_k]
    pairs  = [(query, c.page_content) for c in chunks]
    scores = _reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_k]]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def ask_lynch(
    question: str,
    history:  list[dict] | None = None,
) -> str:
    """
    Main entry point.  Returns Peter Lynch's answer as a string.

    Parameters
    ----------
    question : str
        The user's question.
    history : list[dict], optional
        Previous turns as [{"user": ..., "assistant": ...}, ...].
    """
    if _groq_client is None or _vector_db is None:
        load_pipeline()

    if not question.strip():
        return "Please enter a question."

    # Build retrieval query (contextualise with recent history)
    retrieval_q = question
    if history:
        last = history[-1]
        retrieval_q = f"{last['user']} {last['assistant']} {question}"

    expanded_q = _expand_query(retrieval_q)

    # Retrieve top-20 candidates, deduplicate, rerank to top-4
    raw = _vector_db.similarity_search(expanded_q, k=20)
    seen, unique = set(), []
    for doc in raw:
        txt = doc.page_content.strip()
        if txt not in seen:
            seen.add(txt)
            unique.append(doc)

    top_chunks = _rerank(question, unique, top_k=4)
    if not top_chunks:
        return "That is not something I can speak to from my experience."
        return "That is not something I can speak to from my experience."

    # Build context with source labels
    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content.strip()}"
        for doc in top_chunks
    )

    # Build messages (last 3 turns of history + current question)
    messages = [{"role": "system", "content": LYNCH_SYSTEM}]
    for turn in (history or [])[-3:]:
        messages.append({"role": "user",      "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append({
        "role": "user",
        "content": (
            f"Context from Peter Lynch's writings and interviews:\n"
            f"{context}\n\n"
            f"Question: {question}"
        ),
    })

    resp = _groq_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()