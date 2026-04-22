"""
debug.py — run this to see exactly what's happening inside ask_lynch
"""
import os
from dotenv import load_dotenv
load_dotenv()

import lynch_rag

lynch_rag.load_pipeline()

question = "where were you born?"
print(f"\n{'='*60}")
print(f"QUESTION: {question}")

# Step 1: query expansion
expanded = lynch_rag._expand_query(question)
print(f"\n[1] EXPANDED QUERY:\n{expanded}")

# Step 2: vector search
raw = lynch_rag._vector_db.similarity_search(expanded, k=5)
print(f"\n[2] TOP 5 RAW CHUNKS FROM VECTOR DB:")
for i, doc in enumerate(raw):
    print(f"  [{i+1}] {doc.page_content[:120]}")

# Step 3: reranking
top = lynch_rag._rerank(question, raw, top_k=4)
print(f"\n[3] TOP 4 AFTER RERANKING:")
for i, doc in enumerate(top):
    score = lynch_rag._reranker.predict([(question, doc.page_content)])[0]
    print(f"  [{i+1}] score={score:.3f} | {doc.page_content[:120]}")

# Step 4: show full context sent to LLM
context = "\n\n".join(
    f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content.strip()}"
    for doc in top
)
print(f"\n[4] FULL CONTEXT SENT TO LLM:\n{context}")
print(f"\n{'='*60}")