"""
FAQ RAG Tool
------------
Loads the FAQ knowledge base, embeds it with sentence-transformers
(runs locally, no API cost), and retrieves the most relevant FAQ
entries for a given customer question using cosine similarity.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load a small, fast, free embedding model
_model = SentenceTransformer("all-MiniLM-L6-v2")

_faq_df = None
_faq_embeddings = None


def _load_faq(csv_path="data/faq_docs.csv"):
    """Load and embed the FAQ dataset once, cache in memory."""
    global _faq_df, _faq_embeddings
    if _faq_df is None:
        _faq_df = pd.read_csv(csv_path)
        questions = _faq_df["question"].tolist()
        _faq_embeddings = _model.encode(questions, convert_to_numpy=True)
    return _faq_df, _faq_embeddings


def _cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True))
    return np.dot(b_norm, a_norm)


def search_faq(query: str, top_k: int = 2, csv_path: str = "data/faq_docs.csv") -> str:
    """
    Search the FAQ knowledge base for the most relevant answer(s).

    Args:
        query: The customer's question.
        top_k: Number of top matching FAQ entries to return.
        csv_path: Path to the FAQ CSV file.

    Returns:
        A formatted string with the top matching Q&A pairs, to be
        injected into the LLM's context.
    """
    df, embeddings = _load_faq(csv_path)
    query_embedding = _model.encode(query, convert_to_numpy=True)

    scores = _cosine_similarity(query_embedding, embeddings)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append(
            f"Q: {df.iloc[idx]['question']}\nA: {df.iloc[idx]['answer']}"
        )

    return "\n\n".join(results) if results else "No relevant FAQ found."


if __name__ == "__main__":
    # Quick manual test
    print(search_faq("what time do you close?"))
