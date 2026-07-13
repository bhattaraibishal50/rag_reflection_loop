"""Gemini embeddings via LangChain — shared by ingest (documents) and retriever (queries).

Lazy-initialised so importing this never requires an API key. Task-type asymmetry
(RETRIEVAL_DOCUMENT for the corpus, RETRIEVAL_QUERY for searches) is preserved with two
GoogleGenerativeAIEmbeddings instances — the asymmetry improves retrieval quality and is
recommended by Google.
"""
from __future__ import annotations
from config.config import cfg
from src.llm.client import with_retry
from langchain_google_genai import GoogleGenerativeAIEmbeddings

_BATCH = 100          # embed corpus chunks in batches; retry each batch independently
_doc_embedder = None
_query_embedder = None


def _model_name() -> str:
    """LangChain's embeddings client expects a `models/` prefix on the model id."""
    m = cfg.embedding_model
    return m if m.startswith("models/") else f"models/{m}"


def _make_embedder(task_type: str):
    cfg.validate()

    return GoogleGenerativeAIEmbeddings(
        model=_model_name(),
        google_api_key=cfg.api_key,
        task_type=task_type,
    )


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed corpus chunks for indexing (task_type=retrieval_document)."""
    global _doc_embedder
    if _doc_embedder is None:
        _doc_embedder = _make_embedder("retrieval_document")

    vectors: list[list[float]] = []
    for i in range(0, len(texts), _BATCH):
        batch = texts[i : i + _BATCH]
        vectors.extend(with_retry(lambda b=batch: _doc_embedder.embed_documents(b)))
    return vectors


def embed_query(text: str) -> list[float]:
    """Embed a single search query (task_type=retrieval_query)."""
    global _query_embedder
    if _query_embedder is None:
        _query_embedder = _make_embedder("retrieval_query")
    return with_retry(lambda: _query_embedder.embed_query(text))
