"""Gemini embeddings via LangChain — shared by ingest (documents) and retriever (queries).

Lazy-initialised so importing this never requires an API key. Task-type asymmetry
(RETRIEVAL_DOCUMENT for the corpus, RETRIEVAL_QUERY for searches) is preserved with two
GoogleGenerativeAIEmbeddings instances — the asymmetry improves retrieval quality and is
recommended by Google.
"""
from __future__ import annotations

import threading
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


# --- Query-embedding cache ---------------------------------------------------
# 185 of 187 unit failures in the 2026-09-20 run were embedding 429s. Benchmark
# queries repeat heavily ("What disease affects this tomato?" across dozens of
# cases), and embedContent is deterministic for a fixed model+task_type, so
# memoising by exact query string is scientifically neutral: identical input,
# identical vector. Cuts both cost and the dominant failure surface.
_QUERY_CACHE: dict[str, list[float]] = {}
_QUERY_CACHE_LOCK = threading.Lock()
_CACHE_STATS = {"hits": 0, "misses": 0}

_uncached_embed_query = embed_query


def embed_query(text: str):  # type: ignore[no-redef]
    with _QUERY_CACHE_LOCK:
        hit = _QUERY_CACHE.get(text)
        if hit is not None:
            _CACHE_STATS["hits"] += 1
            return hit
    vec = _uncached_embed_query(text)
    with _QUERY_CACHE_LOCK:
        _QUERY_CACHE[text] = vec
        _CACHE_STATS["misses"] += 1
    return vec


def cache_stats() -> dict:
    with _QUERY_CACHE_LOCK:
        return dict(_CACHE_STATS)
