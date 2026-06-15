"""Gemini embeddings — shared by ingest (documents) and retriever (queries).

Lazy-initialised so importing this never requires an API key. Uses task-type
optimisation: RETRIEVAL_DOCUMENT for the corpus, RETRIEVAL_QUERY for searches —
this asymmetry improves retrieval quality and is recommended by Google.
"""
from __future__ import annotations

from config.config import cfg

_BATCH = 100          # API caps contents per embed_content call; batch to be safe
_client = None


def _get_client():
    global _client
    if _client is None:
        cfg.validate()
        from google import genai

        _client = genai.Client(api_key=cfg.api_key)
    return _client


def _embed(texts: list[str], task_type: str) -> list[list[float]]:
    from google.genai import types

    client = _get_client()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), _BATCH):
        batch = texts[i : i + _BATCH]
        resp = client.models.embed_content(
            model=cfg.embedding_model,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        vectors.extend(e.values for e in resp.embeddings)
    return vectors


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed corpus chunks for indexing."""
    return _embed(texts, "RETRIEVAL_DOCUMENT")


def embed_query(text: str) -> list[float]:
    """Embed a single search query."""
    return _embed([text], "RETRIEVAL_QUERY")[0]
