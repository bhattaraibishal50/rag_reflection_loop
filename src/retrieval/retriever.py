"""Vector search over the Chroma index. Shared by System A and System B.

Quick GATE check:  python -m src.retrieval.retriever "early blight concentric rings tomato"
Inspect the returned passages — are they actually relevant? If not, fix chunking/embeddings
before building agents.
"""
from __future__ import annotations

import sys

import chromadb

from config.config import cfg


class Retriever:
    def __init__(self):
        client = chromadb.PersistentClient(path=str(cfg.chroma_dir))
        self.collection = client.get_collection(cfg.collection_name)

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        from src.llm.embeddings import embed_query

        k = top_k or cfg.top_k
        q_emb = [embed_query(query)]  # task_type=RETRIEVAL_QUERY (matches ingest asymmetry)
        res = self.collection.query(query_embeddings=q_emb, n_results=k)
        return [
            {"text": doc, "source": meta.get("source"), "distance": dist}
            for doc, meta, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def as_context(self, query: str, top_k: int | None = None) -> str:
        hits = self.search(query, top_k)
        return "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "early blight tomato"
    for i, hit in enumerate(Retriever().search(query), 1):
        print(f"\n--- result {i} (dist={hit['distance']:.3f}, {hit['source']}) ---")
        print(hit["text"][:400])
