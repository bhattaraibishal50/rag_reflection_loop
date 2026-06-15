"""Step 2 of the build: PDF knowledge base -> chunks -> embeddings -> Chroma.

Run:  python -m src.ingest.ingest

GATE: after this, run the retriever and confirm it returns the RIGHT passages
before building any agents (bad retrieval poisons both systems — concern #3).
"""
from __future__ import annotations

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config.config import cfg


def _read_pdfs() -> list[tuple[str, str]]:
    """Return list of (source_name, full_text)."""
    docs = []
    pdfs = sorted(cfg.kb_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs in {cfg.kb_dir}. Add FAO/agronomy handbooks first.")
    for pdf in pdfs:
        text = "\n".join((page.extract_text() or "") for page in PdfReader(str(pdf)).pages)
        docs.append((pdf.name, text))
    return docs


def _chunk(docs: list[tuple[str, str]]) -> tuple[list[str], list[dict]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
    )
    texts, metadatas = [], []
    for source, text in docs:
        for i, chunk in enumerate(splitter.split_text(text)):
            texts.append(chunk)
            metadatas.append({"source": source, "chunk": i})
    return texts, metadatas


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed chunks with Gemini (cfg.embedding_model), task_type=RETRIEVAL_DOCUMENT."""
    from src.llm.embeddings import embed_documents

    print(f"Embedding {len(texts)} chunks with {cfg.embedding_model}...")
    return embed_documents(texts)


def main() -> None:
    docs = _read_pdfs()
    texts, metadatas = _chunk(docs)
    print(f"Read {len(docs)} PDFs -> {len(texts)} chunks.")

    embeddings = _embed(texts)

    client = chromadb.PersistentClient(path=str(cfg.chroma_dir))
    client.delete_collection(cfg.collection_name) if cfg.collection_name in [
        c.name for c in client.list_collections()
    ] else None
    collection = client.create_collection(cfg.collection_name)
    collection.add(
        ids=[f"chunk-{i}" for i in range(len(texts))],
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"Indexed {collection.count()} chunks into Chroma at {cfg.chroma_dir}.")


if __name__ == "__main__":
    main()
