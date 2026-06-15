"""System A — single-pass RAG baseline (control group).

retrieve -> diagnose. One model call. No self-check.

Run:  python -m src.system_a_baseline data/images/tomato_earlyblight_001.jpg
"""
from __future__ import annotations

import sys
import time

from config.config import cfg
from src.llm.client import LLMClient, load_prompt
from src.retrieval.retriever import Retriever


def diagnose(image_path: str, query: str = "What disease does this plant have?") -> dict:
    retriever = Retriever()
    actor = LLMClient(cfg.actor_model, temperature=cfg.actor_temperature)
    actor_prompt = load_prompt("actor_system.txt")

    t0 = time.perf_counter()
    context = retriever.as_context(query)
    user_text = f"QUERY: {query}\n\nRETRIEVED CONTEXT:\n{context}"
    diagnosis = actor.generate(actor_prompt, user_text, image_path=image_path)
    latency = time.perf_counter() - t0

    return {
        "system": "A_baseline",
        "image": image_path,
        "query": query,
        "diagnosis": diagnosis,
        "context": context,
        "iterations": 1,
        "latency_s": latency,
    }


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else str(next(cfg.images_dir.glob("*")))
    result = diagnose(img)
    print(result["diagnosis"])
    print(f"\n[latency: {result['latency_s']:.2f}s]")
