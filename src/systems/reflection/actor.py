"""Actor node — the Diagnostician. Same model + base prompt as System A (no confounds)."""
from __future__ import annotations

from config.config import cfg
from src.llm.client import LLMClient, load_prompt
from src.rag.retriever import Retriever
from src.systems.reflection.state import ReflectionState

_actor = LLMClient(cfg.actor_model, temperature=cfg.actor_temperature)
_prompt = load_prompt("actor_system.txt")
_retriever = None  # lazy: don't touch Chroma at import (lets the app start w/o an index)


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def actor_node(state: ReflectionState) -> ReflectionState:
    # On a refinement loop, retrieve with the Critic's improved query.
    query = state.get("refined_query") or state["query"]
    context = _get_retriever().as_context(query)

    user_text = f"QUERY: {state['query']}\n\nRETRIEVED CONTEXT:\n{context}"
    if state.get("critic_feedback"):
        # Feed the Critic's complaints back so the Actor can correct itself.
        fb = state["critic_feedback"]
        user_text += (
            f"\n\nA reviewer flagged these issues with your previous answer — "
            f"address them and revise:\n{fb.get('contradictions')} "
            f"{fb.get('missing_symptoms')}"
        )

    draft = _actor.generate(_prompt, user_text, image_path=state["image_path"])
    return {"context": context, "draft": draft}
