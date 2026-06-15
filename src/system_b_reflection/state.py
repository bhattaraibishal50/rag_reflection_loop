"""Shared state carried through the LangGraph reflection loop."""
from __future__ import annotations

from typing import TypedDict


class ReflectionState(TypedDict, total=False):
    # inputs
    image_path: str
    query: str
    # working memory
    context: str            # currently retrieved passages
    draft: str              # Actor's latest diagnosis
    critic_feedback: dict   # parsed Critic JSON {has_discrepancy, contradictions, ...}
    refined_query: str      # Critic's suggested better retrieval query
    iteration: int          # how many loops so far (capped at cfg.max_iterations)
    # output
    final_diagnosis: str
    latency_s: float
