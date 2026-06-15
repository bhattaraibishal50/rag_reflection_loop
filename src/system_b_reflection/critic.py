"""Critic node — the Reflector. Re-examines the IMAGE against the draft to find
modality-misalignment hallucinations. Higher temperature -> more skeptical."""
from __future__ import annotations

import json

from config.config import cfg
from src.llm.client import LLMClient, load_prompt
from src.system_b_reflection.state import ReflectionState

# NOTE: for the final run, consider a DIFFERENT model family here to reduce self-bias.
_critic = LLMClient(cfg.critic_model, temperature=cfg.critic_temperature)
_prompt = load_prompt("critic_system.txt")


def _parse_json(text: str) -> dict:
    """Critic is asked for strict JSON; be defensive about stray markdown fences."""
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fail safe: if we can't parse, assume a discrepancy so we don't silently pass.
        return {"has_discrepancy": True, "contradictions": ["critic output unparseable"],
                "missing_symptoms": [], "refined_query": None}


def critic_node(state: ReflectionState) -> ReflectionState:
    user_text = (
        f"QUERY: {state['query']}\n\n"
        f"ACTOR'S DRAFT DIAGNOSIS:\n{state['draft']}\n\n"
        f"RETRIEVED CONTEXT:\n{state['context']}"
    )
    raw = _critic.generate(_prompt, user_text, image_path=state["image_path"])
    feedback = _parse_json(raw)
    return {
        "critic_feedback": feedback,
        "refined_query": feedback.get("refined_query") or state["query"],
        "iteration": state.get("iteration", 0) + 1,
    }
