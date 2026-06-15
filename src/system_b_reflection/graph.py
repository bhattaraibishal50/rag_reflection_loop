"""System B — the LangGraph Actor-Critic reflection loop (experimental group).

    ACTOR -> CRITIC -> [discrepancy & under cap?] --yes--> ACTOR (refine)
                                                  --no---> END

Run:  python -m src.system_b_reflection.graph data/images/tomato_earlyblight_001.jpg
"""
from __future__ import annotations

import sys
import time

from langgraph.graph import END, StateGraph

from config.config import cfg
from src.system_b_reflection.actor import actor_node
from src.system_b_reflection.critic import critic_node
from src.system_b_reflection.state import ReflectionState


def _should_continue(state: ReflectionState) -> str:
    """Conditional edge: loop only if the Critic found a problem AND we're under the cap."""
    feedback = state.get("critic_feedback", {})
    if feedback.get("has_discrepancy") and state.get("iteration", 0) < cfg.max_iterations:
        return "refine"
    return "finish"


def build_graph():
    g = StateGraph(ReflectionState)
    g.add_node("actor", actor_node)
    g.add_node("critic", critic_node)
    g.set_entry_point("actor")
    g.add_edge("actor", "critic")
    g.add_conditional_edges("critic", _should_continue,
                            {"refine": "actor", "finish": END})
    return g.compile()


_GRAPH = build_graph()


def diagnose(image_path: str, query: str = "What disease does this plant have?") -> dict:
    t0 = time.perf_counter()
    final = _GRAPH.invoke({"image_path": image_path, "query": query, "iteration": 0})
    latency = time.perf_counter() - t0
    return {
        "system": "B_reflection",
        "image": image_path,
        "query": query,
        "diagnosis": final["draft"],
        "context": final.get("context", ""),
        "iterations": final.get("iteration", 0),
        "critic_feedback": final.get("critic_feedback", {}),
        "latency_s": latency,
    }


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else str(next(cfg.images_dir.glob("*")))
    result = diagnose(img)
    print(result["diagnosis"])
    print(f"\n[iterations: {result['iterations']} | latency: {result['latency_s']:.2f}s]")
