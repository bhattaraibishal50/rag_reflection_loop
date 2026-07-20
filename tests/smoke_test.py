"""End-to-end smoke test with a MOCKED LLM — no API key, no Chroma index needed.

Verifies the parts that don't depend on the network:
  - System A returns a well-formed result in one pass.
  - System B's LangGraph loop runs: Actor -> Critic -> (discrepancy) -> refine -> stop,
    incrementing the iteration counter and honouring the cap.

Run:  python tests/smoke_test.py      (or: make smoke)
Exit code 0 = all checks passed.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Run as a script → put repo root on the path so `import src...` resolves.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class _FakeRetriever:
    def as_context(self, query, top_k=None):
        return "[fake_manual.pdf] Early blight shows concentric-ring lesions on tomato."


def _test_system_a() -> None:
    from src.systems import baseline

    class _FakeLLM:
        def __init__(self, *a, **k): ...
        def generate(self, *a, **k):
            return "Diagnosis: Early Blight\nConfidence: high"

    baseline.Retriever = lambda: _FakeRetriever()   # patch class used inside diagnose()
    baseline.LLMClient = _FakeLLM

    res = baseline.diagnose("fake.jpg", "What disease?")
    assert res["system"] == "A_baseline", res
    assert res["iterations"] == 1, res
    assert "Early Blight" in res["diagnosis"], res
    assert res["latency_s"] >= 0
    print("  [ok] System A: single pass, well-formed result")


def _test_system_b_loops() -> None:
    from src.systems.reflection import actor, critic, graph

    # Actor: canned draft + fake retrieval (bypass Chroma).
    actor._retriever = _FakeRetriever()
    actor._actor.generate = lambda *a, **k: "Diagnosis: Early Blight\nObserved: concentric rings"

    # Critic: flag a discrepancy the first time, then pass — exercises exactly one refine loop.
    state = {"n": 0}

    def _fake_critic(*a, **k):
        state["n"] += 1
        if state["n"] == 1:
            return ('{"has_discrepancy": true, "contradictions": ["no orange pustules"], '
                    '"missing_symptoms": [], "refined_query": "septoria vs early blight"}')
        return ('{"has_discrepancy": false, "contradictions": [], '
                '"missing_symptoms": [], "refined_query": null}')

    critic._critic.generate = _fake_critic

    res = graph.diagnose("fake.jpg", "What disease?")
    assert res["system"] == "B_reflection", res
    assert res["iterations"] == 2, f"expected 2 iterations (one refine), got {res['iterations']}"
    assert res["critic_feedback"]["has_discrepancy"] is False, res
    print("  [ok] System B: Actor->Critic->refine->stop, iterations=2")


def _test_system_b_respects_cap() -> None:
    from src.systems.reflection import actor, critic, graph
    from config.config import cfg

    actor._retriever = _FakeRetriever()
    actor._actor.generate = lambda *a, **k: "Diagnosis: Early Blight"
    # Critic ALWAYS finds a discrepancy → loop must stop at the cap, not run forever.
    critic._critic.generate = lambda *a, **k: (
        '{"has_discrepancy": true, "contradictions": ["x"], '
        '"missing_symptoms": [], "refined_query": "q"}')

    res = graph.diagnose("fake.jpg", "What disease?")
    assert res["iterations"] == cfg.max_iterations, \
        f"expected cap {cfg.max_iterations}, got {res['iterations']}"
    print(f"  [ok] System B: honours the {cfg.max_iterations}-iteration cap under endless critique")


def main() -> int:
    print("Running smoke tests (mocked LLM, no API/index)...")
    for test in (_test_system_a, _test_system_b_loops, _test_system_b_respects_cap):
        test()
    print("\nALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
