"""Dependent variables: Hallucination Rate, Faithfulness, latency, rejection accuracy.

The hallucination judge scores claims using prompts/judge_rubric.md. The SAME judge
function is validated against humans in judge_validation.py before being trusted here.
"""
from __future__ import annotations

import asyncio
import json
import threading

from config.config import cfg
from src.llm.client import LLMClient

# Judge SHOULD be a different model family than Actor/Critic (proposal §3.1).
_judge = LLMClient(cfg.judge_model, temperature=0.0)

_JUDGE_PROMPT = """You are a strict evaluator of crop-disease diagnoses.
Apply this rubric exactly:

{rubric}

Score the DIAGNOSIS below. Consider the retrieved CONTEXT as the textual evidence and
the human-provided TRUE LABEL. Output STRICT JSON:
{{"claims": [{{"text": "...", "label": 0}}], "hallucination_rate": 0.0}}
(label 1 = hallucinated, 0 = supported)
"""


def hallucination_rate(diagnosis: str, context: str, true_label: str,
                       image_path: str | None = None) -> dict:
    """LLM-judge hallucination scoring. Returns {claims, hallucination_rate}.

    NOTE: passing the image lets the judge verify visual claims directly.
    """
    rubric = (cfg.prompts_dir / "judge_rubric.md").read_text()
    user = (f"TRUE LABEL: {true_label}\n\nDIAGNOSIS:\n{diagnosis}\n\nCONTEXT:\n{context}")
    raw = _judge.generate(_JUDGE_PROMPT.format(rubric=rubric), user, image_path=image_path)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"claims": [], "hallucination_rate": None}


# --- RAGAS faithfulness (lazy-initialised; needs an evaluator LLM + API key) ---
_faithfulness = None  # cached (metric, SingleTurnSample-class) tuple


def _get_faithfulness():
    """Build the RAGAS Faithfulness metric backed by the Gemini judge model.

    Faithfulness decomposes the answer into claims and checks each against the
    retrieved context — it needs an LLM but no embeddings. We reuse cfg.judge_model
    (a different family from Actor/Critic) to stay consistent with §3.1.
    """
    global _faithfulness
    if _faithfulness is None:
        cfg.validate()
        from langchain_google_genai import ChatGoogleGenerativeAI
        from ragas import SingleTurnSample
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness

        evaluator_llm = LangchainLLMWrapper(
            ChatGoogleGenerativeAI(
                model=cfg.judge_model, google_api_key=cfg.api_key, temperature=0.0
            )
        )
        _faithfulness = (Faithfulness(llm=evaluator_llm), SingleTurnSample)
    return _faithfulness


# One long-lived loop on one background thread, shared by every RAGAS call.
_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_lock = threading.Lock()


def _get_loop() -> asyncio.AbstractEventLoop:
    """Return the dedicated RAGAS event loop, starting it on first use.

    WHY THIS EXISTS (do not simplify back to `asyncio.run`, and do not close
    the loop between calls) — two distinct failures are being avoided:

    1. `RuntimeError: Timeout should be used inside a task`.
       Importing RAGAS pulls in nest_asyncio, which monkeypatches `asyncio.run`
       and `BaseEventLoop.run_until_complete` with hand-rolled task stepping. On
       CPython 3.14 that stepping leaves `asyncio.current_task()` returning None,
       so the `asyncio.wait_for` inside `Metric.single_turn_ascore` raises and
       every score comes back NaN. Running on our own loop via
       `run_coroutine_threadsafe` creates a genuine Task, so the timeout works.

    2. `RuntimeError: Event loop is closed`.
       The Gemini client underneath RAGAS holds grpc.aio channels bound to the
       loop that created them. A fresh-loop-per-call design scores the first
       sample and then fails on every subsequent one. Hence one persistent loop.

    Verified on ragas 0.2.15 / nest-asyncio 1.6.0 / grpcio aio / CPython 3.14.7.
    """
    global _loop, _loop_thread
    with _loop_lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
            _loop_thread = threading.Thread(
                target=_loop.run_forever, name="ragas-loop", daemon=True
            )
            _loop_thread.start()
    return _loop


def _run_coro_isolated(make_coro, timeout: float = 300.0):
    """Submit a coroutine to the dedicated RAGAS loop and wait for its result."""
    loop = _get_loop()
    future = asyncio.run_coroutine_threadsafe(make_coro(), loop)
    return future.result(timeout=timeout)


def faithfulness_score(diagnosis: str, context: str | list[str], query: str) -> float:
    """RAGAS faithfulness in [0, 1]: fraction of the diagnosis's claims that are
    supported by the retrieved context. Higher = less hallucination.

    `context` may be the joined string from Retriever.as_context (split back into the
    per-passage list it was built from) or an already-split list.
    """
    metric, SingleTurnSample = _get_faithfulness()
    contexts = context.split("\n\n") if isinstance(context, str) else context
    sample = SingleTurnSample(
        user_input=query,
        response=diagnosis,
        retrieved_contexts=[c for c in contexts if c.strip()],
    )
    # Call the inner coroutine directly: the public single_turn_ascore wraps it in
    # asyncio.wait_for, which is the exact call that breaks under nest_asyncio on
    # 3.14. The inner method performs the same scoring without the timeout wrapper.
    return _run_coro_isolated(lambda: metric._single_turn_ascore(sample, callbacks=None))


def rejection_correct(diagnosis: str, true_label: str) -> bool:
    """RQ3: for cross-domain cases (true_label == 'REJECT'), did the system refuse?"""
    refused = diagnosis.strip().upper().startswith("REJECT") or "REJECT" in diagnosis[:60].upper()
    return refused if true_label == "REJECT" else (not refused)
