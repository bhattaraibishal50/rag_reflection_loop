"""Dependent variables: Hallucination Rate, Faithfulness, latency, rejection accuracy.

The hallucination judge scores claims using prompts/judge_rubric.md. The SAME judge
function is validated against humans in judge_validation.py before being trusted here.
"""
from __future__ import annotations

import asyncio
import json

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
    # single_turn_ascore is async; run it on a fresh event loop for CLI/batch use.
    return asyncio.run(metric.single_turn_ascore(sample))


def rejection_correct(diagnosis: str, true_label: str) -> bool:
    """RQ3: for cross-domain cases (true_label == 'REJECT'), did the system refuse?"""
    refused = diagnosis.strip().upper().startswith("REJECT") or "REJECT" in diagnosis[:60].upper()
    return refused if true_label == "REJECT" else (not refused)
