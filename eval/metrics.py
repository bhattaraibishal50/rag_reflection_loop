"""Dependent variables: Hallucination Rate, Faithfulness, latency, rejection accuracy.

The hallucination judge scores claims using prompts/judge_rubric.md. The SAME judge
function is validated against humans in judge_validation.py before being trusted here.
"""
from __future__ import annotations

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


def faithfulness_score(diagnosis: str, context: str) -> float:
    """Faithfulness via RAGAS.

    TODO: wire to ragas.metrics.faithfulness with your LLM + a sample object.
    Returns a 0-1 score. Stubbed until the API run.
    """
    raise NotImplementedError("Wire to RAGAS faithfulness before the benchmark run.")


def rejection_correct(diagnosis: str, true_label: str) -> bool:
    """RQ3: for cross-domain cases (true_label == 'REJECT'), did the system refuse?"""
    refused = diagnosis.strip().upper().startswith("REJECT") or "REJECT" in diagnosis[:60].upper()
    return refused if true_label == "REJECT" else (not refused)
