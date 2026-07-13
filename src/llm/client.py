"""Thin, swappable wrapper around Gemini via LangChain.

All model calls go through LangChain's ChatGoogleGenerativeAI, so swapping models
(or even providers) happens in one place — important for the cross-model judge in
proposal §4.5, and consistent with the RAGAS evaluator (which is also LangChain).
"""
from __future__ import annotations

import base64
import mimetypes
import random
import time
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from config.config import cfg

# Substrings that mark a transient / rate-limit error worth retrying.
_RETRYABLE = ("429", "rate", "quota", "resource_exhausted", "unavailable",
              "503", "500", "deadline", "timeout", "overloaded")


def _is_retryable(exc: Exception) -> bool:
    return any(tok in str(exc).lower() for tok in _RETRYABLE)


def with_retry(fn, *, max_attempts: int = 5, base_delay: float = 2.0):
    """Call `fn()` with exponential backoff + jitter on transient/rate-limit errors.

    A full benchmark makes ~100 x runs_per_case x 2 systems calls; without this the
    run dies the first time Gemini returns 429 (concern #6 in the proposal).
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — LangChain wraps a variety of error types
            if attempt == max_attempts or not _is_retryable(exc):
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
            print(f"  [retry {attempt}/{max_attempts} after {delay:.1f}s: {exc}]")
            time.sleep(delay)


def _image_data_url(image_path: str | Path) -> str:
    """Read an image off disk into a base64 data URL for a multimodal message part."""
    path = Path(image_path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


class LLMClient:
    def __init__(self, model: str, temperature: float = 0.2):
        self.model = model
        self.temperature = temperature
        self._llm = None  # lazy init so importing this file never needs an API key

    def _ensure_llm(self):
        if self._llm is None:
            cfg.validate()
            self._llm = ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=cfg.api_key,
                temperature=self.temperature,
            )
        return self._llm

    def generate(self, system_prompt: str, user_text: str,
                 image_path: str | Path | None = None) -> str:
        """Single (optionally multimodal) generation. Returns the model's text.

        Wrapped in exponential backoff so a long benchmark survives rate limits.
        """
        llm = self._ensure_llm()

        content: list = [{"type": "text", "text": user_text}]
        if image_path is not None:
            content.append({
                "type": "image_url",
                "image_url": {"url": _image_data_url(image_path)},
            })
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=content)]

        def _call():
            result = llm.invoke(messages).content
            # Gemini normally returns a string; coerce block-lists defensively.
            if isinstance(result, list):
                result = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in result
                )
            return result

        return with_retry(_call)


def load_prompt(filename: str) -> str:
    return (cfg.prompts_dir / filename).read_text()
