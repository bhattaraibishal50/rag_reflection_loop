"""Thin, swappable wrapper around the Gemini multimodal API.

Keeping all model calls behind this class means you can swap models (or even providers)
in one place — important for the cross-model judge in proposal §3.1.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from config.config import cfg


class LLMClient:
    def __init__(self, model: str, temperature: float = 0.2):
        self.model = model
        self.temperature = temperature
        self._client = None  # lazy init so importing this file never needs an API key

    def _ensure_client(self):
        if self._client is None:
            cfg.validate()
            # New unified SDK: `pip install google-genai`
            from google import genai

            self._client = genai.Client(api_key=cfg.api_key)
        return self._client

    def generate(self, system_prompt: str, user_text: str,
                 image_path: str | Path | None = None) -> str:
        """Single multimodal generation. Returns the model's text response.

        TODO: add retry/backoff for rate limits (concern #6 in the proposal).
        """
        client = self._ensure_client()
        from google.genai import types

        parts: list = [user_text]
        if image_path is not None:
            parts.append(Image.open(image_path))

        resp = client.models.generate_content(
            model=self.model,
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.temperature,
            ),
        )
        return resp.text


def load_prompt(filename: str) -> str:
    return (cfg.prompts_dir / filename).read_text()
