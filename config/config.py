"""Central configuration. Import `cfg` from here everywhere — never hardcode paths/models."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    # --- Paths ---
    root: Path = ROOT
    images_dir: Path = ROOT / "data" / "images"
    ground_truth_csv: Path = ROOT / "data" / "ground_truth.csv"
    kb_dir: Path = ROOT / "data" / "knowledge_base"
    chroma_dir: Path = ROOT / "chroma_db"
    results_dir: Path = ROOT / "eval" / "results"
    prompts_dir: Path = ROOT / "prompts"

    # --- Models (swappable via env) ---
    actor_model: str = os.getenv("ACTOR_MODEL", "gemini-2.5-pro")
    critic_model: str = os.getenv("CRITIC_MODEL", "gemini-2.5-flash")
    # Judge SHOULD differ from Actor/Critic family to reduce self-bias (see proposal 3.1).
    judge_model: str = os.getenv("JUDGE_MODEL", "gemini-2.5-flash")
    # gemini-embedding-001 is the Gemini Developer API (AI Studio key) embedding model.
    # (text-embedding-005 is a Vertex-only name; use it only if you switch to Vertex.)
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

    api_key: str = os.getenv("GEMINI_API_KEY", "")

    # --- RAG ---
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 5
    collection_name: str = "agronomy_kb"

    # --- Reflection loop ---
    max_iterations: int = 3          # hard cap: prevents over-correction / non-termination
    actor_temperature: float = 0.2   # low → reproducible diagnoses
    critic_temperature: float = 0.7  # higher → divergent, skeptical contradiction-finding

    # --- Benchmark ---
    runs_per_case: int = 3           # repeat each case; report mean ± variance (non-determinism)
    adversarial_fraction: float = 0.2  # 20% cross-domain queries (RQ3)

    def validate(self) -> None:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set. Copy .env.example to .env and fill it in.")


cfg = Config()
