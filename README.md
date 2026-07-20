# Crop Reflection RAG

**Evaluating Multi-Agent Reflection Loops for Reducing Hallucinations in Multimodal Agricultural Diagnostics**

This project benchmarks two diagnostic pipelines on visually ambiguous crop-disease images:

- **System A (baseline):** single-pass Retrieval-Augmented Generation (RAG).
- **System B (experimental):** a multi-agent **Actor–Critic reflection loop** built with LangGraph.

The goal is to measure whether the reflection loop reduces hallucinations (RQ1), at what
latency cost (RQ2), and whether it correctly rejects out-of-scope queries (RQ3).

---

## Architecture

```
Image + query
   ├──────────────► System A: retrieve → diagnose (one pass)
   │
   └──────────────► System B: ACTOR (retrieve+diagnose)
                              → CRITIC (find modality contradictions)
                              → refine & loop (max 3) → final diagnosis
```

Both systems share the **same Actor model, base prompt, retriever, and knowledge base**.
The *only* difference is the Critic + reflection loop — so the comparison isolates the
effect of reflection (see "Confounds" in the proposal).

---

## Project layout

```
.
├── cli.py                  # ← single entry point for every step
├── Makefile                # ← friendly shortcuts (make ingest, make benchmark, …)
├── config/                 # central configuration (models, paths, loop cap)
├── data/
│   ├── images/             # 100 adversarial test images
│   ├── ground_truth.csv    # image filename → verified disease label
│   └── knowledge_base/     # FAO / agronomy PDFs for the RAG corpus
├── prompts/                # Actor, Critic system prompts + judge rubric
├── src/
│   ├── llm/                # Gemini client + embeddings (via LangChain, swappable)
│   ├── rag/                # ingest.py · retriever.py · prepare_ground_truth.py
│   ├── systems/
│   │   ├── baseline.py     # System A — single-pass RAG
│   │   └── reflection/     # System B — LangGraph Actor–Critic loop
│   └── app/                # Streamlit demo
├── eval/
│   ├── preflight.py        # readiness gate before a paid run
│   ├── run_benchmark.py    # run all cases through A and B + paired stats
│   ├── metrics.py          # hallucination rate, faithfulness, latency
│   └── judge_validation.py # Cohen's kappa: LLM judge vs humans
└── notebooks/
```

---

## Setup

### Option A — Docker (recommended; live reload)

```bash
cp .env.example .env          # add your GEMINI_API_KEY
docker compose up app         # demo at http://localhost:8501, auto-reloads on edits
```

Run the pipeline steps in the same image (prefix any `cli.py` command):

```bash
docker compose run --rm app python cli.py ingest
docker compose run --rm app python cli.py benchmark
# or:  make ingest PY="docker compose run --rm app python"
```

The project is bind-mounted into the container, so any source edit on the host
immediately re-runs the Streamlit app (`--server.runOnSave`, poll watcher).

### Option B — local venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then add your GEMINI_API_KEY
```

## Usage — one command surface

Every step runs through `cli.py` (or the matching `make` target):

```bash
python cli.py --help            # list all commands
python cli.py <command> [args]
```

## Build order (validate each step before the next)

| Step | Command | Make | Gate |
|---|---|---|---|
| 0. Scaffold labels from a dataset folder | `python cli.py prepare-data <dataset>` | `make prepare DATASET=…` | |
| 1. Add PDFs to `data/knowledge_base/`, finish `ground_truth.csv` | — | — | |
| 2. Build the index | `python cli.py ingest` | `make ingest` | |
| 3. Inspect retrieval | `python cli.py retrieve "query"` | `make retrieve Q=…` | **confirm good passages** |
| 4. Readiness check | `python cli.py check` | `make check` | **key + KB + index + images** |
| 5. One diagnosis | `python cli.py diagnose <img> --system a\|b` | `make diagnose IMG=… SYS=b` | |
| 6. Validate the judge | `python cli.py validate-judge` | `make validate` | **Cohen's κ ≥ 0.6** |
| 7. Full benchmark | `python cli.py benchmark` | `make benchmark` | |
| 8. Demo | `python cli.py demo` | `make demo` / `make up` | |

## Status

Scaffold complete and hardened for a real run: all API calls retry with backoff,
the benchmark reports paired significance tests (McNemar) and bootstrap CIs, and
`eval/preflight.py` checks readiness before you spend API budget. Still required
before results exist: a `GEMINI_API_KEY`, the KB PDFs, and the labelled image set.
Report results tables in `REPORT.md` stay as `[TBD]` placeholders until the run fills them.
