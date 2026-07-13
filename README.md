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
├── config/                 # central configuration (models, paths, loop cap)
├── data/
│   ├── images/             # 100 adversarial test images
│   ├── ground_truth.csv    # image filename → verified disease label
│   └── knowledge_base/     # FAO / agronomy PDFs for the RAG corpus
├── prompts/                # Actor, Critic system prompts + judge rubric
├── src/
│   ├── ingest/             # PDF → chunk → embed → Chroma
│   ├── retrieval/          # vector search wrapper
│   ├── llm/                # model client (swappable)
│   ├── system_a_baseline.py
│   ├── system_b_reflection/   # LangGraph Actor–Critic graph
│   └── app/                # FastAPI + Streamlit demo
├── eval/
│   ├── run_benchmark.py    # run all cases through A and B
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

Run the pipeline steps in the same image:

```bash
docker compose run --rm app python -m src.ingest.ingest
docker compose run --rm app python -m src.retrieval.retriever "early blight tomato"
docker compose run --rm app python -m eval.run_benchmark
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

## Build order (validate each step before the next)

1. Add images + fill `data/ground_truth.csv`; drop PDFs in `data/knowledge_base/`.
   - Have a class-folder dataset (PlantVillage/Tomato)? Scaffold the CSV automatically:
     `python -m src.ingest.prepare_ground_truth /path/to/dataset --per-class 20`
     then hand-edit to keep ambiguous pairs + add cross-domain `REJECT` rows.
2. `python -m src.ingest.ingest`            → build the Chroma index.
3. `python -m src.retrieval.retriever "test query"`  → **GATE: confirm good passages.**
4. `python -m eval.preflight`               → **GATE: key + KB + index + images all ready.**
5. `python -m src.system_a_baseline <image>` → baseline runs end-to-end.
6. `python -m src.system_b_reflection.graph <image>` → loop runs and reflects.
7. `python -m eval.judge_validation`         → **GATE: Cohen's kappa ≥ 0.6.**
8. `python -m eval.run_benchmark`            → full results + paired McNemar / bootstrap CIs.
9. `streamlit run src/app/app.py`            → demo.

## Status

Scaffold complete and hardened for a real run: all API calls retry with backoff,
the benchmark reports paired significance tests (McNemar) and bootstrap CIs, and
`eval/preflight.py` checks readiness before you spend API budget. Still required
before results exist: a `GEMINI_API_KEY`, the KB PDFs, and the labelled image set.
Report results tables in `REPORT.md` stay as `[TBD]` placeholders until the run fills them.
