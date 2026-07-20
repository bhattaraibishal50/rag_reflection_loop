# Shortcuts for the Crop Reflection RAG pipeline.
#
# Local venv (default):   make ingest
# Through Docker instead:  make ingest PY="docker compose run --rm app python"
#
# Run `make help` to list commands.

PY ?= python

.PHONY: help prepare ingest retrieve check diagnose validate benchmark demo up build test

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

prepare:  ## Scaffold ground_truth.csv:  make prepare DATASET=/path/to/dataset
	$(PY) cli.py prepare-data $(DATASET) --per-class $(or $(PER_CLASS),20)

ingest:  ## Build the Chroma index from KB PDFs
	$(PY) cli.py ingest

retrieve:  ## Inspect retrieval:  make retrieve Q="early blight tomato"
	$(PY) cli.py retrieve "$(Q)"

check:  ## GATE: verify key + KB + index + images are ready
	$(PY) cli.py check

diagnose:  ## Diagnose one image:  make diagnose IMG=path.jpg SYS=b
	$(PY) cli.py diagnose $(IMG) --system $(or $(SYS),b)

validate:  ## GATE: Cohen's kappa of the LLM judge vs humans
	$(PY) cli.py validate-judge

benchmark:  ## Run the full A-vs-B benchmark + stats
	$(PY) cli.py benchmark

demo:  ## Launch the Streamlit app locally
	$(PY) cli.py demo

up:  ## Start the demo via Docker (http://localhost:8501)
	docker compose up app

build:  ## Build the Docker image
	docker compose build

test:  ## Byte-compile all Python (quick sanity check)
	$(PY) -m compileall -q cli.py config src eval
