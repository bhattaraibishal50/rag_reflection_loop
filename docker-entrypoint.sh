#!/usr/bin/env bash
# Makes `docker compose up app` self-contained: if the demo is starting and no
# Chroma index exists yet, build it once (needs KB PDFs + GEMINI_API_KEY). Any
# other command (e.g. `cli.py ingest`) is passed straight through untouched.
set -e

if [ "$1" = "streamlit" ]; then
  if [ ! -d /app/chroma_db ] && ls /app/data/knowledge_base/*.pdf >/dev/null 2>&1 \
     && [ -n "$GEMINI_API_KEY" ]; then
    echo "[entrypoint] No index found — building it once with 'cli.py ingest'..."
    python cli.py ingest || echo "[entrypoint] ingest failed; app starts but diagnosis needs an index."
  else
    echo "[entrypoint] Skipping auto-ingest (index exists, no PDFs, or no API key)."
  fi
fi

exec "$@"
