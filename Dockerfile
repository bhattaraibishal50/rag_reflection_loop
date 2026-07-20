# Single image used for the Streamlit demo AND one-off commands (ingest, benchmark).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# build-essential covers native wheels (chromadb, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install deps first so the layer caches unless requirements.txt changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Source is bind-mounted at runtime (see docker-compose) for live reload;
# this COPY just makes the image usable standalone too.
COPY . .

# Entrypoint auto-builds the index on first `up` so the demo is self-contained.
RUN chmod +x /app/docker-entrypoint.sh
ENTRYPOINT ["/app/docker-entrypoint.sh"]

EXPOSE 8501

# Default: run the demo with auto-rerun on file changes.
# poll watcher is used because inotify is unreliable on macOS/Windows bind mounts.
CMD ["streamlit", "run", "src/app/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.runOnSave=true", \
     "--server.fileWatcherType=poll"]
