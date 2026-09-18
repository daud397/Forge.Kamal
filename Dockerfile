# Flat layout: every module sits at the repo root, so GitHub's web uploader
# can't lose the directory structure.
FROM python:3.12-slim

# trimesh needs these to read GLB and 3MF. STL works without them; the others
# fail at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so editing a module doesn't reinstall scipy every build.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./
COPY index.html styles.css app.js ./

# Mount a persistent volume here or every job is lost on redeploy.
RUN mkdir -p /app/data
VOLUME ["/app/data"]

ENV PORT=8000
EXPOSE 8000

# Shell form so $PORT expands.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
