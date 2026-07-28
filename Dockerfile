# Container image for QueryMancer.
#
# Not needed for Render's native Python runtime (see render.yaml) - this is
# for Hugging Face Spaces, Fly.io, or any host that wants a container.
FROM python:3.12-slim

# psycopg2-binary ships wheels, but libpq is still needed at runtime, and gcc
# covers any dependency without a wheel for this platform.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first so the dependency layer is cached across code edits.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user. Hugging Face Spaces expects uid 1000 specifically.
RUN useradd -m -u 1000 app && chown -R app:app /app
USER app

# Hugging Face Spaces routes to 7860; other hosts set $PORT. Honouring both
# means the same image works on either without an edit.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT} --workers 1"]
