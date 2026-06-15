# requirements first -> Docker layer caching: code edits don't re-install deps
FROM python:3.11-slim

WORKDIR /app

# runtime-only deps (no matplotlib); robust to slow/flaky networks:
# long read timeout + many retries + prefer prebuilt wheels
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --timeout=120 --retries=10 --prefer-binary \
    -r requirements-docker.txt

COPY app/ .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
