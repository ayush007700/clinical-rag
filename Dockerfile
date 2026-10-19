FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY data ./data

RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    DATABASE_URL=sqlite:////app/data/fhir_store.db

EXPOSE 8000

CMD ["uvicorn", "clinical_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
