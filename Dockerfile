FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "model_card_auditor.server.app:app", \
     "--host", "0.0.0.0", "--port", "7860", "--workers", "2"]
