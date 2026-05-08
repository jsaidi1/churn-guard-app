FROM python:3.11-slim AS builder

WORKDIR /app

RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    mlflow \
    pandas \
    scikit-learn \
    pydantic \
    requests

## Fix vulnérabilités Trivy
RUN pip install --no-cache-dir --upgrade \
    wheel>=0.46.2 \
    jaraco.context>=6.1.0

COPY . .


FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

RUN adduser --disabled-password --gecos "" appuser

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/api /app/api
COPY --from=builder /app/churnguard /app/churnguard
COPY --from=builder /app/scripts /app/scripts

RUN mkdir -p /app/data/raw && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

