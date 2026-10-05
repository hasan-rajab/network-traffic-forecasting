FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    NETWORK_ARTIFACT_DIR=/app/artifacts
WORKDIR /app
COPY requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements-web.txt \
    && useradd --create-home --uid 10001 telecom
COPY --chown=telecom:telecom . .
USER telecom
EXPOSE 8501
CMD ["python", "scripts/serve.py"]
