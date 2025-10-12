FROM python:3.11-slim
WORKDIR /app
COPY worker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY worker /app
ENV PYTHONUNBUFFERED=1
ENV WORKER_MODE=loop
CMD ["python", "-m", "ingestion.cli"]
