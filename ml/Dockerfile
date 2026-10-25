FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[mlflow]"

# Run the model comparison (logged to MLflow) and export demo data at build time. Takes a few minutes.
# The service itself needs no trained model: it fits the detector on each company's own history.
ENV MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db MLFLOW_DISABLE_AGENT_HINT=1
RUN python -m journal_anomaly_ml.evaluate --cache /tmp/features --out /app/results/evaluation.json \
    && rm -rf /tmp/features \
    && python -m journal_anomaly_ml.export_demo --out /app/export/demo.json

EXPOSE 8000
CMD ["uvicorn", "journal_anomaly_ml.service:app", "--host", "0.0.0.0", "--port", "8000"]
