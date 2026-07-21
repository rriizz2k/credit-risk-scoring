FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir --timeout 120 -e . --no-deps || true
RUN pip install --no-cache-dir --timeout 120 lightgbm mlflow numpy pandas scikit-learn fastapi uvicorn pydantic joblib
RUN pip install --no-cache-dir --timeout 120 shap

ENV PYTHONPATH=/app/src

CMD ["uvicorn", "credit_risk.api.main:app", "--host", "0.0.0.0", "--port", "8000"]