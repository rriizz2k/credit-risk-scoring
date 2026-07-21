from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd

from credit_risk.features.prepare import add_engineered_features

model = joblib.load('model.pkl')
defaults = joblib.load('defaults.pkl')

app = FastAPI()


class LoanRequest(BaseModel):
    duration: Optional[int] = None
    credit_amount: Optional[float] = None
    installment_commitment: Optional[int] = None
    residence_since: Optional[int] = None
    age: Optional[int] = None
    existing_credits: Optional[int] = None
    num_dependents: Optional[int] = None

    checking_status: Optional[str] = None
    credit_history: Optional[str] = None
    purpose: Optional[str] = None
    savings_status: Optional[str] = None
    employment: Optional[str] = None
    personal_status: Optional[str] = None
    other_parties: Optional[str] = None
    property_magnitude: Optional[str] = None
    other_payment_plans: Optional[str] = None
    housing: Optional[str] = None
    job: Optional[str] = None
    own_telephone: Optional[str] = None
    foreign_worker: Optional[str] = None


@app.get("/")
def home():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: LoanRequest):
    if request.duration == 0:
        raise HTTPException(status_code=400, detail="duration must not be 0")

    provided = {k: v for k, v in request.dict().items() if v is not None}
    input_data = {**defaults, **provided}
    input_df = pd.DataFrame([input_data])
    input_df = add_engineered_features(input_df)

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1][0]

    return {"prediction": int(prediction[0]), "probability": float(probability)}