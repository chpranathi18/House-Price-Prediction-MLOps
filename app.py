import pandas as pd
import joblib
import time

from fastapi import FastAPI
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

app = FastAPI(title="House Price Prediction API")

# Load trained model
model = joblib.load("models/house_price_model.pkl")

# Prometheus metrics
REQUEST_COUNT = Counter(
    "house_price_prediction_requests_total",
    "Total prediction requests"
)

REQUEST_TIME = Histogram(
    "house_price_prediction_duration_seconds",
    "Prediction response time"
)


# Input data
class HouseData(BaseModel):
    area_sqft: float = Field(gt=0)
    bedrooms: int = Field(gt=0)
    bathrooms: int = Field(gt=0)
    property_age_years: float = Field(ge=0)
    location: str
    furnishing_status: str
    parking_spaces: int = Field(ge=0)


@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict_price(data: HouseData):

    start_time = time.time()

    input_data = pd.DataFrame([{
        "Area_sqft": data.area_sqft,
        "Bedrooms": data.bedrooms,
        "Bathrooms": data.bathrooms,
        "Property_Age_years": data.property_age_years,
        "Location": data.location,
        "Furnishing_Status": data.furnishing_status,
        "Parking_Spaces": data.parking_spaces
    }])

    prediction = model.predict(input_data)[0]

    REQUEST_COUNT.inc()
    REQUEST_TIME.observe(time.time() - start_time)

    return {
        "predicted_price_in_lakhs": round(float(prediction), 2)
    }


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )