# predict.py
import pickle

from fastapi import FastAPI
from pydantic import BaseModel

MODEL_FILE = "model.bin"

with open(MODEL_FILE, "rb") as f_in:
    dv, model = pickle.load(f_in)

app = FastAPI(title="hotel-cancellation")


class Booking(BaseModel):
    type_of_meal: str
    room_type: str
    market_segment_type: str
    res_month: str
    number_of_adults: int
    number_of_children: int
    number_of_weekend_nights: int
    number_of_week_nights: int
    car_parking_space: int
    lead_time: int
    repeated: int
    p_c: int
    p_not_c: int
    average_price: float
    special_requests: int


class Prediction(BaseModel):
    cancel_probability: float
    cancel: bool


@app.post("/predict", response_model=Prediction)
def predict(booking: Booking):
    X = dv.transform([booking.model_dump()])
    prob = float(model.predict_proba(X)[0, 1])
    return Prediction(cancel_probability=prob, cancel=prob >= 0.5)
