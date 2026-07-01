
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import numpy as np
import os

app = FastAPI(title="Delivery Time Prediction API", version="1.0")

# Загрузка модели
def load_artifacts():
    model = pickle.load(open('models/model.pkl', 'rb'))
    scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    le = pickle.load(open('models/label_encoder.pkl', 'rb'))
    return model, scaler, le

model, scaler, le = load_artifacts()

# Модель запроса
class PredictionRequest(BaseModel):
    store_id: str
    items_count: int = Field(..., ge=1, le=20)
    order_price: float = Field(..., ge=10, le=10000)
    delivery_distance: float = Field(..., ge=0.1, le=50)
    planned_prep_time: int = Field(..., ge=5, le=120)
    hour: int = Field(..., ge=0, le=23)
    is_weekend: int = Field(..., ge=0, le=1)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Кодирование
        store_encoded = le.transform([request.store_id])[0]
        
        # Подготовка данных
        input_data = np.array([[
            store_encoded, request.items_count, request.order_price,
            request.delivery_distance, request.planned_prep_time,
            request.hour, request.is_weekend
        ]])
        
        # Масштабирование и предсказание
        input_scaled = scaler.transform(input_data)
        prob = model.predict_proba(input_scaled)[0][1]
        pred_class = int(model.predict(input_scaled)[0])
        
        return {
            "probability_on_time": round(prob, 4),
            "class_prediction": pred_class,
            "status": "Доставка вовремя" if pred_class == 1 else "Ожидается задержка"
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/")
async def root():
    return {"message": "Delivery Time Prediction API", "docs": "/docs"}
