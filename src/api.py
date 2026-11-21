from functools import lru_cache
from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title='Mision 11 Iris API')

class Features(BaseModel):
    features: list[float] = Field(min_length=4, max_length=4)

@lru_cache
def get_model():
    path = Path(__file__).resolve().parent.parent / 'models' / 'model.joblib'
    if not path.exists():
        raise FileNotFoundError('Entrena primero con python src/train.py')
    return joblib.load(path)

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/predict')
def predict(payload: Features):
    try:
        prediction = get_model().predict([payload.features])[0]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {'class_id': int(prediction)}
