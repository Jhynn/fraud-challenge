from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from src.inference import MultiModelPredictor

app = FastAPI(title="Sistema Multi-Modal de Detecção de Fraude")
predictor = MultiModelPredictor()

class TabularInput(BaseModel):
    valor: float
    mouse_speed: float

class GraphInput(BaseModel):
    features: List[float]
    vizinhos_ids: List[int]

class SequenceInput(BaseModel):
    trajetoria: List[List[float]] 

@app.post("/predict/random-forest")
def predict_rf(item: TabularInput):
    data = [item.valor, item.mouse_speed]
    return predictor.predict('rf', data)

@app.post("/predict/autoencoder")
def predict_ae(item: TabularInput):
    data = [item.valor, item.mouse_speed]
    return predictor.predict('autoencoder', data)

@app.post("/predict/isolation-forest")
def predict_iso(item: TabularInput):
    data = [item.valor, item.mouse_speed]
    return predictor.predict('iso_forest', data)

@app.post("/predict/lstm")
def predict_lstm(item: SequenceInput):
    if len(item.trajetoria) != 30:
        raise HTTPException(status_code=400, detail="A trajetória deve ter exatamente 30 pontos.")
    return predictor.predict('lstm', item.trajetoria)

@app.post("/predict/gcn")
def predict_gcn(item: GraphInput):
    return predictor.predict('gcn', item.dict())
