"""FastAPI 신용평가 API"""
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, json
from pathlib import Path
from datetime import datetime
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
DATA_SPLITS = PROJECT_ROOT / 'data' / 'splits'

app = FastAPI(title="신용평가 API", version="1.0.0")

try:
    best_model = joblib.load(MODELS_DIR / 'best_model_xgboost_v1.0.pkl')
    with open(MODELS_DIR / 'model_metadata.json') as f:
        model_metadata = json.load(f)
    MODEL_READY = True
except:
    MODEL_READY = False
    model_metadata = {}

@app.get("/health")
async def health():
    return {"status": "healthy" if MODEL_READY else "error", "model_ready": MODEL_READY}

@app.get("/metrics")
async def metrics():
    if not MODEL_READY: return {"error": "Model not ready"}
    test_metrics = model_metadata.get('test_metrics', {})
    return {
        "auc": test_metrics.get('auc', 0),
        "ks": test_metrics.get('ks', 0),
        "f1": test_metrics.get('f1', 0)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
