"""
FastAPI 서버.
엔드포인트: /health, /predict, /explain, /fairness,
/fairness/mitigation, /metrics, /anova
"""
from fastapi import FastAPI

app = FastAPI(title="XAI 기반 대안신용평가 시스템")


@app.get("/health")
def health():
    raise NotImplementedError
