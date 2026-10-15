"""01 고객 심사: /predict"""
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from src.api import model
from src.api.schemas import DISCLAIMER, CustomerInput, PredictResponse

router = APIRouter(tags=['01 고객 심사'])


def require_model():
    if not model.ready():
        raise HTTPException(503, f'모델 파일이 없습니다: {model.MODEL_PATH.name}, {model.SCALER_PATH.name} (models/README.md 참고)')


@router.post('/predict', response_model=PredictResponse, summary='신용점수·승인 여부 예측',
             description='입력 15개 변수를 스케일러로 맞춘 뒤 최종 모델로 부도 확률을 구하고, '
                         '신용점수(0~1000)·리스크 등급(A~E)·승인 여부·씬파일러 판정을 돌려준다.',
             responses={503: {'description': '모델 파일 없음'}})
def predict(customer: CustomerInput):
    require_model()
    features = customer.features()
    prob = model.default_probability(features)
    score = model.credit_score(prob)
    return PredictResponse(
        customer_id=customer.customer_id, credit_score=score, risk_grade=model.risk_grade(score),
        approval_status=model.approval_status(prob), default_probability=round(prob, 4),
        is_thin_filer=model.is_thin_filer(features), approval_threshold=model.APPROVAL_THRESHOLD,
        model_version=model.MODEL_VERSION, timestamp=datetime.now(timezone.utc).isoformat(), disclaimer=DISCLAIMER,
    )
