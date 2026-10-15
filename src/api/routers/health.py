"""시스템: /health"""
from datetime import datetime, timezone

from fastapi import APIRouter

from src.api import model, reports
from src.api.schemas import DISCLAIMER, HealthResponse

router = APIRouter(tags=['시스템'])


@router.get('/health', response_model=HealthResponse, summary='서버·모델 상태',
            description='사이드바 상태 표시용. `reports` 로 각 화면 데이터가 실제 산출물인지 내장 예시인지 알 수 있다.')
def health():
    ready = model.ready()
    test_auc = None
    try:
        test_auc = reports.model_metadata().get('selection', {}).get('test_auc')
    except (OSError, ValueError):
        pass
    return HealthResponse(
        status='healthy' if ready else 'degraded', model_ready=ready,
        model_version=model.MODEL_VERSION, model_stage=model.MODEL_STAGE, test_auc=test_auc,
        approval_threshold=model.APPROVAL_THRESHOLD,
        reports={name: reports.source(name) for name in reports.PATHS},
        timestamp=datetime.now(timezone.utc).isoformat(), disclaimer=DISCLAIMER,
    )
