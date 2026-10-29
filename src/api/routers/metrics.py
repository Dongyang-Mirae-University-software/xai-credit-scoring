"""04 모델 성능: /metrics, /metrics/comparison"""
from fastapi import APIRouter

from src.api import reports
from src.api.schemas import DISCLAIMER, MetricsResponse, ModelComparisonResponse

router = APIRouter(tags=['04 모델 성능'])


@router.get('/metrics', response_model=MetricsResponse, response_model_by_alias=True,
            summary='최종 모델 Test 성능',
            description='AUC·KS·PSI·Precision/Recall/F1, 혼동행렬, ROC·PR 곡선 점, 씬파일러 세그먼트 AUC 향상. '
                        '출처: outputs/evaluation/metrics_<모델>.json (main_step3).')
def metrics():
    return reports.load('metrics')


@router.get('/metrics/comparison', response_model=ModelComparisonResponse, summary='후보 모델 격자 비교',
            description='데이터 묶음(전통/대안/통합) × 알고리즘 × 불균형 기법별 Validation AUC/KS 와 5-Fold CV 평균, 최종 선택 결과. '
                        '출처: models/model_metadata.json (main_step2).')
def comparison():
    meta = reports.model_metadata()
    return {**meta.get('selection', {}), 'cv_runs': meta.get('training_runs', {}).get('runs', {}), 'disclaimer': DISCLAIMER}
