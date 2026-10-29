"""05 통계 검증: /anova"""
from fastapi import APIRouter

from src.api import reports
from src.api.schemas import AnovaResponse

router = APIRouter(tags=['05 통계 검증'])


@router.get('/anova', response_model=AnovaResponse, summary='One-way ANOVA · Tukey HSD',
            description='test_1: 피처 그룹별(전통/대안/통합) AUC 비교, test_2: 알고리즘별(LR/XGBoost/LightGBM) 비교. '
                        'F, p, η², 집단별 평균·표준편차·95% CI, 유의하면 Tukey HSD. 출처: outputs/statistics/anova.json (main_step5).')
def anova():
    return reports.load('anova')
