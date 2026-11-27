"""05 통계 검증: /anova"""
from fastapi import APIRouter

from src.api import reports
from src.api.schemas import AnovaResponse, TwoWayAnovaResponse

router = APIRouter(tags=['05 통계 검증'])


@router.get('/anova', response_model=AnovaResponse, summary='One-way ANOVA · Tukey HSD',
            description='test_1: 피처 그룹별(전통/대안/통합) AUC 비교, test_2: 알고리즘별(LR/XGBoost/LightGBM) 비교. '
                        'F, p, η², 집단별 평균·표준편차·95% CI, 유의하면 Tukey HSD. 출처: outputs/statistics/anova.json (main_step5).')
def anova():
    return reports.load('anova')


@router.get('/anova/two-way', response_model=TwoWayAnovaResponse, summary='Two-way ANOVA (세그먼트 × 모델 구성)',
            description='고객 세그먼트(씬파일러/일반) × 모델 구성(전통/통합) fold AUC 의 Two-way ANOVA, 단순 주효과, 상호작용. '
                        'fold 별 원자료(records)는 뺀다. 출처: outputs/statistics/two_way_anova.json (main_bonus_two_way_anova).')
def two_way_anova():
    report = reports.load('two_way_anova')
    report.pop('records', None)
    return report
