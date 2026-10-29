"""03 공정성: /fairness, /fairness/mitigation"""
from fastapi import APIRouter

from src.api import reports
from src.api.schemas import FairnessMitigationResponse, FairnessResponse

router = APIRouter(tags=['03 공정성'])


@router.get('/fairness', response_model=FairnessResponse, summary='공정성 지표 (DI Ratio · Equalized Odds)',
            description='성별·연령대별 승인율과 DI Ratio(0.8~1.25), Equalized Odds TPR/FPR 차이(≤ 0.10). '
                        '맨 위 di_ratio 는 두 속성 중 기준에서 가장 벗어난 값. 출처: outputs/fairness/fairness.json (main_step4).')
def fairness():
    return reports.load('fairness')


@router.get('/fairness/mitigation', response_model=FairnessMitigationResponse, summary='Bias Mitigation 전/후 비교',
            description='Threshold Adjustment(집단별 임계값, Validation 으로 결정) 적용 전/후의 공정성 지표. '
                        'auc/ks/approval_rate 는 파이프라인이 채우면 함께 나간다. 출처: outputs/fairness/fairness_mitigation.json.')
def fairness_mitigation():
    return reports.load('fairness_mitigation')
