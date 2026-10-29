"""02 XAI 분석: /explain, /explain/summary"""
from fastapi import APIRouter

from src.api import model, reports
from src.api.routers.predict import require_model
from src.api.schemas import DISCLAIMER, CustomerInput, ExplainResponse, ShapSummaryResponse

router = APIRouter(tags=['02 XAI 분석'])


@router.post('/explain', response_model=ExplainResponse, summary='개별 고객 SHAP 설명 · 거절 사유 Top 5',
             description='Waterfall 용 변수별 SHAP 값(log-odds, 양수 = 부도 위험 증가)과 기준값 E[f(x)], '
                         '승인에 불리한 변수 Top 5 의 고객 고지용 문장을 돌려준다. '
                         '기준값(동일 연령대 평균)은 outputs/evaluation/reference_values.json 이 있을 때만 채워진다.',
             responses={503: {'description': '모델 파일 없음'}})
def explain(customer: CustomerInput):
    require_model()
    features = customer.features()
    prob = model.default_probability(features)
    result = model.explain(features, references=reports.load('reference_values', fallback=False))
    return ExplainResponse(
        customer_id=customer.customer_id, approval_status=model.approval_status(prob),
        default_probability=round(prob, 4), base_value=result['base_value'],
        rejection_reasons=result['rejection_reasons'],
        shap_values={k: round(v, 4) for k, v in result['shap_values'].items()},
        total_rejection_factors=len(result['rejection_reasons']), disclaimer=DISCLAIMER,
    )


@router.get('/explain/summary', response_model=ShapSummaryResponse, summary='SHAP 전역 중요도 (Summary)',
            description='검증 표본에서 계산한 변수별 mean |SHAP| 상위 목록. 전통/대안 구분 포함. '
                        '출처: outputs/evaluation/shap_summary.json (main_step3).')
def shap_summary():
    return reports.load('shap_summary')
