"""02 XAI 분석: /counterfactual (보너스 과제 2, 거절 고객이 승인되려면 무엇이 바뀌어야 하나)"""
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.api import model
from src.api.routers.predict import require_model
from src.api.schemas import DISCLAIMER, CustomerInput

router = APIRouter(tags=['02 XAI 분석'])


class CounterfactualChange(BaseModel):
    feature: str
    feature_name_kr: str
    current_value: float = Field(description='현재 값 (unit 단위)')
    suggested_value: float = Field(description='제안 값 (unit 단위)')
    unit: str
    explanation: str
    new_probability: Optional[float] = Field(None, description='이 칸 하나만 바꿨을 때의 부도 확률 (single_changes 에서만)')


class CounterfactualPlan(BaseModel):
    success: bool = Field(description='바꿀 수 있는 칸 안에서 승인 조건을 찾았는가')
    changes: list[CounterfactualChange]
    new_probability: float
    new_credit_score: int


class CounterfactualResponse(BaseModel):
    customer_id: str
    approval_status: str
    default_probability: float
    approval_threshold: float
    already_approved: bool
    combined: Optional[CounterfactualPlan] = Field(None, description='여러 칸을 함께 바꾸는 가장 작은 제안 (이미 승인이면 null)')
    single_changes: list[CounterfactualChange] = Field(description='칸 하나만 바꿔도 승인되는 경우들 (작은 변화 순)')
    immutable_features: list[str] = Field(description='바꿀 수 없어 움직이지 않은 칸 (나이, 과거 연체 등)')
    method: str = '바꿀 수 있는 칸만, Train 1%~99% 분위수 안에서 탐색 (src/explainer/counterfactual.py)'
    disclaimer: str = DISCLAIMER


def _change(features, f, v, prob=None):
    from src.explainer import counterfactual as cf
    from src.explainer.feature_names import korean_name
    mult, unit = cf.DISPLAY_UNITS[f]
    return CounterfactualChange(
        feature=f, feature_name_kr=korean_name(f), current_value=round(features[f] * mult, 4),
        suggested_value=round(v * mult, 4), unit=unit, explanation=cf.describe(f, features[f], v, cf.DISPLAY_UNITS),
        new_probability=None if prob is None else round(prob, 4))


@router.post('/counterfactual', response_model=CounterfactualResponse,
             summary='반사실적 설명: 어떤 조건이 바뀌면 승인되나 (보너스)',
             description='거절된 고객에게 바꿀 수 있는 칸(한도 사용 비율, 부채 비율, 월소득, 대안 변수 5개)만 움직여 '
                         '승인 기준 아래로 내려가는 가장 작은 변화를 찾는다. 나이 · 과거 연체 횟수 등은 움직이지 않는다.',
             responses={503: {'description': '모델 파일 없음'}})
def counterfactual(customer: CustomerInput):
    require_model()
    from src.explainer import counterfactual as cf
    from src.explainer.feature_names import korean_name

    features = customer.features()
    predict_proba = model.batch_default_probability
    prob = model.default_probability(features)
    thr = model.APPROVAL_THRESHOLD
    approved = prob < thr
    combined, singles = None, []
    if not approved:
        comb = cf.combined_changes(features, predict_proba, thr)
        combined = CounterfactualPlan(
            success=comb['success'], changes=[_change(features, f, v) for f, v in comb['changes'].items()],
            new_probability=round(comb['new_probability'], 4), new_credit_score=model.credit_score(comb['new_probability']))
        found = [s for s in cf.single_changes(features, predict_proba, thr).values() if s]
        found.sort(key=lambda s: cf.normalized_distance(features, {s['feature']: s['suggested']}))
        singles = [_change(features, s['feature'], s['suggested'], s['new_probability']) for s in found]
    return CounterfactualResponse(
        customer_id=customer.customer_id, approval_status=model.approval_status(prob),
        default_probability=round(prob, 4), approval_threshold=thr, already_approved=approved,
        combined=combined, single_changes=singles, immutable_features=[korean_name(f) for f in cf.IMMUTABLE])
