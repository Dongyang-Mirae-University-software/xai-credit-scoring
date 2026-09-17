"""
FastAPI 서버 - Mock 버전.

실제 모델 학습·XAI·공정성 계산이 끝나기 전까지, 프론트엔드(Streamlit)가
인터페이스 스펙에 맞춰 개발을 진행할 수 있도록 규칙 기반 가짜(mock) 응답을
반환한다. 엔드포인트 경로와 요청/응답 스키마는 실제 구현과 동일하게 맞춰뒀으므로,
추후 _mock_* 함수들만 실제 모델/SHAP/Fairlearn 호출로 교체하면 됨.

엔드포인트: /health, /predict, /explain, /fairness, /fairness/mitigation,
/metrics, /anova
"""
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="XAI 기반 대안신용평가 시스템 (Mock)")

MODEL_VERSION = "mock-v0.1"

FEATURE_NAME_KR = {
    "telecom_payment_rate": "통신비 정상납부율",
    "utility_payment_rate": "공과금 납부율",
    "spending_consistency": "소비 일관성 점수",
    "regular_payment_count": "정기결제 건수",
    "app_login_frequency": "앱 로그인 빈도",
    "revolving_utilization": "신용한도 소진율",
    "debt_ratio": "부채비율",
    "number_of_times_90_days_late": "90일 이상 연체 횟수",
    "number_of_time_30_59_days_past_due": "30~59일 연체 횟수",
    "number_of_time_60_89_days_past_due": "60~89일 연체 횟수",
    "monthly_income": "월 소득",
    "age": "나이",
}

# 참조값(동일 연령대 평균 등). 실제로는 Train 데이터에서 집계해야 함 - 지금은 목값.
REFERENCE_VALUES = {
    "telecom_payment_rate": 0.92,
    "utility_payment_rate": 0.90,
    "spending_consistency": 70.0,
    "number_of_times_90_days_late": 0,
    "number_of_time_30_59_days_past_due": 0,
    "debt_ratio": 0.35,
    "revolving_utilization": 0.3,
}


class CustomerInput(BaseModel):
    customer_id: Optional[str] = "C000000"
    age: int = Field(..., ge=18, le=100)
    monthly_income: float = Field(..., ge=0)
    debt_ratio: float = Field(..., ge=0)
    revolving_utilization: float = Field(..., ge=0)
    number_of_time_30_59_days_past_due: int = Field(0, ge=0)
    number_of_open_credit_lines_and_loans: int = Field(0, ge=0)
    number_of_times_90_days_late: int = Field(0, ge=0)
    number_real_estate_loans_or_lines: int = Field(0, ge=0)
    number_of_time_60_89_days_past_due: int = Field(0, ge=0)
    number_of_dependents: int = Field(0, ge=0)
    telecom_payment_rate: float = Field(..., ge=0, le=1)
    utility_payment_rate: float = Field(..., ge=0, le=1)
    spending_consistency: float = Field(..., ge=0, le=100)
    regular_payment_count: int = Field(0, ge=0, le=20)
    app_login_frequency: int = Field(0, ge=0, le=30)


class PredictResponse(BaseModel):
    customer_id: str
    credit_score: int
    risk_grade: str
    approval_status: str
    default_probability: float
    model_version: str
    timestamp: str


class RejectionReason(BaseModel):
    rank: int
    feature: str
    feature_name_kr: str
    current_value: float
    reference_value: float
    shap_contribution: float
    explanation: str


class ExplainResponse(BaseModel):
    customer_id: str
    approval_status: str
    rejection_reasons: list[RejectionReason]
    shap_values: dict
    total_rejection_factors: int


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mock_default_probability(c: CustomerInput) -> float:
    """실제 모델 대신 규칙 기반으로 그럴듯한 확률을 계산하는 mock 함수."""
    score = 0.0
    score += 0.08 * c.number_of_times_90_days_late
    score += 0.04 * c.number_of_time_60_89_days_past_due
    score += 0.03 * c.number_of_time_30_59_days_past_due
    score += 0.15 * max(c.debt_ratio - 0.4, 0)
    score += 0.10 * max(c.revolving_utilization - 0.5, 0)
    score -= 0.20 * max(c.telecom_payment_rate - 0.7, 0)
    score -= 0.15 * max(c.utility_payment_rate - 0.7, 0)
    score += 0.05 if c.monthly_income < 2_000_000 else 0
    prob = 1 / (1 + np.exp(-4 * (score - 0.3)))
    return float(np.clip(prob, 0.01, 0.99))


def _risk_grade(prob: float) -> str:
    if prob < 0.05:
        return "A"
    if prob < 0.15:
        return "B"
    if prob < 0.30:
        return "C"
    if prob < 0.50:
        return "D"
    return "E"


@app.get("/health")
def health():
    return {"status": "healthy", "model_version": MODEL_VERSION, "timestamp": _now()}


@app.post("/predict", response_model=PredictResponse)
def predict(customer: CustomerInput):
    prob = _mock_default_probability(customer)
    credit_score = int(round((1 - prob) * 1000))
    approval_status = "APPROVED" if prob < 0.3 else "REJECTED"
    return PredictResponse(
        customer_id=customer.customer_id,
        credit_score=credit_score,
        risk_grade=_risk_grade(prob),
        approval_status=approval_status,
        default_probability=round(prob, 4),
        model_version=MODEL_VERSION,
        timestamp=_now(),
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(customer: CustomerInput):
    prob = _mock_default_probability(customer)
    approval_status = "APPROVED" if prob < 0.3 else "REJECTED"

    candidates = []
    for feat in [
        "telecom_payment_rate", "utility_payment_rate",
        "number_of_times_90_days_late", "debt_ratio",
        "revolving_utilization", "number_of_time_30_59_days_past_due",
    ]:
        current = getattr(customer, feat)
        ref = REFERENCE_VALUES.get(feat, current)
        diff = current - ref
        direction = -1 if feat in ("telecom_payment_rate", "utility_payment_rate") else 1
        contribution = round(direction * diff * 0.3, 4)
        candidates.append((feat, current, ref, contribution))

    candidates.sort(key=lambda x: abs(x[3]), reverse=True)
    top5 = candidates[:5]

    reasons = []
    for i, (feat, current, ref, contribution) in enumerate(top5, start=1):
        name_kr = FEATURE_NAME_KR.get(feat, feat)
        if contribution < 0:
            explanation = f"{name_kr}이(가) {current}로, 평균({ref}) 대비 양호합니다 (기여도: {contribution})"
        else:
            explanation = f"{name_kr}이(가) {current}로, 평균({ref}) 대비 불리합니다 (기여도: {contribution})"
        reasons.append(RejectionReason(
            rank=i, feature=feat, feature_name_kr=name_kr,
            current_value=current, reference_value=ref,
            shap_contribution=contribution, explanation=explanation,
        ))

    shap_values = {feat: contribution for feat, _, _, contribution in candidates}

    return ExplainResponse(
        customer_id=customer.customer_id,
        approval_status=approval_status,
        rejection_reasons=reasons,
        shap_values=shap_values,
        total_rejection_factors=len(reasons),
    )


@app.get("/fairness")
def fairness():
    return {
        "di_ratio": 0.83,
        "equalized_odds_tpr_diff": 0.07,
        "equalized_odds_fpr_diff": 0.05,
    }


@app.get("/fairness/mitigation")
def fairness_mitigation():
    return {
        "before": {"di_ratio": 0.74, "equalized_odds_tpr_diff": 0.12, "equalized_odds_fpr_diff": 0.09},
        "after": {"di_ratio": 0.88, "equalized_odds_tpr_diff": 0.06, "equalized_odds_fpr_diff": 0.04},
    }


@app.get("/metrics")
def metrics():
    return {"auc": 0.81, "ks": 0.32, "precision": 0.68, "recall": 0.61, "f1": 0.64}


@app.get("/anova")
def anova():
    return {
        "test_1": {
            "test_name": "피처 그룹별 AUC 비교 (One-way ANOVA)",
            "groups": ["전통모델", "대안모델", "통합모델"],
            "group_means": {"전통모델": 0.72, "대안모델": 0.68, "통합모델": 0.79},
            "f_statistic": 12.45,
            "p_value": 0.00023,
            "effect_size_eta_squared": 0.34,
            "is_significant": True,
            "post_hoc_tukey": [
                {"comparison": "전통 vs 대안", "mean_diff": 0.04, "p_adj": 0.18, "significant": False},
                {"comparison": "전통 vs 통합", "mean_diff": -0.07, "p_adj": 0.002, "significant": True},
                {"comparison": "대안 vs 통합", "mean_diff": -0.11, "p_adj": 0.0004, "significant": True},
            ],
        },
        "test_2": {
            "test_name": "알고리즘별 AUC 비교 (One-way ANOVA)",
            "groups": ["LogisticRegression", "XGBoost", "LightGBM"],
            "group_means": {"LogisticRegression": 0.71, "XGBoost": 0.79, "LightGBM": 0.78},
            "f_statistic": 8.92,
            "p_value": 0.0012,
            "effect_size_eta_squared": 0.28,
            "is_significant": True,
        },
    }
