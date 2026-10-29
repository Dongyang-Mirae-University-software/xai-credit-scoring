"""
API 요청/응답 스키마. Swagger(/docs)의 예시는 examples/*.json 과 아래 상수에서 온다.

- 입력 변수 이름은 docs/column_spec.md 와 모델 학습 칸 이름(src/explainer/feature_names.py)을 그대로 쓴다.
- 리포트 응답(/metrics, /fairness, /anova ...)은 파이프라인(main_step3~5)이 저장하는 JSON 과 같은 키이며,
  extra='allow' 라 파이프라인이 키를 더 넣어도 그대로 내보낸다.
"""
import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from src.explainer.performance import DISCLAIMER

EXAMPLES_DIR = Path(__file__).parent / 'examples'


def example(name):
    return json.loads((EXAMPLES_DIR / f'{name}.json').read_text(encoding='utf-8'))


def _with_example(data):
    return ConfigDict(extra='allow', json_schema_extra={'examples': [data]})


# ---------- 01 고객 심사 ----------
CUSTOMER_EXAMPLE = {
    "customer_id": "C-10482",
    "RevolvingUtilizationOfUnsecuredLines": 0.48, "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 1, "DebtRatio": 32, "MonthlyIncome": 4200000,
    "NumberOfOpenCreditLinesAndLoans": 6, "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1, "NumberOfTime60-89DaysPastDueNotWorse": 0, "NumberOfDependents": 2,
    "telecom_payment_rate": 90, "utility_payment_rate": 93, "spending_consistency": 75.0,
    "regular_payment_count": 5, "app_login_frequency": 22,
}


# 화면·API 는 % 로 받고 모델은 0~1 을 쓰는 칸 (이슈 #70)
PERCENT_FIELDS = ('DebtRatio', 'telecom_payment_rate', 'utility_payment_rate')


class CustomerInput(BaseModel):
    """고객 1명의 입력 (전통 10개 + 대안 5개). 성별·연령대·민감정보는 받지 않는다.

    단위는 docs/column_spec.md §9 (이슈 #70): 부채비율·납부율은 % 로 받고 서버가 ÷100 해서 모델에 넣는다."""
    model_config = ConfigDict(populate_by_name=True, json_schema_extra={'examples': [CUSTOMER_EXAMPLE]})

    customer_id: str = Field('C000000', description='고객 번호 (표시용)')
    # 전통 금융 데이터 (Give Me Some Credit)
    RevolvingUtilizationOfUnsecuredLines: float = Field(..., ge=0, description='신용한도 소진율 (0~1, 초과 가능)')
    age: int = Field(..., ge=18, le=120, description='나이')
    past_due_30_59: int = Field(0, ge=0, alias='NumberOfTime30-59DaysPastDueNotWorse', description='최근 2년 30~59일 연체 횟수')
    DebtRatio: float = Field(..., ge=0, description='부채비율 % (월 대출 상환액 ÷ 월소득 × 100). 서버가 ÷100')
    MonthlyIncome: float = Field(..., ge=0, description='월소득 (원)')
    NumberOfOpenCreditLinesAndLoans: int = Field(0, ge=0, description='보유 대출·신용 한도 개수')
    NumberOfTimes90DaysLate: int = Field(0, ge=0, description='90일 이상 연체 횟수')
    NumberRealEstateLoansOrLines: int = Field(0, ge=0, description='부동산 담보대출 개수')
    past_due_60_89: int = Field(0, ge=0, alias='NumberOfTime60-89DaysPastDueNotWorse', description='최근 2년 60~89일 연체 횟수')
    NumberOfDependents: float = Field(0, ge=0, description='부양가족 수 (선택, 비우면 0 = 학습 데이터 중앙값)')
    # 대안 데이터 (시뮬레이터 5개)
    telecom_payment_rate: float = Field(..., ge=0, le=100, description='통신비 정상납부율 % (0~100). 서버가 ÷100')
    utility_payment_rate: float = Field(..., ge=0, le=100, description='공과금 납부율 % (0~100). 서버가 ÷100')
    spending_consistency: float = Field(..., ge=0, le=100, description='소비 일관성 점수')
    regular_payment_count: int = Field(0, ge=0, le=20, description='정기결제 건수 (월)')
    app_login_frequency: int = Field(0, ge=0, le=30, description='앱 로그인 빈도 (월)')

    def features(self):
        """모델 입력 칸 이름(alias 포함) → 모델 단위 값. customer_id 제외, % 입력은 ÷100."""
        values = self.model_dump(by_alias=True, exclude={'customer_id'})
        for key in PERCENT_FIELDS:
            values[key] = values[key] / 100
        return values


PREDICT_EXAMPLE = {
    "customer_id": "C-10482", "credit_score": 937, "risk_grade": "A", "approval_status": "APPROVED",
    "default_probability": 0.063, "is_thin_filer": False, "approval_threshold": 0.5,
    "model_version": "lightgbm_is_unbalance_integrated_v1.0", "timestamp": "2026-10-15T10:15:30+00:00",
    "disclaimer": DISCLAIMER,
}


class PredictResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={'examples': [PREDICT_EXAMPLE]})

    customer_id: str
    credit_score: int = Field(description='0~1000, (1 - 부도확률) × 1000')
    risk_grade: str = Field(description='A·B 저위험 / C 중위험 / D·E 고위험')
    approval_status: str = Field(description='APPROVED | REJECTED')
    default_probability: float
    is_thin_filer: bool = Field(description='씬파일러 판정 (src.data.simulator.label_thin_filer 기준)')
    approval_threshold: float = Field(description='부도확률이 이 값 미만이면 승인')
    model_version: str
    timestamp: str
    disclaimer: str = DISCLAIMER


# ---------- 02 XAI 분석 ----------
class RejectionReason(BaseModel):
    rank: int
    feature: str
    feature_name_kr: str
    current_value: float
    reference_value: Optional[float] = Field(None, description='동일 연령대 평균 (reference_values.json 이 없으면 null)')
    shap_contribution: float = Field(description='승인 쪽 기준 기여도 (거절 사유는 음수)')
    explanation: str


EXPLAIN_EXAMPLE = {
    "customer_id": "C-10477", "approval_status": "REJECTED", "default_probability": 0.565,
    "base_value": -2.64, "shap_space": "log-odds",
    "rejection_reasons": [
        {"rank": 1, "feature": "telecom_payment_rate", "feature_name_kr": "통신비 정상납부율",
         "current_value": 0.71, "reference_value": 0.92, "shap_contribution": -0.92,
         "explanation": "통신비 정상납부율이 71%로, 동일 연령대 평균(92%) 대비 낮습니다 (기여도: -0.92)"},
        {"rank": 2, "feature": "RevolvingUtilizationOfUnsecuredLines", "feature_name_kr": "한도 사용 비율",
         "current_value": 0.87, "reference_value": 0.31, "shap_contribution": -0.78,
         "explanation": "한도 사용 비율이 87%로, 동일 연령대 평균(31%) 대비 높습니다 (기여도: -0.78)"},
        {"rank": 3, "feature": "NumberOfTime30-59DaysPastDueNotWorse", "feature_name_kr": "최근 2년 30~59일 연체 횟수",
         "current_value": 2, "reference_value": 0.3, "shap_contribution": -0.61,
         "explanation": "최근 2년 30~59일 연체 횟수가 2회로, 동일 연령대 평균(0.3회) 대비 높습니다 (기여도: -0.61)"},
    ],
    "shap_values": {"telecom_payment_rate": 0.92, "RevolvingUtilizationOfUnsecuredLines": 0.78,
                    "NumberOfTime30-59DaysPastDueNotWorse": 0.61, "DebtRatio": 0.44, "utility_payment_rate": 0.31,
                    "NumberOfTimes90DaysLate": 0.22, "age": -0.18, "MonthlyIncome": -0.12},
    "total_rejection_factors": 3,
}


class ExplainResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={'examples': [EXPLAIN_EXAMPLE]})

    customer_id: str
    approval_status: str
    default_probability: float
    base_value: float = Field(description='SHAP 기준값 E[f(x)] (Waterfall 시작점)')
    shap_space: str = Field('log-odds', description='shap_values 의 단위')
    rejection_reasons: list[RejectionReason] = Field(description='승인에 불리한 변수 Top 5 (승인이면 빈 목록일 수 있음)')
    shap_values: dict[str, float] = Field(description='변수별 부도 쪽 SHAP 값 (양수 = 부도 위험 증가)')
    total_rejection_factors: int
    disclaimer: str = DISCLAIMER


class ShapSummaryFeature(BaseModel):
    feature: str
    mean_abs_shap: float
    group: str = Field(description='financial | alternative')


class ShapSummaryResponse(BaseModel):
    model_config = _with_example(example('shap_summary'))

    model: str
    n_samples: int
    shap_space: str = 'log-odds'
    features: list[ShapSummaryFeature] = Field(description='mean |SHAP| 내림차순')


# ---------- 03 공정성 ----------
class AttributeFairness(BaseModel):
    model_config = ConfigDict(extra='allow')

    protected_group: str
    approval_rates: dict[str, float]
    di_ratio: float
    di_passed: bool
    equalized_odds_tpr_diff: float
    equalized_odds_fpr_diff: float
    equalized_odds_passed: bool


class FairnessSummary(BaseModel):
    model_config = ConfigDict(extra='allow')

    di_ratio: float = Field(description='속성(성별·연령대) 중 가장 나쁜 값')
    equalized_odds_tpr_diff: float
    equalized_odds_fpr_diff: float
    by_attribute: dict[str, AttributeFairness] = Field(description='gender / age_group 별 상세')
    auc: Optional[float] = Field(None, description='mitigation 전/후 성능 비교용 (선택)')
    ks: Optional[float] = None
    approval_rate: Optional[float] = None


class FairnessResponse(FairnessSummary):
    model_config = _with_example(example('fairness'))

    threshold: float
    criteria: dict[str, float] = Field(description='di_min / di_max / eo_max')
    n: Optional[int] = None
    model: Optional[str] = None


class FairnessMitigationResponse(BaseModel):
    model_config = _with_example(example('fairness_mitigation'))

    method: str = Field(description='threshold_adjustment | reweighting')
    threshold: float
    threshold_fit_on: Optional[str] = Field(None, description='validation | evaluation_data')
    before: FairnessSummary
    after: FairnessSummary
    group_thresholds: Optional[dict[str, float]] = None
    target_approval_rate: Optional[float] = None


# ---------- 04 모델 성능 ----------
class ConfusionMatrix(BaseModel):
    model_config = ConfigDict(extra='allow')
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int


class Curve(BaseModel):
    model_config = ConfigDict(extra='allow')


class ThinFilerUplift(BaseModel):
    model_config = ConfigDict(extra='allow')
    n_thin_filer: int
    auc_without_alt: float
    auc_with_alt: float
    uplift: float
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None


class MetricsResponse(BaseModel):
    model_config = _with_example(example('metrics'))

    model: Optional[str] = None
    auc: float
    ks: float
    psi: Optional[float] = Field(None, description='train vs test 점수 분포 안정성')
    precision: float
    recall: float
    f1: float
    confusion_matrix: ConfusionMatrix
    roc_curve: dict[str, list[float]] = Field(description='{fpr: [...], tpr: [...]}')
    pr_curve: dict[str, list[float]] = Field(description='{precision: [...], recall: [...]}')
    n: int
    n_default: int
    pass_: dict[str, bool] = Field(alias='pass', description='기준 통과 여부 (auc/ks/psi/thin_filer_uplift)')
    targets: dict[str, float]
    thin_filer_uplift: Optional[ThinFilerUplift] = Field(None, description='씬파일러 세그먼트 대안 데이터 유/무 AUC')


class ModelCandidate(BaseModel):
    model_config = ConfigDict(extra='allow')
    data_type: str = Field(description='financial_only | alternative_only | integrated')
    algorithm: str
    method: str
    val_auc: float
    val_ks: float


class ModelComparisonResponse(BaseModel):
    model_config = ConfigDict(extra='allow')

    selected_model: str
    selection_metric: str
    val_auc: float
    test_auc: float
    test_ks: float
    val_table: list[ModelCandidate] = Field(description='후보 모델 격자 (데이터 묶음 × 알고리즘 × 불균형 기법)')
    cv_runs: dict[str, dict] = Field(description='후보별 5-Fold CV 평균 (model_metadata.json training_runs.runs)')
    disclaimer: str = DISCLAIMER


# ---------- 05 통계 검증 ----------
class TukeyRow(BaseModel):
    model_config = ConfigDict(extra='allow')
    comparison: str
    mean_diff: float
    p_adj: float
    ci95: Optional[list[float]] = None
    significant: bool


class AnovaTest(BaseModel):
    model_config = ConfigDict(extra='allow')

    test_name: str
    groups: list[str]
    group_means: dict[str, float]
    group_sizes: Optional[dict[str, int]] = None
    group_std: Optional[dict[str, float]] = None
    group_ci95: Optional[dict[str, list[float]]] = None
    f_statistic: float
    p_value: float
    effect_size_eta_squared: float
    effect_size_interpretation: Optional[str] = Field(None, description='small | medium | large')
    is_significant: bool
    post_hoc_tukey: list[TukeyRow] = Field(default_factory=list, description='p < alpha 일 때만 채워짐')


class AnovaResponse(BaseModel):
    model_config = _with_example(example('anova'))

    test_1: AnovaTest = Field(description='피처 그룹별 (전통/대안/통합)')
    test_2: AnovaTest = Field(description='알고리즘별 (LR/XGBoost/LightGBM)')
    source: Optional[dict] = None
    interpretation: Optional[dict[str, list[str]]] = None


# ---------- 시스템 ----------
class HealthResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={'examples': [{
        "status": "healthy", "model_ready": True, "model_version": "lightgbm_is_unbalance_integrated_v1.0",
        "model_stage": "Staging", "test_auc": 0.984, "approval_threshold": 0.5,
        "reports": {"metrics": "example", "fairness": "outputs/fairness/fairness.json"},
        "timestamp": "2026-10-15T10:15:30+00:00"}]})

    status: str = Field(description='healthy | degraded (모델 파일 없음)')
    model_ready: bool
    model_version: str
    model_stage: str
    test_auc: Optional[float] = None
    approval_threshold: float
    reports: dict[str, str] = Field(description='리포트별 데이터 출처: 파일 경로 또는 "example"(내장 예시)')
    timestamp: str
    disclaimer: str = DISCLAIMER
