"""
모델·스케일러·SHAP explainer 로딩과 고객 1명 예측/설명.

무거운 라이브러리(joblib, pandas, shap)는 함수 안에서 불러와 서버 기동과 /health 를 가볍게 유지한다.
MODEL_PATH / SCALER_PATH / APPROVAL_THRESHOLD 는 환경 변수로 바꿀 수 있다.
"""
import logging
import math
import os
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv('MODEL_PATH', PROJECT_ROOT / 'models' / 'lightgbm_is_unbalance_integrated_v1.0.joblib'))
SCALER_PATH = Path(os.getenv('SCALER_PATH', PROJECT_ROOT / 'models' / 'preprocess_scaler_v1.0.joblib'))
# 월소득은 API 바깥에서 원, 모델은 원본 GMSC 단위(달러)로 학습했다. 원 ÷ 이 값 = 모델 단위 (국제달러)
# 기본값: World Bank 한국 PPP 환산계수 (PA.NUS.PPP, GDP 기준, 2025년 816.33원/국제달러, 2026-11 조회)
# 한계: GMSC 시점(2008~2011년경)과의 물가 차이, 미·한 소득 구조 차이는 PPP 로 보정되지 않는다
INCOME_KRW_PER_UNIT = float(os.getenv('INCOME_KRW_PER_UNIT', '816.33'))
APPROVAL_THRESHOLD = float(os.getenv('APPROVAL_THRESHOLD', '0.5'))
MODEL_STAGE = os.getenv('MODEL_STAGE', 'Staging')
MODEL_VERSION = MODEL_PATH.stem

# PDO 스코어카드 (이슈 #157): Score = Offset + Factor·ln(good:bad odds), Factor = PDO/ln2, Offset = 기준점수 − Factor·ln(기준 odds)
# 600점 @ 50:1, PDO 20 은 스코어카드 교과서에서 흔히 쓰는 관례 값이다 (기관마다 다름). 20점 오를 때마다 odds 가 2배
SCORE_BASE, SCORE_BASE_ODDS, SCORE_PDO = 600, 50, 20
SCORE_MIN, SCORE_MAX = 300, 850
SCORE_FACTOR = SCORE_PDO / math.log(2)
SCORE_OFFSET = SCORE_BASE - SCORE_FACTOR * math.log(SCORE_BASE_ODDS)
SCORE_METHOD = f'pdo-{SCORE_BASE}-{SCORE_BASE_ODDS}-{SCORE_PDO}'

# 등급 구간 (등급, 상한): 점수가 상한 미만이면 그 등급. API·대시보드(/health 로 받음)가 이 값 하나를 쓴다.
# 이전 부도확률 경계 0.7·0.5·0.35·0.2 를 새 점수식에 넣은 값 (Test 데이터로 정하지 않음, 승인 여부 · 등급 뜻 유지).
#           모델 확률을 보정하면 다시 정할 것
GRADE_SCALE = [('E', 463), ('D', 487), ('C', 505), ('B', 527), ('A', SCORE_MAX + 1)]


def ready():
    return MODEL_PATH.exists() and SCALER_PATH.exists()


@lru_cache(maxsize=1)
def _artifacts():
    import joblib
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


def _scaled(frame):
    """원래 값 DataFrame(여러 행) → 스케일러·모델이 학습한 칸 순서의 DataFrame."""
    model, scaler = _artifacts()
    cols = list(scaler.feature_names_in_)
    X = frame[cols].astype(float).copy()
    X[cols] = scaler.transform(X)
    return X[list(getattr(model, 'feature_names_in_', cols))]


def _scaled_frame(features):
    """원래 값 dict → 1행 DataFrame (SHAP 설명용)."""
    import pandas as pd
    return _scaled(pd.DataFrame([features]))


def batch_default_probability(frame):
    """원래 값 DataFrame(여러 행) → 부도 확률 배열. /counterfactual 이 후보 여러 개를 한 번에 계산할 때 쓴다."""
    model, _ = _artifacts()
    return model.predict_proba(_scaled(frame))[:, 1]


def default_probability(features):
    import pandas as pd
    return float(batch_default_probability(pd.DataFrame([features]))[0])


def is_thin_filer(features):
    import pandas as pd
    from src.data.simulator import label_thin_filer
    logging.getLogger('src.data.simulator').setLevel(logging.WARNING)  # 판정 때마다 찍히는 집계 로그 끄기
    return bool(label_thin_filer(pd.DataFrame([features]))['is_thin_filer'].iloc[0])


def credit_score(prob):
    prob = min(max(prob, 1e-9), 1 - 1e-9)  # p = 0 또는 1 이면 ln(odds) 가 무한대
    score = SCORE_OFFSET + SCORE_FACTOR * math.log((1 - prob) / prob)
    return int(min(max(round(score), SCORE_MIN), SCORE_MAX))


def approval_score():
    """승인 기준 부도확률(APPROVAL_THRESHOLD)에 해당하는 점수. 대시보드 점선 위치."""
    return credit_score(APPROVAL_THRESHOLD)


def risk_grade(score):
    return next(grade for grade, upper in GRADE_SCALE if score < upper)


def approval_status(prob):
    return 'APPROVED' if prob < APPROVAL_THRESHOLD else 'REJECTED'


@lru_cache(maxsize=1)
def _explainer():
    # shortcut: TreeExplainer 전용(배경 데이터 없음). LR 등 비트리 모델을 쓰면 학습 데이터 표본을 background 로 넘겨야 함
    from src.explainer.shap_explainer import make_explainer
    model, _ = _artifacts()
    return make_explainer(model, background=None)


def _won(features):
    return {**features, 'MonthlyIncome': features['MonthlyIncome'] * INCOME_KRW_PER_UNIT}


def explain(features, references=None):
    """src.explainer.shap_explainer.explain_customer 결과: rejection_reasons, shap_values, base_value."""
    from src.explainer.shap_explainer import explain_customer
    if references:  # 기준 평균도 원 단위로 (reference_values.json 은 모델 단위로 저장돼 있다)
        references = {g: {**t, 'MonthlyIncome': round(t['MonthlyIncome'] * INCOME_KRW_PER_UNIT)} if 'MonthlyIncome' in t else t
                      for g, t in references.items()}
    return explain_customer(_explainer(), _scaled_frame(features), _won(features), references=references)
