"""
모델·스케일러·SHAP explainer 로딩과 고객 1명 예측/설명.

무거운 라이브러리(joblib, pandas, shap)는 함수 안에서 불러와 서버 기동과 /health 를 가볍게 유지한다.
MODEL_PATH / SCALER_PATH / APPROVAL_THRESHOLD 는 환경 변수로 바꿀 수 있다.
"""
import logging
import os
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv('MODEL_PATH', PROJECT_ROOT / 'models' / 'lightgbm_is_unbalance_integrated_v1.0.joblib'))
SCALER_PATH = Path(os.getenv('SCALER_PATH', PROJECT_ROOT / 'models' / 'preprocess_scaler_v1.0.joblib'))
APPROVAL_THRESHOLD = float(os.getenv('APPROVAL_THRESHOLD', '0.5'))
MODEL_STAGE = os.getenv('MODEL_STAGE', 'Staging')
MODEL_VERSION = MODEL_PATH.stem

# 대시보드 config.GRADE_SCALE 과 같은 구간: 점수가 상한 미만이면 그 등급
GRADE_SCALE = [('E', 300), ('D', 500), ('C', 650), ('B', 800), ('A', 1001)]


def ready():
    return MODEL_PATH.exists() and SCALER_PATH.exists()


@lru_cache(maxsize=1)
def _artifacts():
    import joblib
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


def _scaled_frame(features):
    """원래 값 dict → 스케일러·모델이 학습한 칸 순서의 1행 DataFrame."""
    import pandas as pd
    model, scaler = _artifacts()
    cols = list(scaler.feature_names_in_)
    X = pd.DataFrame([features])[cols].astype(float)
    X[cols] = scaler.transform(X)
    return X[list(getattr(model, 'feature_names_in_', cols))]


def default_probability(features):
    model, _ = _artifacts()
    return float(model.predict_proba(_scaled_frame(features))[:, 1][0])


def is_thin_filer(features):
    import pandas as pd
    from src.data.simulator import label_thin_filer
    logging.getLogger('src.data.simulator').setLevel(logging.WARNING)  # 판정 때마다 찍히는 집계 로그 끄기
    return bool(label_thin_filer(pd.DataFrame([features]))['is_thin_filer'].iloc[0])


def credit_score(prob):
    # shortcut: (1-p)*1000 선형 환산, 스코어카드 점수식이 확정되면 교체
    return int(round((1 - prob) * 1000))


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


def explain(features, references=None):
    """src.explainer.shap_explainer.explain_customer 결과: rejection_reasons, shap_values, base_value."""
    from src.explainer.shap_explainer import explain_customer
    return explain_customer(_explainer(), _scaled_frame(features), features, references=references)
