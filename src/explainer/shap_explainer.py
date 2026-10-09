"""
SHAP 설명: 전역(Summary·Dependence Plot)과 지역(Waterfall Plot), 고객별 거절 사유.

원본 과제 기준
- SHAP TreeExplainer 또는 KernelExplainer 사용
- Summary Plot 이미지, 상위 15개 변수 표시
- Dependence Plot 이미지 최소 2개 변수
- 개별 고객 Waterfall Plot 이미지
- /explain 에서 거절 사유 Top 5 (reasons.py)

shap, numpy, pandas, matplotlib 가 필요하다 (requirements.txt). 무거운 라이브러리라 함수 안에서 불러온다.
"""

import os
from pathlib import Path

from src.explainer.feature_names import korean_name
from src.explainer.reasons import build_rejection_reasons

TREE_MODEL_HINTS = ('XGB', 'LGBM', 'LightGBM', 'RandomForest', 'GradientBoosting', 'DecisionTree')


def final_estimator(model):
    """SMOTE 등이 들어간 Pipeline 이면 마지막 모델을 꺼낸다. (SMOTE 는 학습 때만 쓰이므로 설명에는 필요 없다.)"""
    steps = getattr(model, 'steps', None)
    if steps:
        return steps[-1][1]
    return model


def make_explainer(model, background):
    """
    모델 종류에 맞는 SHAP explainer 를 만든다.
    나무 모델(XGBoost, LightGBM 등)은 TreeExplainer, 그 밖(Logistic Regression 등)은
    배경 데이터 일부로 KernelExplainer 를 쓴다.
    """
    import shap

    est = final_estimator(model)
    name = type(est).__name__
    if any(h in name for h in TREE_MODEL_HINTS):
        return shap.TreeExplainer(est)
    sample = background.sample(n=min(100, len(background)), random_state=42) if hasattr(background, 'sample') else background
    return shap.KernelExplainer(lambda x: est.predict_proba(x)[:, 1], sample)


def shap_values_for_default(explainer, X):
    """
    '부도(1)' 쪽 SHAP 값을 (사람 수 × 변수 수) 배열로 돌려준다.
    shap 버전·모델에 따라 클래스별 목록이나 3차원 배열로 나오는 경우를 정리한다.
    """
    import numpy as np

    values = explainer.shap_values(X)
    if isinstance(values, list):
        values = values[1] if len(values) == 2 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    return values


def base_value_for_default(explainer):
    base = explainer.expected_value
    try:
        return float(base[1]) if len(base) == 2 else float(base[0])
    except TypeError:
        return float(base)


# 한글이 들어 있는 글꼴 이름 후보 (앞에 있을수록 먼저 고른다)
KOREAN_FONT_HINTS = ('NanumGothic', 'Noto Sans CJK', 'Noto Sans KR', 'Malgun Gothic',
                     'AppleGothic', 'Nanum', 'Noto Serif CJK', 'Noto Serif KR')
_korean_font_ready = False


def _label(feature):
    """그림에 쓸 이름. 한글 글꼴이 있으면 한글 이름, 없으면 네모로 깨지지 않게 영어 코드 이름."""
    return korean_name(feature) if _korean_font_ready else feature


def _korean_frame(X):
    """그림 축 이름을 _label 로 바꾼 복사본."""
    return X.rename(columns={c: _label(c) for c in X.columns})


def _find_korean_font():
    from matplotlib import font_manager
    names = sorted({f.name for f in font_manager.fontManager.ttflist})
    for hint in KOREAN_FONT_HINTS:
        for name in names:
            if hint.lower() in name.lower():
                return name
    return None


def _setup_font():
    """한글이 깨지지 않게 글꼴을 고른다. 환경 변수 PLOT_FONT 로 직접 정할 수 있다.
    한글 글꼴을 찾지 못하면 그림에는 영어 코드 이름을 쓴다."""
    global _korean_font_ready
    import logging
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    font = os.getenv('PLOT_FONT') or _find_korean_font()
    if font:
        plt.rcParams['font.family'] = font
        _korean_font_ready = True
    else:
        _korean_font_ready = False
        logging.getLogger(__name__).warning(
            "한글 글꼴을 찾지 못해 그림에 영어 변수 이름을 씁니다 (예: sudo apt install fonts-nanum 후 다시 실행)")
    plt.rcParams['axes.unicode_minus'] = False
    return plt


def save_summary_plot(shap_values, X, path, max_display=15):
    """전역 해석: Summary Plot (상위 15개 변수)."""
    import shap
    plt = _setup_font()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_values, _korean_frame(X), max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def top_features(shap_values, columns, k=2):
    """평균 |SHAP| 이 큰 순서로 변수 k 개."""
    import numpy as np
    importance = np.abs(shap_values).mean(axis=0)
    order = importance.argsort()[::-1][:k]
    return [columns[i] for i in order]


def save_dependence_plots(shap_values, X, out_dir, features=None, k=2):
    """전역 해석: Dependence Plot. features 를 주지 않으면 중요도 상위 k 개(기본 2개)."""
    import shap
    plt = _setup_font()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features = features or top_features(shap_values, list(X.columns), k)
    Xk = _korean_frame(X)
    paths = []
    for f in features:
        shap.dependence_plot(_label(f), shap_values, Xk, show=False)
        p = out_dir / f"dependence_{f}.png"
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        paths.append(p)
    return paths


def save_waterfall(explainer, shap_values, X, index, path, max_display=10):
    """지역 해석: 고객 한 명의 Waterfall Plot."""
    import shap
    plt = _setup_font()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    explanation = shap.Explanation(
        values=shap_values[index],
        base_values=base_value_for_default(explainer),
        data=X.iloc[index].values,
        feature_names=[_label(c) for c in X.columns],
    )
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def explain_customer(explainer, x_scaled, x_raw, references=None, top_k=5):
    """
    고객 한 명을 설명한다 (API /explain 재료).
    x_scaled: 모델에 넣는 값(크기 맞춘 값), 1행 DataFrame
    x_raw:    같은 고객의 원래 값 dict (문장에 쓸 값)
    """
    values = shap_values_for_default(explainer, x_scaled)[0]
    shap_by_feature = dict(zip(x_scaled.columns, (float(v) for v in values)))
    return {
        'rejection_reasons': build_rejection_reasons(
            x_raw, shap_by_feature, references=references, age=x_raw.get('age'), top_k=top_k),
        'shap_values': shap_by_feature,
        'base_value': base_value_for_default(explainer),
    }
