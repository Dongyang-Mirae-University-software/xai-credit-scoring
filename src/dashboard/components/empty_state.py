"""데이터 없음 안내: 파이프라인 산출물이 아직 없을 때, 무엇을 실행하면 채워지는지 보여 주는 초기 화면."""
from html import escape
from pathlib import Path

import streamlit as st

import api_client

CSS_PATH = Path(__file__).with_suffix(".css")

# 리포트 이름(/health 의 reports 키) → 안내 문구와 만드는 단계 (실행 예는 각 main_step*.py 설명 참고)
REPORTS = {
    "metrics": {
        "title": "아직 모델 평가 결과가 없어요",
        "message": "평가를 실행하면 AUC·KS·PSI 지표, ROC·PR 곡선, 혼동행렬, 씬파일러 비교가 여기에 표시됩니다.",
        "command": "python main_step3.py --model models/{model}.joblib",
        "output": "outputs/evaluation/metrics_*.json",
    },
    "shap_summary": {
        "title": "아직 전역 중요도 결과가 없어요",
        "message": "평가를 실행하면 피처별 평균 |SHAP| 순위가 표시됩니다.",
        "command": "python main_step3.py --model models/{model}.joblib",
        "output": "outputs/evaluation/shap_summary.json",
    },
    "fairness": {
        "title": "아직 공정성 분석 결과가 없어요",
        "message": "공정성 분석을 실행하면 집단별 승인율, DI Ratio, Equalized Odds 가 여기에 표시됩니다.",
        "command": "python main_step4.py --model models/{model}.joblib",
        "output": "outputs/fairness/fairness.json",
    },
    "fairness_mitigation": {
        "title": "아직 Bias Mitigation 결과가 없어요",
        "message": "공정성 분석을 실행하면 완화 전 / 후 비교가 표시됩니다.",
        "command": "python main_step4.py --model models/{model}.joblib",
        "output": "outputs/fairness/fairness_mitigation.json",
    },
    "anova": {
        "title": "아직 통계 검증 결과가 없어요",
        "message": "MLflow 실험 기록으로 통계 검증을 실행하면 ANOVA 와 Tukey HSD 결과가 여기에 표시됩니다.",
        "command": "python main_step5.py",
        "output": "outputs/statistics/anova.json",
    },
}


def _model_name():
    try:
        return api_client.health().get("model_version") or "<모델 파일>"
    except Exception:
        return "<모델 파일>"


def no_report_html(name, variant="page"):
    """variant: page(남는 높이를 채움) / inline(다른 카드 안, 작게)."""
    r = REPORTS[name]
    command = r["command"].format(model=_model_name())
    return f"""
    <style>{CSS_PATH.read_text(encoding='utf-8')}</style>
    <div class="es-box {variant}">
      <span class="es-icon"></span>
      <div class="es-title">{r["title"]}</div>
      <div class="es-message">{r["message"]}</div>
      <div class="es-steps">
        <span>프로젝트 폴더에서 실행한 뒤 새로고침하세요</span>
        <code>{escape(command)}</code>
        <span class="es-output">결과 파일 · {escape(r["output"])}</span>
      </div>
    </div>"""


def render_no_report(name, variant="page"):
    st.html(no_report_html(name, variant))
