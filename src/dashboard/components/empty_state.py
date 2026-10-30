"""데이터 없음 안내: 파이프라인 산출물이 아직 없을 때의 초기 화면.
페이지 위에 한 줄 안내(무엇을 실행하면 채워지는지)를 두고, 카드 자리는 ghost_card 로 비워 둔다."""
from html import escape
from pathlib import Path

import streamlit as st

import api_client

CSS_PATH = Path(__file__).with_suffix(".css")

# 리포트 이름(/health 의 reports 키) → 안내 문구와 만드는 명령 (실행 예는 각 main_step*.py 설명 참고)
REPORTS = {
    "metrics": ("아직 모델 평가 결과가 없어요", "python main_step3.py --model models/{model}.joblib"),
    "shap_summary": ("아직 변수 중요도 결과가 없어요", "python main_step3.py --model models/{model}.joblib"),
    "fairness": ("아직 공정성 분석 결과가 없어요", "python main_step4.py --model models/{model}.joblib"),
    "fairness_mitigation": ("아직 완화 전 / 후 비교 결과가 없어요", "python main_step4.py --model models/{model}.joblib"),
    "anova": ("아직 통계 검증 결과가 없어요", "python main_step5.py"),
}


def _model_name():
    try:
        return api_client.health().get("model_version") or "<모델 파일>"
    except Exception:
        return "<모델 파일>"


def no_report_html(name, variant="banner"):
    """banner: 페이지 위 한 줄 안내 / inline: 카드 안 가운데 안내."""
    title, command = REPORTS[name]
    command = command.format(model=_model_name())
    return f"""
    <style>{CSS_PATH.read_text(encoding='utf-8')}</style>
    <div class="es-box {variant}">
      <span class="es-icon"></span>
      <div class="es-text">
        <b>{title}</b>
        <span>프로젝트 폴더에서 실행한 뒤 새로고침하세요</span>
      </div>
      <code>{escape(command)}</code>
    </div>"""


def render_no_report(name, variant="banner"):
    st.html(no_report_html(name, variant))


def ghost_card(title, card_cls, title_cls, message="결과가 나오면 여기에 표시됩니다"):
    """결과가 없을 때 카드 자리를 지키는 빈 카드 (페이지의 카드 스타일을 그대로 쓴다)."""
    return (f'<div class="{card_cls} es-ghost"><div class="{title_cls}">{title}</div>'
            f'<div class="es-ghost-body">{message}</div></div>')


def ghost_metric(label, card_cls):
    """값 자리를 '—' 로 둔 지표 카드."""
    return (f'<div class="{card_cls} es-ghost-metric"><span>{label}</span><b>—</b></div>')
