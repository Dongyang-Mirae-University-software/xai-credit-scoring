"""사이드바: 묶음별 페이지 메뉴(아이콘 포함)와 하단 시스템 상태 카드."""
from html import escape
from pathlib import Path

import requests
import streamlit as st

import config

CSS_PATH = Path(__file__).with_suffix(".css")

# 메뉴 묶음: (묶음 이름, [(페이지 제목, 아이콘)]). 아이콘은 Material Symbols 이름
NAV_GROUPS = [
    ("심사 업무", [
        ("고객 심사", ":material/person_check:"),
        ("XAI 분석", ":material/sort:"),
    ]),
    ("모델 검증", [
        ("공정성", ":material/balance:"),
        ("모델 성능", ":material/show_chart:"),
        ("통계 검증", ":material/table_chart:"),
    ]),
]


@st.cache_data(ttl=config.HEALTH_CACHE_TTL, show_spinner=False)
def _fetch_health(base_url):
    """/health 응답을 dict로 반환한다. 연결 실패 시 None."""
    try:
        r = requests.get(f"{base_url}/health", timeout=config.HEALTH_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def _status_card():
    health = _fetch_health(config.API_BASE_URL)
    if health is None:
        server = '<span class="sb-state err">연결 안 됨</span>'
        model = "—"
    else:
        server = '<span class="sb-state">정상</span>'
        model = health.get("model_version") or "—"
    model = escape(str(model))
    # 면책 조항(미션 필수, 이슈 #37)은 모든 페이지에 보이도록 상태 카드 바로 위에 둔다
    st.html(
        '<div class="sb-bottom">'
        f'<div class="sb-disclaimer" role="note">{escape(config.DISCLAIMER)}</div>'
        '<div class="sb-status">'
        '<div class="sb-status-head">시스템 상태</div>'
        f'<div class="sb-row"><span class="sb-row-label">API 서버</span>{server}</div>'
        f'<div class="sb-row"><span class="sb-row-label">모델 버전</span>'
        f'<span class="sb-row-value" title="{model}">{model}</span></div>'
        f'<a class="sb-link" href="{escape(config.MLFLOW_UI_URL)}" target="_blank" rel="noopener">MLflow 실험 기록 열기</a>'
        "</div></div>"
    )


def render_sidebar(pages, current):
    """pages 는 app.py 의 st.Page 목록. 메뉴 묶음과 아이콘은 NAV_GROUPS 에서 정한다."""
    by_title = {page.title: page for page in pages}
    st.html(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>")
    with st.sidebar:
        for group, items in NAV_GROUPS:
            st.html(f'<div class="sb-group">{group}</div>')
            for title, icon in items:
                page = by_title[title]
                state = "active" if page.url_path == current.url_path else "idle"
                with st.container(key=f"nav-{state}-{page.url_path or 'home'}"):
                    st.page_link(page, label=title, icon=icon)
        _status_card()
