"""사이드바: 브랜드, 페이지 이동 메뉴, 하단 서버 상태 카드."""
from pathlib import Path
from urllib.parse import urlparse

import requests
import streamlit as st

import config

CSS_PATH = Path(__file__).with_suffix(".css")


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
    base_url = config.API_BASE_URL
    port = urlparse(base_url).port
    health = _fetch_health(base_url)

    if health is None:
        dot, head = "sb-dot err", "FastAPI 연결 안 됨"
        health = {}
    else:
        dot, head = "sb-dot", "FastAPI 연결됨"
    if port:
        head += f" · :{port}"

    latency = st.session_state.get("predict_latency")
    rows = [
        ("model_version", health.get("model_version", "—")),
        ("/predict 응답", f"{latency:.2f} s" if latency is not None else "—"),
        ("MLflow", health.get("mlflow_stage", "—")),
    ]
    rows_html = "".join(
        f'<div class="sb-row"><span class="sb-row-label">{label}</span>'
        f'<span class="sb-row-value">{value}</span></div>'
        for label, value in rows
    )
    st.html(
        f'<div class="sb-status">'
        f'<div class="sb-status-head"><span class="{dot}"></span>{head}</div>'
        f"{rows_html}</div>"
    )


def render_sidebar(pages, current):
    st.html(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>")
    with st.sidebar:
        st.html(
            '<div class="sb-brand">'
            '<div class="sb-brand-title">XAI 신용평가</div>'
            '<div class="sb-brand-sub">심사역 대시보드 · 대안신용평가</div>'
            "</div>"
        )
        for i, page in enumerate(pages, start=1):
            state = "active" if page.url_path == current.url_path else "idle"
            with st.container(key=f"nav-{state}-{i}"):
                st.page_link(page, label=f"`{i:02d}` {page.title}")
        _status_card()
