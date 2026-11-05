"""사이드바 240px: 묶음별 메뉴(심사 업무 / 모델 검증)와 하단 시스템 상태 (GET /health).

'고객 심사' 는 언제 눌러도 고객 심사 첫 화면으로 간다 (디자인 프로토타입 동작).
page_link 는 누른 것을 알 수 없어, 이 메뉴만 같은 모양의 버튼으로 그린다."""
from html import escape
from pathlib import Path

import requests
import streamlit as st

import api_client
import config
from components import ui
from components.header import go_start

CSS_PATH = Path(__file__).with_suffix(".css")

# 메뉴 묶음: (묶음 이름, [(페이지 제목, 아이콘 이름 — components/ui.py ICONS)])
NAV_GROUPS = [
    ("심사 업무", [("고객 심사", "screening"), ("XAI 분석", "xai")]),
    ("모델 검증", [("공정성", "fairness"), ("모델 성능", "performance"), ("통계 검증", "stats")]),
]


@st.cache_data(ttl=config.HEALTH_CACHE_TTL, show_spinner=False)
def _fetch_health(base_url):
    """/health 응답을 dict로 반환한다. 연결 실패 시 None."""
    try:
        return api_client.request("GET", "/health", timeout=config.HEALTH_TIMEOUT)
    except requests.RequestException:
        return None


def _status_box():
    health = _fetch_health(config.API_BASE_URL)
    if health is None:
        server, model = '<span class="sb-state fail">연결 안 됨</span>', "—"
    else:
        ok = health.get("status") == "healthy"
        server = '<span class="sb-state">정상</span>' if ok else f'<span class="sb-state fail">{escape(str(health.get("status")))}</span>'
        model = escape(str(health.get("model_version") or "—"))
    st.html(
        '<div class="sb-status">'
        '<span class="sb-status-head">시스템 상태</span>'
        f'<span class="sb-row"><span>API 서버</span>{server}</span>'
        f'<span class="sb-row"><span>모델 버전</span><b>{model}</b></span>'
        "</div>"
    )


def _icon_css():
    rules = [ui.mask_css(f'[class*="st-key-nav-{name}"] {target}::before', name)
             for _, items in NAV_GROUPS for _, name in items
             for target in ('a[data-testid="stPageLink-NavLink"]', "button")]
    return "".join(rules)


def render_sidebar(pages, current):
    """pages 는 app.py 의 st.Page 목록. 메뉴 묶음과 아이콘은 NAV_GROUPS 에서 정한다."""
    by_title = {page.title: page for page in pages}
    with st.sidebar:
        # 스타일만 있는 블록도 간격(24px)을 차지하므로 첫 묶음 이름과 함께 넣는다
        style = f"<style>{CSS_PATH.read_text(encoding='utf-8')}{_icon_css()}</style>"
        for i, (group, items) in enumerate(NAV_GROUPS):
            with st.container(key=f"nav-group-{items[0][1]}"):
                st.html((style if i == 0 else "") + f'<span class="sb-group">{group}</span>')
                for title, name in items:
                    page = by_title[title]
                    active = page.url_path == current.url_path
                    with st.container(key=f"nav-{name}" + ("-active" if active else "")):
                        if title == "고객 심사":
                            st.button(title, key="nav-start", on_click=go_start, width="stretch")
                        else:
                            st.page_link(page, label=title)
        _status_box()
