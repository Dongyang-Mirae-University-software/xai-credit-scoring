"""사이드바 240px: 묶음별 메뉴(심사 업무 / 모델 검증)와 하단 시스템 상태 (GET /health, components/connection 이 채운다).

'고객 심사' 는 언제 눌러도 고객 심사 첫 화면으로 간다 (디자인 프로토타입 동작).
page_link 는 누른 것을 알 수 없어, 이 메뉴만 같은 모양의 버튼으로 그린다."""
from html import escape
from pathlib import Path

import streamlit as st

from components import ui
from components.header import go_start

CSS_PATH = Path(__file__).with_suffix(".css")

# 메뉴 묶음: (묶음 이름, [(페이지 제목, 아이콘 이름 - components/ui.py ICONS)])
NAV_GROUPS = [
    ("심사 업무", [("고객 심사", "screening"), ("XAI 분석", "xai")]),
    ("모델 검증", [("공정성", "fairness"), ("모델 성능", "performance"), ("통계 검증", "stats")]),
]


def status_html(state, health=None):
    """하단 시스템 상태. state: loading(연결 중) / ok / down(연결 안 됨). 값은 GET /health 의 status, model_version."""
    if state == "loading":
        server, model = '<span class="sb-state wait">연결 중</span>', '<b class="muted">확인 중</b>'
    elif state == "down":
        server, model = '<span class="sb-state fail">연결 안 됨</span>', '<b class="muted">확인 불가</b>'
    else:
        status = health.get("status")
        server = ('<span class="sb-state">정상</span>' if status == "healthy"
                  else f'<span class="sb-state fail">{escape(str(status))}</span>')
        model = f'<b>{escape(str(health.get("model_version") or "-"))}</b>'
    return ('<div class="sb-status"><span class="sb-status-head">시스템 상태</span>'
            f'<span class="sb-row"><span>API 서버</span>{server}</span>'
            f'<span class="sb-row"><span>모델 버전</span>{model}</span></div>')


def _icon_css():
    rules = [ui.mask_css(f'[class*="st-key-nav-{name}"] {target}::before', name)
             for _, items in NAV_GROUPS for _, name in items
             for target in ('a[data-testid="stPageLink-NavLink"]', "button")]
    return "".join(rules)


def render_sidebar(pages, current):
    """pages 는 app.py 의 st.Page 목록. 메뉴 묶음과 아이콘은 NAV_GROUPS 에서 정한다.
    하단 시스템 상태 자리(st.empty)를 돌려준다 → components/connection 이 연결 확인 결과로 채운다."""
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
        return st.empty()
