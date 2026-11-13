"""화면 맨 위 64px 헤더: 로고·시스템 이름 / 데이터 배지 · 계정. 모든 페이지 공통 (app.py 에서 그린다).

본문 블록 간격에 끼지 않도록 사이드바 안에 그리고 position: fixed 로 화면 맨 위에 띄운다
(사이드바는 transform 이 없어 fixed 가 화면 기준으로 잡힌다).
로고를 누르면 고객 심사 첫 화면으로 간다 (사이드바 '고객 심사' 도 같은 동작, go_start).
"""
from pathlib import Path

import streamlit as st

from components import customer_form, ui

CSS_PATH = Path(__file__).with_suffix(".css")
SCREENING_PAGE = "views/screening.py"


def go_start():
    """로고 · 사이드바 '고객 심사': 고객 심사 첫 화면으로 (버튼 on_click 에서 부른다)."""
    customer_form.clear_result()
    st.session_state.hdr_go = True


def render_header(current_page):
    customer_form.init_state()
    with st.sidebar:
        st.html(f"""
        <style>{CSS_PATH.read_text(encoding='utf-8')}</style>
        <header class="hdr">
          <div class="hdr-brand">
            {ui.logo()}
            <div><b>대안신용평가 심사 시스템</b><span>XAI 기반 씬파일러 대출 심사</span></div>
          </div>
          <div class="hdr-right">
            <span class="hdr-badge">교육용</span>
            <span class="hdr-account" aria-hidden="true">{ui.icon("account", "#FFFFFF", 20)}</span>
          </div>
        </header>
        """)
        st.button("처음 화면으로", key="hdr-home", on_click=go_start)
    if st.session_state.pop("hdr_go", False) and current_page.url_path != "":
        st.switch_page(SCREENING_PAGE)
