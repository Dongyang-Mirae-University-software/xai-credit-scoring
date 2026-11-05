"""화면 맨 위 64px 헤더: 로고·시스템 이름 / 고객 ID 검색 · 데이터 배지 · 계정. 모든 페이지 공통 (app.py 에서 그린다).

본문 블록 간격에 끼지 않도록 사이드바 안에 그리고 position: fixed 로 화면 맨 위에 띄운다
(사이드바는 transform 이 없어 fixed 가 화면 기준으로 잡힌다).
검색창은 실제 입력 위젯(st.text_input)이라 HTML 헤더와 따로 그려 같은 자리에 겹친다.
"""
from pathlib import Path

import streamlit as st

import api_client
from components import customer_form, ui

CSS_PATH = Path(__file__).with_suffix(".css")
SCREENING_PAGE = "views/screening.py"


def _data_badge():
    """리포트 중 하나라도 예시 값이면 '예시 데이터', 아니면 '교육용' (같은 모양)."""
    try:
        reports = api_client.health().get("reports", {})
    except Exception:
        return "API 오류"
    return "예시 데이터" if "example" in reports.values() else "교육용"


def _on_search():
    """고객 ID 검색 (Enter): 심사한 고객이면 그 결과 화면으로, 아니면 오류 안내. (디자인 프로토타입 동작)"""
    ss = st.session_state
    value = (ss.get("hdr_search") or "").strip().upper()
    ss.hdr_search = ""
    if not value:
        return
    if customer_form.show_result(value):
        ss.hdr_toast = (f"{value} 고객의 심사 결과를 열었어요", False)
        ss.hdr_go = True
    else:
        ss.hdr_toast = (f"{value} 고객을 찾을 수 없어요. 고객 ID를 다시 확인해 주세요.", True)


def _on_home():
    """로고: 고객 심사 첫 화면으로."""
    st.session_state.scr_current = None
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
            <span class="hdr-search-slot"></span>
            <span class="hdr-badge">{_data_badge()}</span>
            <span class="hdr-account" role="img" aria-label="계정">{ui.icon("account", "#FFFFFF", 20)}</span>
          </div>
        </header>
        """)
        st.button("처음 화면으로", key="hdr-home", on_click=_on_home)
        with st.container(key="hdr-search"):
            st.text_input("고객 ID 검색", key="hdr_search", placeholder="고객 ID 검색",
                          label_visibility="collapsed", on_change=_on_search)
    toast = st.session_state.pop("hdr_toast", None)
    if st.session_state.pop("hdr_go", False) and current_page.url_path != "":
        if toast:
            st.session_state.scr_toast = toast  # 페이지를 바꾸면 토스트가 사라지므로 고객 심사 화면이 띄운다
        st.switch_page(SCREENING_PAGE)
    if toast:
        st.toast(toast[0], icon=":material/error:" if toast[1] else ":material/check_circle:")
