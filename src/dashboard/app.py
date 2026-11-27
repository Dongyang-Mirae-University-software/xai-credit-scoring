"""
Streamlit 대시보드 진입점.
페이지: 고객 심사 / XAI 분석 / 공정성 / 모델 성능 / 통계 검증
각 페이지는 views/ 아래 파일이며, st.navigation으로 등록한다.
기본 메뉴·헤더는 숨기고 상단 헤더(components/header.py)와 사이드바(components/sidebar.py)를 직접 그린다.
"""
from pathlib import Path

import streamlit as st

from components import connection
from components.header import render_header
from components.sidebar import render_sidebar
from config import DISCLAIMER

CSS_PATH = Path(__file__).with_suffix(".css")

# 파비콘: 헤더 로고(components/ui.LOGO_SVG, 파란 둥근 사각형 + 흰 그래프 선)를 그린 128px PNG (바탕 투명)
st.set_page_config(page_title="대안신용평가 심사 시스템", page_icon=str(Path(__file__).with_name("favicon.png")),
                   layout="wide")
st.html(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>")

pages = [
    st.Page("views/screening.py", title="고객 심사", default=True),
    st.Page("views/xai.py", title="XAI 분석"),
    st.Page("views/fairness.py", title="공정성"),
    st.Page("views/performance.py", title="모델 성능"),
    st.Page("views/statistical_test.py", title="통계 검증"),
]
pg = st.navigation(pages, position="hidden")
# 화면을 옮기면 떠난 화면의 입력·선택 상태를 지운다 (위젯 값은 Streamlit 이 지우고, 위젯이 아닌 상태만 여기서).
# 고객 심사는 1/3 부터 다시 (입력값은 frm_values 에 남지만 단계는 처음으로) (다른 화면에서 맡긴 고객은 frm_pending 으로 남는다)
if st.session_state.get("_page") not in (None, pg.url_path):
    for key in ("scr_step", "idn_notice", "idn_errors"):
        st.session_state.pop(key, None)
st.session_state._page = pg.url_path
render_header(pg)
status_slot = render_sidebar(pages, pg)
connection.check(pg, status_slot)  # 첫 진입·재연결이면 /health 를 기다리는 동안 스켈레톤, 실패하면 경고 띠

pg.run()

# 모든 화면 하단 면책 조항 (미션 필수). 페이지에서 st.stop() 을 쓰면 이 줄이 실행되지 않으므로 페이지는 return 으로 끝낸다
st.html(f'<footer class="ui-footer" role="note">{DISCLAIMER}</footer>')
