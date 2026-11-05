"""
Streamlit 대시보드 진입점.
페이지: 고객 심사 / XAI 분석 / 공정성 / 모델 성능 / 통계 검증
각 페이지는 views/ 아래 파일이며, st.navigation으로 등록한다.
기본 메뉴·헤더는 숨기고 상단 헤더(components/header.py)와 사이드바(components/sidebar.py)를 직접 그린다.
"""
from pathlib import Path

import streamlit as st

from components.header import render_header
from components.sidebar import render_sidebar
from config import DISCLAIMER

CSS_PATH = Path(__file__).with_suffix(".css")

st.set_page_config(page_title="대안신용평가 심사 시스템", layout="wide")
st.html(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>")

pages = [
    st.Page("views/screening.py", title="고객 심사", default=True),
    st.Page("views/xai.py", title="XAI 분석"),
    st.Page("views/fairness.py", title="공정성"),
    st.Page("views/performance.py", title="모델 성능"),
    st.Page("views/statistical_test.py", title="통계 검증"),
]
pg = st.navigation(pages, position="hidden")
render_header(pg)
render_sidebar(pages, pg)

pg.run()

# 모든 화면 하단 면책 조항 (미션 필수). 페이지에서 st.stop() 을 쓰면 이 줄이 실행되지 않으므로 페이지는 return 으로 끝낸다
st.html(f'<footer class="ui-footer" role="note">{DISCLAIMER}</footer>')
