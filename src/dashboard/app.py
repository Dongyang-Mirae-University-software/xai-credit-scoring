"""
Streamlit 대시보드 진입점.
페이지: 고객 심사 / XAI 분석 / 공정성 / 모델 성능 / 통계 검증
각 페이지는 pages/ 아래 파일이며, st.navigation으로 등록한다.
기본 사이드바 메뉴는 숨기고 components/sidebar.py에서 직접 그린다.
"""
from pathlib import Path

import streamlit as st

from components.sidebar import render_sidebar

CSS_PATH = Path(__file__).with_suffix(".css")

st.set_page_config(page_title="XAI 기반 대안신용평가 시스템", layout="wide")
st.html(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>")

pages = [
    st.Page("pages/screening.py", title="고객 심사", default=True),
    st.Page("pages/xai.py", title="XAI 분석"),
    st.Page("pages/fairness.py", title="공정성"),
    st.Page("pages/performance.py", title="모델 성능"),
    st.Page("pages/statistical_test.py", title="통계 검증"),
]
pg = st.navigation(pages, position="hidden")
render_sidebar(pages, pg)

pg.run()
