"""
Streamlit 대시보드 진입점.
페이지: 고객 심사 / XAI 분석 / 공정성 / 모델 성능 / 통계 검증
각 페이지는 pages/ 아래 파일이며, st.navigation으로 사이드바에 등록한다.
"""
import streamlit as st

st.set_page_config(page_title="XAI 기반 대안신용평가 시스템", layout="wide")

pages = [
    st.Page("pages/screening.py", title="고객 심사", default=True),
    st.Page("pages/xai.py", title="XAI 분석"),
    st.Page("pages/fairness.py", title="공정성"),
    st.Page("pages/performance.py", title="모델 성능"),
    st.Page("pages/statistical_test.py", title="통계 검증"),
]
pg = st.navigation(pages)

st.title("XAI 기반 대안신용평가 시스템")

pg.run()
