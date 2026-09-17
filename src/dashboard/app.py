"""
Streamlit 대시보드.
탭: 고객 심사 / XAI 분석 / 공정성 / 모델 성능 / 통계 검증
모델 파일을 직접 로드하지 않고 API 서버와 HTTP 통신으로만 동작한다.
"""
import streamlit as st

st.title("XAI 기반 대안신용평가 시스템")
