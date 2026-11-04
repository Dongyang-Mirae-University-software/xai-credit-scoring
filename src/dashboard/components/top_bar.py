"""화면 맨 위 상단 바: 왼쪽 브랜드, 오른쪽 모델·AUC 배지와 교육용 배지. 모든 페이지 공통 (app.py 에서 그린다)."""
from html import escape
from pathlib import Path

import streamlit as st

import api_client
from config import DISCLAIMER

CSS_PATH = Path(__file__).with_suffix(".css")


def _model_badge():
    try:
        model_version = api_client.health()["model_version"]
        metrics = api_client.metrics()  # 평가 결과가 아직 없으면 None → AUC 없이 모델 이름만
    except Exception:
        return '<span class="tb-badge gray">모델 정보 없음 · API 연결 확인</span>'
    auc = f' · AUC {metrics["auc"]:.3f}' if metrics else ""
    return f'<span class="tb-badge blue" title="{escape(model_version)}">모델 {escape(model_version)}{auc}</span>'


def render_top_bar():
    # 본문 블록 사이 간격(gap)이 생기지 않도록 사이드바 안에 그리고, position: fixed 로 화면 맨 위에 띄운다
    with st.sidebar:
        st.html(f"""
        <style>{CSS_PATH.read_text(encoding='utf-8')}</style>
        <header class="tb-bar">
          <div class="tb-brand">
            <span class="tb-logo">verified_user</span>
            <div>
              <div class="tb-title">XAI 신용평가</div>
              <div class="tb-sub">심사역 대시보드 · 대안신용평가</div>
            </div>
          </div>
          <div class="tb-badges">
            {_model_badge()}
            <span class="tb-badge amber" title="{escape(DISCLAIMER)}">교육용 · 실제 금융 의사결정 불가</span>
          </div>
        </header>
        """)
