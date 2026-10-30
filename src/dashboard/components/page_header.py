"""페이지 상단 헤더: 제목·설명, 모델 버전·AUC 배지, 교육용 면책 배지."""
from html import escape
from pathlib import Path

import streamlit as st

import api_client
from config import DISCLAIMER

CSS_PATH = Path(__file__).with_suffix(".css")


def render_page_header(title, subtitle):
    try:
        model_version = api_client.health()["model_version"]
        auc = api_client.metrics()["auc"]
        model_badge, badge_cls = f"모델 {escape(model_version)} · AUC {auc:.3f}", "blue"
    except Exception:
        model_badge, badge_cls = "모델 정보 없음 · API 연결 확인", "gray"

    st.html(f"""
    <style>{CSS_PATH.read_text(encoding='utf-8')}</style>
    <div class="ph-header">
      <div>
        <div class="ph-title">{escape(title)}</div>
        <div class="ph-subtitle">{escape(subtitle)}</div>
      </div>
      <div class="ph-badges">
        <span class="ph-badge {badge_cls}">{model_badge}</span>
        <span class="ph-badge amber" title="{DISCLAIMER}">교육용 · 실제 금융 의사결정 불가</span>
      </div>
    </div>
    """)
