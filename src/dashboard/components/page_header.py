"""페이지 제목·설명 (모델·교육용 배지는 상단 바 components/top_bar.py 에 있다)."""
from html import escape
from pathlib import Path

import streamlit as st

CSS_PATH = Path(__file__).with_suffix(".css")


def render_page_header(title, subtitle):
    st.html(f"""
    <style>{CSS_PATH.read_text(encoding='utf-8')}</style>
    <div class="ph-header">
      <div class="ph-title">{escape(title)}</div>
      <div class="ph-subtitle">{escape(subtitle)}</div>
    </div>
    """)
