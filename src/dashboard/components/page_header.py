"""페이지 제목(26px) + 설명(14px). 오른쪽에 '라벨 + 굵은 값' 메타를 둘 수 있다."""
from html import escape

import streamlit as st


def render_page_header(title, subtitle, meta=None):
    """meta: [(라벨, 값)] - 예: [('마지막 점검', '2026-11-04')]. 값은 escape 해서 넣는다."""
    meta_html = ""
    if meta:
        meta_html = '<div class="ui-page-meta">' + "".join(
            f"<span>{escape(label)}<b>{escape(str(value))}</b></span>" for label, value in meta) + "</div>"
    st.html(f'<div class="ui-page-head"><div><h1>{escape(title)}</h1>'
            f'<p>{escape(subtitle)}</p></div>{meta_html}</div>')
