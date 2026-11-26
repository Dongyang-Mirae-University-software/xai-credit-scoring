"""승인되려면: 거절 고객에게 바꿀 수 있는 정보만 움직여 승인 기준 아래로 내려가는 가장 작은 변화 (보너스 2, #140).

데이터: POST /counterfactual (#147). 같은 고객 · 같은 입력이면 한 번만 부르고 세션에 둔다 (탐색이라 /predict 보다 느리다).
API 가 아직 없으면(404) 섹션 자리에 'API 준비 중' 안내를 둔다.
"""
import json
from html import escape
from pathlib import Path

import streamlit as st

import api_client
from components import ui
from components.api_error import render_api_error
from components.empty_state import pending_html

CSS_PATH = Path(__file__).with_suffix(".css")
SINGLE_MAX = 4   # '하나만 바꿔도 되는 방법' 최대 개수 (API 는 작은 변화 순으로 준다)


def _pct(p):
    return f"{p * 100:.1f}%"


def _value(v, unit):
    """API 의 unit 단위 값 → 화면 표기. 월소득(원)은 다른 화면처럼 만원으로."""
    if unit == "원":
        return f"{v / 10_000:,.0f}만원"
    if unit == "%":
        return f"{v:.1f}%"
    return f"{v:,.0f}{unit}"


def _fetch(customer_id, payload):
    """성공한 응답과 'API 없음' 만 세션에 둔다. 연결 오류 등은 다음 화면 갱신 때 다시 부른다."""
    cache = st.session_state.setdefault("xai_cf", {})
    key = (customer_id, json.dumps(payload, sort_keys=True, ensure_ascii=False))
    if key not in cache:
        cache[key] = api_client.counterfactual(payload)
    return cache[key]


def _change_row(c, with_prob=False):
    up = c["suggested_value"] > c["current_value"]
    prob = (f'<span class="cf-prob">부도 확률 <b>{_pct(c["new_probability"])}</b></span>'
            if with_prob and c.get("new_probability") is not None else "")
    return (f'<div class="cf-row"><strong>{escape(c["feature_name_kr"])}</strong>'
            f'<span class="cf-from">{_value(c["current_value"], c["unit"])}</span>'
            f'<span class="cf-arrow">{ui.icon("arrow", "#5B6577", 16)}</span>'
            f'<b class="cf-to">{_value(c["suggested_value"], c["unit"])}</b>'
            f'{ui.pill("높이기" if up else "낮추기", "info")}{prob}</div>')


def _combined(cf, result):
    plan = cf.get("combined") or {}
    thr = cf["approval_threshold"]
    changes = plan.get("changes") or []
    if plan.get("success"):
        title = (f"아래 {len(changes)}가지를 함께 바꾸면 승인 기준을 통과해요" if len(changes) > 1
                 else "아래 한 가지만 바꾸면 승인 기준을 통과해요")
        tone = "ok"
    else:
        title = "바꿀 수 있는 정보를 모두 움직여도 승인 기준에 닿지 않아요"
        tone = "fail"
    flow = (f'<div class="cf-flow"><span>부도 확률</span><em>{_pct(cf["default_probability"])}</em>'
            f'{ui.icon("arrow", "#5B6577", 16)}<b class="{tone}">{_pct(plan["new_probability"])}</b>'
            f'<small>승인 기준 {_pct(thr)} 미만</small>'
            f'<span class="cf-sep"></span><span>신용 점수</span><em>{result["credit_score"]}점</em>'
            f'{ui.icon("arrow", "#5B6577", 16)}<b class="{tone}">{plan["new_credit_score"]}점</b></div>')
    rows = "".join(_change_row(c) for c in changes) or '<div class="cf-empty">바꿀 수 있는 정보가 없어요</div>'
    return (f'<div class="cf-col"><span class="cf-col-title">여러 정보를 함께 바꾸기</span>'
            f'<p class="cf-title {tone}">{title}</p>{flow}<div class="cf-rows">{rows}</div></div>')


def _singles(cf):
    singles = cf.get("single_changes") or []
    if singles:
        body = "".join(_change_row(c, with_prob=True) for c in singles[:SINGLE_MAX])
        more = (f'<span class="cf-more">외 {len(singles) - SINGLE_MAX}가지</span>' if len(singles) > SINGLE_MAX else "")
    else:
        body, more = '<div class="cf-empty">정보 하나만 바꿔서는 승인 기준에 닿지 않아요</div>', ""
    return (f'<div class="cf-col"><span class="cf-col-title">하나만 바꿔도 되는 방법</span>'
            f'<div class="cf-rows">{body}</div>{more}</div>')


def render(customer_id, payload, result):
    """거절 고객에게만 그린다. 승인 고객은 아무것도 그리지 않는다."""
    if result["approval_status"] == "APPROVED":
        return
    with st.container(key="bonus-cf", gap="small"):
        _render(customer_id, payload, result)


def _render(customer_id, payload, result):
    st.html(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>"
            + ui.section_head("승인되려면", "바꿀 수 있는 정보만 움직여 찾은 가장 작은 변화예요. 고객 상담에 참고하세요."))
    try:
        with st.spinner("승인 조건을 찾고 있어요"):
            cf = _fetch(customer_id, payload)
    except Exception as e:
        render_api_error("POST /counterfactual", e)
        return
    if cf == api_client.NOT_READY:
        st.html(pending_html("POST /counterfactual", "#147"))
        return
    fixed = ", ".join(cf.get("immutable_features") or [])
    st.html(f'<section class="cf-card"><div class="cf-grid">{_combined(cf, result)}{_singles(cf)}</div>'
            f'<div class="cf-foot">바꿀 수 없어 그대로 둔 정보: {escape(fixed) or "-"}. '
            f'학습 데이터에 있는 값 범위(1~99%) 안에서만 찾았고, 실제 승인을 약속하지 않아요.</div></section>')
