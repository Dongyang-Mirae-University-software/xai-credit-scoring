"""승인되려면: 거절 고객에게 바꿀 수 있는 정보만 움직여 승인 기준 아래로 내려가는 가장 작은 변화 (보너스 2, #140).

데이터: POST /counterfactual (#147). 같은 고객 · 같은 입력이면 한 번만 부르고 세션에 둔다 (탐색이라 /predict 보다 느리다).
API 가 아직 없으면(404) 섹션 자리에 'API 준비 중' 안내를 둔다.
"""
import json
from html import escape
from pathlib import Path

import streamlit as st

import api_client
import features as F
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


def _verb(c):
    return "늘리기" if c["suggested_value"] > c["current_value"] else "줄이기"


def _change_row(c):
    """바꿀 정보 한 줄 (토스식): 항목 ··· 지금 값 → 바꿀 값 · 늘리기/줄이기."""
    return (f'<li><span class="cf2-name">{escape(c["feature_name_kr"])}</span>'
            f'<span class="cf2-move"><em>{_value(c["current_value"], c["unit"])}</em>{ui.icon("arrow", "#B0B8C1", 16)}'
            f'<b>{_value(c["suggested_value"], c["unit"])}</b></span><i class="cf2-verb">{_verb(c)}</i></li>')


def _hero(cf, result):
    """결론 카드: 승인 가능 여부 · 큰 한 줄 · 부도 확률 / 신용 점수 → · 바꿀 정보 목록."""
    plan = cf.get("combined") or {}
    changes = plan.get("changes") or []
    ok = bool(plan.get("success"))
    if ok and len(changes) == 1:
        c = changes[0]
        name = c["feature_name_kr"]
        title = (f'{name}{F.josa(name, "을", "를")} {_value(c["suggested_value"], c["unit"])}로 '
                 f'{"늘리면" if _verb(c) == "늘리기" else "줄이면"} 승인될 수 있어요')
    elif ok:
        title = f"아래 {len(changes)}가지를 함께 바꾸면 승인될 수 있어요"
    else:
        title = "바꿀 수 있는 정보만으로는 승인 기준에 닿지 않아요"
    tone = "ok" if ok else "fail"
    rows = "".join(_change_row(c) for c in changes) or '<li class="cf2-empty">바꿀 수 있는 정보가 없어요</li>'
    return f"""
    <section class="cf2-card cf2-hero">
      <span class="cf2-badge {tone}">{"승인 가능" if ok else "승인 어려움"}</span>
      <h3>{escape(title)}</h3>
      <p class="cf2-sub">바꿀 수 있는 정보만 움직여 찾은 가장 작은 변화예요. 고객 상담에 참고하세요.</p>
      <div class="cf2-figs">
        <div><span>부도 확률</span><p><em>{_pct(cf["default_probability"])}</em>{ui.icon("arrow", "#B0B8C1", 18)}
          <b class="{tone}">{_pct(plan.get("new_probability", cf["default_probability"]))}</b></p>
          <small>승인 기준 {_pct(cf["approval_threshold"])} 미만</small></div>
        <div><span>신용 점수</span><p><em>{result["credit_score"]}점</em>{ui.icon("arrow", "#B0B8C1", 18)}
          <b class="{tone}">{plan.get("new_credit_score", result["credit_score"])}점</b></p></div>
      </div>
      <h4>{"바꿀 정보" if ok else "끝까지 움직여 본 정보"}</h4>
      <ul class="cf2-list">{rows}</ul>
    </section>"""


def _singles(cf):
    """하나만 바꿔도 되는 방법: 항목 · 지금 → 바꿀 값 ··· 그때 부도 확률 (초록)."""
    singles = cf.get("single_changes") or []
    if singles:
        body = "".join(
            f'<li><div><span class="cf2-name">{escape(c["feature_name_kr"])}</span>'
            f'<span class="cf2-move small"><em>{_value(c["current_value"], c["unit"])}</em>{ui.icon("arrow", "#B0B8C1", 14)}'
            f'<b>{_value(c["suggested_value"], c["unit"])}</b><i class="cf2-verb">{_verb(c)}</i></span></div>'
            f'<div class="cf2-prob"><span>부도 확률</span><b>{_pct(c["new_probability"])}</b></div></li>'
            for c in singles[:SINGLE_MAX] if c.get("new_probability") is not None)
        more = (f'<p class="cf2-more">외 {len(singles) - SINGLE_MAX}가지</p>' if len(singles) > SINGLE_MAX else "")
    else:
        body, more = '<li class="cf2-empty">정보 하나만 바꿔서는 승인 기준에 닿지 않아요</li>', ""
    return (f'<section class="cf2-card"><h3 class="cf2-side-title">하나만 바꿔도 되는 방법 '
            f'<b>{len(singles)}가지</b></h3><ul class="cf2-singles">{body}</ul>{more}</section>')


def render(customer_id, payload, result):
    """거절 고객에게만 그린다. 승인 고객은 아무것도 그리지 않는다."""
    if result["approval_status"] == "APPROVED":
        return
    with st.container(key="bonus-cf", gap="small"):
        _render(customer_id, payload, result)


def _render(customer_id, payload, result):
    st.html(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>")
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
    st.html(f'<div class="cf2">{_hero(cf, result)}{_singles(cf)}</div>'
            f'<p class="cf2-foot">바꿀 수 없어 그대로 둔 정보: {escape(fixed) or "-"}. '
            f'학습 데이터에 있는 값 범위(1~99%) 안에서만 찾았고, 실제 승인을 약속하지 않아요.</p>')
