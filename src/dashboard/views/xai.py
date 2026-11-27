"""XAI 분석 페이지 (디자인 05)

결론(한 줄 요약·부도 확률) → 근거(거절 사유 Top 5 · 거절이면 승인되려면) → 상세(SHAP Waterfall · SHAP Summary).
기여도는 모두 '부도 위험' 기준: + 는 위험을 높였다, - 는 낮췄다.
(POST /explain 의 rejection_reasons[].shap_contribution 은 승인 쪽 기준(음수)이라 부호를 뒤집어 보여 준다.)
"""
import hashlib
import math
import re
from html import escape
from pathlib import Path

import streamlit as st

import api_client
import config
import features as F
from components import connection, identity
from components import customer_form as CF
from components import counterfactual
from components import ui
from components.api_error import api_error_html, render_api_error
from components.empty_state import render_no_report

CSS_PATH = Path(__file__).with_suffix(".css")
SUMMARY_TOP = 15          # SHAP Summary 에 보여 줄 변수 수
BEESWARM_MAX_DOTS = 60    # 변수 한 줄에 찍는 점 수 (API 가 더 많이 주면 앞에서부터)
CONTRIBUTION_SUFFIX = re.compile(r"\s*\(기여도:\s*[-+−]?[\d.]+\)\s*$")
GRADE_RISK = {"A": "low", "B": "low", "C": "mid", "D": "high", "E": "high"}


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def _signed(v):
    return f"{v:+.2f}"


def _pct(p):
    return f"{p * 100:.1f}%" if p < 0.1 else f"{p * 100:.0f}%"


# ---------------------------------------------------------------------------
# 상태
# ---------------------------------------------------------------------------
def _init_state():
    st.session_state.setdefault("customers", {})
    st.session_state.setdefault("xai_explain", {})


def _get_explain(customer_id):
    """심사 실행 때 받아 둔 /explain 응답 (이 화면은 API 를 다시 부르지 않는다)."""
    return CF.get_explain(customer_id)


# ---------------------------------------------------------------------------
# 머리: 흐름 경로 + 제목 + 심사 결과로 · 조건 바꿔 재심사 · 상담 마치기
# ---------------------------------------------------------------------------
def _crumb(customer_id):
    """흐름 경로: 고객 심사 › 이름 심사 결과 › 판단 근거. 앞의 두 단계는 눌러서 돌아갈 수 있다."""
    name = st.session_state.customers[customer_id]["profile"].get("name") or "고객"
    with st.container(key="xai-crumb", horizontal=True, vertical_alignment="center", gap=None):
        go_start = st.button("고객 심사", key="xai-crumb-start", type="tertiary")
        st.html('<span class="xai-crumb-sep">›</span>')
        go_result = st.button(f"{name} 심사 결과", key="xai-crumb-result", type="tertiary")
        st.html('<span class="xai-crumb-sep">›</span><b class="xai-crumb-now">판단 근거</b>')
    if go_start:
        CF.clear_result()
        st.switch_page(CF.SCREENING_PAGE)
    if go_result:
        CF.open_result(customer_id)
        st.switch_page(CF.SCREENING_PAGE)


def _head(customer_id):
    """제목 + (상담 중인 고객이 있으면) 흐름 경로 · 심사 결과로 · 조건 바꿔 재심사 · 상담 마치기.
    창구는 한 번에 한 고객이라 다른 고객을 고르는 목록은 두지 않는다."""
    if customer_id:
        _crumb(customer_id)
    name = customer_id and st.session_state.customers[customer_id]["profile"].get("name")
    sub = (f"{escape(name)}님이 이런 결과를 받은 이유예요." if name
           else "이 고객이 왜 이런 결과를 받았는지 설명해요." if customer_id
           else "모델이 평소에 무엇을 보고 판단하는지 보여 줘요.")
    with st.container(key="xai-head", horizontal=True, vertical_alignment="bottom"):
        st.html(f'<div class="ui-page-head"><div><h1>XAI 분석</h1><p>{sub}</p></div></div>')
        if not customer_id:
            with st.container(key="xai-pick", horizontal=True, vertical_alignment="center", gap="small"):
                if st.button("고객 심사로 가기", key="xai-go"):
                    st.switch_page("views/screening.py")
                if st.button("이미 심사한 고객 찾기", key="xai-find-open", type="primary", disabled=connection.is_down()):
                    for key in ("xai_notice", "xai_errors"):
                        st.session_state.pop(key, None)
                    _find_dialog()
            return
        with st.container(key="xai-pick", horizontal=True, vertical_alignment="center", gap="small"):
            if st.button("심사 결과로", key="xai-to-result"):
                CF.open_result(customer_id)
                st.switch_page(CF.SCREENING_PAGE)
            if st.button("조건 바꿔 재심사", key="xai-rescreen", disabled=connection.is_down()):
                # 고객 심사의 3/3 전체 확인으로 (그 고객 값이 채워진다). 심사 실행 → 고객 심사 결과 화면
                CF.edit(st.session_state.customers[customer_id]["profile"])
                st.switch_page(CF.SCREENING_PAGE)
            if st.button("상담 마치기", key="xai-end"):
                CF.end_consult()
                st.switch_page(CF.SCREENING_PAGE)


def _past_lookup():
    """조회: 이름 · 주민등록번호 → 그 고객의 가장 최근 심사 → 판단 근거 (지금 상담 고객이 된다). 찾으면 True."""
    ss = st.session_state
    name = identity.clean_name(ss.get("xai_name"))
    front, back = identity.digits(ss.get("xai_front")), identity.digits(ss.get("xai_back"))
    ss.xai_errors = {"name": identity.name_error(name), "rrn": identity.rrn_error(front, back)}
    ss.xai_notice = None
    if any(ss.xai_errors.values()):
        return False
    try:
        status = CF.load_past(name, front + back, identity.masked(front, back))
    except Exception as e:
        ss.xai_notice = ("error", e)
        return False
    if status != "ok":
        ss.xai_notice = ({None: "not_found", api_client.NO_SCREENING: "no_screening",
                          api_client.NOT_READY: "pending"}[status], None)
        return False
    return True


@st.dialog("이미 심사한 고객의 판단 근거 보기", width="medium")
def _find_dialog():
    """이름 · 주민등록번호를 넣으면 그 고객의 가장 최근 심사의 판단 근거를 바로 연다 (찾으면 창이 닫힌다)."""
    ss = st.session_state
    with st.container(key="xai-find"):
        st.html('<p class="xai-find-sub">고객의 이름과 주민등록번호를 입력하면 가장 최근 심사 결과가 왜 나왔는지 바로 보여 줘요.</p>')
        with st.form("xai-find-form", border=False, enter_to_submit=True):
            st.text_input("이름", key="xai_name", placeholder="홍길동", max_chars=identity.NAME_MAX)
            with st.container(key="xai-find-row", horizontal=True, vertical_alignment="bottom", gap="small"):
                st.text_input("주민등록번호", key="xai_front", placeholder="앞 6자리", max_chars=6)
                st.html('<span class="xai-find-dash">-</span>')
                st.text_input("주민등록번호 뒷자리", key="xai_back", placeholder="뒤 7자리", max_chars=7,
                              type="password", label_visibility="hidden")
            submitted = st.form_submit_button("판단 근거 보기", key="xai-find-go", type="primary",
                                              use_container_width=True)
        if submitted and _past_lookup():
            st.rerun()  # 창을 닫고 그 고객의 판단 근거를 그린다
        for key in ("name", "rrn"):
            message = (ss.get("xai_errors") or {}).get(key)
            if message:
                st.html(f'<p class="xai-find-error">{escape(message)}</p>')
        kind, error = ss.get("xai_notice") or (None, None)
        if kind == "error":
            st.html(api_error_html("POST /screenings/lookup", error, "inline"))
        elif kind:
            text = {"not_found": "일치하는 고객 정보가 없어요. 이름과 주민등록번호를 다시 확인해 주세요.",
                    "no_screening": "이 고객은 아직 심사한 기록이 없어요. 고객 심사를 먼저 해 주세요.",
                    "pending": "심사 기록 조회 API(POST /screenings/lookup)가 아직 서버에 없어요."}[kind]
            st.html(f'<div class="xai-find-notice {kind}">{escape(text)}</div>')


def _model_overview():
    """상담 중인 고객이 없을 때: 모든 고객에게 같은 모델 전체 설명 (SHAP Summary Plot)."""
    _summary_tab()


def _customer_strip(entry):
    profile, result = entry["profile"], entry["result"]
    approved = result["approval_status"] == "APPROVED"
    cells = [
        ("이름", profile.get("name") or "-"),
        ("나이", f'{profile["age"]}세'),
        ("성별", profile.get("gender") or "-"),
        ("월 소득", f'{profile["MonthlyIncome"] / 10_000:,.0f}만원'),
        ("고객 유형", "씬파일러" if result.get("is_thin_filer") else "일반 고객"),
        ("신용 점수", f'{result["credit_score"]}점'),
        ("등급", result["risk_grade"]),
    ]
    inner = "".join(f'<div><span>{label}</span><b>{escape(str(v))}</b></div>' for label, v in cells)
    badge = f'<span class="xai-badge {"ok" if approved else "fail"}">{"승인" if approved else "거절"}</span>'
    return f'<section class="xai-strip">{inner}{badge}</section>'


def _factor_list(items, top, kind):
    """위험을 높인 · 낮춘 정보 상위 3개: 이름 ··· 작은 막대 + 기여도."""
    if not items:
        return '<p class="xs-none">해당하는 정보가 없어요</p>'
    rows = "".join(
        f'<li><span>{escape(F.label(f))}</span><i><em class="{kind}" style="width:{abs(v) / top * 100:.0f}%"></em></i>'
        f'<b class="{kind}">{_signed(v)}</b></li>' for f, v in items[:3])
    more = f'<p class="xs-more">외 {len(items) - 3}개</p>' if len(items) > 3 else ""
    return f'<ul class="xs-factors">{rows}</ul>{more}'


def _overview_tab(explain, entry):
    """요약 탭 (토스식): 결론 한 줄 + 큰 숫자 · 고객 정보 줄 목록 · 위험을 높인 / 낮춘 정보 상위 3개."""
    profile, result = entry["profile"], entry["result"]
    approved = result["approval_status"] == "APPROVED"
    name = profile.get("name") or "고객"
    shap = explain.get("shap_values") or {}
    ups = sorted([(f, v) for f, v in shap.items() if v > 0], key=lambda kv: -kv[1])
    downs = sorted([(f, v) for f, v in shap.items() if v < 0], key=lambda kv: kv[1])
    top = max([abs(v) for _, v in ups[:3] + downs[:3]] or [1]) or 1
    if approved:
        reason = (f"{_pair([F.label(f) for f, _ in downs[:2]])} 승인에 가장 크게 기여했어요." if downs
                  else "부도 위험을 크게 높인 정보 없이 승인 기준을 통과했어요.")
    else:
        reason = (f"{_pair([F.label(f) for f, _ in ups[:2]])} 거절의 가장 큰 이유예요." if ups
                  else "부도 확률이 승인 기준을 넘었어요.")
    prob, base = result["default_probability"], explain.get("base_value")
    avg = _sigmoid(base) if base is not None else None
    avg_mark = (f'<i class="avg" style="left:{avg * 100:.1f}%"></i>'
                f'<span class="avg-label" style="left:{avg * 100:.1f}%">평균 고객 {_pct(avg)}</span>') if avg is not None else ""
    tone = "ok" if approved else "fail"
    info = [("이름", name), ("나이", f'{profile["age"]}세'), ("성별", profile.get("gender") or "-"),
            ("월소득", f'{profile["MonthlyIncome"] / 10_000:,.0f}만원'),
            ("고객 유형", "씬파일러" if result.get("is_thin_filer") else "일반 고객")]
    rows = "".join(f'<li><span>{k}</span><b>{escape(str(v))}</b></li>' for k, v in info)
    return f"""
    <div class="xs">
      <section class="xs-card xs-hero">
        <div class="xs-badges"><span class="xs-badge {tone}">{"승인" if approved else "거절"}</span>
          <span class="xs-grade">{escape(str(result["risk_grade"]))}등급</span></div>
        <h2>{escape(name)}님은 이번 심사에서 {"승인됐어요" if approved else "거절됐어요"}</h2>
        <p class="xs-reason">{escape(reason)}</p>
        <div class="xs-figs">
          <div class="xs-prob"><span>부도 확률</span><strong class="{tone}">{_pct(prob)}</strong>
            <div class="xs-track"><em class="{tone}" style="width:{prob * 100:.1f}%"></em>{avg_mark}</div></div>
          <div class="xs-score"><span>신용 점수</span><strong>{result["credit_score"]}<small>점</small></strong></div>
        </div>
      </section>
      <section class="xs-card xs-info"><h3>고객 정보</h3><ul>{rows}</ul></section>
      <section class="xs-card"><h3>위험을 높인 정보 <b class="up">{len(ups)}개</b></h3>{_factor_list(ups, top, "up")}</section>
      <section class="xs-card"><h3>위험을 낮춘 정보 <b class="down">{len(downs)}개</b></h3>{_factor_list(downs, top, "down")}</section>
    </div>"""


# ---------------------------------------------------------------------------
# 결론: 한 줄 요약 + 지표 3개
# ---------------------------------------------------------------------------
def _pair(names):
    if len(names) == 1:
        return names[0] + F.josa(names[0], "이", "가")
    a, b = names[:2]
    return f'{a}{F.josa(a, "과", "와")} {b}{F.josa(b, "이", "가")}'


def _summary_html(explain, result):
    shap = explain.get("shap_values") or {}
    approved = result["approval_status"] == "APPROVED"
    ups = sorted([(f, v) for f, v in shap.items() if v > 0], key=lambda kv: -kv[1])
    downs = sorted([(f, v) for f, v in shap.items() if v < 0], key=lambda kv: kv[1])
    if approved and downs:
        line = f"{escape(_pair([F.label(f) for f, _ in downs[:2]]))}<br>승인에 가장 크게 기여했어요."
    elif approved:
        line = "부도 위험을 크게 높인 정보 없이<br>승인 기준을 통과했어요."
    elif ups:
        line = f"{escape(_pair([F.label(f) for f, _ in ups[:2]]))}<br>거절의 가장 큰 이유예요."
    else:
        line = "부도 확률이 승인 기준을 넘었어요."

    prob = result["default_probability"]
    base = explain.get("base_value")
    avg = f"평균 고객 {_pct(_sigmoid(base))}" if base is not None else ""
    n_alt = sum(1 for f, _ in ups if F.is_alternative(f))
    down_names = [F.label(f) for f, _ in downs]
    down_sub = (", ".join(down_names[:2]) + (f" 외 {len(down_names) - 2}개" if len(down_names) > 2 else "")
                ) if down_names else "없음"
    return f"""
    <section class="xai-summary">
      <div class="xai-summary-main"><span>한 줄 요약</span><p>{line}</p></div>
      <div class="xai-metric"><span>부도 확률</span>
        <strong class="{'' if approved else 'fail'}">{_pct(prob)}</strong><small>{avg}</small></div>
      <div class="xai-metric"><span>위험을 높인 정보</span><strong>{len(ups)}개</strong>
        <small>이 중 대안 데이터 {n_alt}개</small></div>
      <div class="xai-metric"><span>위험을 낮춘 정보</span><strong>{len(downs)}개</strong>
        <small>{escape(down_sub)}</small></div>
    </section>"""


# ---------------------------------------------------------------------------
# 근거: 거절 사유 Top 5
# ---------------------------------------------------------------------------
def _tag(feature):
    group = F.group_of(feature)
    return f'<span class="xai-tag {"alt" if group == "alternative" else ""}">{F.GROUP_LABELS[group]}</span>'


def _reasons(explain, result, name=None):
    reasons = explain.get("rejection_reasons") or []
    approved = result["approval_status"] == "APPROVED"
    title = "위험을 높인 정보 Top 5" if approved else "거절 사유 Top 5"
    sub = ("승인됐지만 부도 위험을 높인 정보예요." if approved
           else "부도 위험을 가장 크게 높인 순서예요. 설명 문장은 고객 안내에 그대로 쓸 수 있어요.")
    with st.container(key="xai-reasons-head", horizontal=True, vertical_alignment="bottom"):
        st.html(ui.section_head(title, sub))
        if not approved and reasons:
            notice = "\n".join(
                [f"[대출 심사 결과 안내] {name + ' ' if name else ''}고객님",
                 "고객님의 대출 신청은 아래 사유로 승인되지 않았습니다.", ""]
                + [f"{i}. {F.label(r['feature'])}: {CONTRIBUTION_SUFFIX.sub('', r.get('explanation') or '')}"
                   for i, r in enumerate(reasons, 1)]
                + ["", config.DISCLAIMER])
            with st.container(key="xai-copy"):
                ui.copy_button("안내문 복사", notice)

    phis = [-r["shap_contribution"] for r in reasons]  # 부도 위험 기준으로 뒤집는다
    top = max([abs(p) for p in phis] or [1]) or 1
    def ref_text(r):
        ref = r.get("reference_value")
        return f'평균 {F.fmt_value(r["feature"], ref)}' if ref is not None else ""

    # 토스식 목록: 순위 · 항목(+ 묶음 태그) / 설명 한 줄 ··· 지금 값(아래 작게 평균) ··· 기여도 막대.
    # 영어 변수 이름은 항목에 마우스를 올리면 보인다
    rows = "".join(f"""
      <li>
        <i class="xr-rank">{r.get("rank", i)}</i>
        <div class="xr-main">
          <div class="xr-title"><strong title="{escape(r["feature"])}">{escape(F.label(r["feature"]))}</strong>{_tag(r["feature"])}</div>
          <p>{escape(CONTRIBUTION_SUFFIX.sub("", r.get("explanation") or ""))}</p>
        </div>
        <div class="xr-value"><b>{F.fmt_value(r["feature"], r.get("current_value"))}</b><span>{escape(ref_text(r))}</span></div>
        <div class="xr-bar"><i><em class="{"up" if phi >= 0 else "down"}" style="width:{abs(phi) / top * 100:.1f}%"></em></i>
          <strong class="{"up" if phi >= 0 else "down"}">{_signed(phi)}</strong></div>
      </li>""" for i, (r, phi) in enumerate(zip(reasons, phis), 1))
    st.html(f"""
    <ol class="xr" aria-label="{title}">{rows or '<li class="xai-empty">표시할 사유가 없어요</li>'}</ol>
    <p class="xai-note">막대와 숫자는 부도 위험을 얼마나 높였는지예요 (SHAP 기여도, 클수록 위험이 커요).</p>""")


# ---------------------------------------------------------------------------
# 상세 1: SHAP Waterfall (HTML 막대)
# ---------------------------------------------------------------------------
def _nice_step(span):
    for step in (0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5):
        if span / step <= 7:
            return step
    return 10


def _waterfall(explain, entry):
    shap = explain.get("shap_values") or {}
    base = explain.get("base_value")
    if not shap or base is None:
        return '<div class="xai-empty">기여도 정보가 없어 그래프를 그릴 수 없어요.</div>'
    payload, result = entry["payload"], entry["result"]
    ordered = sorted(shap.items(), key=lambda kv: abs(kv[1]), reverse=True)
    shown, rest = ordered[:config.WATERFALL_MAX_DISPLAY], ordered[config.WATERFALL_MAX_DISPLAY:]
    rows = [(f'{escape(F.label(f))} <b>{escape(F.fmt_input(f, payload.get(f)))}</b>', v) for f, v in shown]
    if rest:
        rows.append((f"기타 {len(rest)}개 변수", sum(v for _, v in rest)))

    points, x = [base], base
    for _, v in rows:
        x += v
        points.append(x)
    fx = points[-1]
    lo, hi = min(points), max(points)
    pad = (hi - lo) * 0.12 or 0.5
    lo, hi = lo - pad, hi + pad
    pos = lambda v: (v - lo) / (hi - lo) * 100  # noqa: E731
    step = _nice_step(hi - lo)
    ticks = [round(t * step, 6) for t in range(math.ceil(lo / step), math.floor(hi / step) + 1)]
    grid = "".join(f'<i class="g" style="left:{pos(t):.2f}%"></i>' for t in ticks)

    html = [f'<div class="wf-row"><span class="wf-label muted">평균 고객</span><span class="wf-track">{grid}'
            f'<i class="mark" style="left:{pos(base):.2f}%"></i>'
            f'<em class="mark-label" style="left:calc({pos(base):.2f}% + 10px)">E[f(x)] = {_signed(base).lstrip("+")}</em>'
            f'</span><span></span></div>']
    start = base
    for label, v in rows:
        end = start + v
        left, width = pos(min(start, end)), abs(pos(end) - pos(start))
        kind = "up" if v >= 0 else "down"
        html.append(f'<div class="wf-row"><span class="wf-label">{label}</span><span class="wf-track">{grid}'
                    f'<i class="link" style="left:{pos(end):.2f}%"></i>'
                    f'<i class="bar {kind}" style="left:{left:.2f}%;width:{max(width, 0.4):.2f}%"></i></span>'
                    f'<strong class="{kind}">{_signed(v)}</strong></div>')
        start = end
    right_side = pos(fx) > 60
    label_pos = (f"right:calc({100 - pos(fx):.2f}% + 10px)" if right_side else f"left:calc({pos(fx):.2f}% + 10px)")
    who = entry["profile"].get("name") or "이 고객"
    html.append(f'<div class="wf-row wf-final"><strong class="wf-label">{escape(who)}{"님" if entry["profile"].get("name") else ""}</strong><span class="wf-track">{grid}'
                f'<i class="mark ink" style="left:{pos(fx):.2f}%"></i>'
                f'<em class="mark-label b" style="{label_pos}">f(x) = {_signed(fx).lstrip("+")}</em></span><span></span></div>')
    axis = "".join(f'<em style="left:{pos(t):.2f}%">{t:g}</em>' for t in ticks)
    html.append(f'<div class="wf-row wf-axis"><span></span><span class="wf-ticks">{axis}</span><span></span></div>')

    approved = result["approval_status"] == "APPROVED"
    n_up = sum(1 for v in shap.values() if v > 0)
    n_down = sum(1 for v in shap.values() if v < 0)
    prob = result["default_probability"]
    who_full = f'{escape(entry["profile"]["name"])}님' if entry["profile"].get("name") else "이 고객"
    return f"""
    <div class="wf2-head">
      <div>
        <h3>평균 고객 <b>{_pct(_sigmoid(base))}</b><span class="wf2-arrow">→</span>{who_full} <b class="{'ok' if approved else 'fail'}">{_pct(prob)}</b></h3>
        <p>평균 고객에서 출발해, 정보 하나하나가 부도 위험을 더하고 뺐어요. 높인 정보 {n_up}개 · 낮춘 정보 {n_down}개</p>
      </div>
      <div class="wf2-legend"><span><i class="up"></i>위험 증가</span><span><i class="down"></i>위험 감소</span></div>
    </div>
    <div class="wf" role="img" aria-label="SHAP 워터폴 플롯: 평균 {base:.2f}에서 이 고객 {fx:.2f}까지">{"".join(html)}</div>
    <p class="wf2-note">SHAP Waterfall Plot · 가로축은 SHAP 값(log-odds)이에요. 오른쪽으로 갈수록 부도 위험이 커요.</p>"""


def _dot_color(t):
    """변수 값 0(낮음)~1(높음) → 파랑(#3182F6)~빨강(#F04452). workspace.css 의 --risk-down · --risk-up 과 같다."""
    t = min(max(float(t), 0.0), 1.0)
    a, b = (49, 130, 246), (240, 68, 82)
    return "rgb({},{},{})".format(*(round(a[i] + (b[i] - a[i]) * t) for i in range(3)))


def _jitter(feature, i):
    h = hashlib.md5(f"{feature}:{i}".encode()).digest()
    return 15 + h[0] / 255 * 70  # 줄 높이의 15~85% (점 가운데 기준, xai.css .bs-track i 의 margin-top)


def _share_head(features):
    """머리 오른쪽: 대안 데이터 비중 큰 숫자 + 세 묶음 막대 + 범례 한 줄."""
    totals = {"financial": 0.0, "alternative": 0.0, "basic": 0.0}
    for x in features:
        totals[F.group_of(x["feature"])] += x["mean_abs_shap"]
    total = sum(totals.values()) or 1
    share = {g: v / total * 100 for g, v in totals.items()}
    order = ["financial", "alternative", "basic"]
    bars = "".join(f'<i class="{g}" style="width:{share[g]:.1f}%"></i>' for g in order)
    legend = " · ".join(f'<span><i class="{g}"></i>{F.GROUP_LABELS[g]} {share[g]:.0f}%</span>' for g in order)
    return (f'<div class="bs2-share"><span>대안 데이터 비중 <strong>{share["alternative"]:.0f}%</strong></span>'
            f'<div class="xai-share-bar" role="img" aria-label="영향력 비중">{bars}</div>'
            f'<div class="bs2-share-legend">{legend}</div></div>')


def _how_to_read():
    """그림 읽는 법 (오른쪽 칸)."""
    items = [("위치", "가운데 선 오른쪽은 위험을 높였고, 왼쪽은 낮췄어요"),
             ("퍼진 폭", "넓게 퍼질수록 고객마다 영향이 달라요"),
             ("순서", "위에 있을수록 모델이 많이 봐요"),
             ("파란 이름", "통신비 · 공과금 같은 대안 데이터예요")]
    rows = "".join(f"<li><b>{k}</b><span>{v}</span></li>" for k, v in items)
    return f'<div class="xai-howto"><strong>읽는 법</strong><ul>{rows}</ul></div>'


def _beeswarm_head(summary):
    """카드 머리 (토스식): '모델은 ○○과 ○○를 가장 많이 봐요' 큰 한 줄 + 대안 데이터 비중."""
    feats = sorted(summary.get("features", []), key=lambda x: x["mean_abs_shap"], reverse=True)
    names = [F.label(x["feature"]) for x in feats[:2]]
    if len(names) == 2:
        title = f'모델은 {names[0]}{F.josa(names[0], "과", "와")} {names[1]}{F.josa(names[1], "을", "를")} 가장 많이 봐요'
    elif names:
        title = f'모델은 {names[0]}{F.josa(names[0], "을", "를")} 가장 많이 봐요'
    else:
        title = "모델이 중요하게 보는 정보"
    return (f'<div class="bs2-head"><div><h3>{escape(title)}</h3>'
            f'<p>점 하나가 고객 한 명, 위에 있을수록 영향이 커요 · SHAP Summary Plot 상위 {SUMMARY_TOP}개</p></div>'
            f'{_share_head(summary.get("features", []))}</div>')


def _beeswarm(summary, reason_features):
    features = sorted(summary.get("features", []), key=lambda x: x["mean_abs_shap"], reverse=True)[:SUMMARY_TOP]
    swarm = summary.get("beeswarm") or {}   # {feature: {"shap": [...], "value": [0~1 ...]}}. 예전 서버엔 없다
    if swarm:
        values = [v for f in swarm.values() for v in f.get("shap", [])[:BEESWARM_MAX_DOTS]]
        lo, hi = min(values + [0]), max(values + [0])
    else:
        top = max([x["mean_abs_shap"] for x in features] or [1])
        lo, hi = -top, top * 2
    span = (hi - lo) or 1
    pos = lambda v: (v - lo) / span * 100  # noqa: E731

    rows = []
    for rank, x in enumerate(features, 1):
        f = x["feature"]
        mine = f in reason_features
        name_cls = "alt" if F.is_alternative(f) else ""
        tag = '<i class="xai-mine">이 고객 사유</i>' if mine else ""
        data = swarm.get(f)
        if data:
            dots = "".join(
                f'<i style="left:{pos(s):.2f}%;top:{_jitter(f, i):.1f}%;background:{_dot_color(v)}"></i>'
                for i, (s, v) in enumerate(zip(data.get("shap", [])[:BEESWARM_MAX_DOTS], data.get("value", []))))
        else:  # 고객별 점이 아직 없으면 평균 영향 크기(|SHAP| 평균)를 0 양쪽 띠로 보여 준다
            m = x["mean_abs_shap"]
            dots = f'<b class="band" style="left:{pos(-m):.2f}%;width:{pos(m) - pos(-m):.2f}%"></b>'
        direction = F.direction_text(f, x.get("higher_is_riskier"))
        rows.append(f"""
        <li class="{"mine" if mine else ""}">
          <span class="bs-rank">{rank}</span>
          <span class="bs-name"><span class="{name_cls}" title="{escape(f)}">{escape(F.label(f))}</span>{tag}</span>
          <span class="bs-track"><i class="zero" style="left:{pos(0):.2f}%"></i>{dots}</span>
          <span class="bs-dir">{direction}</span>
        </li>""")
    note = ("" if swarm else
            '<p class="xai-note warn">고객별 점 분포는 API(/explain/summary 의 beeswarm)가 추가되면 표시돼요. '
            "지금은 변수별 평균 영향 크기(|SHAP| 평균)를 띠로 보여줘요.</p>")
    color_legend = ('<div class="bs-color">점 색은 그 고객의 변수 값이에요.<br>낮음<i></i>높음</div>' if swarm else "")
    return f"""
    {_beeswarm_head(summary)}
    <div class="xai-sum-grid">
      <div>
        <ol class="bs" role="img" aria-label="SHAP 요약 플롯, 상위 {len(features)}개 변수">{"".join(rows)}</ol>
        <div class="bs-axis"><span></span><span></span><span class="bs-axis-track">
          <em class="minus" style="right:calc({100 - pos(0):.2f}% + 44px)">← 위험 감소</em>
          <em class="zero" style="left:{pos(0):.2f}%">SHAP 값</em>
          <em class="plus" style="left:calc({pos(0):.2f}% + 44px)">위험 증가 →</em></span><span></span></div>
      </div>
      <aside>{_how_to_read()}{note}{color_legend}</aside>
    </div>"""


def _waterfall_tab(explain, entry):
    """탭: 부도 확률이 정해진 과정 (SHAP Waterfall). 제목 · 설명은 카드 안 머리 한 줄로."""
    st.html(f'<div class="xai-card xai-wf-card">{_waterfall(explain, entry)}</div>')


def _summary_tab(reason_features=frozenset()):
    """모델이 중요하게 보는 정보 (SHAP Summary). 탭 안 · 고객이 없을 때 같은 카드. 이 고객의 사유인 변수는 표시한다."""
    try:
        summary = api_client.shap_summary()
    except Exception as e:
        render_api_error("GET /explain/summary", e)
        return
    if not summary:
        render_no_report("shap_summary")
        return
    st.html(f'<div class="xai-card xai-bs-card">{_beeswarm(summary, set(reason_features))}</div>')


# ---------------------------------------------------------------------------
# 페이지
# ---------------------------------------------------------------------------
st.html(f"<style>{CF.WORKSPACE_CSS}{CSS_PATH.read_text(encoding='utf-8')}</style>")
_init_state()
customer_id = CF.current_customer()
_head(customer_id)

if customer_id is None:
    _model_overview()
else:
    entry = st.session_state.customers[customer_id]
    explain_error = None
    try:
        explain = _get_explain(customer_id)
    except Exception as e:
        explain, explain_error = None, e
    overview = (f'<section class="xai-overview">{_customer_strip(entry)}'
                f'{_summary_html(explain, entry["result"]) if explain else ""}</section>')
    if explain is None:
        st.html(overview)
        render_api_error("POST /explain", explain_error)
    else:
        # 고객 정보 + 한 줄 요약도 탭 하나로 (아래 탭들이 화면 높이를 넉넉히 쓰게)
        rejected = entry["result"]["approval_status"] != "APPROVED"
        names = ["요약", "거절 사유 Top 5" if rejected else "위험을 높인 정보 Top 5"]
        names += ["승인되려면"] if rejected else []
        names += ["부도 확률이 정해진 과정", "모델이 중요하게 보는 정보"]
        tabs = dict(zip(names, st.tabs(names)))
        with tabs["요약"]:
            st.html(_overview_tab(explain, entry))
        with tabs[names[1]]:
            _reasons(explain, entry["result"], entry["profile"].get("name"))
        if rejected:
            with tabs["승인되려면"]:
                counterfactual.render(customer_id, entry["payload"], entry["result"])
        with tabs["부도 확률이 정해진 과정"]:
            _waterfall_tab(explain, entry)
        with tabs["모델이 중요하게 보는 정보"]:
            _summary_tab({r["feature"] for r in explain.get("rejection_reasons") or []})
