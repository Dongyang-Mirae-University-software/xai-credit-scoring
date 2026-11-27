"""고객 심사 페이지 (은행 창구 기준, 한 화면씩 넘긴다)

흐름 (components/customer_form)
- 1/3 본인 확인: 이름 + 주민등록번호 → 정보 조회
- 2/3 신청 정보: 월소득 · 부양가족 수 → 동의(개인 · 신용 · 대안정보) → 불러오기 연출
- 3/3 전체 확인: 모든 정보 → 틀린 항목만 고치고 심사 실행 (결과는 서버 이력에 바로 남는다)
- 결과 화면: 정보 수정 → 3/3, 이전 → 1/3
고객 번호(customer_id)는 화면에 보이지 않는다. 고객 구분 · 서버 이력 · XAI 목록 같은 내부 구분에만 쓴다.
- 로고 · 사이드바 '고객 심사' → 언제나 1/3
"""
import math
import re
from datetime import datetime
from html import escape
from pathlib import Path

import streamlit as st

import api_client
import config
import features as F
from components import connection
from components import customer_form as CF
from components import ui
from components.api_error import api_error_html

CSS_PATH = Path(__file__).with_suffix(".css")
REASON_LIMIT = 3  # 결과 화면의 주요 거절 사유 수 (전체 5개는 XAI 분석)

# 등급 → 위험도 (A·B 저위험 / C 중위험 / D·E 고위험) → 색 (ok / mid / fail)
GRADE_RISK = {"A": "low", "B": "low", "C": "mid", "D": "high", "E": "high"}
RISK_LABELS = {"low": "저위험", "mid": "중위험", "high": "고위험"}
RISK_TONE = {"low": "ok", "mid": "mid", "high": "fail"}
CONTRIBUTION_SUFFIX = re.compile(r"\s*\(기여도:\s*[-+−]?[\d.]+\)\s*$")



# ---------------------------------------------------------------------------
# 첫 화면 (디자인 01)
# ---------------------------------------------------------------------------
def _start_view():
    """1/3 본인 확인 카드."""
    CF.render_identify(disabled=connection.is_down())  # API 가 꺼져 있으면 조회 · 진행 버튼을 끈다 (시안 #api-down)


# ---------------------------------------------------------------------------
# 결과 화면 (디자인 03 승인 / 04 거절)
# ---------------------------------------------------------------------------
def _back():
    """이전: 결과 화면을 닫고 1/3 본인 확인으로."""
    CF.clear_result()


def _completed_at(result):
    try:
        stamp = datetime.fromisoformat(result["timestamp"]).astimezone()
    except (KeyError, ValueError):
        stamp = datetime.now()
    return stamp.strftime("%Y.%m.%d %H:%M")


def _summary_html(profile, result):
    counts = [profile[f] for f, _ in CF.DELINQUENCY_FIELDS if profile[f] is not None]  # None = 기록 없음
    late = sum(int(c) for c in counts)
    cells = [
        ("이름", profile.get("name") or "-"),
        ("나이", f'{profile["age"]}세'),
        ("성별", profile.get("gender") or "-"),
        ("월 소득", f'{profile["MonthlyIncome"] / 10_000:,.0f}만원'),
        ("부채비율", f'{profile["DebtRatio"]:.0f}%'),
        ("연체", f"{late}회" if late else ("기록 없음" if not counts else "없음")),
        ("고객 유형", "씬파일러" if result.get("is_thin_filer") else "일반 고객"),
    ]
    return '<div class="scr-summary">' + "".join(
        f'<div><span>{label}</span><b>{escape(str(value))}</b></div>' for label, value in cells) + "</div>"


def _health():
    try:
        return api_client.health()
    except Exception:
        return {}


def _approval_score(result):
    """승인 기준 점수: 그 결과를 낸 /predict 응답의 approval_score, 없으면 /health 의 것. 둘 다 없으면 None (점선 없음)."""
    return result.get("approval_score", _health().get("approval_score"))


def _bar_pos(score, scale, low):
    """점수 → 막대 위치(%). 다섯 칸을 같은 폭으로 그리고 칸 안에서는 점수 비율로 놓는다
    (확률 보정 전이라 C·D 구간이 20점 안팎으로 좁아서, 점수 길이 그대로면 칸이 안 보인다)."""
    lowers = [low] + [upper for _, upper in scale[:-1]]
    for i, ((_, upper), lower) in enumerate(zip(scale, lowers)):
        if score < upper or i == len(scale) - 1:
            return (i + min(max((score - lower) / (upper - lower), 0), 1)) / len(scale) * 100


def _score_bar(score, grade, approval_score):
    """E~A 다섯 칸(구간은 API /health 의 grade_scale)과 점수 위치, 승인 기준 점선을 그린다. 구간을 못 받으면 막대 없이 문구만."""
    health = _health()
    scale, (low, high) = health.get("grade_scale"), health.get("score_range") or (300, 850)
    note = f'<p class="scr-bar-note">점수는 PDO 스코어카드 방식({low}~{high}점)이에요. 20점 오를 때마다 부도 위험(odds)이 절반이 돼요.'
    if not scale:
        return note + "</p>"
    widths = " ".join("1fr" for _ in scale)
    tone = RISK_TONE[GRADE_RISK.get(grade, "high")]
    segs = "".join(f'<i class="seg-{g}"></i>' for g, _ in scale)
    labels = "".join(f'<b class="{tone}">{g}</b>' if g == grade else f"<span>{g}</span>" for g, _ in scale)
    line = ""
    if approval_score is not None:
        line = f'<u style="left:{_bar_pos(approval_score, scale, low):.1f}%"></u>'
        note += f" 점선은 승인 기준 {approval_score}점이에요."
    return (f'<div class="scr-bar" style="grid-template-columns:{widths}">{segs}'
            f'{line}<em style="left:{_bar_pos(score, scale, low):.1f}%"></em></div>'
            f'<div class="scr-bar-labels" style="grid-template-columns:{widths}">{labels}</div>{note}</p>')


def _decision_html(result):
    approved = result["approval_status"] == "APPROVED"
    grade, score, prob = result["risk_grade"], result["credit_score"], result["default_probability"]
    risk = GRADE_RISK.get(grade, "high")
    message = ("신용 점수와 부도 확률 모두 승인 기준을 충족했어요." if approved
               else "부도 확률이 승인 기준을 넘어 대출이 어려워요. 아래 사유를 고객에게 안내해 주세요.")
    mark = ui.icon("check" if approved else "x", "#FFFFFF", 30)
    return f"""
    <section class="scr-decision {'ok' if approved else 'fail'}">
      <div class="scr-banner">
        <div class="scr-banner-main">
          <span class="scr-mark">{mark}</span>
          <div><strong>{'승인' if approved else '거절'} <small>{'APPROVED' if approved else 'REJECTED'}</small></strong>
            <p>{message}</p></div>
        </div>
        <span class="scr-grade {RISK_TONE[risk]}"><b>{escape(grade)}등급</b><small>{RISK_LABELS[risk]}</small></span>
      </div>
      <div class="scr-figures">
        <div class="scr-score">
          <div class="scr-score-top"><span>신용 점수</span><strong>{score}<small> / {_health().get("score_range", [300, 850])[1]}</small></strong></div>
          <div class="scr-bar-wrap">{_score_bar(score, grade, _approval_score(result))}</div>
        </div>
        <div class="scr-prob">
          <span>부도 확률</span>
          <strong class="{'' if approved else 'fail'}">{prob * 100:.1f}<small>%</small></strong>
          <p>100명 중 약 {round(prob * 100)}명이 2년 내 연체할 수준</p>
        </div>
      </div>
    </section>"""


def _reason_text(reason):
    """거절 사유 문장 (대안정보는 XAI 분석 · 3/3 과 같은 원천 기록 모양으로)."""
    return F.reason_sentence(reason["feature"], reason.get("current_value"), reason.get("reference_value"),
                             CONTRIBUTION_SUFFIX.sub("", reason.get("explanation") or ""))


# /predict 요청은 % 로 보내고 /explain 의 current_value 는 0~1 로 오는 칸
PERCENT_FIELDS = {"DebtRatio", "telecom_payment_rate", "utility_payment_rate"}
FACTOR_LIMIT = 3  # 결과 화면의 주요 승인 요인 수


def _risk_shap(explain, reason):
    """부도 위험 기준 SHAP 기여도 (양수 = 위험 증가). shap_values 가 부도 쪽 값이라 그것을 쓰고,
    없으면 rejection_reasons 의 승인 쪽 값(shap_contribution)을 뒤집는다."""
    value = (explain.get("shap_values") or {}).get(reason["feature"])
    if value is None and reason.get("shap_contribution") is not None:
        value = -reason["shap_contribution"]
    return value


def _approval_factors(explain, profile):
    """주요 승인 요인 3개: /explain 의 top_factors. 서버가 아직 주지 않으면 shap_values 에서
    부도 위험을 가장 많이 낮춘(음수) 변수 3개를 고르고, 현재 값은 입력값에서 가져온다 (기준 값은 없음)."""
    factors = explain.get("top_factors")
    if factors:
        return factors[:FACTOR_LIMIT]
    lowest = sorted((v, f) for f, v in (explain.get("shap_values") or {}).items() if v < 0)[:FACTOR_LIMIT]
    rows = []
    for _, feature in lowest:
        current = profile.get(feature)
        if current is not None and feature in PERCENT_FIELDS:
            current = current / 100
        rows.append({"feature": feature, "current_value": current, "reference_value": None,
                     "explanation": f"{F.name(feature)} 값이 부도 위험을 낮췄어요."})
    return rows


def _shap_cell(value):
    if value is None:
        return '<strong class="num">-</strong>'
    return f'<strong class="num {"up" if value >= 0 else "down"}">{"+" if value >= 0 else "-"}{abs(value):.2f}</strong>'


def _factor_table(label, explain, rows):
    body = "".join(
        f'<div class="scr-rr-row"><span class="scr-rr-rank">{r.get("rank", i)}</span>'
        f'<strong>{escape(F.name(r["feature"]))}</strong>'
        f'<span class="scr-rr-text">{escape(_reason_text(r))}</span>'
        f'<b class="num">{F.fmt_value(r["feature"], r.get("current_value"))}</b>'
        f'<span class="num scr-rr-ref">{F.fmt_value(r["feature"], r.get("reference_value"))}</span>'
        f'{_shap_cell(_risk_shap(explain, r))}</div>'
        for i, r in enumerate(rows, 1))
    st.html(f"""
    <div class="scr-rr" role="table" aria-label="{label}">
      <div class="scr-rr-row scr-rr-headrow"><span>순위</span><span>항목</span><span>설명</span>
        <span class="num">현재</span><span class="num">기준</span><span class="num">SHAP 기여도</span></div>
      {body or f'<div class="scr-rr-empty">{label}가 없어요</div>'}
    </div>
    <p class="scr-rr-note">SHAP 기여도는 부도 위험 기준이에요. 양수는 위험을 높이고, 음수는 위험을 낮췄다는 뜻이에요.</p>""")


def _factors_section(result, profile, approved):
    """승인: 주요 승인 요인 3개 (top_factors). 거절: 주요 거절 사유 3개 (rejection_reasons) + 안내문 복사."""
    try:
        explain = CF.get_explain(result["customer_id"])
    except Exception as e:
        st.html(api_error_html("POST /explain", e, "banner"))
        return
    if approved:
        with st.container(key="scr-reasons-head", horizontal=True, vertical_alignment="bottom"):
            st.html(ui.section_head("주요 승인 요인", "부도 위험을 가장 많이 낮춘 정보예요. "
                                                    "모든 심사 결과에는 SHAP 기반 설명을 함께 제공해요."))
        _factor_table("주요 승인 요인", explain, _approval_factors(explain, profile))
        return
    reasons = (explain.get("rejection_reasons") or [])[:REASON_LIMIT]
    with st.container(key="scr-reasons-head", horizontal=True, vertical_alignment="bottom"):
        st.html(ui.section_head("주요 거절 사유",
                                "자동화 평가 결과는 고객에게 주요 사유를 설명해야 해요 (신용정보법 제36조의2)."))
        notice = "\n".join(
            [f"[대출 심사 결과 안내] {profile['name'] + ' ' if profile.get('name') else ''}고객님",
             "고객님의 대출 신청은 아래 사유로 승인되지 않았습니다.", ""]
            + [f"{i}. {F.name(r['feature'])}: {_reason_text(r)}" for i, r in enumerate(reasons, 1)]
            + ["", config.DISCLAIMER])
        with st.container(key="scr-copy"):
            ui.copy_button("안내문 복사", notice)
    _factor_table("주요 거절 사유", explain, reasons)


def _xai_preview(result):
    """판단 근거 미리 보기: 평균 고객 → 이 고객 부도 확률, 위험을 높인 · 낮춘 정보 수, XAI 화면에 있는 것.
    카드 전체가 눌리는 버튼 (보이는 내용은 HTML, 위에 투명 버튼을 덮는다). 누르면 이 고객의 XAI 분석."""
    approved = result["approval_status"] == "APPROVED"
    try:
        explain = CF.get_explain(result["customer_id"])
    except Exception:
        explain = {}
    shap = explain.get("shap_values") or {}
    base = explain.get("base_value")
    flow = ""
    if base is not None:
        avg = 1 / (1 + math.exp(-base))
        flow = (f'<div class="scr-xp-flow"><span>평균 고객<b>{avg * 100:.0f}%</b></span>'
                f'{ui.icon("arrow", "#8B95A1", 18)}'
                f'<span>이 고객<b class="{"" if approved else "fail"}">{result["default_probability"] * 100:.0f}%</b></span>'
                f'<em>위험을 높인 정보 {sum(v > 0 for v in shap.values())}개 · '
                f'낮춘 정보 {sum(v < 0 for v in shap.values())}개</em></div>')
    chips = "".join(f"<i>{c}</i>" for c in ("SHAP Waterfall", "사유 Top 5 전체", "승인되려면", "모델 전체 중요도"))
    with st.container(key="scr-act-xai"):
        st.html(f'<div class="scr-act scr-xp"><div class="scr-xp-main">'
                f'<b>판단 근거 <small>XAI 분석</small></b>'
                f'<span>부도 확률이 어떻게 {"정해졌는지" if approved else "올라갔는지"} 정보 하나하나의 영향으로 보여 줘요.</span>'
                f'{flow}<div class="scr-xp-chips">{chips}</div></div>'
                f'<span class="scr-xp-go">자세히 보기{ui.icon("chevron", "#3182F6", 18)}</span></div>')
        return st.button("판단 근거 자세히 보기", key="scr-act-btn-xai")


def _result_view():
    current = st.session_state.scr_current
    profile, result = current["profile"], current["result"]
    approved = result["approval_status"] == "APPROVED"

    with st.container(key="scr-result-head", horizontal=True, vertical_alignment="bottom"):
        st.html(f'<div class="ui-page-head"><div><h1>심사 결과</h1>'
                f'<p>{_completed_at(result)} 심사 완료</p></div></div>')
        with st.container(key="scr-result-actions", horizontal=True, gap="small"):
            st.button("이전", key="scr-back", on_click=_back)
            st.button("상담 마치기", key="scr-end", on_click=CF.end_consult)

    with st.container(key="scr-summary-card", horizontal=True, vertical_alignment="center"):
        st.html(_summary_html(profile, result))
        st.button("정보 수정", key="scr-edit", on_click=CF.edit, args=(profile,),
                  disabled=connection.is_down())

    st.html(_decision_html(result))

    _factors_section(result, profile, approved)

    if _xai_preview(result):
        st.switch_page("views/xai.py")


# ---------------------------------------------------------------------------
# 페이지
# ---------------------------------------------------------------------------
st.html(f"<style>{CF.workspace_css()}{CSS_PATH.read_text(encoding='utf-8')}{CF.css()}"
        + ui.mask_css(".st-key-frm-back button::before", "chevron", 16)
        + ui.mask_css(".st-key-apl-back button::before", "chevron", 16)
        + ui.mask_css(".st-key-agr-back button::before", "chevron", 16)
        + ui.mask_css(".st-key-scr-back button::before", "chevron", 16)
        + ui.mask_css(".st-key-scr-edit button::before", "pencil", 14)
        + "</style>")
CF.init_state()
CF.take_pending()
_toast = st.session_state.pop("consult_toast", None)
if _toast:
    st.toast(_toast, icon=":material/check_circle:")


if st.session_state.scr_current is not None:
    _result_view()
elif st.session_state.scr_step == "apply":
    CF.render_apply(disabled=connection.is_down())
elif st.session_state.scr_step == "agree":
    CF.render_agree(disabled=connection.is_down())
elif st.session_state.scr_step == "loading":
    CF.render_loading()
elif st.session_state.scr_step == "judging":
    CF.render_judging()
elif st.session_state.scr_step == "confirm":
    CF.render_confirm(disabled=connection.is_down())
else:
    _start_view()
