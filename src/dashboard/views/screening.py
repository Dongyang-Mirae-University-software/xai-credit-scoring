"""고객 심사 페이지 (디자인 01~04)

흐름 (디자인 시안의 프로토타입 동작과 같다)
- 첫 화면: 불러오기 · 시연용 샘플 칩 → 값이 채워진 입력 창 (제목 옆 고객 ID), 신규 고객 심사 → 빈 입력 창
- 입력 창(components/customer_form)에서 심사 실행 → 결과 화면. API 를 부르는 것은 심사 실행뿐이다
- 오늘 심사한 고객 줄 → 그 고객의 결과 화면 바로
- 결과 화면: 정보 수정 → 입력 창, 새 심사 → 첫 화면
- 로고 · 사이드바 '고객 심사' → 언제나 첫 화면
'불러오기'·샘플은 API 를 부르지 않고 로컬 고객 파일(samples.json)과 이번 세션에서 심사한 고객에서 찾는다.
오늘 심사한 고객은 이번 세션에서 심사한 것이다 (심사 결과 저장 기능은 없다).
"""
import re
from datetime import datetime
from html import escape
from pathlib import Path

import streamlit as st

import config
import features as F
from components import connection
from components import customer_form as CF
from components import ui
from components.api_error import api_error_html

CSS_PATH = Path(__file__).with_suffix(".css")
HISTORY_LIMIT = 5
RISK_DOT = {"low": "#0E7C5A", "mid": "#E39A1B", "high": "#C62828"}
REASON_LIMIT = 3  # 결과 화면의 주요 거절 사유 수 (전체 5개는 XAI 분석)

# 등급 → 위험도 (A·B 저위험 / C 중위험 / D·E 고위험) → 색 (ok / mid / fail)
GRADE_RISK = {"A": "low", "B": "low", "C": "mid", "D": "high", "E": "high"}
RISK_LABELS = {"low": "저위험", "mid": "중위험", "high": "고위험"}
RISK_TONE = {"low": "ok", "mid": "mid", "high": "fail"}
CONTRIBUTION_SUFFIX = re.compile(r"\s*\(기여도:\s*[-+−]?[\d.]+\)\s*$")



# ---------------------------------------------------------------------------
# 첫 화면 (디자인 01)
# ---------------------------------------------------------------------------
def _toast_later(message, error=False):
    """콜백 안에서 띄운 안내를 다음 렌더에서 토스트로 보여 준다."""
    st.session_state.scr_toast = (message, error)


def _load_by_id():
    """불러오기: 찾으면 '불러왔어요' 입력 창, 비었거나 없으면 오류 토스트 + 입력칸 빨간 테두리."""
    ss = st.session_state
    customer_id = (ss.scr_load_id or "").strip().upper()
    ss.scr_load_invalid = True
    if not customer_id:
        _toast_later(f"고객 ID를 입력해 주세요. 예: {CF.SAMPLE_CHIPS[0][0]}", error=True)
        return
    profile = CF.find_customer(customer_id)
    if not profile:
        _toast_later(f"{customer_id} 고객을 찾을 수 없어요. 고객 ID를 다시 확인해 주세요.", error=True)
        return
    ss.scr_load_invalid = False
    CF.open_form(profile, loaded=True)


def _open_sample(customer_id):
    """시연용 샘플 칩: 그 고객 값이 채워진 입력 창을 연다 (고객 파일에서 읽는다)."""
    CF.open_form(CF.find_customer(customer_id), loaded=True)


def _open_history(customer_id):
    """오늘 심사한 고객 줄: 그 고객의 결과 화면으로."""
    if not CF.show_result(customer_id):
        _toast_later(f"{customer_id} 고객의 심사 결과를 찾지 못했어요.", error=True)


def _history_entries():
    """오늘 심사한 고객 (이번 세션). 오늘 것만, 고객당 최신 1건, 최신순 5건."""
    today = datetime.now().date()
    rows = [{**h["result"], "at": h["at"]} for h in st.session_state.scr_history if h["at"].date() == today]
    return sorted(rows, key=lambda r: r["at"], reverse=True)[:HISTORY_LIMIT]


def _customer_type(r):
    """고객 유형: /predict 응답의 is_thin_filer."""
    thin = r.get("is_thin_filer")
    return "—" if thin is None else ("씬파일러" if thin else "일반")


def _history_row(r):
    approved = r["approval_status"] == "APPROVED"
    grade = str(r["risk_grade"])
    risk = GRADE_RISK.get(grade, "high")
    return (f'<tr><td class="scr-id">{escape(r["customer_id"])}</td>'
            f'<td class="scr-type">{_customer_type(r)}</td>'
            f'<td>{ui.pill("승인" if approved else "거절", "ok" if approved else "fail")}</td>'
            f'<td><span class="scr-risk"><i style="background:{RISK_DOT[risk]}"></i><b>{escape(grade)}</b>'
            f'<span>{RISK_LABELS[risk]}</span></span></td>'
            f'<td class="num">{r["credit_score"]}</td>'
            f'<td class="scr-time">오늘 {r["at"].strftime("%H:%M")}</td>'
            f'<td class="scr-go">{ui.icon("chevron", "#8A94A6", 16)}</td></tr>')


def _history():
    """오늘 심사한 고객 표. 줄 전체가 눌리고 (위에 투명 버튼), 누르면 그 고객의 결과 화면이 열린다."""
    colgroup = "<colgroup>" + "<col>" * 6 + '<col style="width:40px"></colgroup>'
    entries = _history_entries()
    with st.container(key="scr-history"):
        st.html('<div class="scr-history-head"><h2>오늘 심사한 고객</h2>'
                '<span>줄을 누르면 심사 결과를 다시 볼 수 있어요</span></div>')
        with st.container(key="scr-history-card"):
            st.html(f'<table class="scr-htable">{colgroup}<thead><tr><th>고객 ID</th><th>고객 유형</th><th>결과</th>'
                    '<th>리스크 등급</th><th class="num">신용 점수</th><th>심사 시각</th><th></th></tr></thead></table>')
            if not entries:
                st.html('<p class="scr-empty">오늘 심사한 고객이 아직 없어요. 심사를 실행하면 여기에 쌓여요.</p>')
            for i, r in enumerate(entries):
                customer_id = r["customer_id"]
                with st.container(key=f"scr-hrow-{i}"):
                    st.html(f'<table class="scr-htable">{colgroup}<tbody>{_history_row(r)}</tbody></table>')
                    st.button(f"{customer_id} 심사 결과 열기", key=f"scr-hrow-btn-{i}",
                              on_click=_open_history, args=(customer_id,))


def _start_view():
    down = connection.is_down()  # API 가 꺼져 있으면 심사로 이어지는 버튼을 끈다 (시안 #api-down)
    with st.container(key="scr-start-head", horizontal=True, vertical_alignment="bottom"):
        st.html('<div class="ui-page-head"><div><h1>고객 심사</h1>'
                '<p>기존 고객은 ID로 불러오고, 처음 오신 고객은 신규 심사로 시작하세요.</p></div></div>')
        st.button("신규 고객 심사", key="scr-new", type="primary", disabled=down,
                  on_click=lambda: CF.open_form(CF.new_profile(), loaded=False))
    if st.session_state.get("scr_load_invalid"):
        st.html("<style>.st-key-scr-load [data-baseweb=\"input\"]{border-color:var(--fail)!important;"
                "box-shadow:0 0 0 3px rgba(198,40,40,.15)}</style>")

    with st.container(key="scr-load-card", horizontal=True, vertical_alignment="bottom"):
        with st.container(key="scr-load-box"):
            st.html('<span class="scr-load-label">고객 ID로 불러오기</span>')
            with st.form("scr-load-form", border=False, enter_to_submit=True):
                with st.container(key="scr-load", horizontal=True, gap="small"):
                    st.text_input("고객 ID", key="scr_load_id", label_visibility="collapsed",
                                  placeholder=f"고객 ID 입력 (예: {CF.SAMPLE_CHIPS[0][0]})")
                    st.form_submit_button("불러오기", on_click=_load_by_id, disabled=down)
        with st.container(key="scr-samples"):
            st.html('<span class="scr-samples-label">시연용 샘플 불러오기</span>')
            with st.container(key="scr-sample-chips", horizontal=True, gap="small"):
                for customer_id, label in CF.SAMPLE_CHIPS:
                    st.button(label, key=f"scr-sample-{customer_id}", on_click=_open_sample, args=(customer_id,),
                              disabled=down)

    _history()

# ---------------------------------------------------------------------------
# 결과 화면 (디자인 03 승인 / 04 거절)
# ---------------------------------------------------------------------------
def _new_screening():
    st.session_state.scr_current = None
    st.session_state.scr_load_invalid = False


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
        ("고객 ID", profile["customer_id"]),
        ("나이", f'{profile["age"]}세'),
        ("성별", profile.get("gender") or "—"),
        ("월 소득", f'{profile["MonthlyIncome"] / 10_000:,.0f}만원'),
        ("부채비율", f'{profile["DebtRatio"]:.0f}%'),
        ("연체", f"{late}회" if late else ("기록 없음" if not counts else "없음")),
        ("고객 유형", "씬파일러" if result.get("is_thin_filer") else "일반 고객"),
    ]
    return '<div class="scr-summary">' + "".join(
        f'<div><span>{label}</span><b>{escape(str(value))}</b></div>' for label, value in cells) + "</div>"


def _score_bar(score, grade):
    """E~A 다섯 칸을 점수 구간 길이 비율(45:20:15:10:10)로 그리고, 점수 위치와 승인 기준(650점) 점선을 표시한다."""
    scale = config.GRADE_SCALE  # [(등급, 구간 상한)] 낮은 등급부터
    lowers = [0] + [upper for _, upper in scale[:-1]]
    widths = " ".join(f"{upper - lower}fr" for (_, upper), lower in zip(scale, lowers))
    top = scale[-1][1]
    marker = min(max(score / top, 0), 1) * 100
    cut = config.APPROVAL_SCORE / top * 100
    tone = RISK_TONE[GRADE_RISK.get(grade, "high")]
    segs = "".join(f'<i class="seg-{g}"></i>' for g, _ in scale)
    labels = "".join(f'<b class="{tone}">{g}</b>' if g == grade else f"<span>{g}</span>" for g, _ in scale)
    return (f'<div class="scr-bar" style="grid-template-columns:{widths}">{segs}'
            f'<u style="left:{cut:.1f}%"></u><em style="left:{marker:.1f}%"></em></div>'
            f'<div class="scr-bar-labels" style="grid-template-columns:{widths}">{labels}</div>'
            f'<p class="scr-bar-note">점수는 (1 − 부도 확률) × 1000이에요. 점선은 승인 기준 {config.APPROVAL_SCORE}점이에요.</p>')


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
          <div class="scr-score-top"><span>신용 점수</span><strong>{score}<small> / 1000</small></strong></div>
          <div class="scr-bar-wrap">{_score_bar(score, grade)}</div>
        </div>
        <div class="scr-prob">
          <span>부도 확률</span>
          <strong class="{'' if approved else 'fail'}">{prob * 100:.1f}<small>%</small></strong>
          <p>100명 중 약 {round(prob * 100)}명이 2년 내 연체할 수준</p>
        </div>
      </div>
    </section>"""


def _reason_text(reason):
    return CONTRIBUTION_SUFFIX.sub("", reason.get("explanation") or "")


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
                     "explanation": f"{F.label(feature)} 값이 부도 위험을 낮췄어요."})
    return rows


def _shap_cell(value):
    if value is None:
        return '<strong class="num">—</strong>'
    return f'<strong class="num {"up" if value >= 0 else "down"}">{"+" if value >= 0 else "−"}{abs(value):.2f}</strong>'


def _factor_table(label, explain, rows):
    body = "".join(
        f'<div class="scr-rr-row"><span class="scr-rr-rank">{r.get("rank", i)}</span>'
        f'<strong>{escape(F.label(r["feature"]))}</strong>'
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
            [f"[대출 심사 결과 안내] 고객 ID {result['customer_id']}",
             "고객님의 대출 신청은 아래 사유로 승인되지 않았습니다.", ""]
            + [f"{i}. {F.label(r['feature'])}: {_reason_text(r)}" for i, r in enumerate(reasons, 1)]
            + ["", config.DISCLAIMER])
        with st.container(key="scr-copy"):
            ui.copy_button("안내문 복사", notice)
    _factor_table("주요 거절 사유", explain, reasons)


def _action_card(key, title, caption):
    """카드 전체가 눌리는 버튼 (보이는 내용은 HTML, 위에 투명 버튼을 덮는다)."""
    with st.container(key=f"scr-act-{key}"):
        st.html(f'<div class="scr-act"><div><b>{escape(title)}</b><span>{escape(caption)}</span></div>'
                f'{ui.icon("chevron", "#5B6577", 20)}</div>')
        return st.button(title, key=f"scr-act-btn-{key}")


def _result_view():
    current = st.session_state.scr_current
    profile, result = current["profile"], current["result"]
    approved = result["approval_status"] == "APPROVED"

    with st.container(key="scr-result-head", horizontal=True, vertical_alignment="bottom"):
        st.html(f'<div class="ui-page-head"><div><h1>심사 결과</h1>'
                f'<p>{_completed_at(result)} 심사 완료</p></div></div>')
        st.button("새 심사", key="scr-new-screening", type="primary", on_click=_new_screening)

    with st.container(key="scr-summary-card", horizontal=True, vertical_alignment="center"):
        st.html(_summary_html(profile, result))
        st.button("정보 수정", key="scr-edit", on_click=CF.open_form, args=(profile, True),
                  disabled=connection.is_down())

    st.html(_decision_html(result))

    _factors_section(result, profile, approved)

    title, caption = (("전체 판단 근거 보기", "어떤 정보가 점수에 영향을 줬는지 XAI 분석에서 확인") if approved
                      else ("사유 5개 전체와 근거 보기", "XAI 분석에서 SHAP 기여도와 원본 플롯 확인"))
    if _action_card("xai", title, caption):
        st.session_state.xai_customer = result["customer_id"]
        st.switch_page("views/xai.py")


# ---------------------------------------------------------------------------
# 페이지
# ---------------------------------------------------------------------------
st.html(f"<style>{CSS_PATH.read_text(encoding='utf-8')}{CF.CSS}"
        + ui.mask_css(".st-key-scr-new button::before", "plus", 16)
        + ui.mask_css(".st-key-scr-new-screening button::before", "plus", 16)
        + ui.mask_css(".st-key-scr-edit button::before", "pencil", 14)
        + "</style>")
CF.init_state()

toast = st.session_state.pop("scr_toast", None)
if toast:
    st.toast(toast[0], icon=":material/error:" if toast[1] else ":material/check_circle:")

if st.session_state.scr_current is None:
    _start_view()
else:
    _result_view()

CF.render(on_screening=True)
