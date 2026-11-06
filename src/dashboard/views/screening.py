"""고객 심사 페이지 (디자인 01~04)

흐름 (디자인 시안의 프로토타입 동작과 같다)
- 첫 화면: 불러오기 · 시연용 샘플 칩 → 값이 채워진 입력 창 (제목 옆 고객 ID), 신규 고객 심사 → 빈 입력 창
- 입력 창(components/customer_form)에서 심사 실행 → 결과 화면 (결과는 서버 이력에 바로 남는다)
- 심사 이력 줄 → 그 고객의 결과 화면 바로
- 결과 화면: 정보 수정 → 입력 창, 이전 → 첫 화면. 주소는 ?customer=<고객 ID> 라 새로고침·공유해도 그 결과가 열린다
- 로고 · 사이드바 '고객 심사' → 언제나 첫 화면
'불러오기'·샘플은 API 를 부르지 않고 로컬 고객 파일(samples.json)과 이번 세션에서 심사한 고객에서 찾는다.
심사 이력은 서버 이력(GET /screenings)과 이번 세션에서 심사한 것을 합쳐 보여 준다 (새로고침해도 남는다).
조회 기간은 오늘(기본) · 최근 7일 · 최근 30일 · 직접 설정(시작일~종료일 + 조회) 중에서 고른다.
"""
import re
from datetime import date, datetime, timedelta
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
HISTORY_FETCH = 100  # GET /screenings 최대치 (서버 le=100, 고객당 최신 1건). 기간은 받아 온 목록에서 거른다
# 조회 기간 버튼: (이름, 며칠 — 오늘 포함). 마지막 '직접 설정' 은 날짜 두 칸 + 조회
HISTORY_PRESETS = [("오늘", 1), ("최근 7일", 7), ("최근 30일", 30)]
HISTORY_CUSTOM = "직접 설정"
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
    """심사 이력 줄: 그 고객의 결과 화면으로.
    이번 세션에서 심사한 고객이면 그 결과를, 서버 이력에만 있으면 고객 파일의 입력값으로 다시 심사해 보여 준다."""
    if CF.show_result(customer_id):
        return
    profile = CF.find_customer(customer_id)
    if not profile:
        _toast_later(f"{customer_id} 고객의 입력값을 찾을 수 없어 결과를 열지 못했어요.", error=True)
        return
    try:
        CF.run_predict(profile, record=False)
    except Exception as e:
        _toast_later(f"{customer_id} 고객의 결과를 불러오지 못했어요: {type(e).__name__}", error=True)


def _parse_time(text):
    try:
        return datetime.fromisoformat(text).astimezone().replace(tzinfo=None)
    except (TypeError, ValueError):
        return None


def _history_entries(start, end):
    """심사 이력 = 서버 이력(GET /screenings) + 이번 세션에서 심사한 것.
    start~end (날짜, 양끝 포함) 것만, 고객당 최신 1건, 최신순."""
    rows = [{**h["result"], "at": h["at"]} for h in st.session_state.scr_history]
    try:
        saved = [] if connection.is_down() else api_client.screenings(HISTORY_FETCH)
    except Exception:
        saved = []  # 이력 조회가 실패해도 첫 화면은 그린다
    for r in saved:
        at = _parse_time(r.get("screened_at"))
        if at and r.get("customer_id"):
            rows.append({**r, "at": at})
    latest = {}
    for r in sorted(rows, key=lambda r: r["at"], reverse=True):
        if start <= r["at"].date() <= end:
            latest.setdefault(r["customer_id"], r)
    return list(latest.values())


def _customer_type(r):
    """고객 유형: /predict 응답의 is_thin_filer. 서버 이력 목록에는 없어서, 이번 세션에서 심사한 결과로 찾고 그래도 없으면 '—'."""
    thin = r.get("is_thin_filer")
    if thin is None:
        thin = st.session_state.customers.get(r["customer_id"], {}).get("result", {}).get("is_thin_filer")
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
            f'<td class="scr-time">{_time_label(r["at"])}</td>'
            f'<td class="scr-go">{ui.icon("chevron", "#8A94A6", 16)}</td></tr>')


def _time_label(at):
    """심사 시각: 오늘 09:12 / 어제 18:40 / 11.03 14:05"""
    days = (datetime.now().date() - at.date()).days
    day = "오늘" if days == 0 else "어제" if days == 1 else at.strftime("%m.%d")
    return f"{day} {at.strftime('%H:%M')}"


def _preset_range(days):
    today = date.today()
    return today - timedelta(days=days - 1), today


def _history_range():
    """적용 중인 조회 기간 (시작, 끝). 처음엔 오늘.
    화면을 옮겨도 남도록 위젯이 아닌 키(scr_period_sel, scr_range)에 둔다. 위젯 키(scr_period_w)와 이름을 겹치지 않는다
    (위젯 키였던 이름은 그 위젯이 안 그려지면 Streamlit 이 지운다). 둘 중 하나만 없어도 각각 채운다."""
    ss = st.session_state
    ss.setdefault("scr_period_sel", HISTORY_PRESETS[0][0])
    ss.setdefault("scr_range", _preset_range(HISTORY_PRESETS[0][1]))
    return ss.scr_range


def _pick_period():
    """기간 버튼: 오늘·최근 7일·최근 30일은 바로 적용, 직접 설정은 날짜 칸만 연다 (조회를 눌러야 적용)."""
    ss = st.session_state
    ss.scr_period_sel = ss.scr_period_w
    days = dict(HISTORY_PRESETS).get(ss.scr_period_sel)
    if days:
        ss.scr_range = _preset_range(days)


def _apply_custom():
    """직접 설정의 조회: 시작일이 끝일보다 늦으면 안내만 하고 기간은 그대로."""
    ss = st.session_state
    start, end = ss.scr_from_w, ss.scr_to_w
    if not start or not end:
        _toast_later("시작일과 종료일을 모두 골라 주세요.", error=True)
    elif start > end:
        _toast_later("시작일이 종료일보다 늦어요. 날짜를 다시 골라 주세요.", error=True)
    else:
        ss.scr_range = (start, end)


def _range_label(start, end):
    fmt = lambda d: d.strftime("%Y.%m.%d")
    return fmt(start) if start == end else f"{fmt(start)} ~ {fmt(end)}"


def _history():
    """심사 이력 표 (조회 기간 선택). 줄 전체가 눌리고 (위에 투명 버튼), 누르면 그 고객의 결과 화면이 열린다."""
    ss = st.session_state
    colgroup = "<colgroup>" + "<col>" * 6 + '<col style="width:40px"></colgroup>'
    start, end = _history_range()
    entries = _history_entries(start, end)
    names = [name for name, _ in HISTORY_PRESETS] + [HISTORY_CUSTOM]
    with st.container(key="scr-history"):
        with st.container(key="scr-history-head", horizontal=True, vertical_alignment="center"):
            st.html(f'<div class="scr-history-title"><h2>심사 이력 <span>{len(entries)}건</span></h2>'
                    '<p>줄을 누르면 심사 결과를 다시 볼 수 있어요</p></div>')
            st.html(f'<span class="scr-history-range">{_range_label(start, end)}</span>')
            if ss.scr_period_sel == HISTORY_CUSTOM:
                # 날짜 칸 처음 값: 적용 중인 기간. 하루짜리(오늘)면 최근 7일로 채워 둔다 (시안)
                pick_from, pick_to = (start, end) if start != end else _preset_range(7)
                st.date_input("시작일", value=pick_from, max_value=date.today(), key="scr_from_w",
                              format="YYYY.MM.DD", label_visibility="collapsed")
                st.html('<span class="scr-history-tilde">~</span>')
                st.date_input("종료일", value=pick_to, max_value=date.today(), key="scr_to_w",
                              format="YYYY.MM.DD", label_visibility="collapsed")
                st.button("조회", key="scr-range-go", on_click=_apply_custom)
            st.radio("조회 기간", names, index=names.index(ss.scr_period_sel) if ss.scr_period_sel in names else 0, key="scr_period_w",
                     horizontal=True, label_visibility="collapsed", on_change=_pick_period)
        with st.container(key="scr-history-card"):
            st.html(f'<table class="scr-htable">{colgroup}<thead><tr><th>고객 ID</th><th>고객 유형</th><th>결과</th>'
                    '<th>리스크 등급</th><th class="num">신용 점수</th><th>심사 시각</th><th></th></tr></thead></table>')
            if not entries:
                st.html('<p class="scr-empty">이 기간에 심사한 고객이 없어요. 심사를 실행하면 여기에 쌓여요.</p>')
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
def _back():
    """이전: 결과 화면을 닫고 첫 화면으로 (주소의 ?customer= 도 지운다)."""
    CF.clear_result()
    st.session_state.scr_load_invalid = False


def _sync_query():
    """결과 화면과 주소(?customer=)를 맞춘다. 브라우저 뒤로·앞으로 가기도 이걸로 따라간다.
    - 주소에 고객 ID 가 있는데 지금 결과와 다르면 그 고객 결과를 연다 (이번 세션 결과 → 없으면 고객 파일 값으로 다시 심사)
    - 주소에서 고객 ID 가 빠졌는데 직전에 주소에 붙여 둔 결과를 보고 있으면 = 뒤로 가기 → 첫 화면
    - 결과를 보고 있으면 주소에 그 고객 ID 를 붙인다 (새로고침·공유해도 같은 결과)
    st.query_params 에 값을 넣을 때마다 브라우저 기록이 하나 쌓이므로 (pushState), 주소와 다를 때만 넣는다."""
    ss = st.session_state
    wanted = (st.query_params.get(CF.RESULT_QUERY) or "").strip().upper()
    current = ss.scr_current["result"]["customer_id"] if ss.scr_current else None
    if wanted and wanted != current:
        _open_history(wanted)
        if ss.scr_current is None or ss.scr_current["result"]["customer_id"] != wanted:
            st.query_params.pop(CF.RESULT_QUERY, None)  # 열 수 없는 고객이면 첫 화면 (안내는 토스트)
    elif not wanted and current and ss.get("scr_query_synced") == current:
        CF.clear_result()  # 뒤로 가기로 주소의 ?customer= 가 빠졌다
        ss.scr_query_synced = None
        st.rerun()
    if ss.scr_current:
        customer_id = ss.scr_current["result"]["customer_id"]
        if st.query_params.get(CF.RESULT_QUERY) != customer_id:
            st.query_params[CF.RESULT_QUERY] = customer_id
        ss.scr_query_synced = customer_id
    else:
        ss.scr_query_synced = None


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
        st.button("이전", key="scr-back", on_click=_back)

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
        + ui.mask_css(".st-key-scr-back button::before", "chevron", 16)
        + ui.mask_css(".st-key-scr-edit button::before", "pencil", 14)
        + "</style>")
CF.init_state()

_sync_query()
toast = st.session_state.pop("scr_toast", None)
if toast:
    st.toast(toast[0], icon=":material/error:" if toast[1] else ":material/check_circle:")

if st.session_state.scr_current is None:
    _start_view()
else:
    _result_view()

CF.render(on_screening=True)
