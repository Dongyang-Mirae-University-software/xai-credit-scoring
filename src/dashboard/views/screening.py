"""고객 심사 페이지 (디자인 01~04)

흐름: 첫 화면(고객 불러오기 · 신규 입력 · 샘플) → 고객 정보 입력 창(st.dialog) → 심사 결과 화면.
'불러오기'는 GET /customers/{id} 를 먼저 부르고, API 가 아직 없으면 이번 세션에서 심사한 고객과 SAMPLE_CUSTOMERS 에서 찾는다.
"""
import re
from datetime import datetime
from html import escape
from pathlib import Path

import streamlit as st

import api_client
import config
import features as F
from components import ui
from components.api_error import api_error_html

CSS_PATH = Path(__file__).with_suffix(".css")
HISTORY_LIMIT = 5
REASON_LIMIT = 3  # 결과 화면의 주요 거절 사유 수 (전체 5개는 XAI 분석)

# 기본값은 백엔드 Swagger 예시 고객(src/api/schemas.py CUSTOMER_EXAMPLE)과 같다.
# 요청 키·단위는 실제 API(CustomerInput)와 같다: 부채비율·통신비·공과금 납부율은 % (서버가 ÷100)
DEFAULT_CUSTOMER = {
    "age": 45, "MonthlyIncome": 4_200_000, "DebtRatio": 32.0, "RevolvingUtilizationOfUnsecuredLines": 0.48,
    "NumberOfDependents": 2, "NumberOfTime30-59DaysPastDueNotWorse": 1, "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfTimes90DaysLate": 0, "NumberOfOpenCreditLinesAndLoans": 6, "NumberRealEstateLoansOrLines": 1,
    "telecom_payment_rate": 90.0, "utility_payment_rate": 93.0, "spending_consistency": 75.0,
    "regular_payment_count": 5, "app_login_frequency": 22,
}

# 샘플 고객 (첫 화면 '샘플로 바로 심사' · ID 불러오기). GET /customers 가 생기면 고객 DB 로 대체된다.
SAMPLE_CUSTOMERS = {
    "C100001": ("일반 고객", "여성", {
        **DEFAULT_CUSTOMER, "age": 42, "MonthlyIncome": 5_200_000, "DebtRatio": 25.0,
        "RevolvingUtilizationOfUnsecuredLines": 0.15, "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "NumberOfOpenCreditLinesAndLoans": 8, "telecom_payment_rate": 99.0, "utility_payment_rate": 99.0,
        "spending_consistency": 88.0, "regular_payment_count": 9, "app_login_frequency": 25}),
    "C100002": ("씬파일러", "남성", {
        **DEFAULT_CUSTOMER, "age": 27, "MonthlyIncome": 2_800_000, "DebtRatio": 10.0,
        "RevolvingUtilizationOfUnsecuredLines": 0.10, "NumberOfDependents": 0,
        "NumberOfTime30-59DaysPastDueNotWorse": 0, "NumberOfOpenCreditLinesAndLoans": 1,
        "NumberRealEstateLoansOrLines": 0, "telecom_payment_rate": 98.0, "utility_payment_rate": 97.0,
        "spending_consistency": 85.0, "regular_payment_count": 8, "app_login_frequency": 27}),
    "C100003": ("거절 예상", "남성", {
        **DEFAULT_CUSTOMER, "age": 33, "MonthlyIncome": 1_800_000, "DebtRatio": 85.0,
        "RevolvingUtilizationOfUnsecuredLines": 0.95, "NumberOfDependents": 3,
        "NumberOfTime30-59DaysPastDueNotWorse": 3, "NumberOfTime60-89DaysPastDueNotWorse": 1,
        "NumberOfTimes90DaysLate": 1, "NumberOfOpenCreditLinesAndLoans": 4, "NumberRealEstateLoansOrLines": 0,
        "telecom_payment_rate": 70.0, "utility_payment_rate": 65.0, "spending_consistency": 40.0,
        "regular_payment_count": 2, "app_login_frequency": 8}),
}

# 입력 창 칸: (요청 키, 라벨)
DELINQUENCY_FIELDS = [
    ("NumberOfTime30-59DaysPastDueNotWorse", "30–59일"),
    ("NumberOfTime60-89DaysPastDueNotWorse", "60–89일"),
    ("NumberOfTimes90DaysLate", "90일 이상"),
]
EXTRA_FIELDS = [
    ("NumberOfOpenCreditLinesAndLoans", "신용 개설 건수"),
    ("NumberRealEstateLoansOrLines", "부동산 대출"),
    ("NumberOfDependents", "부양가족 수"),
]
# 대안 데이터: (요청 키, 라벨, 최소, 최대, 표시 형식)
ALT_FIELDS = [
    ("telecom_payment_rate", "통신비 정상납부율", 0, 100, "{:.0f}%"),
    ("utility_payment_rate", "공과금 납부율", 0, 100, "{:.0f}%"),
    ("spending_consistency", "소비 일관성 점수", 0, 100, "{:.0f} / 100"),
    ("regular_payment_count", "정기결제 건수", 0, 20, "{:.0f}건"),
    ("app_login_frequency", "앱 로그인 빈도", 0, 30, "월 {:.0f}회"),
]
STEP_MAX = 50
INT_FIELDS = {f for f, _ in DELINQUENCY_FIELDS + EXTRA_FIELDS} | {"age", "regular_payment_count", "app_login_frequency"}

# 등급 → 위험도 (A·B 저위험 / C 중위험 / D·E 고위험) → 색 (ok / mid / fail)
GRADE_RISK = {"A": "low", "B": "low", "C": "mid", "D": "high", "E": "high"}
RISK_LABELS = {"low": "저위험", "mid": "중위험", "high": "고위험"}
RISK_TONE = {"low": "ok", "mid": "mid", "high": "fail"}
CONTRIBUTION_SUFFIX = re.compile(r"\s*\(기여도:\s*[-+−]?[\d.]+\)\s*$")


# ---------------------------------------------------------------------------
# 상태
# ---------------------------------------------------------------------------
def _init_state():
    st.session_state.setdefault("scr_seq", 1)
    st.session_state.setdefault("scr_current", None)   # 지금 보고 있는 심사 결과 {profile, result}
    st.session_state.setdefault("scr_history", [])     # 심사할 때마다 쌓이는 최근 심사 (세션 안에서만 유지)
    st.session_state.setdefault("scr_error", None)     # 샘플 바로 심사 실패
    st.session_state.setdefault("scr_saved", set())    # '심사 결과 저장' 을 누른 고객
    st.session_state.setdefault("customers", {})       # XAI 분석 페이지가 고객 목록으로 쓴다 {id: {payload, result, profile}}
    st.session_state.setdefault("xai_explain", {})     # /explain 응답 (XAI 분석 페이지와 함께 쓴다)


def _new_id():
    return f"C{st.session_state.scr_seq:06d}"


def _find_customer(customer_id):
    """고객 DB(GET /customers/{id}) → 이번 세션에서 심사한 고객 → 샘플 고객 순서로 찾는다. 없으면 None."""
    try:
        found = api_client.customer(customer_id)
    except Exception:
        found = None
    if found:
        return {"customer_id": customer_id, "gender": found.get("gender"),
                **{k: found[k] for k in DEFAULT_CUSTOMER if k in found}}
    entry = st.session_state.customers.get(customer_id)
    if entry and "profile" in entry:
        return entry["profile"]
    if customer_id in SAMPLE_CUSTOMERS:
        _, gender, values = SAMPLE_CUSTOMERS[customer_id]
        return {"customer_id": customer_id, "gender": gender, **values}
    return None


def _payload(profile):
    """심사 요청 본문 (GMSC 10 + 대안 5). 성별은 예측에 쓰지 않으므로 보내지 않는다."""
    return {k: v for k, v in profile.items() if k != "gender"}


def _run_predict(profile):
    """성공하면 결과 화면으로 넘어간다. 실패하면 예외를 그대로 올린다."""
    payload = _payload(profile)
    result = api_client.predict(payload)
    customer_id = profile["customer_id"]
    result = {**result, "customer_id": customer_id}
    st.session_state.customers.pop(customer_id, None)  # 다시 심사하면 XAI 고객 목록 맨 앞으로
    st.session_state.customers[customer_id] = {"payload": payload, "result": result, "profile": profile}
    st.session_state.xai_explain.pop(customer_id, None)
    st.session_state.scr_saved.discard(customer_id)
    st.session_state.scr_current = {"profile": profile, "result": result}
    history = [h for h in st.session_state.scr_history if h["result"]["customer_id"] != customer_id]
    st.session_state.scr_history = ([{"result": result, "at": datetime.now()}] + history)[:HISTORY_LIMIT]
    if customer_id == _new_id():
        st.session_state.scr_seq += 1


def _thin_filer(open_lines, real_estate, late_total):
    """씬파일러 판정 미리보기 (src/data/simulator.label_thin_filer 기준 2와 같다). 실제 판정은 /predict 응답."""
    return open_lines <= 1 and real_estate == 0 and late_total == 0


# ---------------------------------------------------------------------------
# 고객 정보 입력 창 (디자인 02)
# ---------------------------------------------------------------------------
def _fill_form(profile, loaded):
    income = profile["MonthlyIncome"] / 10_000
    ss = st.session_state
    ss.dlg_profile, ss.dlg_loaded, ss.dlg_alt_edit, ss.dlg_error = profile, loaded, False, None
    ss.dlg_age = int(profile["age"])
    ss.dlg_gender = profile.get("gender")
    ss.dlg_income = int(round(income))
    ss.dlg_repay = int(round(profile["DebtRatio"] / 100 * income))
    ss.dlg_util = int(round(min(profile["RevolvingUtilizationOfUnsecuredLines"], 1.0) * 100))
    for field, *_ in DELINQUENCY_FIELDS + EXTRA_FIELDS + ALT_FIELDS:
        ss[f"dlg_{field}"] = int(profile[field]) if field in INT_FIELDS else float(profile[field])


def _form_profile():
    ss = st.session_state
    income = ss.dlg_income or 0
    profile = {
        "customer_id": ss.dlg_profile["customer_id"],
        "gender": ss.dlg_gender,
        "age": ss.dlg_age,
        "MonthlyIncome": float(income * 10_000),
        # 부채비율 = 월 상환액 ÷ 월 소득 (API 는 % 로 받는다)
        "DebtRatio": round(ss.dlg_repay / income * 100, 1) if income else 0.0,
        "RevolvingUtilizationOfUnsecuredLines": ss.dlg_util / 100,
    }
    for field, *_ in DELINQUENCY_FIELDS + EXTRA_FIELDS + ALT_FIELDS:
        profile[field] = ss[f"dlg_{field}"]
    return profile


def _open_form(profile, loaded):
    _fill_form(profile, loaded)
    st.session_state.scr_open_form = True


def _reset_form():
    _fill_form(st.session_state.dlg_profile, st.session_state.dlg_loaded)


def _bump(field, delta):
    key = f"dlg_{field}"
    st.session_state[key] = min(max(int(st.session_state[key]) + delta, 0), STEP_MAX)


def _steppers(fields, key):
    """'라벨  − 값 +' 칸 3개 (디자인의 증감 버튼). 값은 위젯이 아닌 세션 값(dlg_<필드>)."""
    with st.container(key=f"dlg-steps-{key}", horizontal=True, gap="small"):
        for field, label in fields:
            with st.container(key=f"dlg-step-{field}", horizontal=True, vertical_alignment="center", gap=None):
                st.html(f'<span class="dlg-step-label">{escape(label)}</span>')
                st.space("stretch")
                st.button("−", key=f"dlg-dec-{field}", on_click=_bump, args=(field, -1),
                          disabled=st.session_state[f"dlg_{field}"] <= 0)
                st.html(f'<b class="dlg-step-value">{st.session_state[f"dlg_{field}"]}</b>')
                st.button("\\+", key=f"dlg-inc-{field}", on_click=_bump, args=(field, 1))


def _group_head(title, right=""):
    st.html(f'<div class="dlg-group-head"><span>{title}</span>{right}</div>')


@st.dialog("고객 정보 입력", width="medium")
def _customer_form():
    ss = st.session_state
    customer_id = ss.dlg_profile["customer_id"]
    st.html('<p class="dlg-sub">'
            + ("불러온 값은 그대로 두고 바뀐 항목만 수정하세요." if ss.dlg_loaded else "고객 정보를 입력하고 심사를 실행하세요.")
            + "</p>")

    if ss.dlg_loaded:
        with st.container(key="dlg-loaded", horizontal=True, vertical_alignment="center"):
            st.html(f'<span class="dlg-loaded-text">{ui.icon("check", "#0B5E44", 16)}'
                    f'<b>{escape(customer_id)}</b> 고객 정보를 불러왔어요</span>')
            if st.button("다른 고객 불러오기", key="dlg-other", type="tertiary"):
                ss.scr_current = None
                ss.scr_focus_load = True
                st.rerun()

    # 기본 정보
    with st.container(key="dlg-basic"):
        _group_head("기본 정보")
        c1, c2, c3 = st.columns(3, gap="small")
        with c1, st.container(key="dlg-unit-age"):
            st.number_input("나이", min_value=18, max_value=100, step=1, key="dlg_age")
        with c2, st.container(key="dlg-gender-box"):
            st.radio("성별", ["여성", "남성"], index=None, horizontal=True, key="dlg_gender")
            st.html('<span class="dlg-hint">공정성 점검용이며 심사 점수에는 쓰지 않아요</span>')
        with c3, st.container(key="dlg-unit-income"):
            st.number_input("월 소득", min_value=0, step=10, key="dlg_income")

    # 금융 이력
    with st.container(key="dlg-finance"):
        late_total = sum(int(ss[f"dlg_{f}"]) for f, _ in DELINQUENCY_FIELDS)
        thin = _thin_filer(ss["dlg_NumberOfOpenCreditLinesAndLoans"], ss["dlg_NumberRealEstateLoansOrLines"], late_total)
        _group_head("금융 이력", f'<em>씬파일러 자동 판정 <b>{"해당" if thin else "해당 없음"}</b></em>')
        c1, c2 = st.columns(2, gap="medium")
        with c1, st.container(key="dlg-unit-repay"):
            st.number_input("월 대출 상환액", min_value=0, step=1, key="dlg_repay")
            ratio = f"{ss.dlg_repay / ss.dlg_income * 100:.0f}%" if ss.dlg_income else "—"
            st.html(f'<span class="dlg-hint">부채비율 <b>{ratio}</b> 자동 계산</span>')
        with c2, st.container(key="dlg-util"):
            st.html(f'<div class="dlg-util-head"><span>신용한도 사용률</span><b>{ss.dlg_util}%</b></div>')
            st.slider("신용한도 사용률", 0, 100, key="dlg_util", label_visibility="collapsed")
        st.html('<span class="dlg-label">최근 2년 연체 횟수</span>')
        _steppers(DELINQUENCY_FIELDS, "due")

    # 추가 정보
    with st.container(key="dlg-extra"):
        _group_head("추가 정보", '<em>불러온 값이 기본으로 채워져요</em>' if ss.dlg_loaded else "")
        _steppers(EXTRA_FIELDS, "extra")

    # 대안 데이터 (불러온 고객은 자동 연동 값 표시, '직접 수정' 을 누르면 입력칸)
    with st.container(key="dlg-alt"):
        linked = ss.dlg_loaded and not ss.dlg_alt_edit
        with st.container(key="dlg-alt-head", horizontal=True, vertical_alignment="center"):
            tag = '<i class="dlg-tag">자동 연동</i>' if linked else ""
            st.html(f'<div class="dlg-group-head"><span>대안 데이터{tag}</span></div>')
            if linked and st.button("직접 수정", key="dlg-alt-edit", type="tertiary"):
                ss.dlg_alt_edit = True
                st.rerun(scope="fragment")
        if linked:
            cells = "".join(
                f'<div><span>{label}</span><b>{fmt.format(ss[f"dlg_{field}"])}</b></div>'
                for field, label, _, _, fmt in ALT_FIELDS)
            st.html(f'<div class="dlg-alt-list">{cells}</div>')
        else:
            with st.container(key="dlg-alt-edit-box"):
                for row in (ALT_FIELDS[:3], ALT_FIELDS[3:]):
                    cols = st.columns(3, gap="small")
                    for col, (field, label, lo, hi, _) in zip(cols, row):
                        with col:
                            st.number_input(label, min_value=lo, max_value=hi, step=1, key=f"dlg_{field}")

    if ss.dlg_error is not None:
        st.html(api_error_html("POST /predict", ss.dlg_error, "inline"))

    with st.container(key="dlg-foot", horizontal=True, vertical_alignment="center"):
        st.button("초기화", key="dlg-reset", on_click=_reset_form)
        st.space("stretch")
        if st.button("취소", key="dlg-cancel", type="tertiary"):
            st.rerun()
        if st.button("심사 실행", key="dlg-submit", type="primary"):
            try:
                _run_predict(_form_profile())
            except Exception as e:
                ss.dlg_error = e
                st.rerun(scope="fragment")
            st.rerun()


# ---------------------------------------------------------------------------
# 첫 화면 (디자인 01)
# ---------------------------------------------------------------------------
def _lookup(customer_id):
    customer_id = (customer_id or "").strip().upper()
    profile = _find_customer(customer_id) if customer_id else None
    st.session_state.scr_load_msg = None if profile else (customer_id or None)
    if profile:
        _open_form(profile, loaded=True)


def _load_by_id():
    _lookup(st.session_state.scr_load_id)


def _new_customer():
    _open_form({"customer_id": _new_id(), "gender": None, **DEFAULT_CUSTOMER}, loaded=False)


def _run_sample(customer_id):
    st.session_state.scr_error = None
    try:
        _run_predict(_find_customer(customer_id))
    except Exception as e:
        st.session_state.scr_error = e


def _history_table():
    today = datetime.now().date()
    rows = []
    for h in st.session_state.scr_history:
        r, at = h["result"], h["at"]
        approved = r["approval_status"] == "APPROVED"
        tone = RISK_TONE[GRADE_RISK.get(r["risk_grade"], "high")] if approved else "fail"
        tone = "ok" if approved and tone == "fail" else tone
        when = ("오늘 " if at.date() == today else at.strftime("%m.%d ")) + at.strftime("%H:%M")
        rows.append(f'<tr><td class="scr-id">{escape(r["customer_id"])}</td>'
                    f'<td>{ui.pill("승인" if approved else "거절", tone)}</td>'
                    f'<td>{escape(r["risk_grade"])}</td><td>{r["credit_score"]}</td>'
                    f'<td class="scr-time">{when}</td></tr>')
    body = "".join(rows) or ('<tr><td colspan="5" class="scr-empty">아직 심사한 고객이 없어요. '
                             '심사를 실행하면 여기에 쌓여요.</td></tr>')
    return f"""
    <section class="scr-history">
      <h2>최근 심사</h2>
      <table class="ui-table">
        <colgroup><col style="width:24%"><col style="width:19%"><col style="width:12.5%"><col style="width:20%"><col></colgroup>
        <thead><tr><th>고객 ID</th><th>결과</th><th>등급</th><th>신용 점수</th><th>심사 시각</th></tr></thead>
        <tbody>{body}</tbody>
      </table>
    </section>"""


def _start_view():
    st.html('<div class="ui-page-head"><div><h1>고객 심사</h1>'
            '<p>고객을 불러오거나 새로 입력해 심사를 시작하세요.</p></div></div>')
    if st.session_state.scr_error is not None:
        st.html(api_error_html("POST /predict", st.session_state.scr_error, "banner"))

    with st.container(key="scr-start"):
        st.html(f"""
        <div class="scr-start-icon">{ui.icon("doc", "#1F4FD8", 30)}</div>
        <div class="scr-start-text"><h2>진행 중인 심사가 없어요</h2>
          <p>기존 고객은 ID로 불러오면 정보가 자동으로 채워져요.</p></div>
        """)
        with st.form("scr-load-form", border=False, enter_to_submit=True):
            with st.container(key="scr-load", horizontal=True, gap="small"):
                st.text_input("고객 ID", key="scr_load_id", label_visibility="collapsed",
                              placeholder="고객 ID 입력 (예: C100001)")
                st.form_submit_button("불러오기", type="primary", on_click=_load_by_id)
        missing = st.session_state.get("scr_load_msg")
        if missing:
            st.html(f'<p class="scr-load-msg"><b>{escape(missing)}</b> 고객을 찾지 못했어요. '
                    "ID를 다시 확인하거나 신규 고객으로 입력해 주세요.</p>")
        st.html('<div class="scr-or"><i></i>또는<i></i></div>')
        st.button("신규 고객 직접 입력", key="scr-new", on_click=_new_customer)
        with st.container(key="scr-samples", horizontal=True, horizontal_alignment="center",
                          vertical_alignment="center", gap="small"):
            st.html('<span class="scr-samples-label">샘플로 바로 심사</span>')
            for customer_id, (label, _, _) in SAMPLE_CUSTOMERS.items():
                st.button(label, key=f"scr-sample-{customer_id}", on_click=_run_sample, args=(customer_id,))

    st.html(_history_table())


# ---------------------------------------------------------------------------
# 결과 화면 (디자인 03 승인 / 04 거절)
# ---------------------------------------------------------------------------
def _new_screening():
    st.session_state.scr_current = None
    st.session_state.scr_error = None


def _completed_at(result):
    try:
        stamp = datetime.fromisoformat(result["timestamp"]).astimezone()
    except (KeyError, ValueError):
        stamp = datetime.now()
    return stamp.strftime("%Y.%m.%d %H:%M")


def _summary_html(profile, result):
    late = sum(int(profile[f]) for f, _ in DELINQUENCY_FIELDS)
    cells = [
        ("고객 ID", profile["customer_id"]),
        ("나이", f'{profile["age"]}세'),
        ("성별", profile.get("gender") or "—"),
        ("월 소득", f'{profile["MonthlyIncome"] / 10_000:,.0f}만원'),
        ("부채비율", f'{profile["DebtRatio"]:.0f}%'),
        ("연체", f"{late}회" if late else "없음"),
        ("고객 유형", "씬파일러" if result.get("is_thin_filer") else "일반 고객"),
    ]
    return '<div class="scr-summary">' + "".join(
        f'<div><span>{label}</span><b>{escape(str(value))}</b></div>' for label, value in cells) + "</div>"


def _score_bar(score, grade):
    """E~A 다섯 칸을 같은 폭으로 그리고, 점수는 해당 등급 칸 안에서의 위치로 표시한다."""
    scale = config.GRADE_SCALE  # [(등급, 구간 상한)] 낮은 등급부터
    lower, marker = 0, 0.0
    for i, (g, upper) in enumerate(scale):
        if g == grade:
            frac = min(max((score - lower) / (upper - lower), 0), 1)
            marker = (i + frac) / len(scale) * 100
            break
        lower = upper
    tone = RISK_TONE[GRADE_RISK.get(grade, "high")]
    segs = "".join(f'<i class="seg-{g}"></i>' for g, _ in scale)
    labels = "".join(f'<b class="{tone}">{g}</b>' if g == grade else f"<span>{g}</span>" for g, _ in scale)
    return (f'<div class="scr-bar">{segs}<em style="left:{marker:.1f}%"></em></div>'
            f'<div class="scr-bar-labels">{labels}</div>')


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


def _explain(customer_id):
    cache = st.session_state.xai_explain
    if customer_id not in cache:
        cache[customer_id] = api_client.explain(st.session_state.customers[customer_id]["payload"])
    return cache[customer_id]


def _reason_text(reason):
    return CONTRIBUTION_SUFFIX.sub("", reason.get("explanation") or "")


def _reasons_section(result):
    """주요 거절 사유 3개 (POST /explain rejection_reasons[:3])."""
    try:
        reasons = (_explain(result["customer_id"]).get("rejection_reasons") or [])[:REASON_LIMIT]
    except Exception as e:
        st.html(api_error_html("POST /explain", e, "banner"))
        return
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
    rows = "".join(
        f'<div class="scr-rr-row"><span class="scr-rr-rank">{r.get("rank", i)}</span>'
        f'<strong title="{escape(r["feature"])}">{escape(F.label(r["feature"]))}</strong>'
        f'<span class="scr-rr-text">{escape(_reason_text(r))}</span>'
        f'<b class="num">{F.fmt_value(r["feature"], r.get("current_value"))}</b>'
        f'<span class="num scr-rr-ref">{F.fmt_value(r["feature"], r.get("reference_value"))}</span></div>'
        for i, r in enumerate(reasons, 1))
    st.html(f"""
    <div class="scr-rr" role="table" aria-label="주요 거절 사유">
      <div class="scr-rr-row scr-rr-headrow"><span>순위</span><span>항목</span><span>설명</span>
        <span class="num">현재</span><span class="num">기준</span></div>
      {rows or '<div class="scr-rr-empty">거절 사유가 없어요</div>'}
    </div>""")


def _action_card(key, title, caption):
    """카드 전체가 눌리는 버튼 (보이는 내용은 HTML, 위에 투명 버튼을 덮는다)."""
    with st.container(key=f"scr-act-{key}"):
        st.html(f'<div class="scr-act"><div><b>{escape(title)}</b><span>{escape(caption)}</span></div>'
                f'{ui.icon("chevron", "#5B6577", 20)}</div>')
        return st.button(title, key=f"scr-act-btn-{key}")


def _save(profile, result):
    """POST /screenings 로 저장. API 가 아직 없으면 이번 세션 이력에만 남았다고 알린다."""
    customer_id = result["customer_id"]
    try:
        saved = api_client.save_screening({"input": profile, "result": result})
    except Exception as e:
        st.toast(f"저장하지 못했어요: {type(e).__name__}")
        return
    st.session_state.scr_saved.add(customer_id)
    st.toast("심사 이력에 저장했어요" if saved else "저장 API(POST /screenings)가 아직 없어 이번 세션 이력에만 남았어요")


def _result_view():
    current = st.session_state.scr_current
    profile, result = current["profile"], current["result"]
    approved = result["approval_status"] == "APPROVED"

    with st.container(key="scr-result-head", horizontal=True, vertical_alignment="bottom"):
        st.html(f'<div class="ui-page-head"><div><h1>고객 심사</h1>'
                f'<p>{_completed_at(result)} 심사 완료</p></div></div>')
        st.button("새 심사", key="scr-new-screening", type="primary", on_click=_new_screening)

    with st.container(key="scr-summary-card", horizontal=True, vertical_alignment="center"):
        st.html(_summary_html(profile, result))
        st.button("정보 수정", key="scr-edit", on_click=_open_form, args=(profile, False))

    st.html(_decision_html(result))

    if not approved:
        _reasons_section(result)

    left, right = st.columns(2, gap="small")
    with left:
        title, caption = (("판단 근거 보기", "어떤 정보가 점수에 영향을 줬는지 XAI 분석에서 확인") if approved
                          else ("사유 5개 전체와 근거 보기", "XAI 분석에서 SHAP 기여도와 원본 플롯 확인"))
        if _action_card("xai", title, caption):
            st.session_state.xai_customer = result["customer_id"]
            st.switch_page("views/xai.py")
    with right:
        saved = result["customer_id"] in st.session_state.scr_saved
        if _action_card("save", "심사 결과 저장" + (" 완료" if saved else ""), "입력값과 결과를 심사 이력에 기록"):
            _save(profile, result)
            st.rerun()


# ---------------------------------------------------------------------------
# 페이지
# ---------------------------------------------------------------------------
st.html(f"<style>{CSS_PATH.read_text(encoding='utf-8')}"
        + ui.mask_css(".st-key-scr-new button::before", "plus", 16)
        + ui.mask_css(".st-key-scr-new-screening button::before", "plus", 16)
        + ui.mask_css(".st-key-scr-edit button::before", "pencil", 14)
        + "</style>")
_init_state()

# 헤더의 고객 ID 검색 → 바로 불러오기
lookup = st.session_state.pop("scr_lookup", None)
if lookup:
    st.session_state.scr_current = None
    _lookup(lookup)

# XAI 분석의 '조건 바꿔 재심사' → 그 고객 결과 화면에서 입력 창을 연다
edit_id = st.session_state.pop("scr_edit_request", None)
if edit_id and edit_id in st.session_state.customers:
    entry = st.session_state.customers[edit_id]
    st.session_state.scr_current = {"profile": entry["profile"], "result": entry["result"]}
    _open_form(entry["profile"], loaded=False)

if st.session_state.scr_current is None:
    _start_view()
else:
    _result_view()

if st.session_state.pop("scr_open_form", False):
    _customer_form()
