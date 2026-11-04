"""고객 심사 페이지

흐름: 첫 화면(고객 불러오기 · 신규 입력 · 샘플) → 고객 정보 입력 창(st.dialog) → 심사 결과 화면.
고객 DB API 가 없어서, '불러오기'는 이번 세션에서 심사한 고객과 SAMPLE_CUSTOMERS 만 찾는다.
"""
from datetime import datetime
from html import escape
from pathlib import Path

import streamlit as st

import api_client
import config
from components.api_error import api_error_html
from components.page_header import render_page_header

CSS_PATH = Path(__file__).with_suffix(".css")
HISTORY_LIMIT = 5

# 기본값은 백엔드 Swagger 예시 고객(src/api/schemas.py CUSTOMER_EXAMPLE)과 같다.
# 요청 키·단위는 실제 API(CustomerInput)와 같다: 부채비율·통신비·공과금 납부율은 % (서버가 ÷100, 이슈 #70)
DEFAULT_CUSTOMER = {
    "age": 45, "MonthlyIncome": 4_200_000, "DebtRatio": 32.0, "RevolvingUtilizationOfUnsecuredLines": 0.48,
    "NumberOfDependents": 2, "NumberOfTime30-59DaysPastDueNotWorse": 1, "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfTimes90DaysLate": 0, "NumberOfOpenCreditLinesAndLoans": 6, "NumberRealEstateLoansOrLines": 1,
    "telecom_payment_rate": 90.0, "utility_payment_rate": 93.0, "spending_consistency": 75.0,
    "regular_payment_count": 5, "app_login_frequency": 22,
}

# 샘플 고객 (첫 화면 '샘플로 바로 심사' · ID 불러오기). 결과는 실제 모델로 확인한 값:
# C100001 승인 A / C100002 씬파일러 승인 / C100003 거절 E
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

# 입력 창의 연체·계좌 칸: (요청 키, 라벨)
DELINQUENCY_FIELDS = [
    ("NumberOfTime30-59DaysPastDueNotWorse", "30–59일"),
    ("NumberOfTime60-89DaysPastDueNotWorse", "60–89일"),
    ("NumberOfTimes90DaysLate", "90일 이상"),
]
ACCOUNT_FIELDS = [
    ("NumberOfOpenCreditLinesAndLoans", "보유 대출·한도"),
    ("NumberRealEstateLoansOrLines", "부동산 대출"),
    ("NumberOfDependents", "부양가족"),
]
# 대안 데이터: (요청 키, 라벨, 최소, 최대, 표시 형식)
ALT_FIELDS = [
    ("telecom_payment_rate", "통신비 정상납부율", 0, 100, "{:.0f}%"),
    ("utility_payment_rate", "공과금 납부율", 0, 100, "{:.0f}%"),
    ("spending_consistency", "소비 일관성 점수", 0, 100, "{:.0f} / 100"),
    ("regular_payment_count", "정기결제 건수", 0, 20, "{:.0f}건"),
    ("app_login_frequency", "앱 로그인 빈도", 0, 30, "월 {:.0f}회"),
]
INT_FIELDS = {f for f, _ in DELINQUENCY_FIELDS + ACCOUNT_FIELDS} | {"age", "regular_payment_count", "app_login_frequency"}

# 등급 → 위험도 (A·B 저위험 / C 중위험 / D·E 고위험)
GRADE_RISK = {"A": "low", "B": "low", "C": "mid", "D": "high", "E": "high"}
RISK_LABELS = {"low": "저위험", "mid": "중위험", "high": "고위험"}


# ---------------------------------------------------------------------------
# 상태
# ---------------------------------------------------------------------------
def _init_state():
    st.session_state.setdefault("scr_seq", 1)
    st.session_state.setdefault("scr_current", None)   # 지금 보고 있는 심사 결과 {profile, result, saved}
    st.session_state.setdefault("scr_history", [])     # '심사 결과 저장' 으로 남긴 이력
    st.session_state.setdefault("scr_error", None)     # 샘플 바로 심사 실패
    st.session_state.setdefault("customers", {})       # XAI 분석 페이지가 고객 목록으로 쓴다 {id: {payload, result, profile}}


def _new_id():
    return f"C{st.session_state.scr_seq:06d}"


def _find_customer(customer_id):
    """이번 세션에서 심사한 고객 → 샘플 고객 순서로 찾는다. 없으면 None."""
    entry = st.session_state.customers.get(customer_id)
    if entry and "profile" in entry:
        return entry["profile"]
    if customer_id in SAMPLE_CUSTOMERS:
        _, gender, values = SAMPLE_CUSTOMERS[customer_id]
        return {"customer_id": customer_id, "gender": gender, **values}
    return None


def _payload(profile):
    """심사 요청 본문. 성별은 예측에 쓰지 않으므로 보내지 않는다."""
    return {k: v for k, v in profile.items() if k != "gender"}


def _run_predict(profile):
    """성공하면 결과 화면으로 넘어가고 True. 실패하면 예외를 그대로 올린다."""
    payload = _payload(profile)
    result = api_client.predict(payload)
    customer_id = profile["customer_id"]
    st.session_state.customers.pop(customer_id, None)  # 다시 심사하면 XAI 고객 목록 맨 앞으로
    st.session_state.customers[customer_id] = {"payload": payload, "result": result, "profile": profile}
    st.session_state.scr_current = {"profile": profile, "result": result, "saved": False}
    if customer_id == _new_id():
        st.session_state.scr_seq += 1


# ---------------------------------------------------------------------------
# 고객 정보 입력 창
# ---------------------------------------------------------------------------
def _fill_form(profile, loaded):
    """입력 창 위젯 값을 profile 로 채운다 (창을 열기 전, 또는 초기화 때)."""
    income = profile["MonthlyIncome"] / 10_000
    ss = st.session_state
    ss.dlg_profile, ss.dlg_loaded, ss.dlg_alt_edit, ss.dlg_error = profile, loaded, False, None
    ss.dlg_age = int(profile["age"])
    ss.dlg_gender = profile.get("gender")
    ss.dlg_income = int(round(income))
    ss.dlg_repay = int(round(profile["DebtRatio"] / 100 * income))
    ss.dlg_util = int(round(min(profile["RevolvingUtilizationOfUnsecuredLines"], 1.0) * 100))
    for field, *_ in DELINQUENCY_FIELDS + ACCOUNT_FIELDS + ALT_FIELDS:
        ss[f"dlg_{field}"] = int(profile[field]) if field in INT_FIELDS else float(profile[field])


def _form_profile():
    ss = st.session_state
    income = ss.dlg_income or 0
    profile = {
        "customer_id": ss.dlg_profile["customer_id"],
        "gender": ss.dlg_gender,
        "age": ss.dlg_age,
        "MonthlyIncome": float(income * 10_000),
        "DebtRatio": round(ss.dlg_repay / income * 100, 1) if income else 0.0,
        "RevolvingUtilizationOfUnsecuredLines": ss.dlg_util / 100,
    }
    for field, *_ in DELINQUENCY_FIELDS + ACCOUNT_FIELDS + ALT_FIELDS:
        profile[field] = ss[f"dlg_{field}"]
    return profile


def _open_form(profile, loaded):
    _fill_form(profile, loaded)
    st.session_state.scr_open_form = True


def _reset_form():
    _fill_form(st.session_state.dlg_profile, st.session_state.dlg_loaded)


def _stepper_row(fields):
    for col, (field, label) in zip(st.columns(len(fields), gap="small"), fields):
        with col:
            st.number_input(label, min_value=0, step=1, key=f"dlg_{field}")


@st.dialog("고객 정보 입력", width="medium")
def _customer_form():
    ss = st.session_state
    customer_id = ss.dlg_profile["customer_id"]
    st.html('<div class="dlg-sub">'
            + ("불러온 값은 그대로 두고 바뀐 항목만 수정하세요." if ss.dlg_loaded else "고객 정보를 입력하고 심사를 실행하세요.")
            + "</div>")

    if ss.dlg_loaded:
        with st.container(key="dlg-loaded", horizontal=True, vertical_alignment="center"):
            st.html(f'<div class="dlg-loaded-text"><b>{escape(customer_id)}</b> 고객 정보를 불러왔어요</div>')
            if st.button("다른 고객 불러오기", key="dlg-other", type="tertiary"):
                ss.scr_current = None
                st.rerun()

    st.html('<div class="dlg-section">기본 정보</div>')
    c1, c2, c3 = st.columns(3, gap="small")
    with c1, st.container(key="dlg-unit-age"):
        st.number_input("나이", min_value=18, max_value=100, step=1, key="dlg_age")
    with c2, st.container(key="dlg-gender-box"):
        st.radio("성별", ["여성", "남성"], index=None, horizontal=True, key="dlg_gender")
    with c3, st.container(key="dlg-unit-income"):
        st.number_input("월 소득", min_value=0, step=10, key="dlg_income")
    st.html('<div class="dlg-hint">성별은 공정성 검증용으로만 기록하고 심사(예측)에는 쓰지 않아요.</div>')

    st.html('<div class="dlg-section">금융 이력</div>')
    c1, c2 = st.columns(2, gap="medium")
    with c1, st.container(key="dlg-unit-repay"):
        st.number_input("월 대출 상환액", min_value=0, step=1, key="dlg_repay")
        ratio = f"{ss.dlg_repay / ss.dlg_income * 100:.0f}%" if ss.dlg_income else "—"
        st.html(f'<div class="dlg-hint">부채비율 <b>{ratio}</b> 자동 계산</div>')
    with c2, st.container(key="dlg-util"):
        st.html(f'<div class="dlg-util-head"><span>신용한도 사용률</span><b>{ss.dlg_util}%</b></div>')
        st.slider("신용한도 사용률", 0, 100, key="dlg_util", label_visibility="collapsed")

    st.html('<div class="dlg-label">최근 2년 연체 횟수</div>')
    with st.container(key="dlg-steps-due"):
        _stepper_row(DELINQUENCY_FIELDS)
    st.html('<div class="dlg-label">보유 계좌 · 가족</div>')
    with st.container(key="dlg-steps-acct"):
        _stepper_row(ACCOUNT_FIELDS)

    with st.container(key="dlg-alt-head", horizontal=True, vertical_alignment="center"):
        tag = '<span class="dlg-tag">자동 연동</span>' if ss.dlg_loaded and not ss.dlg_alt_edit else ""
        st.html(f'<div class="dlg-section">대안 데이터 {tag}</div>')
        if ss.dlg_loaded and not ss.dlg_alt_edit:
            if st.button("직접 수정", key="dlg-alt-edit", type="tertiary"):
                ss.dlg_alt_edit = True
                st.rerun(scope="fragment")
    if ss.dlg_loaded and not ss.dlg_alt_edit:
        cells = "".join(
            f'<div class="dlg-alt-cell"><span>{label}</span><b>{fmt.format(ss[f"dlg_{field}"])}</b></div>'
            for field, label, _, _, fmt in ALT_FIELDS
        )
        st.html(f'<div class="dlg-alt">{cells}</div>')
    else:
        with st.container(key="dlg-alt-edit-box"):
            for row in (ALT_FIELDS[:3], ALT_FIELDS[3:]):
                for col, (field, label, lo, hi, _) in zip(st.columns(3, gap="small"), row):
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
# 첫 화면
# ---------------------------------------------------------------------------
def _load_by_id():
    customer_id = st.session_state.scr_load_id.strip().upper()
    profile = _find_customer(customer_id) if customer_id else None
    if profile is None:
        st.session_state.scr_load_msg = customer_id or None
        return
    st.session_state.scr_load_msg = None
    _open_form(profile, loaded=True)


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
        r, saved_at = h["result"], h["saved_at"]
        approved = r["approval_status"] == "APPROVED"
        tone = ("mid" if GRADE_RISK.get(r["risk_grade"]) == "mid" else "low") if approved else "high"
        when = ("오늘 " if saved_at.date() == today else saved_at.strftime("%m.%d ")) + saved_at.strftime("%H:%M")
        rows.append(f'<tr><th>{escape(r["customer_id"])}</th>'
                    f'<td><span class="scr-pill {tone}">{"승인" if approved else "거절"}</span></td>'
                    f'<td>{escape(r["risk_grade"])}</td><td>{r["credit_score"]}</td>'
                    f'<td class="scr-time">{when}</td></tr>')
    body = "".join(rows) or '<tr><td colspan="5" class="scr-history-empty">아직 저장한 심사가 없어요. 심사 결과 화면에서 저장하면 여기에 남아요.</td></tr>'
    return f"""
    <div class="scr-card-title">최근 심사</div>
    <table class="scr-history">
      <thead><tr><th>고객 ID</th><th>결과</th><th>등급</th><th>신용 점수</th><th>심사 시각</th></tr></thead>
      <tbody>{body}</tbody>
    </table>"""


def _start_view():
    render_page_header("고객 심사", "고객을 불러오거나 새로 입력해 심사를 시작하세요")
    if st.session_state.scr_error is not None:
        st.html(api_error_html("POST /predict", st.session_state.scr_error, "banner"))

    with st.container(key="scr-start"):
        st.html("""
        <div class="scr-start-icon">description</div>
        <div class="scr-start-title">진행 중인 심사가 없어요</div>
        <div class="scr-start-sub">기존 고객은 ID로 불러오면 정보가 자동으로 채워져요.</div>
        """)
        with st.container(key="scr-load"):
            with st.form("scr-load-form", border=False, enter_to_submit=True):
                id_col, btn_col = st.columns([3, 1], gap="small", vertical_alignment="bottom")
                id_col.text_input("고객 ID", key="scr_load_id", label_visibility="collapsed",
                                  placeholder="고객 ID 입력 (예: C100001)")
                btn_col.form_submit_button("불러오기", type="primary", on_click=_load_by_id, use_container_width=True)
            missing = st.session_state.get("scr_load_msg")
            if missing:
                st.html(f'<div class="scr-load-msg"><b>{escape(missing)}</b> 고객을 찾지 못했어요. '
                        "이번에 심사한 고객이나 샘플 고객(C100001~C100003)만 불러올 수 있어요.</div>")
            st.html('<div class="scr-or"><span>또는</span></div>')
        with st.container(key="scr-new-wrap", horizontal=True, horizontal_alignment="center"):
            st.button("신규 고객 직접 입력", key="scr-new", icon=":material/add:", on_click=_new_customer)
        with st.container(key="scr-samples", horizontal=True, horizontal_alignment="center",
                          vertical_alignment="center"):
            st.html('<span class="scr-samples-label">샘플로 바로 심사</span>')
            for customer_id, (label, _, _) in SAMPLE_CUSTOMERS.items():
                st.button(label, key=f"scr-sample-{customer_id}", on_click=_run_sample, args=(customer_id,))

    with st.container(key="scr-history-card"):
        st.html(_history_table())


# ---------------------------------------------------------------------------
# 결과 화면
# ---------------------------------------------------------------------------
def _new_screening():
    st.session_state.scr_current = None
    st.session_state.scr_error = None


def _save_current():
    current = st.session_state.scr_current
    current["saved"] = True
    st.session_state.scr_history = (
        [{"result": current["result"], "saved_at": datetime.now()}] + st.session_state.scr_history
    )[:HISTORY_LIMIT]


def _to_xai():
    st.session_state.xai_customer = st.session_state.scr_current["result"]["customer_id"]


def _completed_at(result):
    try:
        stamp = datetime.fromisoformat(result["timestamp"]).astimezone()
    except (KeyError, ValueError):
        stamp = datetime.now()
    return stamp.strftime("%Y.%m.%d %H:%M")


def _summary_html(profile, result):
    late = sum(int(profile[f]) for f, _ in DELINQUENCY_FIELDS)
    age = f'{profile["age"]}세' + (f' {profile["gender"]}' if profile.get("gender") else "")
    items = [
        ("고객 ID", profile["customer_id"]),
        ("나이 · 성별", age),
        ("월 소득", f'{profile["MonthlyIncome"] / 10_000:,.0f}만원'),
        ("부채비율", f'{profile["DebtRatio"]:.0f}%'),
        ("연체", f"{late}회" if late else "없음"),
        ("고객 유형", "씬파일러" if result.get("is_thin_filer") else "일반 고객"),
    ]
    return '<div class="scr-summary">' + "".join(
        f'<div class="scr-summary-item"><span>{label}</span><b>{escape(str(value))}</b></div>' for label, value in items
    ) + "</div>"


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
    segs = "".join(f'<i class="seg-{g}"></i>' for g, _ in scale)
    labels = "".join(f'<span class="{"on" if g == grade else ""}">{g}</span>' for g, _ in scale)
    return (f'<div class="scr-grade-bar">{segs}<b class="scr-marker" style="left:{marker:.1f}%"></b></div>'
            f'<div class="scr-grade-labels">{labels}</div>')


def _decision_html(result):
    approved = result["approval_status"] == "APPROVED"
    grade, score, prob = result["risk_grade"], result["credit_score"], result["default_probability"]
    risk = GRADE_RISK.get(grade, "high")
    threshold = result.get("approval_threshold", config.APPROVAL_THRESHOLD)
    message = (f"부도 확률이 승인 기준({threshold:.0%} 미만)을 충족했어요." if approved
               else f"부도 확률이 승인 기준({threshold:.0%} 미만)을 넘었어요.")
    return f"""
    <div class="scr-decision {'ok' if approved else 'ng'}">
      <div class="scr-decision-head">
        <span class="scr-decision-icon">{'check' if approved else 'close'}</span>
        <div class="scr-decision-text">
          <div><b>{'승인' if approved else '거절'}</b><span>{'APPROVED' if approved else 'REJECTED'}</span></div>
          <p>{message}</p>
        </div>
        <span class="scr-risk {risk}">{RISK_LABELS[risk]} · {escape(grade)}등급</span>
      </div>
      <div class="scr-decision-body">
        <div class="scr-score">
          <div class="scr-score-top"><span>신용 점수</span><div><b>{score}</b><em>/ 1000</em></div></div>
          {_score_bar(score, grade)}
        </div>
        <div class="scr-prob">
          <span>부도 확률</span>
          <div><b>{prob * 100:.1f}</b><em>%</em></div>
          <p>100명 중 약 {round(prob * 100)}명이 2년 내 연체할 수준</p>
        </div>
      </div>
    </div>"""


def _action_card(key, title, caption, done=False):
    """카드 전체가 눌리는 버튼 (보이는 내용은 HTML, 위에 투명 버튼을 덮는다)."""
    with st.container(key=f"scr-act-{key}"):
        icon = "check" if done else "chevron_right"
        st.html(f'<div class="scr-act {"done" if done else ""}"><div><b>{title}</b><span>{caption}</span></div>'
                f'<i>{icon}</i></div>')
        return st.button(title, key=f"scr-act-btn-{key}", disabled=done)


def _result_view():
    current = st.session_state.scr_current
    profile, result = current["profile"], current["result"]

    with st.container(key="scr-result-head", horizontal=True, vertical_alignment="center"):
        render_page_header("고객 심사", f"{_completed_at(result)} 심사 완료")
        st.button("새 심사", key="scr-new-screening", type="primary", icon=":material/add:", on_click=_new_screening)

    with st.container(key="scr-summary-card", horizontal=True, vertical_alignment="center"):
        st.html(_summary_html(profile, result))
        st.button("정보 수정", key="scr-edit", icon=":material/edit:", on_click=_open_form, args=(profile, False))

    st.html(_decision_html(result))

    left, right = st.columns(2, gap="small")
    with left:
        if _action_card("xai", "판단 근거 보기", "어떤 정보가 점수에 영향을 줬는지 XAI 분석에서 확인"):
            _to_xai()
            st.switch_page("views/xai.py")
    with right:
        if current["saved"]:
            _action_card("save", "심사 이력에 저장했어요", "새 심사를 시작하면 첫 화면의 최근 심사에서 볼 수 있어요", done=True)
        elif _action_card("save", "심사 결과 저장", "입력값과 결과를 심사 이력에 기록"):
            _save_current()
            st.rerun()


# ---------------------------------------------------------------------------
# 페이지
# ---------------------------------------------------------------------------
st.html(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>")
_init_state()

if st.session_state.scr_current is None:
    _start_view()
else:
    _result_view()

if st.session_state.pop("scr_open_form", False):
    _customer_form()
