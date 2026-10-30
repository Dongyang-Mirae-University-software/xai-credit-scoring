"""고객 심사 페이지"""
import json
import time
from html import escape
from pathlib import Path

import streamlit as st

import api_client
import config
from components.api_error import api_error_html
from components.page_header import render_page_header

CSS_PATH = Path(__file__).with_suffix(".css")
RECENT_LIMIT = 5

# (요청 키, 라벨, 기본값, 최소, 최대, 증감 단위) — 요청 키·단위는 실제 API(src/api/schemas.py CustomerInput)와 동일
# 부채비율·통신비·공과금 납부율은 % 로 보내고 서버가 ÷100 한다 (이슈 #70)
# 기본값은 백엔드 Swagger 예시 고객(src/api/schemas.py CUSTOMER_EXAMPLE)과 같다
TRADITIONAL_FIELDS = [
    ("age", "나이", 45, 18, 100, 1),
    ("MonthlyIncome", "월소득 (원)", 4_200_000, 0, None, None),
    ("DebtRatio", "부채비율 (%)", 32.0, 0.0, None, 1.0),
    ("RevolvingUtilizationOfUnsecuredLines", "리볼빙 이용률", 0.48, 0.0, None, 0.01),
    ("NumberOfDependents", "부양가족 수", 2, 0, None, 1),
    ("NumberOfTime30-59DaysPastDueNotWorse", "30–59일 연체 횟수", 1, 0, None, 1),
    ("NumberOfTime60-89DaysPastDueNotWorse", "60–89일 연체 횟수", 0, 0, None, 1),
    ("NumberOfTimes90DaysLate", "90일 이상 연체", 0, 0, None, 1),
    ("NumberOfOpenCreditLinesAndLoans", "개설 신용계좌 수", 6, 0, None, 1),
    ("NumberRealEstateLoansOrLines", "부동산 대출 건수", 1, 0, None, 1),
]
ALTERNATIVE_FIELDS = [
    ("telecom_payment_rate", "통신비 정상납부율 (%)", 90.0, 0.0, 100.0, 1.0),
    ("utility_payment_rate", "공과금 납부율 (%)", 93.0, 0.0, 100.0, 1.0),
    ("spending_consistency", "소비 일관성", 75.0, 0.0, 100.0, 1.0),
    ("regular_payment_count", "정기결제 건수", 5, 0, 20, 1),
    ("app_login_frequency", "앱 로그인 빈도 (월)", 22, 0, 30, 1),
]
ALL_FIELDS = TRADITIONAL_FIELDS + ALTERNATIVE_FIELDS

# 등급 → 위험도 (A·B 저위험 / C 중위험 / D·E 고위험)
GRADE_RISK = {"A": "low", "B": "low", "C": "mid", "D": "high", "E": "high"}


# ---------------------------------------------------------------------------
# 상태
# ---------------------------------------------------------------------------
def _key(field):
    return f"scr_{field}"


def _format_won(value):
    return f"{int(value):,}"


def _init_state():
    st.session_state.setdefault("scr_seq", 1)
    st.session_state.setdefault("last_customer", None)
    st.session_state.setdefault("last_predict", None)
    st.session_state.setdefault("predict_error", None)
    st.session_state.setdefault("recent_predictions", [])
    st.session_state.setdefault("customers", {})
    for field, _, default, *_ in ALL_FIELDS:
        if _key(field) not in st.session_state:
            st.session_state[_key(field)] = _format_won(default) if field == "MonthlyIncome" else default


def _reset_form():
    for field, _, default, *_ in ALL_FIELDS:
        st.session_state[_key(field)] = _format_won(default) if field == "MonthlyIncome" else default


def _reformat_income():
    raw = st.session_state[_key("MonthlyIncome")]
    digits = "".join(ch for ch in raw if ch.isdigit())
    st.session_state[_key("MonthlyIncome")] = _format_won(digits) if digits else ""


def _customer_id():
    return f"C{st.session_state.scr_seq:06d}"


def _build_payload():
    payload = {"customer_id": _customer_id()}
    for field, *_ in ALL_FIELDS:
        value = st.session_state[_key(field)]
        if field == "MonthlyIncome":
            digits = "".join(ch for ch in value if ch.isdigit())
            value = float(digits) if digits else 0.0
        payload[field] = value
    return payload


# ---------------------------------------------------------------------------
# 화면 조각
# ---------------------------------------------------------------------------
def _section_title(title, caption):
    st.html(f'<div class="scr-section"><span class="scr-section-title">{title}</span>'
            f'<span class="scr-section-caption">{caption}</span></div>')


def _field_grid(fields):
    for row_start in range(0, len(fields), 5):
        cols = st.columns(5, gap="small")
        for col, (field, label, default, min_v, max_v, step) in zip(cols, fields[row_start:row_start + 5]):
            with col:
                if field == "MonthlyIncome":
                    st.text_input(label, key=_key(field), on_change=_reformat_income)
                else:
                    st.number_input(
                        label, min_value=min_v, max_value=max_v, step=step,
                        format=("%.2f" if step < 1 else "%.0f") if isinstance(default, float) else None,
                        key=_key(field),
                    )


def _request_preview(payload):
    body = escape(json.dumps(payload, ensure_ascii=False, indent=2))
    st.html(f"""
    <div class="scr-section">
      <span class="scr-section-title">요청 미리보기</span>
      <span class="scr-section-caption">POST {escape(config.API_BASE_URL)}/predict · 대시보드는 이 JSON만 보내고 모델은 만지지 않음</span>
    </div>
    <pre class="scr-code">{body}</pre>
    """)


def _run_predict(payload):
    try:
        started = time.perf_counter()
        result = api_client.predict(payload)
        st.session_state.predict_latency = time.perf_counter() - started
    except Exception as e:
        st.session_state.predict_error = e  # 심사 결과 카드 자리에 안내한다 (_result_card)
        return

    st.session_state.predict_error = None
    st.session_state.last_customer = payload
    st.session_state.last_predict = result
    st.session_state.customers[result["customer_id"]] = {"payload": payload, "result": result}
    st.session_state.recent_predictions = (
        [result] + st.session_state.recent_predictions
    )[:RECENT_LIMIT]
    st.session_state.scr_seq += 1
    st.rerun()  # 사이드바(응답 시간)와 씬파일러 배지가 이번 결과로 바로 보이도록 한 번 더 그린다


def _thin_filer_badge():
    """마지막 심사 결과의 씬파일러 판정 (/predict 의 is_thin_filer)."""
    result = st.session_state.last_predict
    if not result or "is_thin_filer" not in result:
        return
    label = "예" if result["is_thin_filer"] else "아니오"
    st.html(f'<span class="scr-thin">씬파일러 판정: {label} · 고객 #{escape(result["customer_id"])}</span>')


def _status_pill(status):
    if status == "APPROVED":
        return '<span class="scr-pill low">승인</span>'
    return '<span class="scr-pill high">거절</span>'


def _result_card():
    result = st.session_state.last_predict
    if st.session_state.predict_error is not None:
        st.html('<div class="scr-card-head"><span class="scr-card-title">심사 결과</span></div>'
                + api_error_html("POST /predict", st.session_state.predict_error, "inline"))
        return
    if result is None:
        st.html("""
        <div class="scr-card-head"><span class="scr-card-title">심사 결과</span></div>
        <div class="scr-empty">고객 정보를 입력하고 <b>심사 실행</b>을 누르면 결과가 표시됩니다.</div>
        """)
        return

    grade = result["risk_grade"]
    risk = GRADE_RISK.get(grade, "high")
    approved = result["approval_status"] == "APPROVED"
    score = result["credit_score"]
    scale_labels = "".join(
        f"<span>{escape(g)} {upper}</span>" for g, upper in config.GRADE_SCALE
    )

    st.html(f"""
    <div class="scr-card-head">
      <span class="scr-card-title">심사 결과</span>
      <span class="scr-badge blue">고객 #{escape(result["customer_id"])}</span>
    </div>
    <div class="scr-decision">
      <span class="scr-decision-badge {'low' if approved else 'high'}">{'승인' if approved else '거절'}</span>
      <span class="scr-decision-raw">approval_status<br>{escape(result["approval_status"])}</span>
    </div>
    <div class="scr-score">
      <span class="scr-score-value">{score}</span>
      <span class="scr-score-max">/ 1000</span>
      <span class="scr-score-label">신용점수 (credit_score)</span>
    </div>
    <div class="scr-bar"><div class="scr-bar-fill {risk}" style="width:{max(0, min(score, 1000)) / 10}%"></div></div>
    <div class="scr-scale"><span>0</span>{scale_labels}</div>
    <div class="scr-tiles">
      <div class="scr-tile"><div class="scr-tile-label">부도 확률</div>
        <div class="scr-tile-value">{result["default_probability"] * 100:.1f}%</div></div>
      <div class="scr-tile"><div class="scr-tile-label">리스크 등급</div>
        <div class="scr-tile-value risk-{risk}">{escape(grade)}</div></div>
      <div class="scr-tile"><div class="scr-tile-label">승인 임계값</div>
        <div class="scr-tile-value">p &lt; {result.get("approval_threshold", config.APPROVAL_THRESHOLD):.2f}</div></div>
    </div>
    <div class="scr-legend">
      <span class="scr-legend-label">등급 색상</span>
      <span class="scr-pill low">A·B 저위험</span>
      <span class="scr-pill mid">C 중위험</span>
      <span class="scr-pill high">D·E 고위험</span>
    </div>
    """)
    if st.button("이 결과의 근거 보기 → XAI 분석 탭", key="scr-to-xai", use_container_width=True):
        st.session_state.xai_customer = result["customer_id"]
        st.switch_page("views/xai.py")


def _recent_card():
    recent = st.session_state.recent_predictions
    rows = "".join(
        f'<div class="scr-recent-row">'
        f'<span class="scr-recent-id">{escape(r["customer_id"])}</span>'
        f'<span class="scr-recent-score">{r["credit_score"]}</span>'
        f'<span class="scr-recent-grade">등급 {escape(r["risk_grade"])}</span>'
        f'{_status_pill(r["approval_status"])}</div>'
        for r in recent
    ) or '<div class="scr-empty">아직 심사 이력이 없습니다.</div>'
    st.html(f"""
    <div class="scr-card-head">
      <span class="scr-card-title">최근 심사</span>
      <span class="scr-card-caption">세션 기준 {RECENT_LIMIT}건</span>
    </div>
    {rows}
    """)


# ---------------------------------------------------------------------------
# 페이지
# ---------------------------------------------------------------------------
st.html(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>")
_init_state()
render_page_header("고객 심사", "고객 정보를 입력하면 FastAPI /predict 를 호출해 승인 여부·신용점수·리스크 등급을 반환합니다")

left, right = st.columns([1.45, 1], gap="medium")

with left:
    with st.container(key="scr-form"):
        _section_title("전통 신용 데이터", "Give Me Some Credit 10개 변수")
        _field_grid(TRADITIONAL_FIELDS)
        _section_title("대안 데이터", "시뮬레이터 생성 5개 변수 · 씬파일러 판정 근거")
        _field_grid(ALTERNATIVE_FIELDS)
        st.html('<div class="scr-notice">ⓘ 민감정보(인종·종교·장애 여부)는 입력받지 않습니다. '
                '성별·연령대는 공정성 검증에만 사용되며 예측 피처에서 제외됩니다.</div>')

        payload = _build_payload()
        _request_preview(payload)

        thin_col, reset_col, submit_col = st.columns([3.2, 1, 1.3], gap="small", vertical_alignment="center")
        thin_slot = thin_col.empty()  # 심사 결과가 나온 뒤 채운다 (아래)
        with reset_col:
            st.button("초기화", key="scr-reset", on_click=_reset_form, use_container_width=True)
        with submit_col:
            submitted = st.button("심사 실행 →", key="scr-submit", type="primary", use_container_width=True)

if submitted:
    _run_predict(payload)
with thin_slot.container():
    _thin_filer_badge()

with right:
    with st.container(key="scr-result"):
        _result_card()
    with st.container(key="scr-recent"):
        _recent_card()
