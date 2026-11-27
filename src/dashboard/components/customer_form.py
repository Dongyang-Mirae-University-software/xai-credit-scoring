"""고객 심사 흐름 (은행 창구 기준, 한 화면씩 넘긴다). 고객 심사 · XAI 분석 두 화면에서 쓴다.

1/3 본인 확인: 이름 + 주민등록번호 → POST /customers/lookup (은행이 가진 신용정보 · 대안정보를 이때 받아 둔다)
2/3 신청 정보: 월소득 · 부양가족 수를 고객에게 받아 입력 (은행 기록으로 미리 채우지 않는다)
    → 불러오기 연출: 신용정보 · 대안정보를 가져오는 기관을 보여 준다 (config.CB_LOADING_SECONDS)
3/3 전체 확인: 모든 정보를 보여 주고, 틀린 것만 고쳐 심사 실행 → POST /predict + /explain → 결과 화면
    (결과는 POST /screenings 로 서버 이력에 바로 남는다)
- 조회 결과가 없거나 조회 API 가 아직 없으면 '직접 입력'(나이 · 성별은 주민등록번호로 채운다, 연출 없이 3/3) 또는 시연용 샘플
- 결과 화면 '정보 수정' · XAI '조건 바꿔 재심사' → 그 고객 값이 채워진 3/3 화면

주민등록번호는 조회 요청에만 쓴다. 심사 요청 · 이력에는 넣지 않고 화면에는 가린 값(900101-1******)만 남긴다.
고객 번호(customer_id)는 화면에 보이지 않는다 (내부 구분용).
시연용 샘플은 API 가 아니라 로컬 파일 samples.json(테스트셋 고객 입력값)에서 읽는다. 모델이 아니라 입력값이라
'대시보드는 모델을 직접 읽지 않는다' 규칙에 맞다.
"""
import json
import time
from base64 import b64encode
from functools import lru_cache
from html import escape
from pathlib import Path

import streamlit as st

import api_client
import config
import features as F
from components import identity, ui
from components.api_error import api_error_html

CSS_PATH = Path(__file__).with_suffix(".css")
WORKSPACE_CSS_PATH = Path(__file__).with_name("workspace.css")


def css():
    """customer_form.css. 그릴 때마다 파일을 읽는다 (처음 불러올 때만 읽으면 CSS 를 고쳐도 재시작 전까지 안 바뀐다)."""
    return CSS_PATH.read_text(encoding="utf-8")


def workspace_css():
    """심사 업무 화면 공통 톤 (workspace.css). 그릴 때마다 읽는다."""
    return WORKSPACE_CSS_PATH.read_text(encoding="utf-8")


SCREENING_PAGE = "views/screening.py"
NEW_ID_LOOKUP = 100  # 신규 고객 ID 를 고를 때 피할 서버 이력 수 (GET /screenings 최대치)

# 월 대출 상환액(원, CB 대출 정보). 부채비율(DebtRatio %) = 상환액 ÷ 고객이 적은 월소득 × 100 으로 만들어 /predict 에 넣는다
REPAY_FIELD = "MonthlyRepayment"

# 입력 목록 칸: (요청 키, 라벨, 단위, 화면 배율, 소수 자릿수, 최소, 최대, 필수)
# 화면 값 = 요청 값 × 배율. 요청 단위는 실제 API(CustomerInput)와 같다: 월소득 원, 부채비율 · 납부율 %, 한도 사용률 0~1
BASIC_FIELDS = [
    ("age", "나이", "세", 1, 0, identity.MIN_AGE, 120, True),
]
# 신청 정보: 대출 신청서에 고객이 직접 적는 값. 은행 기록으로 미리 채우지 않는다 (시연용 샘플 · 다시 심사는 채운다)
APPLY_FIELDS = [
    ("MonthlyIncome", "월소득", "만원", 1e-4, 0, 0, None, True),
    ("NumberOfDependents", "부양가족 수", "명", 1, 0, 0, 20, True),
]
APPLY_KEYS = {f for f, *_ in APPLY_FIELDS}
APPLY_SOURCES = {"bank", "manual"}  # 이 출처에서는 신청 정보를 빈칸으로 시작한다
CREDIT_FIELDS = [
    (REPAY_FIELD, "월 대출 상환액", "만원", 1e-4, 0, 0, None, False),
    ("RevolvingUtilizationOfUnsecuredLines", "신용한도 사용률", "%", 100, 1, 0, None, False),
    ("NumberOfOpenCreditLinesAndLoans", "보유 대출 · 신용 한도", "개", 1, 0, 0, None, False),
    ("NumberRealEstateLoansOrLines", "부동산 담보대출", "개", 1, 0, 0, None, False),
    ("NumberOfTime30-59DaysPastDueNotWorse", "30~59일 연체", "회", 1, 0, 0, None, False),
    ("NumberOfTime60-89DaysPastDueNotWorse", "60~89일 연체", "회", 1, 0, 0, None, False),
    ("NumberOfTimes90DaysLate", "90일 이상 연체", "회", 1, 0, 0, None, False),
]
ALT_FIELDS = [
    ("telecom_payment_rate", "통신비 정상납부율", "%", 1, 1, 0, 100, True),
    ("utility_payment_rate", "공과금 납부율", "%", 1, 1, 0, 100, True),
    ("spending_consistency", "소비 일관성 점수", "점", 1, 1, 0, 100, True),
    ("regular_payment_count", "정기결제 건수", "건", 1, 0, 0, 20, True),
    ("app_login_frequency", "앱 로그인 빈도", "회", 1, 0, 0, 30, True),
]
ALL_FIELDS = BASIC_FIELDS + APPLY_FIELDS + CREDIT_FIELDS + ALT_FIELDS
# 결과 화면 요약에서 연체 합계를 낼 때 쓴다
DELINQUENCY_FIELDS = [(f, label) for f, label, *_ in CREDIT_FIELDS if "DaysPastDue" in f or "DaysLate" in f]
FORM_KEYS = [f for f, *_ in ALL_FIELDS]                                     # 2/3 · 3/3 화면 칸
REQUEST_KEYS = [f for f in FORM_KEYS if f != REPAY_FIELD] + ["DebtRatio"]    # 심사 요청 칸
DISPLAY_KEYS = ("gender", "name", "rrn_masked", REPAY_FIELD)  # 화면 표시 · 계산용. 심사 요청에는 넣지 않는다

# 정보 출처 → 3/3 화면 머리 배지
SOURCES = {"bank": "은행 보유 정보", "manual": "직접 입력", "sample": "시연용 샘플", "edit": "다시 심사"}

SAMPLES_PATH = Path(__file__).resolve().parent.parent / "samples.json"
CUSTOMERS = {c["customer_id"]: c for c in json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))}
SAMPLE_CHIPS = [(cid, c["sample_label"]) for cid, c in CUSTOMERS.items() if c.get("sample_label")]


# ---------------------------------------------------------------------------
# 상태 · 고객 찾기 · 심사 실행
# ---------------------------------------------------------------------------
def init_state():
    ss = st.session_state
    ss.setdefault("scr_current", None)   # 지금 보고 있는 심사 결과 {profile, result}. 있으면 고객 심사 = 결과 화면
    ss.setdefault("scr_step", "identify")  # 결과가 없을 때: identify 1/3 · apply 2/3 · agree 동의 · loading 연출 · confirm 3/3 · judging 심사 연출
    ss.setdefault("customers", {})       # 지금 상담 중인 고객 {id: {payload, result, profile}} (한 명)
    ss.setdefault("xai_explain", {})     # /explain 응답 캐시


def new_id():
    """신규 고객 ID: 아직 아무도 안 쓴 가장 작은 C000001 형식 번호.
    고객 파일, 이번 세션에서 심사한 고객, 서버 이력(GET /screenings)의 ID 를 모두 피한다."""
    used = set(CUSTOMERS) | set(st.session_state.customers)
    try:
        used |= {r.get("customer_id") for r in api_client.screenings(NEW_ID_LOOKUP)}
    except Exception:
        pass  # 이력을 못 읽어도 화면은 연다 (파일 · 세션 ID 는 피한다)
    n = 1
    while f"C{n:06d}" in used:
        n += 1
    return f"C{n:06d}"


def _blank_profile():
    """직접 입력: 필수 칸은 비우고(None), 횟수 · 비율 칸은 0 으로 시작한다."""
    profile = {"customer_id": new_id(), "gender": None, **{f: None for f in FORM_KEYS + ["DebtRatio"]}}
    for field, _, _, _, _, _, _, required in CREDIT_FIELDS:
        if not required:
            profile[field] = 0
    return profile


def find_customer(customer_id):
    """시연용 샘플 파일 → 이번 세션에서 심사한 고객 순서로 찾는다. 없으면 None. API 는 부르지 않는다."""
    found = CUSTOMERS.get(customer_id)
    if found:
        rrn = found.get("rrn") or ""   # 시연용 가짜 번호 (화면에는 가린 값만)
        masked = identity.masked(rrn[:6], rrn[6:]) if len(rrn) == 13 else None
        return {"customer_id": customer_id, "name": found.get("name"), "gender": found.get("gender"),
                "rrn_masked": masked, **{k: found.get(k) for k in REQUEST_KEYS}}
    entry = st.session_state.customers.get(customer_id)
    if entry and "profile" in entry:
        return entry["profile"]
    return None


def clear_result():
    """첫 화면(1/3 본인 확인)으로: 보고 있던 결과를 닫는다 (버튼 on_click 에서 불러도 된다)."""
    st.session_state.scr_current = None
    st.session_state.scr_step = "identify"


# ---------------------------------------------------------------------------
# 상담 단위: 창구는 한 번에 한 고객. 새 고객을 시작하거나 '상담 마치기' 를 누르면 앞 고객 정보를 세션에서 지운다
# ---------------------------------------------------------------------------
_FORM_KEYS = ("frm_profile", "frm_values", "frm_shown", "frm_source", "frm_gender", "frm_missing", "frm_error",
              "frm_agree", "apl_missing", "idn_notice", "idn_errors")


def _clear_consult():
    """지금 고객의 입력 · 결과 · 설명 · '승인되려면' 캐시를 지운다."""
    ss = st.session_state
    ss.customers, ss.xai_explain, ss.xai_cf, ss.scr_current = {}, {}, {}, None
    for key in _FORM_KEYS:
        ss.pop(key, None)


def current_customer():
    """지금 상담 중인 고객 ID (심사를 마친 고객). 없으면 None."""
    return next(iter(st.session_state.get("customers") or {}), None)


def end_consult():
    """상담 마치기: 지금 고객 정보를 모두 지우고 1/3 본인 확인(빈 칸)으로. 다른 화면에서 부르면 뒤에 st.switch_page."""
    ss = st.session_state
    _clear_consult()
    for key in ("idn_name", "idn_front", "idn_back"):
        ss[key] = ""
    ss.scr_step = "identify"
    ss.consult_toast = "상담을 마쳤어요. 다음 고객을 맞이해 주세요."


def load_past(name, rrn, rrn_masked):
    """XAI 분석: 이미 심사한 고객의 가장 최근 심사를 불러와 지금 상담 고객으로 둔다 (판단 근거를 바로 보여 준다).
    그때 입력값으로 /explain 을 다시 부른다. 돌려주는 값: "ok" · api_client.NO_SCREENING · NOT_READY · None(고객 없음)."""
    record = api_client.latest_screening(name, rrn)
    if not isinstance(record, dict):
        return record
    ss = st.session_state
    _clear_consult()
    payload = record["input"]
    customer_id = record["customer_id"]
    result = {**record["result"], "customer_id": customer_id}
    profile = {**payload, "customer_id": customer_id, "name": record.get("name") or name,
               "gender": record.get("gender"), "rrn_masked": rrn_masked}
    ss.customers = {customer_id: {"payload": payload, "result": result, "profile": profile}}
    try:
        ss.xai_explain[customer_id] = api_client.explain(payload)
    except Exception as e:
        ss.xai_explain[customer_id] = e
    return "ok"


def open_result(customer_id):
    """XAI 분석 → 고객 심사 결과 화면: 이미 심사한 고객의 결과를 연다. 다른 화면에서 부르면 뒤에 st.switch_page."""
    entry = st.session_state.customers.get(customer_id)
    if entry:
        st.session_state.scr_current = {"profile": entry["profile"], "result": entry["result"]}


def _payload(profile):
    """심사 요청 본문 (GMSC 10 + 대안 5). 성별 · 이름 · 가린 주민등록번호는 보내지 않는다."""
    return {k: v for k, v in profile.items() if k not in DISPLAY_KEYS}


def _save(payload, result):
    """POST /screenings: 서버 이력에 남긴다. 저장이 실패해도 심사 결과는 보여 준다."""
    try:
        api_client.save_screening({"input": payload, "result": result})
    except Exception:
        pass


def run_predict(profile, record=True):
    """심사 실행: POST /predict 와 POST /explain 을 함께 부른다 (승인이든 거절이든).
    결과를 지금 상담 중인 고객으로 두고, record 면 서버 이력(POST /screenings)에 남긴다. /predict 가 실패하면 예외를 올린다.
    /explain 이 실패하면 그 예외를 캐시에 넣어 두고 결과 화면이 안내한다 (화면을 그릴 때 다시 부르지 않는다)."""
    ss = st.session_state
    payload = _payload(profile)
    result = {**api_client.predict(payload), "customer_id": profile["customer_id"]}
    customer_id = profile["customer_id"]
    ss.customers = {customer_id: {"payload": payload, "result": result, "profile": profile}}  # 한 번에 한 고객
    try:
        ss.xai_explain[customer_id] = api_client.explain(payload)
    except Exception as e:
        ss.xai_explain[customer_id] = e
    ss.scr_current = {"profile": profile, "result": result}
    if record:
        _save(payload, result)


def get_explain(customer_id):
    """심사 실행 때 받아 둔 /explain 응답. 그때 실패했으면 그 예외를 다시 올린다."""
    explain = st.session_state.xai_explain.get(customer_id)
    if isinstance(explain, Exception):
        raise explain
    if explain is None:
        raise LookupError("이 고객의 설명(/explain)이 없어요. 다시 심사해 주세요.")
    return explain


# ---------------------------------------------------------------------------
# 1/3 본인 확인
# ---------------------------------------------------------------------------
def _identity_input():
    """입력칸의 이름 · 주민등록번호 (숫자만). 칸마다 안내 문구를 함께 돌려준다."""
    ss = st.session_state
    name = identity.clean_name(ss.get("idn_name"))
    front, back = identity.digits(ss.get("idn_front")), identity.digits(ss.get("idn_back"))
    return name, front, back, {"name": identity.name_error(name), "rrn": identity.rrn_error(front, back)}


def _lookup():
    """정보 조회: 형식 확인 → POST /customers/lookup. 찾으면 2/3, 없으면 · API 가 없으면 안내 + 직접 입력 버튼."""
    ss = st.session_state
    name, front, back, errors = _identity_input()
    ss.idn_errors = errors
    ss.idn_notice = None
    if any(errors.values()):
        return
    try:
        found = api_client.lookup_customer(name, f"{front}{back}")
    except Exception as e:
        ss.idn_notice = ("error", e)
        return
    if found == api_client.NOT_READY:
        ss.idn_notice = ("pending", None)
    elif not found:
        ss.idn_notice = ("not_found", None)
    else:
        profile = {"customer_id": found.get("customer_id") or new_id(),
                   "gender": found.get("gender") or identity.gender(back),
                   **{k: found.get(k) for k in FORM_KEYS}}
        if profile["age"] is None:
            profile["age"] = identity.age(front, back)
        _open_flow({**profile, "name": found.get("name") or name, "rrn_masked": identity.masked(front, back)}, "bank")


def _manual():
    """직접 입력: 은행 기록이 없는 고객. 나이 · 성별은 주민등록번호로 채운 빈 목록을 연다."""
    ss = st.session_state
    name, front, back, errors = _identity_input()
    ss.idn_errors = errors
    if any(errors.values()):
        return
    profile = {**_blank_profile(), "age": identity.age(front, back), "gender": identity.gender(back),
               "name": name, "rrn_masked": identity.masked(front, back)}
    _open_flow(profile, "manual")


def _sample(customer_id):
    _open_flow(find_customer(customer_id), "sample")


def _field_error(key):
    message = (st.session_state.get("idn_errors") or {}).get(key)
    if message:
        st.html(f'<p class="idn-error">{escape(message)}</p>')


def render_identify(disabled=False):
    """1/3 본인 확인 카드. disabled 면 (API 연결 끊김) 조회 · 진행 버튼을 끈다."""
    ss = st.session_state
    with st.container(key="idn-card"):
        st.html('<div class="tz-step">1 / 3</div>'
                '<h1 class="tz-title">고객님의 이름과<br>주민등록번호를 입력해 주세요</h1>'
                '<p class="tz-sub">신분증을 확인한 뒤 입력해 주세요.</p>')
        with st.form("idn-form", border=False, enter_to_submit=True):
            with st.container(key="idn-name"):
                st.text_input("이름", key="idn_name", placeholder="홍길동", max_chars=identity.NAME_MAX)
            _field_error("name")
            with st.container(key="idn-rrn", horizontal=True, vertical_alignment="bottom", gap="small"):
                st.text_input("주민등록번호", key="idn_front", placeholder="앞 6자리", max_chars=6)
                st.html('<span class="idn-dash" aria-hidden="true"></span>')
                st.text_input("주민등록번호 뒷자리", key="idn_back", placeholder="뒤 7자리", max_chars=7,
                              type="password", label_visibility="hidden")
            _field_error("rrn")
            st.form_submit_button("다음", key="idn-submit", type="primary", on_click=_lookup,
                                  disabled=disabled, use_container_width=True)

        kind, error = ss.get("idn_notice") or (None, None)
        if kind == "error":
            st.html(api_error_html("POST /customers/lookup", error, "inline"))
        elif kind in ("not_found", "pending"):
            text = ("일치하는 고객 정보가 없어요. 이름과 주민등록번호를 다시 확인하거나, 처음 오신 고객이면 직접 입력해 주세요."
                    if kind == "not_found" else
                    "정보 조회 API(POST /customers/lookup)가 아직 서버에 없어요. 직접 입력하거나 시연용 샘플로 진행해 주세요.")
            st.html(f'<div class="idn-notice {kind}">{escape(text)}</div>')
            st.button("고객 정보 직접 입력하기", key="idn-manual", on_click=_manual, disabled=disabled,
                      use_container_width=True)

    if SAMPLE_CHIPS:
        with st.container(key="idn-samples", horizontal=True, vertical_alignment="center", gap="small"):
            st.html('<span class="idn-samples-label">시연용 샘플</span>')
            for customer_id, label in SAMPLE_CHIPS:
                st.button(label, key=f"idn-sample-{customer_id}", on_click=_sample, args=(customer_id,),
                          disabled=disabled)


# ---------------------------------------------------------------------------
# 입력값 보관 (2/3 · 3/3 공통)
# 화면에 없는 입력칸 값은 Streamlit 이 지운다 → 값은 frm_values 에 두고, 입력칸은 그릴 때 채우고 바뀌면 옮겨 담는다
# ---------------------------------------------------------------------------
def _shown(value, scale, decimals):
    """요청 값 → 화면 값 (배율 · 반올림). 정수 칸은 int, 소수 칸은 float."""
    if value is None:
        return None
    shown = round(float(value) * scale, decimals)
    return int(shown) if decimals == 0 else shown


def _with_repayment(profile):
    """월 대출 상환액이 없는 고객(시연용 샘플 · 예전 심사)은 부채비율 × 월소득으로 채운다."""
    if profile.get(REPAY_FIELD) is None and profile.get("DebtRatio") is not None and profile.get("MonthlyIncome"):
        return {**profile, REPAY_FIELD: profile["DebtRatio"] / 100 * profile["MonthlyIncome"]}
    return profile


def _debt_ratio(repayment, income):
    """부채비율 % = 월 대출 상환액 ÷ 월소득 × 100. 소득이 0 이면 1만원으로 나눈다 (상환액이 있으면 아주 높게 나온다)."""
    return round((repayment or 0) / max(income or 0, 10_000) * 100, 4)


def _val(field):
    return st.session_state.frm_values.get(field)


def _keep(field):
    st.session_state.frm_values[field] = st.session_state[f"frm_{field}"]


def _number(field, label, scale, decimals, lo, hi, required, **kwargs):
    """frm_values 에 묶인 숫자 입력칸. 그릴 때마다 보관한 값을 넣는다 (고친 값은 on_change 가 먼저 옮겨 둔다).
    단계를 오가는 사이 입력칸 상태가 지워지거나 다른 값으로 남아 있어도 보관한 값이 기준이다."""
    key = f"frm_{field}"
    st.session_state[key] = _val(field)
    num = int if decimals == 0 else float  # Streamlit 은 최소 · 최대 · 간격의 숫자 형이 같아야 한다
    st.number_input(label, key=key, min_value=num(lo), max_value=None if hi is None else num(hi),
                    step=num(1) if decimals == 0 else 0.1, format="%d" if decimals == 0 else "%.1f",
                    placeholder="입력" if required else "-", on_change=_keep, args=(field,), **kwargs)


def _open_flow(profile, source):
    """고객 값으로 입력값을 채우고 다음 단계로 (버튼 on_click 에서 불러도 된다).
    다시 심사(edit)는 3/3 으로 바로, 나머지는 2/3 신청 정보부터."""
    ss = st.session_state
    if source != "edit":  # 새 고객 시작: 앞 고객 정보는 남기지 않는다
        _clear_consult()
    profile = _with_repayment(profile)
    ss.frm_profile, ss.frm_source, ss.frm_missing, ss.frm_error = profile, source, [], None
    ss.pop("frm_snap", None)
    ss.frm_shown, ss.frm_values = {}, {}
    for field, _, _, scale, decimals, *_ in ALL_FIELDS:
        value = None if source in APPLY_SOURCES and field in APPLY_KEYS else profile.get(field)
        ss.frm_values[field] = ss.frm_shown[field] = _shown(value, scale, decimals)
    ss.frm_gender = profile.get("gender")
    ss.frm_agree = {k: source == "edit" for k, *_ in AGREEMENTS}   # 다시 심사는 지난 심사 때 이미 동의했다
    ss.scr_step = "confirm" if source == "edit" else "apply"
    for key in ("idn_notice", "idn_errors", "apl_missing", "agr_missing"):
        ss.pop(key, None)


def edit(profile):
    """결과 화면 '정보 수정' · XAI '조건 바꿔 재심사': 그 고객 값이 채워진 3/3 화면.
    다른 화면에서 부를 때는 이 함수 뒤에 st.switch_page(SCREENING_PAGE) (화면을 옮기면 입력칸 값이 지워지므로
    값은 고객 심사 화면이 그려질 때 채운다)."""
    st.session_state.frm_pending = (profile, "edit")


def take_pending():
    """edit() 로 맡겨 둔 고객이 있으면 3/3 화면을 연다. 고객 심사 화면 맨 앞에서 부른다."""
    pending = st.session_state.pop("frm_pending", None)
    if pending:
        st.session_state.scr_current = None
        _open_flow(*pending)


def _name_of():
    return st.session_state.frm_profile.get("name")


# ---------------------------------------------------------------------------
# 2/3 신청 정보 (월소득 · 부양가족 수)
# ---------------------------------------------------------------------------
# 2/3 동의 항목: (키, 글, 자세히). 모두 필수 (대안정보도 모델 입력이라 빠지면 심사할 수 없다)
AGREEMENTS = [
    ("collect", "개인(신용)정보 수집 · 이용 동의", "대출 심사를 위해 이름 · 주민등록번호 · 소득 정보를 수집해요"),
    ("cb", "개인(신용)정보 조회 동의", "신용정보원 · CB사(KCB · NICE)에서 대출 · 연체 · 카드 기록을 조회해요"),
    ("alt", "마이데이터 전송 요구 동의", "통신사 · 공공기관 · 카드사에서 납부 · 결제 기록을 받아 심사에 써요"),
]


def _agreed(key):
    return st.session_state.get("frm_agree", {}).get(key, False)


def _agree_one(key):
    st.session_state.frm_agree[key] = st.session_state[f"agr_{key}"]


def _agree_all():
    ss = st.session_state
    for key, *_ in AGREEMENTS:
        ss.frm_agree[key] = ss.agr_all


def _after_apply():
    """2/3 다음 단계: 은행 기록이 있는 고객은 불러오기 연출, 직접 입력 고객은 바로 3/3."""
    return "loading" if st.session_state.frm_source in ("bank", "sample") else "confirm"


def _apply_next():
    """2/3 다음: 두 칸을 다 채웠으면 동의 화면으로 (이미 모두 동의했으면 — 다시 심사 등 — 바로 다음 단계)."""
    ss = st.session_state
    ss.apl_missing = [label for f, label, *_ in APPLY_FIELDS if _val(f) is None]
    if not ss.apl_missing:
        ss.scr_step = _after_apply() if all(_agreed(k) for k, *_ in AGREEMENTS) else "agree"


def render_apply(disabled=False):
    ss = st.session_state
    name = _name_of()
    who = f"{escape(name)}님의" if name else "고객님의"
    with st.container(key="apl-card"):
        st.button("이전", key="apl-back", type="tertiary", on_click=_go, args=("identify",))
        st.html(f'<div class="tz-step">2 / 3</div><h1 class="tz-title">{who}<br>월소득과 부양가족 수를 알려 주세요</h1>'
                '<p class="tz-sub">원천징수영수증과 가족관계증명서를 확인한 뒤 입력해 주세요.</p>')
        for (field, label, unit, scale, decimals, lo, hi, required), key in zip(APPLY_FIELDS, ("apl-income", "apl-dep")):
            with st.container(key=key):
                _number(field, label, scale, decimals, lo, hi, required)
        if ss.get("apl_missing"):
            st.html(f'<p class="idn-error">{escape(", ".join(ss.apl_missing))} 칸을 채워 주세요</p>')
        st.button("다음", key="apl-next", type="primary", on_click=_apply_next, disabled=disabled,
                  use_container_width=True)


# ---------------------------------------------------------------------------
# 동의 (2/3 → 불러오기 연출 사이): 정보를 불러오기 전에 받는 필수 동의
# ---------------------------------------------------------------------------
def _agree_go():
    """동의하고 정보 불러오기: 모두 동의했으면 다음 단계, 아니면 안내만 (버튼은 늘 켜 둔다)."""
    ss = st.session_state
    ss.agr_missing = not all(_agreed(k) for k, *_ in AGREEMENTS)
    if not ss.agr_missing:
        ss.scr_step = _after_apply()


def render_agree(disabled=False):
    """동의 화면: '전체 동의' + 필수 항목. 모두 동의해야 [동의하고 정보 불러오기] 가 켜진다.
    값은 frm_agree 에 두고 체크칸은 그릴 때 채운다 (단계를 오가도 남게)."""
    ss = st.session_state
    ss.setdefault("frm_agree", {})
    ss.agr_all = all(_agreed(k) for k, *_ in AGREEMENTS)
    name = _name_of()
    with st.container(key="agr-card"):
        st.button("이전", key="agr-back", type="tertiary", on_click=_go, args=("apply",))
        st.html(f'<div class="tz-step">동의</div><h1 class="tz-title">{escape(name) + "님의 " if name else ""}정보를 불러오려면<br>'
                '동의가 필요해요</h1><p class="tz-sub">고객님이 동의한 신용정보와 대안정보를 심사에 써요.</p>')
        with st.container(key="apl-agree"):
            st.checkbox("필수 항목에 모두 동의해요", key="agr_all", on_change=_agree_all)
            for key, label, detail in AGREEMENTS:
                ss[f"agr_{key}"] = _agreed(key)
                with st.container(key=f"apl-agree-{key}"):
                    st.checkbox(f"(필수) {label}", key=f"agr_{key}", on_change=_agree_one, args=(key,))
                    st.html(f'<p class="apl-agree-detail">{escape(detail)}</p>')
        if ss.get("agr_missing") and not ss.agr_all:
            st.html('<p class="idn-error agr-error">필수 항목에 모두 동의해 주세요</p>')
        st.button("동의하고 정보 불러오기", key="agr-go", type="primary", on_click=_agree_go, use_container_width=True,
                  disabled=disabled)


# ---------------------------------------------------------------------------
# 불러오기 연출 (2/3 → 3/3): 기관별로 신용정보 · 대안정보를 가져오는 장면
# 실제 조회는 1/3 에서 끝났다. 시연에서 '은행이 정보를 가져온다' 는 흐름이 보이게 하는 화면이다
# ---------------------------------------------------------------------------
LOGO_DIR = Path(__file__).resolve().parents[1] / "assets" / "screening-logos"
LOGO_MIME = {".svg": "image/svg+xml", ".png": "image/png", ".gif": "image/gif"}
LOADING_STEPS = (
    ("신용정보 · CB사", (
        ("KCB", "대출 · 카드 · 연체 기록", ("kcb_footer.png",), ""),
        ("NICE평가정보", "대출 · 카드 · 연체 기록", (), "NICE"),
    )),
    ("대안정보 · 마이데이터", (
        ("통신사", "SKT · KT · LG U+ 통신요금 납부 기록", ("skt.png", "kt.png", "lgu.svg"), ""),
        ("국민건강보험공단", "건강보험료 납부 기록", ("nhis.svg",), ""),
        ("카드사 · 은행", "신한 · 삼성 · KB국민 · 현대 소비 · 정기결제 내역",
         ("shinhan.gif", "samsung.png", "kb.svg", "hyundai.svg"), ""),
        ("남궁 은행", "앱 로그인 기록", ("namgung.png",), "남궁은행"),
    )),
)


@lru_cache(maxsize=None)
def _logo_img(filename):
    path = LOGO_DIR / filename
    encoded = b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:{LOGO_MIME[path.suffix]};base64,{encoded}" alt="">'


def _loading_row(state, label, detail, logos, badge):
    if not logos:
        artwork = ui.icon("trends", "#FFFFFF", 22)
        style = " ld-logo-nice"
    elif len(logos) == 1:
        artwork = _logo_img(logos[0])
        style = " ld-logo-namgung" if badge else ""
    else:
        images = "".join(f'<span>{_logo_img(filename)}</span>' for filename in logos)
        artwork = (f'<span class="ld-logo-window" style="--logo-count:{len(logos)}">'
                   f'<span class="ld-logo-track">{images}</span></span>')
        style = ""
    wordmark = f'<b>{escape(badge)}</b>' if badge else ""
    check = '<span class="ld-logo-check">✓</span>' if state == "done" else ""
    status = {"todo": "대기", "doing": "가져오는 중", "done": "받음"}[state]
    return (f'<li class="ld-org {state}">'
            f'<div class="ld-logo{style}" aria-hidden="true">{artwork}{wordmark}{check}</div>'
            f'<div class="ld-org-name">{escape(label)}<small>{escape(detail)}</small></div>'
            f'<span class="ld-org-status">{status}</span></li>')


def _loading_html(done, name):
    index = 0
    groups = []
    for title, steps in LOADING_STEPS:
        rows = []
        for step in steps:
            state = "done" if index < done else "doing" if index == done else "todo"
            rows.append(_loading_row(state, *step))
            index += 1
        groups.append(f'<h2 class="ld-org-group">{title}</h2><ul class="ld-org-list">{"".join(rows)}</ul>')
    who = f"{escape(name)}님의" if name else "고객님의"
    action = "모두 가져왔어요" if done == index else "가져오고 있어요"
    return (f'<div class="ld-card ld-org-card"><h1 class="tz-title">{who} 정보를<br>{action}</h1>'
            '<p class="tz-sub">고객님이 동의한 기관에서만 받아 와요</p>'
            f'{"".join(groups)}<progress class="ld-progress" aria-label="정보 불러오기 진행률" value="{done}" max="{index}"></progress>'
            '<p class="ld-logo-credit">로고 출처: 각 회사 · 기관 (교육용 시연)</p></div>')


def render_loading():
    """총 연출 시간 안에서 로고가 여럿인 기관에 더 오래 머문다."""
    if config.CB_LOADING_SECONDS > 0:
        slot = st.empty()
        name = _name_of()
        weights = [max(len(logos), 1) for _, steps in LOADING_STEPS for _, _, logos, _ in steps]
        weights.append(1)  # 모든 기관의 '받음' 상태도 잠깐 보여 준다
        cycle = config.CB_LOADING_SECONDS / sum(weights)
        for done, weight in enumerate(weights):
            slot.html(_loading_html(done, name))
            time.sleep(cycle * weight)
    st.session_state.scr_step = "confirm"
    st.rerun()


# 심사 실행 연출 (3/3 → 결과): 실제 /predict · /explain 호출이 이 화면 뒤에서 돈다
JUDGING_STEPS = [
    ("입력한 정보를 확인하고 있어요", "입력한 정보를 확인했어요"),
    ("모델이 부도 확률을 계산하고 있어요", "부도 확률을 계산했어요"),
    ("판단 근거(SHAP)를 정리하고 있어요", "판단 근거를 정리했어요"),
    ("심사 결과를 만들고 있어요", "심사 결과가 나왔어요"),
]


def _judging_html(done, name):
    rows = []
    for i, (doing, finished) in enumerate(JUDGING_STEPS):
        state = "done" if i < done else "doing" if i == done else "todo"
        mark = ui.icon("check", "#FFFFFF", 14) if state == "done" else ""
        rows.append(f'<li class="{state}"><i>{mark}</i><span>{escape(finished if state == "done" else doing)}</span></li>')
    who = f"{escape(name)}님의" if name else "고객님의"
    all_done = done >= len(JUDGING_STEPS)
    mark = (f'<div class="ld-done">{ui.icon("check", "#FFFFFF", 28)}</div>' if all_done   # 끝나면 체크가 든 파란 원
            else '<div class="ld-spin"></div>')
    action = "심사가<br>끝났어요" if all_done else "심사를<br>진행하고 있어요"
    return (f'<div class="ld-card">{mark}'
            f'<h1 class="tz-title">{who} {action}</h1>'
            f'<ul class="ld-steps">{"".join(rows)}</ul></div>')


# ---------------------------------------------------------------------------
# 3/3 전체 확인 → 심사 실행
# ---------------------------------------------------------------------------
def _form_profile():
    """입력값 → 심사 요청 프로필. 화면 값을 고치지 않은 칸은 반올림한 값이 아니라 불러온 원래 값을 보낸다."""
    ss = st.session_state
    original = ss.frm_profile
    profile = {k: original.get(k) for k in ("customer_id",) + DISPLAY_KEYS}
    profile["gender"] = ss.frm_gender
    for field, _, _, scale, decimals, *_ in ALL_FIELDS:
        shown = _val(field)
        if shown is not None and shown == ss.frm_shown[field] and original.get(field) is not None:
            profile[field] = original[field]
        elif shown is None:
            profile[field] = None
        else:
            value = shown / scale
            profile[field] = int(round(value)) if decimals == 0 and scale == 1 else float(value)
    # 부채비율: 월소득 · 상환액을 그대로 두었으면 원래 값, 바뀌었으면 다시 계산
    same = all(_val(f) is not None and _val(f) == ss.frm_shown[f] for f in ("MonthlyIncome", REPAY_FIELD))
    if same and original.get("DebtRatio") is not None:
        profile["DebtRatio"] = original["DebtRatio"]
    else:
        profile["DebtRatio"] = _debt_ratio(profile[REPAY_FIELD], profile["MonthlyIncome"])
    return profile


def _missing_fields():
    return [label for field, label, *_, required in ALL_FIELDS if required and _val(field) is None]


def _submit():
    """심사 실행: 빈 칸이 없으면 심사 진행 연출 화면으로 (실제 심사는 그 화면에서 부른다)."""
    ss = st.session_state
    ss.frm_missing = _missing_fields()
    ss.frm_error = None
    if not ss.frm_missing:
        ss.scr_step = "judging"


def render_judging():
    """심사 진행 연출: 첫 단계를 보여 주는 동안 실제 심사(POST /predict + /explain)를 부르고,
    남은 단계를 하나씩 체크한 뒤 결과 화면으로 (총 config.SCREENING_LOADING_SECONDS 초 이상).
    심사가 실패하면 3/3 으로 돌아가 오류를 보여 준다."""
    ss = st.session_state
    slot = st.empty()
    name = _name_of()
    show = lambda done: slot.html(_judging_html(done, name))  # noqa: E731
    show(0)
    started = time.monotonic()
    try:
        run_predict(_form_profile())
    except Exception as e:
        ss.frm_error, ss.scr_current, ss.scr_step = e, None, "confirm"
        st.rerun()
    left = max(config.SCREENING_LOADING_SECONDS - (time.monotonic() - started), 0)
    for done in range(1, len(JUDGING_STEPS) + 1):
        show(done)
        time.sleep(left / len(JUDGING_STEPS))
    ss.scr_step = "identify"
    st.rerun()


def _go(step):
    st.session_state.scr_step = step
    st.session_state.pop("frm_snap", None)


# ---------------------------------------------------------------------------
# 3/3 출처별 타일 카드 (#178 선택안): 서류 · 신용정보 · 대안정보 카드 3장, 값은 같은 크기 타일.
# 고칠 수 있는 값(서류의 월소득 · 부양가족, 직접 입력 고객은 전부)만 파란 테두리 입력칸이다.
# ---------------------------------------------------------------------------
def _tile_html(label, value, unit):
    unit_html = f"<em>{escape(unit)}</em>" if unit else ""
    return f'<div class="frm-tile"><span>{escape(label)}</span><div><b>{escape(value)}</b>{unit_html}</div></div>'


def _edit_tile(field, label, unit, scale, decimals, lo, hi, required):
    """고칠 수 있는 타일: 라벨 + 숫자 입력칸(파란 테두리) + 단위."""
    with st.container(key=f"frm-tile-{field}"):
        st.html(f'<span class="frm-tile-label">{escape(label)}</span>')
        with st.container(key=f"frm-tile-in-{field}", horizontal=True, vertical_alignment="center", gap=None):
            _number(field, label, scale, decimals, lo, hi, required, label_visibility="collapsed")
            st.html(f'<em class="frm-tile-unit">{escape(unit)}</em>')


def _tiles(items):
    """타일을 두 칸씩 그린다. items: ("html", 글) 또는 ("edit", 칸 정의)."""
    for i in range(0, len(items), 2):
        cols = st.columns(2, gap="small")
        for col, (kind, item) in zip(cols, items[i:i + 2]):
            with col:
                if kind == "edit":
                    _edit_tile(*item)
                else:
                    st.html(item)


def _fmt(value, decimals=0):
    if value is None:
        return "-"
    return f"{value:,.{decimals}f}" if decimals else f"{round(value):,}"


def _card(key, no, title, chip, caption, items, note=""):
    with st.container(key=f"frm-card-{key}"):
        st.html(f'<div class="frm-card-head"><i>{no}</i><b>{escape(title)}</b><span class="frm-chip">{escape(chip)}</span></div>'
                f'<p class="frm-card-sub">{escape(caption)}</p>')
        _tiles(items)
        if note:
            st.html(f'<p class="frm-card-note">{escape(note)}</p>')


def _confirm_cards(source):
    """서류 · 신용정보 · 대안정보 타일 목록."""
    ss = st.session_state
    manual = source == "manual"   # 은행 기록이 없는 고객은 신용 · 대안정보도 직접 입력
    spec = {f: (f, label, unit, scale, decimals, lo, hi, required)
            for f, label, unit, scale, decimals, lo, hi, required in ALL_FIELDS}
    age = _val("age")
    docs = [("html", _tile_html("나이", _fmt(age), "세")), ("html", _tile_html("성별", ss.frm_gender or "-", "")),
            ("edit", spec["MonthlyIncome"][:1] + ("월소득",) + spec["MonthlyIncome"][2:]),
            ("edit", spec["NumberOfDependents"][:1] + ("부양가족",) + spec["NumberOfDependents"][2:])]
    income, repay = _val("MonthlyIncome"), _val(REPAY_FIELD)
    ratio = "-" if not income or repay is None else _fmt(repay / income * 100)
    credit = []
    for f, label, unit, scale, decimals, *_ in CREDIT_FIELDS:
        credit.append(("edit", spec[f]) if manual else ("html", _tile_html(label, _fmt(_val(f), decimals), unit)))
        if f == REPAY_FIELD:   # 월 대출 상환액 바로 옆에 자동 계산한 부채비율
            credit.append(("html", _tile_html("부채비율 (자동 계산)", ratio, "%")))
    alt = []
    for f, label, unit, *_ in ALT_FIELDS:
        if manual:
            alt.append(("edit", spec[f]))
        else:
            name, number, record_unit = F.source_record(f, _val(f))
            alt.append(("html", _tile_html(name, number, record_unit)))
    return docs, credit, alt


def _snapshot():
    """3/3 에 들어온 순간의 값 (여기서 고친 항목 수를 세는 기준). 3/3 을 떠나면 지운다."""
    ss = st.session_state
    if "frm_snap" not in ss:
        ss.frm_snap = {f: _val(f) for f, *_ in ALL_FIELDS}
    return ss.frm_snap


def render_confirm(disabled=False):
    """3/3 전체 확인 (#178 선택안: 출처별 타일 카드). 서류 / 신용정보 / 대안정보 카드 3장과 아래 줄(확인 표시 · 심사 실행).
    1440 × 900 한 화면에 들어오게 맞췄다. 대안정보는 화면에서만 원천 기록 모양으로 보여 준다 (features.source_record)."""
    ss = st.session_state
    profile, source = ss.frm_profile, ss.frm_source
    name = profile.get("name")
    title = (f"{escape(name)}님의 정보를 모두 불러왔어요" if name and source in ("bank", "sample")
             else f"{escape(name)}님의 나머지 정보를 입력해 주세요" if name and source == "manual"
             else "고객 정보를 확인해 주세요")
    sub = ("은행에 기록이 없는 고객이에요. 파란 테두리 칸을 모두 채워 주세요." if source == "manual"
           else "파란 테두리 값은 서류로 받은 값이라 눌러서 고칠 수 있어요.")
    snap = _snapshot()
    edited = sum(_val(f) != snap.get(f) for f, *_ in ALL_FIELDS)
    docs, credit, alt = _confirm_cards(source)

    with st.container(key="frm-wrap"):
        with st.container(key="frm-top", horizontal=True, vertical_alignment="bottom"):
            with st.container(key="frm-top-left"):
                with st.container(key="frm-top-step", horizontal=True, vertical_alignment="center", gap="small"):
                    st.button("이전", key="frm-back", type="tertiary", on_click=_go,
                              args=("identify" if source == "edit" else "apply",))
                    st.html('<div class="tz-step">3 / 3</div>')
                st.html(f'<h1 class="tz-title">{title}</h1><p class="tz-sub">{sub}</p>')
            # 고객 번호는 보여 주지 않는다 (내부 구분용)
            rows = [("이름", name or "-"), ("주민등록번호", profile.get("rrn_masked") or "-")]
            cells = "".join(f'<div><span>{label}</span><b>{escape(str(value))}</b></div>' for label, value in rows)
            st.html(f'<div class="frm-id-card"><div class="frm-id-cells">{cells}</div>'
                    f'<i class="frm-badge {source}">{SOURCES[source]}</i></div>')

        c1, c2, c3 = st.columns([1, 1.25, 1], gap="medium")
        with c1:
            _card("docs", 1, "서류", "담당자 입력", "신분증 · 원천징수영수증 · 가족관계증명서", docs)
        with c2:
            _card("credit", 2, "신용정보", "직접 입력" if source == "manual" else "KCB · NICE 조회",
                  "대출 · 카드 · 연체 기록 (연체는 최근 2년)", credit)
        with c3:
            _card("alt", 3, "대안정보", "직접 입력" if source == "manual" else "마이데이터",
                  "통신사 · 공공기관 · 카드사 · 남궁 은행", alt,
                  "" if source == "manual" else "대안정보는 원천 기록 형태로 환산해 보여 줘요")

        f1, f2 = st.columns([2.25, 1], gap="medium")
        with f1, st.container(key="frm-foot-row", horizontal=True, vertical_alignment="center"):
            checks = "".join(f'<span><i>✓</i>{label} {n}</span>' for label, n in
                             (("서류", len(docs)), ("신용정보", len(credit)), ("대안정보", len(alt))))
            if ss.frm_missing:
                note = f'<span class="frm-foot-miss">{escape(", ".join(ss.frm_missing))} 칸을 채워 주세요</span>'
            else:
                note = f'<span class="frm-foot-note">{f"{edited}개 항목을 고쳤어요" if edited else "고친 항목이 없어요"}</span>'
            st.html(f'<div class="frm-checks">{checks}</div>{note}')
        with f2:
            st.button("심사 실행", key="frm-submit", type="primary", on_click=_submit, disabled=disabled,
                      use_container_width=True)
        if ss.frm_error is not None:
            st.html(api_error_html("POST /predict", ss.frm_error, "inline"))
