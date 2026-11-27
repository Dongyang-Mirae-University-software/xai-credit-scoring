"""고객 심사 흐름 (은행 창구 기준, 한 화면씩 넘긴다). 고객 심사 · XAI 분석 두 화면에서 쓴다.

1/3 본인 확인: 이름 + 주민등록번호 → POST /customers/lookup (은행이 가진 신용정보 · 대안정보를 이때 받아 둔다)
2/3 신청 정보: 월소득 · 부양가족 수를 고객에게 받아 입력 (은행 기록으로 미리 채우지 않는다)
    → 불러오기 연출: '신용정보원(CB) · 대안정보를 가져오는' 장면을 잠깐 보여 준다 (config.CB_LOADING_SECONDS)
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
from html import escape
from pathlib import Path

import streamlit as st

import api_client
import config
from components import identity, ui
from components.api_error import api_error_html

CSS = Path(__file__).with_suffix(".css").read_text(encoding="utf-8")
# 심사 업무 화면(고객 심사 · XAI 분석) 공통 톤. 두 화면이 맨 앞에 넣는다
WORKSPACE_CSS = Path(__file__).with_name("workspace.css").read_text(encoding="utf-8")
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
FIELD_HINTS = {
    "NumberOfTime30-59DaysPastDueNotWorse": "최근 2년",
    "NumberOfTime60-89DaysPastDueNotWorse": "최근 2년",
    "app_login_frequency": "한 달 기준",
    "regular_payment_count": "한 달 기준",
}
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
    ss.setdefault("scr_step", "identify")  # 결과가 없을 때: identify 1/3 · apply 2/3 · loading 연출 · confirm 3/3
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
        return {"customer_id": customer_id, "name": found.get("name"), "gender": found.get("gender"),
                **{k: found.get(k) for k in REQUEST_KEYS}}
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
              "apl_missing", "idn_notice", "idn_errors")


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
                st.html('<span class="idn-dash">-</span>')
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
    ss.frm_shown, ss.frm_values = {}, {}
    for field, _, _, scale, decimals, *_ in ALL_FIELDS:
        value = None if source in APPLY_SOURCES and field in APPLY_KEYS else profile.get(field)
        ss.frm_values[field] = ss.frm_shown[field] = _shown(value, scale, decimals)
    ss.frm_gender = profile.get("gender")
    ss.scr_step = "confirm" if source == "edit" else "apply"
    for key in ("idn_notice", "idn_errors", "apl_missing"):
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
def _apply_next():
    """다음: 두 칸을 다 채웠으면 은행 기록이 있는 고객은 불러오기 연출, 직접 입력 고객은 3/3 으로."""
    ss = st.session_state
    ss.apl_missing = [label for f, label, *_ in APPLY_FIELDS if _val(f) is None]
    if not ss.apl_missing:
        ss.scr_step = "loading" if ss.frm_source in ("bank", "sample") else "confirm"


def render_apply(disabled=False):
    ss = st.session_state
    name = _name_of()
    who = f"{escape(name)}님의" if name else "고객님의"
    with st.container(key="apl-card"):
        st.button("이전", key="apl-back", type="tertiary", on_click=_go, args=("identify",))
        st.html(f'<div class="tz-step">2 / 3</div><h1 class="tz-title">{who}<br>월소득과 부양가족 수를 알려 주세요</h1>'
                '<p class="tz-sub">대출 신청서에 고객님이 직접 적는 정보예요.</p>')
        for (field, label, unit, scale, decimals, lo, hi, required), key in zip(APPLY_FIELDS, ("apl-income", "apl-dep")):
            with st.container(key=key):
                _number(field, label, scale, decimals, lo, hi, required)
        if ss.get("apl_missing"):
            st.html(f'<p class="idn-error">{escape(", ".join(ss.apl_missing))} 칸을 채워 주세요</p>')
        st.button("다음", key="apl-next", type="primary", on_click=_apply_next, disabled=disabled,
                  use_container_width=True)


# ---------------------------------------------------------------------------
# 불러오기 연출 (2/3 → 3/3): 신용정보원(CB) · 대안정보를 가져오는 장면
# 실제 조회는 1/3 에서 끝났다. 시연에서 '은행이 정보를 가져온다' 는 흐름이 보이게 하는 화면이다
# ---------------------------------------------------------------------------
LOADING_STEPS = [
    ("본인 확인", "본인 확인을 마쳤어요"),
    ("신용정보원(CB)에서 금융 기록을 가져오고 있어요", f"금융 기록 {len(CREDIT_FIELDS)}건을 가져왔어요"),
    ("통신 · 공과금 · 결제 기록을 가져오고 있어요", f"대안정보 {len(ALT_FIELDS)}건을 가져왔어요"),
    ("심사에 쓸 정보를 정리하고 있어요", "심사 준비를 마쳤어요"),
]


def _loading_html(done, name):
    rows = []
    for i, (doing, finished) in enumerate(LOADING_STEPS):
        state = "done" if i < done else "doing" if i == done else "todo"
        mark = ui.icon("check", "#FFFFFF", 14) if state == "done" else ""
        rows.append(f'<li class="{state}"><i>{mark}</i><span>{escape(finished if state == "done" else doing)}</span></li>')
    who = f"{escape(name)}님의 정보를" if name else "고객님의 정보를"
    return (f'<div class="ld-card"><div class="ld-spin{" ok" if done >= len(LOADING_STEPS) else ""}"></div>'
            f'<h1 class="tz-title">{who}<br>가져오고 있어요</h1>'
            f'<ul class="ld-steps">{"".join(rows)}</ul></div>')


def render_loading():
    """단계를 하나씩 체크하며 보여 준 뒤 3/3 으로 넘어간다 (총 config.CB_LOADING_SECONDS 초)."""
    slot = st.empty()
    pause = config.CB_LOADING_SECONDS / len(LOADING_STEPS)
    name = _name_of()
    for done in range(1, len(LOADING_STEPS) + 1):
        slot.html(_loading_html(done, name))
        time.sleep(pause)
    st.session_state.scr_step = "confirm"
    st.rerun()


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


def _edited(field):
    """불러온 값을 고친 칸인가. 신청 정보는 고객이 적는 칸이라 '수정됨' 으로 세지 않는다."""
    ss = st.session_state
    if ss.frm_source == "manual" or (ss.frm_source in APPLY_SOURCES and field in APPLY_KEYS):
        return False
    return _val(field) != ss.frm_shown.get(field)


def _row(field, label, unit, scale, decimals, lo, hi, required):
    """한 줄: 라벨(+ 도움말 · 수정됨) ··· 값 단위. 값은 그 자리에서 바로 고친다."""
    edited = '<i class="frm-edited">수정됨</i>' if _edited(field) else ""
    hint = FIELD_HINTS.get(field)
    if field == REPAY_FIELD:  # 부채비율 자동 계산 (화면 값은 만원 단위)
        income, repay = _val("MonthlyIncome"), _val(REPAY_FIELD)
        hint = (f"부채비율 {(repay or 0) / income * 100:.0f}% 자동 계산 (상환액 ÷ 월소득)"
                if income else "월소득을 입력하면 부채비율을 계산해요")
    with st.container(key=f"frm-row-{field}", horizontal=True, vertical_alignment="center", gap="small"):
        st.html(f'<div class="frm-label"><span>{escape(label)}{edited}</span>'
                + (f'<small>{escape(hint)}</small>' if hint else "") + "</div>")
        _number(field, label, scale, decimals, lo, hi, required, label_visibility="collapsed")
        st.html(f'<span class="frm-unit">{escape(unit)}</span>')


def _section(key, title, caption, fields):
    with st.container(key=f"frm-sec-{key}"):
        st.html(f'<div class="frm-sec-head"><h2>{escape(title)}</h2><p>{escape(caption)}</p></div>')
        for spec in fields:
            _row(*spec)


def _submit():
    ss = st.session_state
    ss.frm_missing = _missing_fields()
    ss.frm_error = None
    if ss.frm_missing:
        return
    try:
        run_predict(_form_profile())
    except Exception as e:
        ss.frm_error = e
        ss.scr_current = None
        return
    ss.scr_step = "identify"


def _go(step):
    st.session_state.scr_step = step


def _fixed_row(label, hint, value, unit):
    """고칠 수 없는 값 한 줄: 입력 상자 없이 글자만 (고칠 수 있는 칸은 회색 상자라 구분된다)."""
    small = f"<small>{escape(hint)}</small>" if hint else ""
    st.html(f'<div class="frm-fixed-row"><div class="frm-label"><span>{escape(label)}</span>{small}</div>'
            f'<b class="frm-fixed">{escape(value)}</b><span class="frm-unit">{escape(unit)}</span></div>')


def _basic_section(profile):
    """기본 정보: 주민등록번호로 정해진 값이라 고칠 수 없다 ('변경 불가' 표시)."""
    ss = st.session_state
    with st.container(key="frm-sec-basic"):
        caption = "주민등록번호로 확인한 정보예요" if profile.get("rrn_masked") else "고객 기본 정보예요"
        st.html(f'<div class="frm-sec-head"><h2>기본 정보<i class="frm-lock">변경 불가</i></h2><p>{caption}</p></div>')
        age = _val("age")
        _fixed_row("나이", "만 나이", "-" if age is None else str(age), "세")
        _fixed_row("성별", "공정성 점검용, 점수 미반영", ss.frm_gender or "-", "")


def _apply_section():
    """신청 정보: 2/3 에서 받은 값. 여기서는 글자로만 보여 준다 (바꾸려면 '이전' 으로 2/3)."""
    with st.container(key="frm-sec-apply"):
        st.html('<div class="frm-sec-head"><h2>신청 정보</h2><p>2단계에서 고객님께 받은 정보예요</p></div>')
        for field, label, unit, *_ in APPLY_FIELDS:
            value = _val(field)
            _fixed_row(label, "", "-" if value is None else f"{value:,}", unit)


def render_confirm(disabled=False):
    """3/3 전체 확인 화면. 한 화면(1440 × 900)에 스크롤 없이 들어오도록 좌우로 나눈다:
    왼쪽 기본 정보 · 신청 정보 · 대안정보 | 오른쪽 신용정보 · 심사 실행. 좁은 화면에서는 위아래로 쌓인다."""
    ss = st.session_state
    profile, source = ss.frm_profile, ss.frm_source
    name = profile.get("name")
    title = (f"{escape(name)}님의 정보를 모두 불러왔어요" if name and source in ("bank", "sample")
             else f"{escape(name)}님의 나머지 정보를 입력해 주세요" if name and source == "manual"
             else "고객 정보를 확인해 주세요")
    sub = {"bank": "틀린 정보가 있으면 고친 뒤 심사를 실행하세요.",
           "manual": "은행에 기록이 없는 고객이에요. 나이와 성별은 주민등록번호로 채웠어요.",
           "sample": "시연용 샘플 고객의 정보예요. 값을 바꿔 가며 심사해 볼 수 있어요.",
           "edit": "지난 심사에 쓴 정보예요. 바꿀 항목만 고친 뒤 다시 심사하세요."}[source]
    edited = sum(_edited(f) for f, *_ in ALL_FIELDS)

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

        left, right = st.columns(2, gap="medium")
        with left:
            _basic_section(profile)
            _apply_section()
            _section("alt", "대안정보", "통신 · 공과금 · 결제 기록에서 모은 정보예요", ALT_FIELDS)
        with right:
            _section("credit", "신용정보", "신용정보원(CB)에 있는 금융 기록이에요", CREDIT_FIELDS)
            with st.container(key="frm-foot"):
                if ss.frm_missing:
                    st.html(f'<p class="frm-missing">{escape(", ".join(ss.frm_missing))} 칸을 채운 뒤 '
                            '심사를 실행해 주세요.</p>')
                if ss.frm_error is not None:
                    st.html(api_error_html("POST /predict", ss.frm_error, "inline"))
                note = ("필수 항목을 모두 채워 주세요" if source == "manual"
                        else f"{edited}개 항목을 고쳤어요" if edited else "고친 항목이 없어요")
                st.html(f'<p class="frm-foot-note">{note}</p>')
                st.button("심사 실행", key="frm-submit", type="primary", on_click=_submit, disabled=disabled,
                          use_container_width=True)
