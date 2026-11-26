"""고객 심사 흐름 (은행 창구 기준, 한 화면씩 넘긴다). 고객 심사 · XAI 분석 두 화면에서 쓴다.

1/2 본인 확인: 이름 + 주민등록번호 → POST /customers/lookup → 은행이 가진 신용정보 · 대안정보
2/2 정보 확인: 불러온 값이 채워진 입력 목록. 틀린 정보만 고치고 심사 실행 → POST /predict + /explain → 결과 화면
    (결과는 POST /screenings 로 서버 이력에 바로 남는다)
- 조회 결과가 없거나 조회 API 가 아직 없으면 '직접 입력'(빈 목록, 나이 · 성별은 주민등록번호로 채운다) 또는 시연용 샘플
- 결과 화면 '정보 수정' · XAI '조건 바꿔 재심사' → 그 고객 값이 채워진 2/2 화면

주민등록번호는 조회 요청에만 쓴다. 심사 요청 · 이력에는 넣지 않고 화면에는 가린 값(900101-1******)만 남긴다.
시연용 샘플은 API 가 아니라 로컬 파일 samples.json(테스트셋 고객 입력값)에서 읽는다. 모델이 아니라 입력값이라
'대시보드는 모델을 직접 읽지 않는다' 규칙에 맞다.
"""
import json
from datetime import datetime
from html import escape
from pathlib import Path

import streamlit as st

import api_client
from components import identity
from components.api_error import api_error_html

CSS = Path(__file__).with_suffix(".css").read_text(encoding="utf-8")
# 심사 업무 화면(고객 심사 · XAI 분석) 공통 톤. 두 화면이 맨 앞에 넣는다
WORKSPACE_CSS = Path(__file__).with_name("workspace.css").read_text(encoding="utf-8")
SCREENING_PAGE = "views/screening.py"
HISTORY_LIMIT = 5
NEW_ID_LOOKUP = 100  # 신규 고객 ID 를 고를 때 피할 서버 이력 수 (GET /screenings 최대치)

# 신용카드 이용 기간(개월). 원래 과제의 씬파일러 기준 '카드 거래 이력 12개월 미만' 을 API 가 판정할 수 있게 보낸다.
# 비우면 null. (API 에 이 칸이 아직 없으면 서버가 무시한다)
CARD_FIELD = "credit_card_months"

# 입력 목록 칸: (요청 키, 라벨, 단위, 화면 배율, 소수 자릿수, 최소, 최대, 필수)
# 화면 값 = 요청 값 × 배율. 요청 단위는 실제 API(CustomerInput)와 같다: 월소득 원, 부채비율 · 납부율 %, 한도 사용률 0~1
BASIC_FIELDS = [
    ("age", "나이", "세", 1, 0, identity.MIN_AGE, 120, True),
]
CREDIT_FIELDS = [
    ("MonthlyIncome", "월소득", "만원", 1e-4, 0, 0, None, True),
    ("NumberOfDependents", "부양가족", "명", 1, 0, 0, 20, False),
    ("DebtRatio", "부채비율", "%", 1, 1, 0, None, False),
    ("RevolvingUtilizationOfUnsecuredLines", "신용한도 사용률", "%", 100, 1, 0, None, False),
    ("NumberOfOpenCreditLinesAndLoans", "보유 대출 · 신용 한도", "개", 1, 0, 0, None, False),
    ("NumberRealEstateLoansOrLines", "부동산 담보대출", "개", 1, 0, 0, None, False),
    ("NumberOfTime30-59DaysPastDueNotWorse", "30~59일 연체", "회", 1, 0, 0, None, False),
    ("NumberOfTime60-89DaysPastDueNotWorse", "60~89일 연체", "회", 1, 0, 0, None, False),
    ("NumberOfTimes90DaysLate", "90일 이상 연체", "회", 1, 0, 0, None, False),
    (CARD_FIELD, "신용카드 이용 기간", "개월", 1, 0, 0, 600, False),
]
ALT_FIELDS = [
    ("telecom_payment_rate", "통신비 정상납부율", "%", 1, 1, 0, 100, True),
    ("utility_payment_rate", "공과금 납부율", "%", 1, 1, 0, 100, True),
    ("spending_consistency", "소비 일관성 점수", "점", 1, 1, 0, 100, True),
    ("regular_payment_count", "정기결제 건수", "건", 1, 0, 0, 20, True),
    ("app_login_frequency", "앱 로그인 빈도", "회", 1, 0, 0, 30, True),
]
ALL_FIELDS = BASIC_FIELDS + CREDIT_FIELDS + ALT_FIELDS
FIELD_HINTS = {
    "DebtRatio": "월 대출 상환액 ÷ 월소득",
    "NumberOfTime30-59DaysPastDueNotWorse": "최근 2년",
    "NumberOfTime60-89DaysPastDueNotWorse": "최근 2년",
    CARD_FIELD: "선택",
    "app_login_frequency": "한 달 기준",
    "regular_payment_count": "한 달 기준",
}
# 결과 화면 요약에서 연체 합계를 낼 때 쓴다
DELINQUENCY_FIELDS = [(f, label) for f, label, *_ in CREDIT_FIELDS if "DaysPastDue" in f or "DaysLate" in f]
REQUEST_KEYS = [f for f, *_ in ALL_FIELDS]
DISPLAY_KEYS = ("gender", "name", "rrn_masked")  # 화면 표시용. 심사 요청에는 넣지 않는다

# 정보 출처 → 2/2 화면 머리 배지
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
    ss.setdefault("scr_step", "identify")  # 결과가 없을 때: identify 1/2 본인 확인 · confirm 2/2 정보 확인
    ss.setdefault("scr_history", [])     # 이번 세션에서 심사한 것 [{profile, result, at}]. 서버 이력과 합쳐 보인다
    ss.setdefault("customers", {})       # 심사한 고객 {id: {payload, result, profile}} (XAI 분석 고객 목록)
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
    profile = {"customer_id": new_id(), "gender": None, **{f: None for f in REQUEST_KEYS}}
    for field, _, _, _, _, _, _, required in CREDIT_FIELDS:
        if not required and field != CARD_FIELD:
            profile[field] = 0
    return profile


def find_customer(customer_id):
    """시연용 샘플 파일 → 이번 세션에서 심사한 고객 순서로 찾는다. 없으면 None. API 는 부르지 않는다."""
    found = CUSTOMERS.get(customer_id)
    if found:
        return {"customer_id": customer_id, "gender": found.get("gender"), **{k: found.get(k) for k in REQUEST_KEYS}}
    entry = st.session_state.customers.get(customer_id)
    if entry and "profile" in entry:
        return entry["profile"]
    return None


RESULT_QUERY = "customer"  # 결과 화면 주소: 고객 심사?customer=C001234


def clear_result():
    """첫 화면(1/2 본인 확인)으로: 보고 있던 결과와 주소의 ?customer= 를 지운다 (버튼 on_click 에서 불러도 된다)."""
    st.session_state.scr_current = None
    st.session_state.scr_step = "identify"
    st.query_params.pop(RESULT_QUERY, None)


def show_result(customer_id):
    """이미 심사한 고객의 결과를 고객 심사 화면에 띄운다. 심사한 적 없으면 False."""
    entry = st.session_state.customers.get(customer_id)
    if not entry:
        return False
    st.session_state.scr_current = {"profile": entry["profile"], "result": entry["result"]}
    return True


def _payload(profile):
    """심사 요청 본문 (GMSC 10 + 대안 5 + 신용카드 이용 기간). 성별 · 이름 · 가린 주민등록번호는 보내지 않는다."""
    return {k: v for k, v in profile.items() if k not in DISPLAY_KEYS}


def _save(payload, result):
    """POST /screenings: 서버 이력에 남긴다. 저장이 실패해도 심사 결과는 보여 준다 (이번 세션 목록에는 남는다)."""
    try:
        api_client.save_screening({"input": payload, "result": result})
    except Exception:
        pass


def run_predict(profile, record=True):
    """심사 실행: POST /predict 와 POST /explain 을 함께 부른다 (승인이든 거절이든).
    결과를 현재 심사 · XAI 고객 목록에 넣고, record 면 심사 이력에도 쌓는다. /predict 가 실패하면 예외를 올린다.
    /explain 이 실패하면 그 예외를 캐시에 넣어 두고 결과 화면이 안내한다 (화면을 그릴 때 다시 부르지 않는다)."""
    ss = st.session_state
    payload = _payload(profile)
    result = {**api_client.predict(payload), "customer_id": profile["customer_id"]}
    customer_id = profile["customer_id"]
    ss.customers.pop(customer_id, None)  # 다시 심사하면 XAI 고객 목록 맨 앞으로
    ss.customers[customer_id] = {"payload": payload, "result": result, "profile": profile}
    try:
        ss.xai_explain[customer_id] = api_client.explain(payload)
    except Exception as e:
        ss.xai_explain[customer_id] = e
    ss.scr_current = {"profile": profile, "result": result}
    if record:
        _save(payload, result)
        history = [h for h in ss.scr_history if h["result"]["customer_id"] != customer_id]
        ss.scr_history = ([{"profile": profile, "result": result, "at": datetime.now()}] + history)[:HISTORY_LIMIT]


def get_explain(customer_id):
    """심사 실행 때 받아 둔 /explain 응답. 그때 실패했으면 그 예외를 다시 올린다."""
    explain = st.session_state.xai_explain.get(customer_id)
    if isinstance(explain, Exception):
        raise explain
    if explain is None:
        raise LookupError(f"{customer_id} 고객의 설명(/explain)이 없어요. 다시 심사해 주세요.")
    return explain


# ---------------------------------------------------------------------------
# 1/2 본인 확인
# ---------------------------------------------------------------------------
def _identity_input():
    """입력칸의 이름 · 주민등록번호 (숫자만). 칸마다 안내 문구를 함께 돌려준다."""
    ss = st.session_state
    name = identity.clean_name(ss.get("idn_name"))
    front, back = identity.digits(ss.get("idn_front")), identity.digits(ss.get("idn_back"))
    return name, front, back, {"name": identity.name_error(name), "rrn": identity.rrn_error(front, back)}


def _lookup():
    """정보 조회: 형식 확인 → POST /customers/lookup. 찾으면 2/2, 없으면 · API 가 없으면 안내 + 직접 입력 버튼."""
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
                   **{k: found.get(k) for k in REQUEST_KEYS}}
        if profile["age"] is None:
            profile["age"] = identity.age(front, back)
        _open_confirm({**profile, "name": found.get("name") or name, "rrn_masked": identity.masked(front, back)}, "bank")


def _manual():
    """직접 입력: 은행 기록이 없는 고객. 나이 · 성별은 주민등록번호로 채운 빈 목록을 연다."""
    ss = st.session_state
    name, front, back, errors = _identity_input()
    ss.idn_errors = errors
    if any(errors.values()):
        return
    profile = {**_blank_profile(), "age": identity.age(front, back), "gender": identity.gender(back),
               "name": name, "rrn_masked": identity.masked(front, back)}
    _open_confirm(profile, "manual")


def _sample(customer_id):
    _open_confirm(find_customer(customer_id), "sample")


def _field_error(key):
    message = (st.session_state.get("idn_errors") or {}).get(key)
    if message:
        st.html(f'<p class="idn-error">{escape(message)}</p>')


def render_identify(disabled=False):
    """1/2 본인 확인 카드. disabled 면 (API 연결 끊김) 조회 · 진행 버튼을 끈다."""
    ss = st.session_state
    with st.container(key="idn-card"):
        st.html('<div class="tz-step">1 / 2</div>'
                '<h1 class="tz-title">고객님의 이름과<br>주민등록번호를 입력해 주세요</h1>'
                '<p class="tz-sub">은행이 가진 신용정보와 대안정보를 불러와요.</p>')
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
            st.form_submit_button("정보 조회", key="idn-submit", type="primary", on_click=_lookup,
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
# 2/2 정보 확인
# ---------------------------------------------------------------------------
def _shown(value, scale, decimals):
    """요청 값 → 화면 값 (배율 · 반올림). 정수 칸은 int, 소수 칸은 float."""
    if value is None:
        return None
    shown = round(float(value) * scale, decimals)
    return int(shown) if decimals == 0 else shown


def _open_confirm(profile, source):
    """2/2 정보 확인으로 넘어간다. 입력 목록을 그 고객 값으로 채운다 (버튼 on_click 에서 불러도 된다)."""
    ss = st.session_state
    ss.frm_profile, ss.frm_source, ss.frm_missing, ss.frm_error = profile, source, [], None
    ss.frm_shown = {}
    for field, _, _, scale, decimals, *_ in ALL_FIELDS:
        ss[f"frm_{field}"] = ss.frm_shown[field] = _shown(profile.get(field), scale, decimals)
    ss.frm_gender = profile.get("gender")
    ss.scr_step = "confirm"
    ss.pop("idn_notice", None)
    ss.pop("idn_errors", None)


def edit(profile):
    """결과 화면 '정보 수정' · XAI '조건 바꿔 재심사': 그 고객 값이 채워진 2/2 화면.
    다른 화면에서 부를 때는 이 함수 뒤에 st.switch_page(SCREENING_PAGE) (화면을 옮기면 입력칸 값이 지워지므로
    값은 고객 심사 화면이 그려질 때 채운다)."""
    st.session_state.frm_pending = (profile, "edit")


def take_pending():
    """edit() 로 맡겨 둔 고객이 있으면 2/2 화면을 연다. 고객 심사 화면 맨 앞에서 부른다."""
    pending = st.session_state.pop("frm_pending", None)
    if pending:
        st.session_state.scr_current = None
        _open_confirm(*pending)


def _form_profile():
    """입력 목록 → 심사 요청 프로필. 화면 값을 고치지 않은 칸은 반올림한 값이 아니라 불러온 원래 값을 보낸다."""
    ss = st.session_state
    original = ss.frm_profile
    profile = {k: original.get(k) for k in ("customer_id",) + DISPLAY_KEYS}
    profile["gender"] = ss.frm_gender
    for field, _, _, scale, decimals, *_ in ALL_FIELDS:
        shown = ss[f"frm_{field}"]
        if shown == ss.frm_shown[field] and original.get(field) is not None:
            profile[field] = original[field]
        elif shown is None:
            profile[field] = None
        else:
            value = shown / scale
            profile[field] = int(round(value)) if decimals == 0 and scale == 1 else float(value)
    return profile


def _missing_fields():
    ss = st.session_state
    return [label for field, label, *_, required in ALL_FIELDS if required and ss.get(f"frm_{field}") is None]


def _edited(field):
    ss = st.session_state
    return ss.frm_source != "manual" and ss.get(f"frm_{field}") != ss.frm_shown.get(field)


def _row(field, label, unit, scale, decimals, lo, hi, required):
    """한 줄: 라벨(+ 도움말 · 수정됨) ··· 값 단위. 값은 그 자리에서 바로 고친다."""
    edited = '<i class="frm-edited">수정됨</i>' if _edited(field) else ""
    hint = FIELD_HINTS.get(field)
    with st.container(key=f"frm-row-{field}", horizontal=True, vertical_alignment="center", gap="small"):
        st.html(f'<div class="frm-label"><span>{escape(label)}{edited}</span>'
                + (f'<small>{escape(hint)}</small>' if hint else "") + "</div>")
        num = int if decimals == 0 else float  # Streamlit 은 최소 · 최대 · 간격의 숫자 형이 같아야 한다
        st.number_input(label, key=f"frm_{field}", min_value=num(lo), max_value=None if hi is None else num(hi),
                        step=num(1) if decimals == 0 else 0.1, format="%d" if decimals == 0 else "%.1f",
                        placeholder="입력" if required else "-", label_visibility="collapsed")
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


def _back_to_identify():
    st.session_state.scr_step = "identify"


def render_confirm(disabled=False):
    """2/2 정보 확인 화면."""
    ss = st.session_state
    profile, source = ss.frm_profile, ss.frm_source
    name = profile.get("name")
    title = (f"{escape(name)}님의 정보를 불러왔어요" if name and source == "bank"
             else f"{escape(name)}님의 정보를 입력해 주세요" if name and source == "manual"
             else "고객 정보를 확인해 주세요")
    sub = {"bank": "은행이 가진 정보예요. 틀린 정보가 있으면 고친 뒤 심사를 실행하세요.",
           "manual": "은행에 기록이 없는 고객이에요. 나이와 성별은 주민등록번호로 채웠어요.",
           "sample": "시연용 샘플 고객의 정보예요. 값을 바꿔 가며 심사해 볼 수 있어요.",
           "edit": "지난 심사에 쓴 정보예요. 바꿀 항목만 고친 뒤 다시 심사하세요."}[source]
    edited = sum(_edited(f) for f, *_ in ALL_FIELDS)

    with st.container(key="frm-wrap"):
        st.button("이전", key="frm-back", type="tertiary", on_click=_back_to_identify)
        st.html(f'<div class="tz-step">2 / 2</div><h1 class="tz-title">{title}</h1><p class="tz-sub">{sub}</p>')

        rows = [("이름", name), ("주민등록번호", profile.get("rrn_masked")), ("고객번호", profile["customer_id"])]
        rows = [(label, value) for label, value in rows if value]  # 시연용 샘플은 이름 · 주민등록번호가 없다
        cells = "".join(f'<div><span>{label}</span><b>{escape(str(value))}</b></div>' for label, value in rows)
        badge = f'<i class="frm-badge {source}">{SOURCES[source]}</i>'
        st.html(f'<div class="frm-id-card">{cells}{badge}</div>')

        with st.container(key="frm-sec-basic"):
            st.html('<div class="frm-sec-head"><h2>기본 정보</h2><p>주민등록번호로 확인한 정보예요</p></div>')
            _row(*BASIC_FIELDS[0])
            with st.container(key="frm-row-gender", horizontal=True, vertical_alignment="center", gap="small"):
                st.html('<div class="frm-label"><span>성별</span><small>공정성 점검용, 점수 미반영</small></div>')
                st.radio("성별", ["여성", "남성"], key="frm_gender", horizontal=True, index=None,
                         label_visibility="collapsed", disabled=bool(profile.get("rrn_masked")))
        _section("credit", "신용정보", "신용정보원(CB)에 있는 금융 기록이에요", CREDIT_FIELDS)
        _section("alt", "대안정보", "통신 · 공과금 · 결제 기록에서 모은 정보예요", ALT_FIELDS)

        if ss.frm_missing:
            st.html(f'<p class="frm-missing">{escape(", ".join(ss.frm_missing))} 칸을 채운 뒤 심사를 실행해 주세요.</p>')
        if ss.frm_error is not None:
            st.html(api_error_html("POST /predict", ss.frm_error, "inline"))

        with st.container(key="frm-foot"):
            note = f"{edited}개 항목을 고쳤어요" if edited else "고친 항목이 없어요"
            st.html(f'<p class="frm-foot-note">{note if source != "manual" else "필수 항목을 모두 채워 주세요"}</p>')
            st.button("심사 실행", key="frm-submit", type="primary", on_click=_submit, disabled=disabled,
                      use_container_width=True)
