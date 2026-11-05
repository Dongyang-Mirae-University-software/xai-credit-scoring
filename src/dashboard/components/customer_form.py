"""고객 정보 입력 창 (디자인 02, st.dialog) 과 심사 실행. 고객 심사·XAI 분석 두 화면에서 쓴다.

흐름 (디자인 시안의 프로토타입 동작과 같다)
- 불러오기 · 시연용 샘플 · 정보 수정 · 조건 바꿔 재심사 → 제목 옆에 고객 ID 태그가 붙은 창 (값이 채워져 있다)
- 신규 고객 심사 → 태그 없이 빈 칸으로 시작하는 창
- 심사 실행 → POST /predict + /explain → 고객 심사의 결과 화면. 결과는 POST /screenings 로 서버 이력에 바로 남는다
  (따로 누르는 저장 버튼은 없다. 오늘 심사한 고객이 새로고침해도 남게 하려는 것)
- 취소 · 닫기 → 창만 닫고 지금 화면에 남는다

고객 ID 불러오기와 시연용 샘플은 API 를 부르지 않고 로컬 파일 samples.json(테스트셋 고객 입력값)에서 읽는다.
모델이 아니라 입력값이라 '대시보드는 모델을 직접 읽지 않는다' 규칙에 맞다. API 를 부르는 것은 심사 실행뿐이다.
"""
import json
from datetime import datetime
from html import escape
from pathlib import Path

import streamlit as st

import api_client
from components.api_error import api_error_html

CSS = Path(__file__).with_suffix(".css").read_text(encoding="utf-8")
SCREENING_PAGE = "views/screening.py"
HISTORY_LIMIT = 5

# 기본값은 백엔드 Swagger 예시 고객(src/api/schemas.py CUSTOMER_EXAMPLE)과 같다.
# 요청 키·단위는 실제 API(CustomerInput)와 같다: 월 소득은 원 (서버가 ÷INCOME_KRW_PER_UNIT 해서 모델 단위로),
# 부채비율·통신비·공과금 납부율은 % (서버가 ÷100), 신용한도 사용률은 0~1
DEFAULT_CUSTOMER = {
    "age": 45, "MonthlyIncome": 4_200_000, "DebtRatio": 32.0, "RevolvingUtilizationOfUnsecuredLines": 0.48,
    "NumberOfDependents": 2, "NumberOfTime30-59DaysPastDueNotWorse": 1, "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfTimes90DaysLate": 0, "NumberOfOpenCreditLinesAndLoans": 6, "NumberRealEstateLoansOrLines": 1,
    "telecom_payment_rate": 90.0, "utility_payment_rate": 93.0, "spending_consistency": 75.0,
    "regular_payment_count": 5, "app_login_frequency": 22, "credit_card_months": 36,
}
# 신용카드 이용 기간(개월). 원래 과제의 씬파일러 기준 '카드 거래 이력 12개월 미만' 을 API 가 판정할 수 있게 보낸다.
# 비우면 null. (API 에 이 칸이 아직 없으면 서버가 무시한다)
CARD_FIELD = "credit_card_months"

# 고객 파일: [{customer_id, gender, sample_label?, + POST /predict 입력}]. sample_label 이 있으면 첫 화면 샘플 칩 이름.
# 고객 유형(씬파일러 여부)은 넣지 않는다 → /predict 응답의 is_thin_filer 로만 보여 준다.
# 연체 칸이 null 이면 '기록 없음' (0회와 다르다)
SAMPLES_PATH = Path(__file__).resolve().parent.parent / "samples.json"
CUSTOMERS = {c["customer_id"]: c for c in json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))}
SAMPLE_CHIPS = [(cid, c["sample_label"]) for cid, c in CUSTOMERS.items() if c.get("sample_label")]

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
# 대안 데이터 묶음 머리의 표시: 모드 → (글자, 색)
ALT_TAGS = {"linked": ("자동 연동", "ok"), "empty": ("직접 입력", "ok"),
            "editing": ("직접 입력 중", "mid"), "edited": ("직접 수정됨", "info")}
STEP_MAX = 50
INT_FIELDS = ({f for f, _ in DELINQUENCY_FIELDS + EXTRA_FIELDS}
              | {"age", "regular_payment_count", "app_login_frequency", CARD_FIELD})


# ---------------------------------------------------------------------------
# 상태 · 고객 찾기 · 심사 실행
# ---------------------------------------------------------------------------
def init_state():
    ss = st.session_state
    ss.setdefault("scr_seq", 1)
    ss.setdefault("scr_current", None)   # 지금 보고 있는 심사 결과 {profile, result}. 있으면 고객 심사 = 결과 화면
    ss.setdefault("scr_history", [])     # 이번 세션에서 심사한 것 [{profile, result, at}]. 서버 이력(GET /screenings)과 합쳐 보인다
    ss.setdefault("customers", {})       # 심사한 고객 {id: {payload, result, profile}} (XAI 분석 고객 목록)
    ss.setdefault("xai_explain", {})     # /explain 응답 캐시


def new_id():
    return f"C{st.session_state.scr_seq:06d}"


def new_profile():
    """신규 고객: 입력칸은 비우고(None) 슬라이더·증감 칸은 0 으로 시작한다 (시안의 신규 고객 창)."""
    profile = {"customer_id": new_id(), "gender": None, **{k: None for k in DEFAULT_CUSTOMER}}
    profile["RevolvingUtilizationOfUnsecuredLines"] = 0.0
    for field, _ in DELINQUENCY_FIELDS + EXTRA_FIELDS:
        profile[field] = 0
    return profile


def find_customer(customer_id):
    """고객 파일(samples.json) → 이번 세션에서 새로 심사한 고객 순서로 찾는다. 없으면 None. API 는 부르지 않는다."""
    found = CUSTOMERS.get(customer_id)
    if found:
        return {"customer_id": customer_id, "gender": found.get("gender"), **{k: found.get(k) for k in DEFAULT_CUSTOMER}}
    entry = st.session_state.customers.get(customer_id)
    if entry and "profile" in entry:
        return entry["profile"]
    return None


RESULT_QUERY = "customer"  # 결과 화면 주소: 고객 심사?customer=C001234


def clear_result():
    """첫 화면으로: 보고 있던 결과와 주소의 ?customer= 를 지운다 (버튼 on_click 에서 불러도 된다)."""
    st.session_state.scr_current = None
    st.query_params.pop(RESULT_QUERY, None)


def show_result(customer_id):
    """이미 심사한 고객의 결과를 고객 심사 화면에 띄운다. 심사한 적 없으면 False."""
    entry = st.session_state.customers.get(customer_id)
    if not entry:
        return False
    st.session_state.scr_current = {"profile": entry["profile"], "result": entry["result"]}
    return True


def _payload(profile):
    """심사 요청 본문 (GMSC 10 + 대안 5 + 신용카드 이용 기간). 성별은 공정성 점검용이라 보내지 않는다.
    '기록 없음' 연체 칸과 비운 카드 이용 기간은 null 로 보낸다."""
    return {k: v for k, v in profile.items() if k != "gender"}


def _save(payload, result):
    """POST /screenings: 서버 이력에 남긴다. 저장이 실패해도 심사 결과는 보여 준다 (이번 세션 목록에는 남는다)."""
    try:
        api_client.save_screening({"input": payload, "result": result})
    except Exception:
        pass


def run_predict(profile, record=True):
    """심사 실행: POST /predict 와 POST /explain 을 함께 부른다 (승인이든 거절이든).
    결과를 현재 심사·XAI 고객 목록에 넣고, record 면 오늘 심사한 고객에도 쌓는다. /predict 가 실패하면 예외를 올린다.
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
    if customer_id == new_id():
        ss.scr_seq += 1


# ---------------------------------------------------------------------------
# 입력 창
# ---------------------------------------------------------------------------
def get_explain(customer_id):
    """심사 실행 때 받아 둔 /explain 응답. 그때 실패했으면 그 예외를 다시 올린다."""
    explain = st.session_state.xai_explain.get(customer_id)
    if isinstance(explain, Exception):
        raise explain
    if explain is None:
        raise LookupError(f"{customer_id} 고객의 설명(/explain)이 없어요. 다시 심사해 주세요.")
    return explain


def _num(value, as_int):
    if value is None:
        return None
    return int(value) if as_int else float(value)


def _fill_form(profile, loaded):
    ss = st.session_state
    ss.dlg_profile, ss.dlg_loaded, ss.dlg_error, ss.dlg_missing = profile, loaded, None, []
    ss.dlg_alt_mode = "linked" if loaded else "empty"  # linked 자동 연동 / empty 직접 입력 / editing 입력 중 / edited 직접 수정됨
    income = None if profile["MonthlyIncome"] is None else profile["MonthlyIncome"] / 10_000
    ss.dlg_age = _num(profile["age"], True)
    ss.dlg_gender = profile.get("gender")
    ss.dlg_income = None if income is None else int(round(income))
    ss.dlg_repay = (None if income is None or profile["DebtRatio"] is None
                    else int(round(profile["DebtRatio"] / 100 * income)))
    ss.dlg_shown = (ss.dlg_income, ss.dlg_repay)  # 창에 보여 준 만원 단위 값 (고치지 않으면 원래 값을 그대로 보낸다)
    ss.dlg_util = int(round(min(profile["RevolvingUtilizationOfUnsecuredLines"] or 0, 1.0) * 100))
    for field, *_ in DELINQUENCY_FIELDS + EXTRA_FIELDS + ALT_FIELDS:
        ss[f"dlg_{field}"] = _num(profile[field], field in INT_FIELDS)
    for field, _ in DELINQUENCY_FIELDS:  # null = 기록 없음 (값 칸은 0 으로 두고 체크만 켠다)
        ss[f"dlg_none_{field}"] = profile[field] is None
        if profile[field] is None:
            ss[f"dlg_{field}"] = 0
    ss.dlg_card = _num(profile.get(CARD_FIELD), True)


def _form_profile():
    ss = st.session_state
    income = ss.dlg_income or 0
    original = ss.dlg_profile
    if (ss.dlg_income, ss.dlg_repay) == ss.dlg_shown and original.get("MonthlyIncome") is not None:
        # 월 소득·상환액을 고치지 않았으면 만원 단위로 반올림한 값이 아니라 불러온 원래 값을 보낸다
        monthly_income, debt_ratio = float(original["MonthlyIncome"]), float(original["DebtRatio"])
    else:
        monthly_income = float(income * 10_000)
        # 부채비율 = 월 상환액 ÷ 월 소득 (API 는 % 로 받는다)
        debt_ratio = round((ss.dlg_repay or 0) / income * 100, 1) if income else 0.0
    util = original.get("RevolvingUtilizationOfUnsecuredLines")
    profile = {
        "customer_id": original["customer_id"],
        "gender": ss.dlg_gender,
        "age": ss.dlg_age,
        "MonthlyIncome": monthly_income,
        "DebtRatio": debt_ratio,
        # 슬라이더는 1% 단위라, 움직이지 않았으면 원래 값 (1 초과 값도 그대로)
        "RevolvingUtilizationOfUnsecuredLines": (
            float(util) if util is not None and ss.dlg_util == int(round(min(util, 1.0) * 100)) else ss.dlg_util / 100),
    }
    for field, *_ in DELINQUENCY_FIELDS + EXTRA_FIELDS + ALT_FIELDS:
        profile[field] = ss[f"dlg_{field}"]
    for field, _ in DELINQUENCY_FIELDS:
        if ss[f"dlg_none_{field}"]:
            profile[field] = None
    profile[CARD_FIELD] = ss.dlg_card
    return profile


def open_form(profile, loaded):
    """다음 렌더에서 입력 창을 연다 (버튼 on_click 에서 불러도 된다)."""
    _fill_form(profile, loaded)
    st.session_state.dlg_open = True


def _missing_fields():
    """심사 실행 전에 채워야 하는 칸 (비어 있는 칸의 라벨). 월 대출 상환액은 비우면 0 으로 본다."""
    ss = st.session_state
    required = [("dlg_age", "나이"), ("dlg_income", "월 소득")] + [(f"dlg_{f}", label) for f, label, *_ in ALT_FIELDS]
    return [label for key, label in required if ss.get(key) is None]


def _reset_form():
    """초기화: 입력칸·슬라이더·증감 칸·기록 없음을 모두 비운다 (시안 동작). 성별과 대안 데이터는 그대로 둔다."""
    ss = st.session_state
    ss.dlg_age = ss.dlg_income = ss.dlg_repay = ss.dlg_card = None
    ss.dlg_util = 0
    for field, _ in DELINQUENCY_FIELDS + EXTRA_FIELDS:
        ss[f"dlg_{field}"] = 0
    for field, _ in DELINQUENCY_FIELDS:
        ss[f"dlg_none_{field}"] = False
    ss.dlg_missing, ss.dlg_error = [], None
    ss.dlg_toast = "입력값을 모두 지웠어요"


def _open_alt_inputs():
    """대안 데이터 입력칸을 연다. 입력칸(dlg_in_<필드>)은 닫히면 Streamlit 이 값을 지우므로
    값은 dlg_<필드> 에 두고, 열 때 채우고 바뀔 때마다 옮겨 담는다 (_keep_alt)."""
    ss = st.session_state
    for field, *_ in ALT_FIELDS:
        ss[f"dlg_in_{field}"] = ss[f"dlg_{field}"]
    ss.dlg_alt_mode = "editing"


def _keep_alt(field):
    st.session_state[f"dlg_{field}"] = st.session_state[f"dlg_in_{field}"]


def _toggle_alt():
    """대안 데이터 '직접 수정' ↔ '수정 완료'."""
    ss = st.session_state
    if ss.dlg_alt_mode == "editing":
        ss.dlg_alt_mode = "edited"
        ss.dlg_toast = "대안 데이터를 직접 입력한 값으로 바꿨어요"
    else:
        _open_alt_inputs()


def _bump(field, delta):
    key = f"dlg_{field}"
    st.session_state[key] = min(max(int(st.session_state[key]) + delta, 0), STEP_MAX)


def _stepper(field, label, disabled=False):
    """'라벨  − 값 +' 한 줄 (디자인의 증감 버튼). 값은 위젯이 아닌 세션 값(dlg_<필드>). disabled 면 값 대신 '–'."""
    ss = st.session_state
    with st.container(key=f"dlg-step-{field}", horizontal=True, vertical_alignment="center", gap=None):
        st.html(f'<span class="dlg-step-label">{escape(label)}</span>')
        st.space("stretch")
        st.button("−", key=f"dlg-dec-{field}", on_click=_bump, args=(field, -1),
                  disabled=disabled or ss[f"dlg_{field}"] <= 0)
        st.html(f'<b class="dlg-step-value">{"–" if disabled else ss[f"dlg_{field}"]}</b>')
        st.button("\\+", key=f"dlg-inc-{field}", on_click=_bump, args=(field, 1), disabled=disabled)


def _delinquency():
    """연체 3칸: 증감 + '기록 없음' 체크 (체크하면 null 로 보낸다)."""
    ss = st.session_state
    with st.container(key="dlg-steps-due", horizontal=True, gap="small"):
        for field, label in DELINQUENCY_FIELDS:
            none = ss[f"dlg_none_{field}"]
            with st.container(key=f"dlg-due-{field}" + ("-none" if none else ""), gap=None):
                _stepper(field, label, disabled=none)
                st.checkbox("기록 없음", key=f"dlg_none_{field}")


def _extra():
    """추가 정보 2×2: 신용카드 이용 기간(입력칸) + 신용 개설 건수 · 부동산 대출 · 부양가족 수 (증감)."""
    cells = [None] + EXTRA_FIELDS
    for r, row in enumerate((cells[:2], cells[2:])):
        with st.container(key=f"dlg-steps-extra-{r}", horizontal=True, gap="small"):
            for cell in row:
                if cell is None:
                    with st.container(key="dlg-step-card", horizontal=True, vertical_alignment="center", gap=None):
                        st.html('<span class="dlg-step-label">신용카드 이용 기간</span>')
                        st.space("stretch")
                        st.number_input("신용카드 이용 기간", min_value=0, max_value=600, step=1, key="dlg_card",
                                        placeholder="0", label_visibility="collapsed")
                        st.html('<span class="dlg-step-unit">개월</span>')
                else:
                    _stepper(*cell)


def _group_head(title, right=""):
    st.html(f'<div class="dlg-group-head"><span>{title}</span>{right}</div>')


@st.dialog("고객 정보 입력", width="medium")
def _dialog(on_screening):
    ss = st.session_state
    customer_id = ss.dlg_profile["customer_id"]
    if ss.dlg_loaded:
        # 제목 옆 고객 ID 태그 (창 제목은 Streamlit 이 그리므로 제목 글 뒤에 CSS 로 붙인다)
        tag = "".join(ch for ch in customer_id if ch.isalnum())
        st.html(f'<style>div[role="dialog"]:has(.dlg-sub) > div:first-child p::after{{content:"{tag}"}}</style>'
                '<p class="dlg-sub">불러온 값은 그대로 두고 바뀐 항목만 수정하세요.</p>')
    else:
        st.html('<p class="dlg-sub">고객 정보를 입력한 뒤 심사를 실행하세요.</p>')
    # 창 안 버튼 콜백에서 st.toast 를 부르면 창 밖에 쓰여 페이지 스타일이 지워지므로, 콜백은 글만 남기고 여기서 띄운다
    toast = ss.pop("dlg_toast", None)
    if toast:
        st.toast(toast, icon=":material/check_circle:")

    # 기본 정보
    with st.container(key="dlg-basic"):
        _group_head("기본 정보")
        c1, c2, c3 = st.columns(3, gap="small")
        with c1, st.container(key="dlg-unit-age"):
            st.number_input("나이", min_value=18, max_value=100, step=1, key="dlg_age", placeholder="0")
        with c2, st.container(key="dlg-gender-box"):
            st.radio("성별", ["여성", "남성"], index=None, horizontal=True, key="dlg_gender")
            st.html('<span class="dlg-hint">공정성 점검용, 점수 미반영</span>')
        with c3, st.container(key="dlg-unit-income"):
            st.number_input("월 소득", min_value=0, step=10, key="dlg_income", placeholder="0")

    # 금융 이력 (씬파일러 판정은 심사 실행 후 API 가 한다 → 여기서는 미리 보여 주지 않는다)
    with st.container(key="dlg-finance"):
        _group_head("금융 이력", "<em>기록이 없는 항목은 ‘기록 없음’을 선택하세요</em>")
        c1, c2 = st.columns(2, gap="medium")
        with c1, st.container(key="dlg-unit-repay"):
            st.number_input("월 대출 상환액", min_value=0, step=1, key="dlg_repay", placeholder="0")
            ratio = f"{ss.dlg_repay / ss.dlg_income * 100:.0f}%" if ss.dlg_income and ss.dlg_repay is not None else "-"
            st.html(f'<span class="dlg-hint">부채비율 <b>{ratio}</b> 자동 계산</span>')
        with c2, st.container(key="dlg-util"):
            st.html(f'<div class="dlg-util-head"><span>신용한도 사용률</span><b>{ss.dlg_util}%</b></div>')
            st.slider("신용한도 사용률", 0, 100, key="dlg_util", label_visibility="collapsed")
        st.html('<span class="dlg-label">최근 2년 연체 횟수</span>')
        _delinquency()

    # 추가 정보
    with st.container(key="dlg-extra"):
        _group_head("추가 정보", '<em>불러온 값이 기본으로 채워져요</em>')
        _extra()

    # 대안 데이터: 불러온 고객은 '자동 연동' 값, 신규 고객은 '입력 필요'. '직접 수정' 을 누르면 입력칸
    with st.container(key="dlg-alt"):
        mode = ss.dlg_alt_mode
        tag_text, tag_tone = ALT_TAGS[mode]
        with st.container(key="dlg-alt-head", horizontal=True, vertical_alignment="center"):
            st.html(f'<div class="dlg-group-head"><span>대안 데이터<i class="dlg-tag {tag_tone}">{tag_text}</i></span></div>')
            st.button("수정 완료" if mode == "editing" else "직접 수정", key="dlg-alt-edit", type="tertiary",
                      on_click=_toggle_alt)
        if mode == "editing":
            with st.container(key="dlg-alt-edit-box"):
                for row in (ALT_FIELDS[:3], ALT_FIELDS[3:]):
                    cols = st.columns(3, gap="small")
                    for col, (field, label, lo, hi, _) in zip(cols, row):
                        with col:
                            st.number_input(label, min_value=lo, max_value=hi, step=1, key=f"dlg_in_{field}",
                                            placeholder="0", on_change=_keep_alt, args=(field,))
        else:
            cells = "".join(
                f'<div><span>{label}</span>'
                + ('<b class="dlg-alt-empty">입력 필요</b>' if ss[f"dlg_{field}"] is None
                   else f'<b>{fmt.format(ss[f"dlg_{field}"])}</b>') + '</div>'
                for field, label, _, _, fmt in ALT_FIELDS)
            st.html(f'<div class="dlg-alt-list">{cells}</div>')

    if ss.dlg_missing:
        st.html(f'<p class="dlg-missing">{escape(", ".join(ss.dlg_missing))} 칸을 채운 뒤 심사를 실행해 주세요.</p>')
    if ss.dlg_error is not None:
        st.html(api_error_html("POST /predict", ss.dlg_error, "inline"))

    with st.container(key="dlg-foot", horizontal=True, vertical_alignment="center"):
        st.button("초기화", key="dlg-reset", on_click=_reset_form)
        st.space("stretch")
        if st.button("취소", key="dlg-cancel", type="tertiary"):
            st.rerun()
        if st.button("심사 실행", key="dlg-submit", type="primary"):
            ss.dlg_missing = _missing_fields()
            if ss.dlg_missing:
                if ss.dlg_alt_mode != "editing" and any(label in ss.dlg_missing for _, label, *_ in ALT_FIELDS):
                    _open_alt_inputs()  # 비어 있는 대안 데이터 칸을 바로 채울 수 있게 연다
                st.rerun(scope="fragment")
            try:
                with st.spinner("심사 중"):
                    run_predict(_form_profile())
            except Exception as e:
                ss.dlg_error = e
                st.rerun(scope="fragment")
            if on_screening:
                st.rerun()
            st.switch_page(SCREENING_PAGE)


def render(on_screening):
    """open_form 으로 요청된 창이 있으면 연다. 페이지 맨 끝에서 부른다. CSS 는 페이지가 CSS 를 함께 넣는다."""
    if st.session_state.pop("dlg_open", False):
        _dialog(on_screening)
