"""입력 변수 15개(GMSC 10 + 대안 5)의 화면 이름·단위·묶음. 심사·XAI 화면이 함께 쓴다.

API 단위 (src/api/schemas.py):
- 요청: DebtRatio·telecom_payment_rate·utility_payment_rate 는 % (서버가 ÷100), RevolvingUtilization 은 0~1
- /explain 응답의 current_value·reference_value: 비율 변수는 0~1
"""

# 화면에 보여줄 이름 (은행원 용어). 영어 코드 이름은 마우스 오버로 보여 준다.
FEATURE_NAMES_KR = {
    "RevolvingUtilizationOfUnsecuredLines": "신용한도 사용률",
    "age": "나이",
    "NumberOfTime30-59DaysPastDueNotWorse": "30-59일 연체 횟수",
    "DebtRatio": "부채비율",
    "MonthlyIncome": "월 소득",
    "NumberOfOpenCreditLinesAndLoans": "신용 개설 건수",
    "NumberOfTimes90DaysLate": "90일 이상 연체 횟수",
    "NumberRealEstateLoansOrLines": "부동산 대출",
    "NumberOfTime60-89DaysPastDueNotWorse": "60-89일 연체 횟수",
    "NumberOfDependents": "부양가족 수",
    "telecom_payment_rate": "통신비 정상납부율",
    "utility_payment_rate": "공과금 납부율",
    "spending_consistency": "소비 일관성 점수",
    "regular_payment_count": "정기결제 건수",
    "app_login_frequency": "앱 로그인 빈도",
}

ALTERNATIVE_FEATURES = [
    "telecom_payment_rate", "utility_payment_rate", "spending_consistency",
    "regular_payment_count", "app_login_frequency",
]
DELINQUENCY_FEATURES = [
    "NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTimes90DaysLate",
]

# 값 표시 형식: 비율(0~1 → %), 금액(원 → 만원), 횟수·건수 단위
_RATE = {"RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "telecom_payment_rate", "utility_payment_rate"}
_UNITS = {
    "age": "{}세",
    "NumberOfTime30-59DaysPastDueNotWorse": "{}회",
    "NumberOfTime60-89DaysPastDueNotWorse": "{}회",
    "NumberOfTimes90DaysLate": "{}회",
    "NumberOfOpenCreditLinesAndLoans": "{}건",
    "NumberRealEstateLoansOrLines": "{}건",
    "NumberOfDependents": "{}명",
    "spending_consistency": "{}점",
    "regular_payment_count": "{}건",
    "app_login_frequency": "월 {}회",
}


def label(feature):
    return FEATURE_NAMES_KR.get(feature, feature)


def is_alternative(feature):
    return feature in ALTERNATIVE_FEATURES


def _count(v):
    v = float(v)
    return f"{v:.0f}" if abs(v - round(v)) < 0.05 else f"{v:.1f}"


def fmt_value(feature, value):
    """/explain 의 current_value·reference_value 를 화면 표기로 (비율 변수는 0~1 로 온다)."""
    if value is None:
        return "-"
    if feature in _RECORD_FEATURES:   # 대안정보: 원천 기록 모양 (납부율은 0~1 로 온다)
        return record_text(feature, float(value) * 100 if feature in _RATE else value)
    if feature in _RATE:
        return f"{float(value) * 100:.0f}%"
    if feature == "MonthlyIncome":
        return f"{float(value) / 10_000:,.0f}만원"
    return _UNITS.get(feature, "{}").format(_count(value))


# 영향력 비중 막대의 묶음: 기본 정보(인적 사항) / 금융 이력 / 대안 데이터
BASIC_FEATURES = ["age", "MonthlyIncome", "NumberOfDependents"]
GROUP_LABELS = {"financial": "금융 이력", "alternative": "대안 데이터", "basic": "기본 정보"}


def group_of(feature):
    if feature in ALTERNATIVE_FEATURES:
        return "alternative"
    if feature in BASIC_FEATURES:
        return "basic"
    return "financial"


# 방향 문구에 '많을수록 / 적을수록' 을 쓰는 건수형 변수
_COUNT_FEATURES = set(DELINQUENCY_FEATURES) | {
    "NumberOfOpenCreditLinesAndLoans", "NumberRealEstateLoansOrLines", "NumberOfDependents",
    "regular_payment_count", "app_login_frequency",
}


def direction_text(feature, higher_is_riskier):
    if higher_is_riskier is None:
        return ""
    if feature in _COUNT_FEATURES:
        return "많을수록 위험↑" if higher_is_riskier else "적을수록 위험↑"
    return "높을수록 위험↑" if higher_is_riskier else "낮을수록 위험↑"


def fmt_input(feature, value):
    """/predict 요청 단위의 값(부채비율·납부율은 %, 한도 사용률은 0~1, 소득은 원)을 화면 표기로."""
    if value is None:
        return "-"
    if feature in _RECORD_FEATURES:   # 대안정보: 원천 기록 모양 (요청 단위 납부율은 %)
        return record_text(feature, value)
    if feature in {"DebtRatio", "telecom_payment_rate", "utility_payment_rate"}:
        return f"{float(value):.0f}%"
    return fmt_value(feature, value)


def josa(word, with_batchim, without):
    """한글 받침에 따라 조사를 고른다: josa('사용률', '이', '가') → '이'."""
    last = word.rstrip()[-1:] if word else ""
    if "가" <= last <= "힣":
        return with_batchim if (ord(last) - 0xAC00) % 28 else without
    if last.isdigit():
        return with_batchim if last in "013678" else without
    return with_batchim if last.upper() in "LMN" else without  # 엘·엠·엔


# ---------------------------------------------------------------------------
# 대안정보 원천 기록 표시 (#178): 모델 입력값은 그대로 두고, 화면에서만 '받아 온 기록' 모양으로 환산해 보여 준다.
# 납부율 % → 최근 24개월 중 납부한 달, 앱 로그인 → 최근 30일 중 로그인한 날. (라벨, 숫자, 단위)
# 고객 심사 3/3 과 XAI 분석이 같은 함수를 써야 화면마다 값이 다르게 보이지 않는다.
# ---------------------------------------------------------------------------
RECORD_MONTHS, RECORD_DAYS = 24, 30
_RECORDS = {
    "telecom_payment_rate": ("통신요금", lambda v: round(v / 100 * RECORD_MONTHS), f"/ {RECORD_MONTHS}개월"),
    "utility_payment_rate": ("건강보험", lambda v: round(v / 100 * RECORD_MONTHS), f"/ {RECORD_MONTHS}개월"),
    "spending_consistency": ("소비 일관성", lambda v: round(v), "점"),
    "regular_payment_count": ("정기결제", lambda v: round(v), "건"),
    "app_login_frequency": ("앱 로그인", lambda v: round(v), f"/ {RECORD_DAYS}일"),
}


def source_record(feature, value):
    """대안 변수 값(납부율은 %, 0~100) → (라벨, 숫자 글자, 단위). 대안 변수가 아니면 None."""
    if feature not in _RECORDS:
        return None
    name, convert, unit = _RECORDS[feature]
    return name, "-" if value is None else f"{convert(float(value)):,}", unit


_RECORD_FEATURES = set(_RECORDS)


def name(feature):
    """심사 업무 화면(고객 심사 · XAI 분석)의 이름: 대안정보는 원천 기록 이름(통신요금 · 건강보험 · 앱 로그인 …),
    나머지는 label(). 모델 검증 화면은 변수 이름 그대로 label() 을 쓴다."""
    return _RECORDS[feature][0] if feature in _RECORDS else label(feature)


def record_text(feature, value):
    """대안 변수 값(납부율은 %, 0~100) → '19 / 24개월', '45점' 처럼 한 덩어리 글자."""
    rec = source_record(feature, value)
    if rec is None:
        return "-" if value is None else str(value)
    _, number, unit = rec
    return f"{number} {unit}" if unit.startswith("/") else f"{number}{unit}"


def reason_sentence(feature, current, reference, api_text):
    """거절 사유 문장. 대안정보는 API 문장(통신비 정상납부율이 78%로 …) 대신 화면과 같은 원천 기록 모양으로 다시 쓴다.
    current · reference 는 /explain 단위 (납부율 0~1)."""
    if feature not in _RECORDS or current is None:
        return api_text
    word = name(feature)
    now = fmt_value(feature, current)
    if reference is None:
        return f"{word}{josa(word, '이', '가')} {now}이에요"
    lower = float(current) < float(reference)
    return (f"{word}{josa(word, '이', '가')} {now}{euro(now)}, 동일 연령대 평균({fmt_value(feature, reference)}) 대비 "
            f"{_compare_word(feature, lower)}")


def euro(word):
    """'으로' / '로': 마지막 글자에 받침이 없거나 ㄹ 받침이면 '로' (예: 24개월로, 2건으로)."""
    last = word[-1] if word else ""
    if not ("가" <= last <= "힣"):
        return "로"
    batchim = (ord(last) - ord("가")) % 28
    return "로" if batchim in (0, 8) else "으로"


def _compare_word(feature, lower):
    """점수는 낮아요 / 높아요, 횟수 · 달 수는 적어요 / 많아요."""
    if feature == "spending_consistency":
        return "낮아요" if lower else "높아요"
    return "적어요" if lower else "많아요"
