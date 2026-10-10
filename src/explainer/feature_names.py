"""
입력 변수 이름 정의 (영어 코드 이름 ↔ 한글 이름).

거절 사유 문장, SHAP 그림, 대시보드 표시에 같은 이름을 쓰기 위해 한곳에 모아 둔다.
데이터 묶음(금융데이터만 / 대안데이터만 / 통합)은 src/models/train.py 의 분류와 같다.
"""

# 금융데이터 10개 (Kaggle Give Me Some Credit 원본 칸)
FINANCIAL_COLS = [
    'RevolvingUtilizationOfUnsecuredLines',
    'age',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents',
]

# 대안데이터 5개 (원본 과제가 정한 시뮬레이터 변수)
ALTERNATIVE_COLS = [
    'telecom_payment_rate',
    'utility_payment_rate',
    'spending_consistency',
    'regular_payment_count',
    'app_login_frequency',
]

# 모델 입력에서 빼야 하는 칸 (정답, 보호 속성, 씬파일러 표시)
NON_FEATURE_COLS = [
    'SeriousDlqin2yrs',
    'gender', 'Gender',
    'age_group', 'AgeGroup',
    'is_thin_filer',
    'Unnamed: 0',
]

FEATURE_NAMES_KR = {
    'RevolvingUtilizationOfUnsecuredLines': '한도 사용 비율',
    'age': '나이',
    'NumberOfTime30-59DaysPastDueNotWorse': '최근 2년 30~59일 연체 횟수',
    'DebtRatio': '부채 비율',
    'MonthlyIncome': '월소득',
    'NumberOfOpenCreditLinesAndLoans': '보유 대출·신용 한도 개수',
    'NumberOfTimes90DaysLate': '90일 이상 연체 횟수',
    'NumberRealEstateLoansOrLines': '부동산 담보대출 개수',
    'NumberOfTime60-89DaysPastDueNotWorse': '최근 2년 60~89일 연체 횟수',
    'NumberOfDependents': '부양가족 수',
    'telecom_payment_rate': '통신비 정상납부율',
    'utility_payment_rate': '공과금 납부율',
    'spending_consistency': '소비 일관성 점수',
    'regular_payment_count': '정기결제 건수',
    'app_login_frequency': '앱 로그인 빈도',
}

# 문장에 값을 쓰는 방식: 'percent' 는 0~1 값을 % 로, 그 밖에는 단위를 붙인다
VALUE_FORMAT = {
    'RevolvingUtilizationOfUnsecuredLines': 'percent',
    'telecom_payment_rate': 'percent',
    'utility_payment_rate': 'percent',
    'DebtRatio': 'ratio',
    'age': '세',
    'MonthlyIncome': '',
    'spending_consistency': '점',
    'regular_payment_count': '건',
    'app_login_frequency': '회',
    'NumberOfTime30-59DaysPastDueNotWorse': '회',
    'NumberOfTime60-89DaysPastDueNotWorse': '회',
    'NumberOfTimes90DaysLate': '회',
    'NumberOfOpenCreditLinesAndLoans': '개',
    'NumberRealEstateLoansOrLines': '개',
    'NumberOfDependents': '명',
}

FEATURE_GROUPS = {
    'financial_only': FINANCIAL_COLS,
    'alternative_only': ALTERNATIVE_COLS,
    'integrated': FINANCIAL_COLS + ALTERNATIVE_COLS,
}


def korean_name(feature):
    """코드 이름을 한글 이름으로 바꾼다. 모르는 이름은 그대로 돌려준다."""
    return FEATURE_NAMES_KR.get(feature, feature)


def format_value(feature, value):
    """거절 사유 문장에 넣을 값 표기. 예: 0.78 → '78%', 3 → '3회'."""
    kind = VALUE_FORMAT.get(feature, '')
    if value is None:
        return '정보 없음'
    if kind == 'percent':
        return f"{value * 100:.0f}%"
    if kind == 'ratio':
        return f"{value:.2f}"
    if float(value).is_integer():
        return f"{int(value):,}{kind}"
    return f"{value:,.1f}{kind}"
