"""
대안 데이터 시뮬레이터.
telecom_payment_rate, utility_payment_rate, spending_consistency,
regular_payment_count, app_login_frequency 5개 변수를 생성한다.

thin_filer_ratio: 씬파일러 비율 조정 파라미터
bias_ratio: 성별/연령대별 승인율 편향 조정 파라미터
"""


def generate_alternative_data(df, target_col="SeriousDlqin2yrs",
                               thin_filer_ratio: float = 0.3,
                               bias_ratio: float = 0.1):
    raise NotImplementedError


def label_thin_filer(df):
    """금융 연체 관련 컬럼 2개 이상 결측 또는 신용카드 거래 이력 12개월 미만 -> 씬파일러."""
    raise NotImplementedError
