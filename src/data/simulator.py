"""
대안 데이터 시뮬레이터.
telecom_payment_rate, utility_payment_rate, spending_consistency,
regular_payment_count, app_login_frequency 5개 변수를 생성한다.

방법 1: 공분산 행렬 기반 생성 - 잠재 상환 성실도 요인 모델 (이슈 #26)
- 다섯 변수가 보이지 않는 하나의 요인 "상환 성실도" L에서 나온다고 본다 (공통 요인 모델)
    L   = z(w*X + (1-w)*S)
    A_i = -k*L + sqrt(1-k^2) * (sqrt(rho)*F + sqrt(1-rho)*Z_i)   (연체자일수록 낮게)
  X: 정규화된 Target, S: 정답 없이 금융 칸으로만 만든 위험 지표 (납부 이력 35 : 부채 사용 30, FICO 공개 구성비)
  F: 다섯 변수가 함께 갖는 공통 성향, Z_i: 변수별 노이즈
  w: 성실도에서 정답 신호 비중, rho: 공통 성향 비중, k: 최종 값의 상관이 목표가 되도록 이분법으로 맞춤
- 이전 방식(각 변수가 정답에서 따로 신호를 받음)은 변수를 합치면 노이즈가 상쇄되어
  대안 정보만으로 AUC가 0.9를 넘었다. 공통 성향과 금융 정보 연결로 이 과장을 줄인다.
- 결과: 생성된 대안 변수와 실제 Target 간 상관계수 약 -0.32 (과제 조건: 절댓값 0.30~0.50)
- w, rho는 이론값이 아니라 보정값이다. 근거와 민감도는 docs/alt_data_simulation.md 참고

thin_filer_ratio: 씬파일러 비율 조정 파라미터 (None 권장 - 기준대로만 판정)
bias_ratio: 성별/연령대별 승인율 편향 조정 파라미터

씬파일러 판정 기준 (이슈 #15):
- 기준 1: 금융 연체 관련 컬럼 2개 이상 결측 (손잡이를 쓰지 않으면 GMSC 해당자 0명,
         thin_filer_ratio를 쓰면 손잡이가 연체 칸을 지운 사람이 여기에 해당)
- 기준 2: 아래 3개 조건을 모두(AND) 만족 - "금융 기록이 거의 없는 사람"
    (1) 보유 대출·신용 한도 1개 이하
    (2) 부동산 담보대출 없음
    (3) 연체 기록 3개 컬럼 모두 0
- 두 기준 중 하나라도 만족하면 씬파일러 (GMSC 기준 약 4,110명)
- 임계값은 아래 상수로 관리하며, None을 넣으면 해당 조건을 끈다.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ===== 씬파일러 판정 기준 상수 (이슈 #15) =====
# 연체 기록 컬럼 (기준 1의 결측 검사 대상이자, 기준 2의 "연체 없음" 검사 대상)
DELINQUENCY_COLS = [
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate',
]
OPEN_CREDIT_COL = 'NumberOfOpenCreditLinesAndLoans'     # 보유 대출·신용 한도 개수
REAL_ESTATE_COL = 'NumberRealEstateLoansOrLines'        # 부동산 담보대출 건수
REVOLVING_COL = 'RevolvingUtilizationOfUnsecuredLines'  # 신용한도 이용률
DEBT_RATIO_COL = 'DebtRatio'                            # 부채비율

# 기준 1
MIN_MISSING_DELINQUENCY = 2      # 연체 컬럼 2개 이상 결측

# 기준 2 임계값 (None으로 두면 해당 조건을 사용하지 않음)
MAX_OPEN_CREDIT_LINES = 1        # 보유 대출·신용 한도 개수 상한
MAX_REAL_ESTATE_LOANS = 0        # 부동산 담보대출 건수 상한

# 씬파일러 집단이 평가에 쓸 수 있을 만큼 큰지 확인하는 경고 기준
MIN_THIN_FILER_POSITIVES = 30    # Test(15%) 기준 부도 표본이 이보다 적으면 경고

# ===== 대안 변수 생성 상수 (이슈 #26: 잠재 상환 성실도 요인 모델) =====
ALT_TARGET_CORR = 0.32           # 생성된 각 대안 변수와 정답의 상관 절댓값 (조건 0.30~0.50, 분할 표본 오차 여유)
LATENT_TARGET_WEIGHT = 0.3       # w: 상환 성실도에서 정답 신호 비중 (나머지는 금융 위험 지표)
COMMON_FACTOR_SHARE = 0.8        # rho: 다섯 변수가 함께 갖는 공통 성향 비중
FICO_PAYMENT_WEIGHT = 0.35       # 금융 위험 지표: 납부 이력 비중 (FICO 공개 구성비)
FICO_AMOUNTS_WEIGHT = 0.30       # 금융 위험 지표: 부채 사용 비중 (FICO 공개 구성비)
DELINQUENCY_COUNT_CAP = 20       # 연체 횟수의 96·98 같은 특수 코드가 지표를 끌고 가지 않게 자르는 상한
UTILIZATION_CAP = 2              # 한도 이용률 극단값 상한


def _setup_logger():
    """로깅 설정"""
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


def _z(values):
    """평균 0, 표준편차 1로 바꾼다 (값이 모두 같으면 0)."""
    values = np.asarray(values, dtype=float)
    std = values.std()
    return (values - values.mean()) / std if std > 0 else np.zeros_like(values)


def _minmax(values):
    return (values - values.min()) / (values.max() - values.min() + 1e-8)


def financial_risk_index(df):
    """
    정답을 쓰지 않고 금융 칸으로만 만든 위험 지표 S (표준화, 클수록 위험).
    신용평가의 두 큰 요소인 납부 이력과 부채 사용을 FICO 공개 구성비(35:30)로 더한다.
    필요한 칸이 없으면 None을 돌려준다.
    """
    if not all(col in df.columns for col in DELINQUENCY_COLS + [REVOLVING_COL]):
        return None
    past_due = df[DELINQUENCY_COLS].fillna(0).clip(upper=DELINQUENCY_COUNT_CAP).sum(axis=1)
    utilization = df[REVOLVING_COL].fillna(0).clip(upper=UTILIZATION_CAP).rank(pct=True)
    score = (FICO_PAYMENT_WEIGHT * _z(np.log1p(past_due.values))
             + FICO_AMOUNTS_WEIGHT * _z(utilization.values))
    return _z(score)


# 대안 변수별 값 범위로 바꾸는 방법 (Y는 평균 0, 표준편차 1인 성실도 점수, 클수록 성실)
ALT_TRANSFORMS = {
    'telecom_payment_rate': lambda y: np.clip(0.90 + 0.06 * y, 0, 1),                     # 0~1
    'utility_payment_rate': lambda y: np.clip(0.85 + 0.08 * y, 0, 1),                     # 0~1
    'spending_consistency': lambda y: np.clip(75 + 25 * _minmax(y), 0, 100),              # 0~100
    'regular_payment_count': lambda y: np.maximum(0, np.round(8 + 6 * _minmax(y))).astype(int),   # 0~20
    'app_login_frequency': lambda y: np.maximum(0, np.round(15 + 10 * _minmax(y))).astype(int),   # 0~30
}


def _calibrated_variable(latent, noise, target, transform, target_corr, iterations=40):
    """
    최종 값(범위 변환 후)과 정답의 상관이 -target_corr가 되도록 k를 이분법으로 찾는다.
    잘라내기·반올림을 거치면 상관이 줄어들기 때문에, 변환 전이 아니라 변환 후 값으로 맞춘다.
    """
    def build(k):
        return transform(-k * latent + np.sqrt(1 - k ** 2) * noise)

    def corr_of(k):
        return -np.corrcoef(build(k), target)[0, 1]

    if corr_of(1.0) < target_corr:
        raise ValueError(
            f"성실도와 정답의 상관이 너무 약해 목표 상관 {target_corr:.2f}를 만들 수 없습니다. "
            f"latent_weight를 높이세요."
        )
    low, high = 0.0, 1.0
    for _ in range(iterations):
        mid = (low + high) / 2
        if corr_of(mid) < target_corr:
            low = mid
        else:
            high = mid
    return build(high), high


def generate_alternative_data(df, target_col="SeriousDlqin2yrs",
                               thin_filer_ratio: float = None,
                               bias_ratio: float = 0,
                               random_seed: int = None,
                               latent_weight: float = LATENT_TARGET_WEIGHT,
                               common_factor: float = COMMON_FACTOR_SHARE,
                               target_corr: float = ALT_TARGET_CORR):
    """
    5개 대안 변수 생성 (Target과의 상관계수 절댓값 target_corr, 조건 0.30~0.50)

    방법: 잠재 상환 성실도 요인 모델 (공분산 행렬 기반, 이슈 #26)
    - 성실도 L = z(w*X + (1-w)*S): X는 정규화된 Target, S는 금융 위험 지표 (정답 미사용)
    - 대안 변수 A_i = -k*L + sqrt(1-k^2)*(sqrt(rho)*F + sqrt(1-rho)*Z_i) (연체자일수록 낮게)
    - F는 다섯 변수의 공통 성향, Z_i는 변수별 노이즈, k는 상관이 목표가 되도록 맞춘 값
    - 금융 칸(연체 3칸, 한도 이용률)이 없으면 S 없이 X만 쓴다

    Args:
        df: 기존 데이터프레임
        target_col: 타겟 변수명 (기본값: SeriousDlqin2yrs)
        thin_filer_ratio: 씬파일러 목표 비율 (0-1). 값을 주면 일반 고객의 연체 칸을 지워 기준 1로 판정됨 (#61).
                          None(기본값)이면 판정 기준대로만 분류한다 (권장).
        bias_ratio: 편향 강도 (0: 편향 없음, 1: 최대 편향)
        random_seed: 무작위 생성 시드. 같은 시드면 같은 결과가 나온다 (None이면 매번 다름)
        latent_weight: w. 성실도에서 정답 신호 비중 (0~1). 높을수록 대안 정보가 금융 정보에 없는 신호를 더 가진다
        common_factor: rho. 다섯 변수의 공통 성향 비중 (0~1). 높을수록 변수끼리 비슷하게 움직인다
        target_corr: 생성된 각 변수와 Target의 상관 절댓값

    Returns:
        대안 변수가 추가된 데이터프레임
    """
    _setup_logger()

    df_new = df.copy()
    n = len(df_new)
    rng = np.random.default_rng(random_seed)

    logger.info(f"대안 데이터 시뮬레이터 실행 중...")
    logger.info(f"   샘플 크기: {n:,}")
    logger.info(f"   생성 방법: 잠재 상환 성실도 요인 모델 (이슈 #26)")
    logger.info(f"   목표 상관계수: -{target_corr:.2f}, w(정답 비중)={latent_weight}, rho(공통 성향)={common_factor}")

    # ===== 단계 1: 실제 Target 표준화 (평균 0, 표준편차 1) =====
    target = df_new[target_col].values.astype(float)
    target_standardized = _z(target)

    # ===== 단계 2: 상환 성실도 L = 정답 신호 + 금융 위험 지표 =====
    risk_index = financial_risk_index(df_new)
    if risk_index is None:
        logger.info("   금융 칸이 없어 정답 신호만으로 성실도를 만듭니다")
        latent = target_standardized
    else:
        latent = _z(latent_weight * target_standardized + (1 - latent_weight) * risk_index)
        logger.info(f"   금융 위험 지표와 정답의 상관: {np.corrcoef(risk_index, target)[0, 1]:.3f} (정답 미사용으로 계산)")
    logger.info(f"   성실도와 정답의 상관: {np.corrcoef(latent, target)[0, 1]:.3f}")

    # ===== 단계 3: 공통 성향 F와 변수별 노이즈로 다섯 변수 생성 =====
    common = rng.standard_normal(n)
    for name, transform in ALT_TRANSFORMS.items():
        noise = np.sqrt(common_factor) * common + np.sqrt(1 - common_factor) * rng.standard_normal(n)
        values, k = _calibrated_variable(latent, noise, target, transform, target_corr)
        df_new[name] = values
        logger.info(f"      {name}: k={k:.4f}")

    # ===== 단계 5: 인구통계 변수 추가 =====
    df_new = add_demographic_features(df_new, bias_ratio, rng=rng)

    # ===== 단계 6: 씬파일러 레이블 생성 =====
    df_new = label_thin_filer(df_new, thin_filer_ratio, target_col=target_col, random_seed=random_seed)

    # ===== 단계 7: 상관계수 검증 =====
    # 생성된 대안 변수와 실제 Target의 상관계수 계산
    _validate_correlations(df_new, target_col)

    # ===== 단계 8: 시뮬레이션 데이터 저장 =====
    import os
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'intermediate')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, '02_simulated.csv')
    df_new.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"시뮬레이션 데이터 저장: {output_path} ({len(df_new):,} rows)\n")

    logger.info(f"대안 데이터 생성 완료\n")

    return df_new


def _validate_correlations(df, target_col):
    """상관계수 검증 및 리포트 생성"""
    _setup_logger()

    alt_vars = ['telecom_payment_rate', 'utility_payment_rate',
                'spending_consistency', 'regular_payment_count',
                'app_login_frequency']

    # 타겟과의 상관계수 계산 (사후 검증)
    logger.info(f"\n{'='*75}")
    logger.info(f"생성된 대안 변수와 타겟(SeriousDlqin2yrs) 간 실제 상관계수 (사후 검증)")
    logger.info(f"{'='*75}")
    logger.info(f"{'변수명':<30} {'실제 상관계수':<18} {'범위 확인':<20}")
    logger.info("-" * 75)

    correlations = {}
    in_range_count = 0

    for var in alt_vars:
        if var in df.columns:
            corr = df[var].corr(df[target_col])
            correlations[var] = corr

            abs_corr = abs(corr)
            if 0.30 <= abs_corr <= 0.50:
                status = "OK (0.30~0.50)"
                in_range_count += 1
            else:
                status = f"OUT ({abs_corr:.4f})"

            logger.info(f"{var:<30} {corr:>14.4f}    {status:<20}")

    logger.info("-" * 75)
    avg_abs_corr = np.mean([abs(c) for c in correlations.values()])
    logger.info(f"{'평균 상관계수 절대값':<30} {avg_abs_corr:>14.4f}")
    logger.info(f"{'범위 내 변수 개수':<30} {in_range_count}/5")
    logger.info(f"{'='*75}\n")

    # 변수 간 상관계수 행렬
    logger.info(f"대안 변수 간 상관계수 행렬:")
    corr_df = df[alt_vars].corr()
    logger.info(f"\n{corr_df.to_string()}\n")

    return correlations


def add_demographic_features(df, bias_ratio: float = 0.0, rng=None):
    """
    인구통계 변수 추가 (성별, 연령대)

    Args:
        df: 입력 데이터프레임
        bias_ratio: 성별/연령대별 부도 편향 강도 (0-1). 0보다 크면 남성의 앱 로그인 빈도(성별 대리 변수)도 높인다 (#41)
                   0: 편향 없음 (성별/연령대와 부도가 독립)
                   1: 최대 편향 (특정 성별/연령대에 높은 부도율)

    Returns:
        gender, age_group 컬럼이 추가된 데이터프레임
    """
    _setup_logger()

    n = len(df)
    if rng is None:
        rng = np.random.default_rng()

    # 성별 생성 (0: Female, 1: Male)
    df['gender'] = rng.binomial(1, 0.5, n)

    # 연령대 생성 (age 컬럼 기반)
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 35, 55, np.inf],
        labels=[0, 1, 2],
        right=False,
        include_lowest=True
    ).fillna(0).astype(int)

    # 성별 대리 변수 (#41): bias_ratio > 0이면 남성의 앱 로그인 빈도를 높인다.
    # 정답(연체 여부)과 무관하게 넣으므로, 모델은 이 변수를 통해 성별에 따른 승인율 차이를 배우게 된다.
    # bias_ratio = 0이면 아무것도 바꾸지 않는다 (기본 데이터와 같음).
    if bias_ratio > 0 and 'app_login_frequency' in df.columns:
        shift = int(round(30 * bias_ratio))
        male = df['gender'] == 1
        df.loc[male, 'app_login_frequency'] = np.clip(
            df.loc[male, 'app_login_frequency'] + shift, 0, 30
        )
        logger.info(f"   성별 대리 변수: 남성 app_login_frequency +{shift} (0~30 범위 유지)")
    # bias_ratio 적용: 특정 성별·연령대에 부도율 편향
    if bias_ratio > 0 and 'SeriousDlqin2yrs' in df.columns:
        # 남성(1) 또는 55세 이상(2)에 편향 적용
        bias_mask = (df['gender'] == 1) | (df['age_group'] == 2)
        bias_indices = df[bias_mask].index

        if len(bias_indices) > 0:
            # 편향된 샘플 중 일부에 부도 레이블 추가
            n_bias = int(len(bias_indices) * bias_ratio)
            if n_bias > 0:
                bias_sample_indices = rng.choice(
                    bias_indices,
                    size=min(n_bias, len(bias_indices)),
                    replace=False
                )
                df.loc[bias_sample_indices, 'SeriousDlqin2yrs'] = 1

    # 통계 로깅
    gender_counts = df['gender'].value_counts().sort_index()
    age_group_counts = df['age_group'].value_counts().sort_index()

    logger.info(f"인구통계 변수 생성 완료:")
    logger.info(f"   gender: {gender_counts.get(0, 0):,} (여성), {gender_counts.get(1, 0):,} (남성)")
    logger.info(f"   age_group:")
    logger.info(f"      0 (20~34): {age_group_counts.get(0, 0):,}")
    logger.info(f"      1 (35~54): {age_group_counts.get(1, 0):,}")
    logger.info(f"      2 (55+):   {age_group_counts.get(2, 0):,}")
    if bias_ratio > 0:
        logger.info(f"   (편향 강도: {bias_ratio*100:.1f}%)\n")
    else:
        logger.info(f"   (편향 없음)\n")

    return df


def label_thin_filer(df, thin_filer_ratio: float = None,
                     target_col: str = 'SeriousDlqin2yrs',
                     random_seed: int = None):
    """
    씬파일러 판정 (이슈 #15 기준)

    원본 과제 정의
        "금융 연체 관련 컬럼 2개 이상 결측치이거나 신용카드 거래 이력이 12개월 미만"

    GMSC 데이터에 그대로 적용할 수 없는 이유
        - 기준 1: 연체 컬럼 3개에 결측이 없어 해당자 0명
        - 기준 2: 카드 사용 "기간" 정보가 데이터에 없음

    따라서 용어 정의("금융 이력이 부족하여 기존 CB 모델로는 적절한 신용평가가
    어려운 개인")에 맞춰, 데이터 안에서 금융 기록이 가장 적은 사람을 씬파일러로 본다.

    판정 기준 (기준 1 OR 기준 2)
        기준 1 : 연체 컬럼 2개 이상 결측              (thin_filer_ratio=None이면 GMSC 해당자 0명)
        기준 2 : 아래 3개 조건을 모두(AND) 만족        (GMSC 해당자 약 4,110명)
                 - NumberOfOpenCreditLinesAndLoans      <= 1     보유 대출·신용 한도
                 - NumberRealEstateLoansOrLines         == 0     부동산 담보대출
                 - 연체 기록 3개 컬럼                    == 0     연체 이력 없음

    기준 2의 근거
        - 업계: Experian은 활성 계좌가 1~2개이거나 5개 미만이면 씬파일러로 분류
        - 데이터: 부동산 담보대출·연체 기록이 없는 사람 중 계좌 0개는 연체율 13.0%,
                 1개는 5.3%이고, 2개부터는 2-3%로 평평함 (1개 이하에서 구분선)
        - 데이터: 계좌 1개 이하 집단의 35세 미만 비율 40% (전체 13%)
                 → 사회초년생 비중이 높아 씬파일러 정의와 부합

    한계 (리포트에 명시 필요)
        - 기간이 아닌 "개수" 기준이므로, 카드 1개를 오래 사용한 사람도 포함될 수 있다.

    Args:
        df: 입력 데이터프레임
        thin_filer_ratio: 씬파일러 목표 비율 (0-1). None(기본값)이면 기준대로만 판정한다.
                          값을 주면 일반 고객 일부의 연체 칸 3개를 결측으로 지운 뒤
                          기준 1로 다시 판정한다 (이슈 #61). 이미 목표보다 많으면 그대로 둔다.
                          후보는 연체 기록이 0인 일반 고객으로 한정한다 (기록을 지워도 중앙값과 같아짐).
                          입력 df의 연체 칸은 바꾸지 않고 복사본을 만들어 수정한다.
        target_col: 타겟 컬럼명 (판정 로그의 연체율 계산에만 사용, 판정 자체에는 미사용)
        random_seed: thin_filer_ratio 사용 시 고객을 고르는 시드 (같은 값이면 같은 결과)

    Returns:
        is_thin_filer 컬럼이 추가된 데이터프레임
    """
    _setup_logger()

    # ===== 기준 1: 연체 관련 컬럼 2개 이상 결측 =====
    delinquency_cols = [col for col in DELINQUENCY_COLS if col in df.columns]

    if len(delinquency_cols) >= MIN_MISSING_DELINQUENCY:
        criteria_1 = df[delinquency_cols].isna().sum(axis=1) >= MIN_MISSING_DELINQUENCY
    else:
        logger.warning(
            f"기준 1 적용 불가 - 연체 컬럼 부족 "
            f"(필요 {MIN_MISSING_DELINQUENCY}개 이상, 발견 {len(delinquency_cols)}개)"
        )
        criteria_1 = pd.Series(False, index=df.index)

    # ===== 기준 2: 금융 기록이 거의 없는 사람 (모든 조건 AND) =====
    # (상한, 컬럼, 설명) - 상한이 None이면 해당 조건을 건너뛴다
    upper_bound_rules = [
        (MAX_OPEN_CREDIT_LINES, OPEN_CREDIT_COL, "보유 대출·신용 한도"),
        (MAX_REAL_ESTATE_LOANS, REAL_ESTATE_COL, "부동산 담보대출"),
    ]

    criteria_2 = pd.Series(True, index=df.index)
    applied_rules = []       # 로그용: (설명, 그 조건만 만족하는 인원)
    skipped_rules = []       # 로그용: 적용하지 못한 조건

    for threshold, col, name in upper_bound_rules:
        if threshold is None:
            skipped_rules.append(f"{name} (상한 None - 사용 안 함)")
            continue
        if col not in df.columns:
            skipped_rules.append(f"{name} (컬럼 없음: {col})")
            continue
        if col == REAL_ESTATE_COL:
            # 부동산 담보대출은 == 0 (정확히 0)
            rule = df[col] == threshold
            applied_rules.append((f"{name} == {threshold}", int(rule.sum())))
        else:
            # 나머지는 <= (이하)
            rule = df[col] <= threshold          # 결측은 False → 자동 제외
            applied_rules.append((f"{name} <= {threshold}", int(rule.sum())))
        criteria_2 &= rule

    # 연체 기록 3개 컬럼이 모두 0
    if len(delinquency_cols) == len(DELINQUENCY_COLS):
        rule = (df[DELINQUENCY_COLS] == 0).all(axis=1)   # 결측은 False → 자동 제외
        criteria_2 &= rule
        applied_rules.append(("연체 기록 3개 컬럼 모두 0", int(rule.sum())))
    else:
        skipped_rules.append(
            f"연체 기록 0 (컬럼 부족: {len(delinquency_cols)}/{len(DELINQUENCY_COLS)})"
        )

    # 적용된 조건이 하나도 없으면 전원이 씬파일러가 되므로 차단한다
    if not applied_rules:
        logger.warning("기준 2를 적용할 수 있는 조건이 없습니다. 기준 2는 건너뜁니다.")
        criteria_2 = pd.Series(False, index=df.index)

    df['is_thin_filer'] = (criteria_1 | criteria_2).astype(int)

    # ===== thin_filer_ratio 조정 (#61) =====
    # 목표 비율에 못 미치면 연체 기록이 0인 일반 고객을 골라 연체 칸 3개를 결측으로 지운 뒤,
    # 기준 1로 다시 판정한다. 재판정(재귀 호출)이 한 번만 로그를 남기도록 여기서 먼저 처리한다.
    if thin_filer_ratio is not None:
        n = len(df)
        target_count = int(n * thin_filer_ratio)
        diff = target_count - int(df['is_thin_filer'].sum())
        if diff > 0:
            if len(delinquency_cols) != len(DELINQUENCY_COLS):
                raise ValueError("thin_filer_ratio를 쓰려면 연체 칸 3개가 모두 있어야 합니다")
            clean = (df['is_thin_filer'] == 0) & (df[DELINQUENCY_COLS] == 0).all(axis=1)
            candidates = df.index[clean]
            rng = np.random.default_rng(random_seed)
            picked = rng.choice(candidates, size=min(diff, len(candidates)), replace=False)
            logger.info(f"   목표 {thin_filer_ratio:.1%}({target_count:,}명): 연체 기록 0인 일반 고객 "
                        f"{len(picked):,}명의 연체 칸 {len(DELINQUENCY_COLS)}개를 지운 뒤 다시 판정")
            df = df.copy()
            df.loc[picked, DELINQUENCY_COLS] = np.nan
            return label_thin_filer(df, thin_filer_ratio=None, target_col=target_col)

    # ===== 판정 결과 로그 =====
    n = len(df)
    n_c1 = int(criteria_1.sum())
    n_c2 = int(criteria_2.sum())
    n_thin = int(df['is_thin_filer'].sum())

    logger.info("=" * 75)
    logger.info("씬파일러 판정 결과 (이슈 #15 기준)")
    logger.info("=" * 75)
    logger.info(f"   기준 1 (연체 컬럼 {MIN_MISSING_DELINQUENCY}개 이상 결측): "
                f"{n_c1:,}명 ({n_c1/n*100:.2f}%)")
    logger.info(f"   기준 2 (아래 조건 모두 AND): {n_c2:,}명 ({n_c2/n*100:.2f}%)")
    for description, single_count in applied_rules:
        logger.info(f"      - {description:<30} 이 조건만 만족 {single_count:,}명")
    for description in skipped_rules:
        logger.info(f"      - [미적용] {description}")
    logger.info(f"   최종 씬파일러 (기준1 OR 기준2): {n_thin:,}명 ({n_thin/n*100:.2f}%)")
    logger.info(f"   일반 고객: {n - n_thin:,}명 ({(n - n_thin)/n*100:.2f}%)")

    # 집단별 연체율
    if target_col in df.columns and 0 < n_thin < n:
        rate_thin = df.loc[df['is_thin_filer'] == 1, target_col].mean() * 100
        rate_normal = df.loc[df['is_thin_filer'] == 0, target_col].mean() * 100
        logger.info(f"   연체율: 씬파일러 {rate_thin:.2f}% / 일반 {rate_normal:.2f}%")

    # 계좌 수 분포 (기준 2가 적용된 경우)
    if OPEN_CREDIT_COL in df.columns and n_thin > 0:
        account_dist = df.loc[df['is_thin_filer'] == 1, OPEN_CREDIT_COL].value_counts().sort_index()
        dist_text = ", ".join(f"{int(k)}개 {int(v):,}명" for k, v in account_dist.head(5).items())
        logger.info(f"   씬파일러 계좌 수 분포: {dist_text}")

    if 'age' in df.columns and n_thin > 0:
        young_thin = (df.loc[df['is_thin_filer'] == 1, 'age'] < 35).mean() * 100
        young_all = (df['age'] < 35).mean() * 100
        logger.info(f"   35세 미만 비율: 씬파일러 {young_thin:.1f}% / 전체 {young_all:.1f}%")

    # ===== 평가 가능성 경고: Test(15%) 기준 부도 표본 수 =====
    if target_col in df.columns and n_thin > 0:
        thin_positives = int(df.loc[df['is_thin_filer'] == 1, target_col].sum())
        expected_test_positives = thin_positives * 0.15
        if expected_test_positives < MIN_THIN_FILER_POSITIVES:
            logger.warning(
                f"씬파일러 부도 표본이 적습니다 (전체 {thin_positives:,}명, "
                f"Test 15% 기준 약 {expected_test_positives:.0f}명). "
                f"씬파일러 AUC 향상(+0.03) 측정이 불안정할 수 있으니 "
                f"임계값을 완화하거나 부트스트랩 신뢰구간을 함께 보고하세요."
            )

    return df
