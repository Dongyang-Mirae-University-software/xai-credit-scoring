"""
대안 데이터 시뮬레이터.
telecom_payment_rate, utility_payment_rate, spending_consistency,
regular_payment_count, app_login_frequency 5개 변수를 생성한다.

방법 1: 공분산 행렬 기반 생성
- 공분산 행렬로 정의된 상관관계를 유지
- 공식: Y = r*X + sqrt(1-r^2)*Z
  (r: 목표 상관계수, X: 정규화된 Target, Z: 독립적인 노이즈)
- 결과: 생성된 대안 변수와 실제 Target 간 0.30~0.50 상관계수

thin_filer_ratio: 씬파일러 비율 조정 파라미터 (None 권장 - 기준대로만 판정)
bias_ratio: 성별/연령대별 승인율 편향 조정 파라미터

씬파일러 판정 기준 (이슈 #15):
- 기준 1: 금융 연체 관련 컬럼 2개 이상 결측 (GMSC에는 해당자 0명, 형식상 유지)
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


def generate_alternative_data(df, target_col="SeriousDlqin2yrs",
                               thin_filer_ratio: float = None,
                               bias_ratio: float = 0.1):
    """
    5개 대안 변수 생성 (Target과의 상관계수: 0.30~0.50 범위)

    방법: 공분산 행렬 기반 생성
    - 공식: Y = r*X + sqrt(1-r^2)*Z
    - X: 정규화된 실제 Target
    - Z: 독립적인 표준정규분포
    - r: 공분산 행렬에 정의된 목표 상관계수

    Args:
        df: 기존 데이터프레임
        target_col: 타겟 변수명 (기본값: SeriousDlqin2yrs)
        thin_filer_ratio: 씬파일러 비율 강제 조정 (0-1).
                          None(기본값)이면 판정 기준대로만 분류한다 (권장).
        bias_ratio: 편향 강도 (0: 편향 없음, 1: 최대 편향)

    Returns:
        대안 변수가 추가된 데이터프레임
    """
    _setup_logger()

    df_new = df.copy()
    n = len(df_new)

    logger.info(f"대안 데이터 시뮬레이터 실행 중...")
    logger.info(f"   샘플 크기: {n:,}")
    logger.info(f"   목표 상관계수 범위: 0.30~0.50")
    logger.info(f"   생성 방법: 공분산 행렬 기반 + 실제 Target 연결")

    # ===== 단계 1: 실제 Target 표준화 (평균 0, 표준편차 1) =====
    target = df_new[target_col].values.astype(float)
    target_standardized = (target - target.mean()) / (target.std() + 1e-8)

    # ===== 단계 2: 공분산 행렬 정의 =====
    # Target과의 상관계수 설정 (0.30~0.50 범위)
    target_correlations = {
        'telecom_payment_rate': np.random.uniform(0.30, 0.50),
        'utility_payment_rate': np.random.uniform(0.30, 0.50),
        'spending_consistency': np.random.uniform(0.30, 0.50),
        'regular_payment_count': np.random.uniform(0.30, 0.50),
        'app_login_frequency': np.random.uniform(0.30, 0.50)
    }

    logger.info(f"\n   설정된 목표 상관계수:")
    for var, corr in target_correlations.items():
        logger.info(f"      {var}: {corr:.4f}")

    # 공분산 행렬 구성 (6x6: target + 5개 변수)
    cov_matrix = np.eye(6)

    # 타겟과의 상관관계 정의
    for idx, corr in enumerate(target_correlations.values(), start=1):
        cov_matrix[0, idx] = corr
        cov_matrix[idx, 0] = corr

    # 변수 간 상관관계 (0.20~0.40)
    for i in range(1, 6):
        for j in range(i+1, 6):
            cov_matrix[i, j] = np.random.uniform(0.20, 0.40)
            cov_matrix[j, i] = cov_matrix[i, j]

    # ===== 단계 3: 공분산 행렬에서 5개 변수만 샘플링 =====
    # 5x5 부분 공분산 행렬 (변수 간만)
    cov_alt = cov_matrix[1:, 1:]

    # 표준편차 설정
    std_devs = np.array([0.15, 0.15, 12, 4, 6])
    scaled_cov_alt = np.diag(std_devs) @ cov_alt @ np.diag(std_devs)

    try:
        # 5개 변수의 기본값
        mean_alt = np.array([0.85, 0.80, 75, 8, 15])
        samples_alt = np.random.multivariate_normal(mean_alt, scaled_cov_alt, n)
        logger.info(f"   공분산 행렬 기반 샘플링 성공")
    except np.linalg.LinAlgError:
        logger.warning("공분산 행렬이 양정부호가 아닙니다. 표준 정규분포 사용")
        samples_alt = np.random.standard_normal((n, 5)) * std_devs + np.array([0.85, 0.80, 75, 8, 15])

    # ===== 단계 4: 공식으로 실제 Target과 상관관계 유지 =====
    # Y = r*X + sqrt(1-r^2)*Z
    # X: 정규화된 Target (0~1)
    # Z: 독립적인 샘플 (정규화됨)
    # r: 목표 상관계수

    logger.info(f"   공식 적용: Y = r*X + sqrt(1-r^2)*Z")

    # 각 변수별로 공식 적용
    var_names = list(target_correlations.keys())
    r_values = list(target_correlations.values())

    # 1. telecom_payment_rate (0~1)
    r = r_values[0]
    Z = (samples_alt[:, 0] - samples_alt[:, 0].mean()) / (samples_alt[:, 0].std() + 1e-8)
    Y = r * target_standardized + np.sqrt(1 - r**2) * Z
    df_new['telecom_payment_rate'] = np.clip(
        (Y - Y.min()) / (Y.max() - Y.min() + 1e-8),
        0, 1
    )

    # 2. utility_payment_rate (0~1)
    r = r_values[1]
    Z = (samples_alt[:, 1] - samples_alt[:, 1].mean()) / (samples_alt[:, 1].std() + 1e-8)
    Y = r * target_standardized + np.sqrt(1 - r**2) * Z
    df_new['utility_payment_rate'] = np.clip(
        (Y - Y.min()) / (Y.max() - Y.min() + 1e-8),
        0, 1
    )

    # 3. spending_consistency (0~100)
    r = r_values[2]
    Z = (samples_alt[:, 2] - samples_alt[:, 2].mean()) / (samples_alt[:, 2].std() + 1e-8)
    Y = r * target_standardized + np.sqrt(1 - r**2) * Z
    df_new['spending_consistency'] = np.clip(
        75 + 25 * ((Y - Y.min()) / (Y.max() - Y.min() + 1e-8)),
        0, 100
    )

    # 4. regular_payment_count (0~20)
    r = r_values[3]
    Z = (samples_alt[:, 3] - samples_alt[:, 3].mean()) / (samples_alt[:, 3].std() + 1e-8)
    Y = r * target_standardized + np.sqrt(1 - r**2) * Z
    df_new['regular_payment_count'] = np.maximum(
        0, np.round(8 + 6 * ((Y - Y.min()) / (Y.max() - Y.min() + 1e-8))).astype(int)
    )

    # 5. app_login_frequency (0~30)
    r = r_values[4]
    Z = (samples_alt[:, 4] - samples_alt[:, 4].mean()) / (samples_alt[:, 4].std() + 1e-8)
    Y = r * target_standardized + np.sqrt(1 - r**2) * Z
    df_new['app_login_frequency'] = np.maximum(
        0, np.round(15 + 10 * ((Y - Y.min()) / (Y.max() - Y.min() + 1e-8))).astype(int)
    )

    # ===== 단계 5: 인구통계 변수 추가 =====
    df_new = add_demographic_features(df_new, bias_ratio)

    # ===== 단계 6: 씬파일러 레이블 생성 =====
    df_new = label_thin_filer(df_new, thin_filer_ratio, target_col=target_col)

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


def add_demographic_features(df, bias_ratio: float = 0.0):
    """
    인구통계 변수 추가 (성별, 연령대)

    Args:
        df: 입력 데이터프레임
        bias_ratio: 성별/연령대별 부도 편향 강도 (0-1)
                   0: 편향 없음 (성별/연령대와 부도가 독립)
                   1: 최대 편향 (특정 성별/연령대에 높은 부도율)

    Returns:
        Gender, AgeGroup 컬럼이 추가된 데이터프레임
    """
    _setup_logger()

    n = len(df)

    # 성별 생성 (0: Female, 1: Male)
    df['Gender'] = np.random.binomial(1, 0.5, n)

    # 연령대 생성 (age 컬럼 기반)
    df['AgeGroup'] = pd.cut(
        df['age'],
        bins=[0, 29, 39, 49, 59, 101],
        labels=[0, 1, 2, 3, 4],
        right=False,
        include_lowest=True
    ).fillna(0).astype(int)

    # bias_ratio 적용: 특정 성별·연령대에 부도율 편향
    if bias_ratio > 0 and 'SeriousDlqin2yrs' in df.columns:
        # 남성(1) 또는 60세 이상(4)에 편향 적용
        bias_mask = (df['Gender'] == 1) | (df['AgeGroup'] == 4)
        bias_indices = df[bias_mask].index

        if len(bias_indices) > 0:
            # 편향된 샘플 중 일부에 부도 레이블 추가
            n_bias = int(len(bias_indices) * bias_ratio)
            if n_bias > 0:
                bias_sample_indices = np.random.choice(
                    bias_indices,
                    size=min(n_bias, len(bias_indices)),
                    replace=False
                )
                df.loc[bias_sample_indices, 'SeriousDlqin2yrs'] = 1

    # 통계 로깅
    gender_counts = df['Gender'].value_counts().sort_index()
    age_group_counts = df['AgeGroup'].value_counts().sort_index()

    logger.info(f"인구통계 변수 생성 완료:")
    logger.info(f"   Gender: {gender_counts.get(0, 0):,} (여성), {gender_counts.get(1, 0):,} (남성)")
    logger.info(f"   AgeGroup:")
    logger.info(f"      0 (20~29): {age_group_counts.get(0, 0):,}")
    logger.info(f"      1 (30~39): {age_group_counts.get(1, 0):,}")
    logger.info(f"      2 (40~49): {age_group_counts.get(2, 0):,}")
    logger.info(f"      3 (50~59): {age_group_counts.get(3, 0):,}")
    logger.info(f"      4 (60+):   {age_group_counts.get(4, 0):,}")
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
        기준 1 : 연체 컬럼 2개 이상 결측              (GMSC 해당자 0명, 형식상 유지)
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
        thin_filer_ratio: 씬파일러 비율 강제 조정 (0-1).
                          None(기본값)이면 무작위 조정 없이 기준대로만 판정한다.
                          값을 주면 무작위로 레이블을 추가/제거하므로
                          씬파일러 AUC 향상 측정이 왜곡된다 (디버깅 용도로만 사용).
        target_col: 타겟 컬럼명 (판정 로그의 연체율 계산에만 사용, 판정 자체에는 미사용)
        random_seed: thin_filer_ratio 사용 시 무작위 선택 시드

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

    # ===== thin_filer_ratio 강제 조정 (기본값 None이면 건너뜀) =====
    if thin_filer_ratio is None:
        logger.info("   조정 없음 (판정 기준대로만 분류)")
        logger.info("=" * 75 + "\n")
        return df

    logger.warning(
        f"thin_filer_ratio={thin_filer_ratio}가 지정되어 무작위로 레이블을 조정합니다. "
        f"기준과 무관한 샘플이 씬파일러로 섞이므로 "
        f"씬파일러 AUC 향상 측정이 왜곡됩니다. 평가용으로는 None을 사용하세요."
    )

    rng = np.random.default_rng(random_seed)
    target_count = int(n * thin_filer_ratio)
    diff = target_count - n_thin

    if diff > 0:
        candidates = df.index[df['is_thin_filer'] == 0]
        picked = rng.choice(candidates, size=min(diff, len(candidates)), replace=False)
        df.loc[picked, 'is_thin_filer'] = 1
    elif diff < 0:
        candidates = df.index[df['is_thin_filer'] == 1]
        picked = rng.choice(candidates, size=min(-diff, len(candidates)), replace=False)
        df.loc[picked, 'is_thin_filer'] = 0

    adjusted = int(df['is_thin_filer'].sum())
    logger.info(f"   조정 후 씬파일러: {adjusted:,}명 ({adjusted/n*100:.2f}%) "
                f"- 이 중 기준 충족 {n_thin:,}명")
    logger.info("=" * 75 + "\n")

    return df