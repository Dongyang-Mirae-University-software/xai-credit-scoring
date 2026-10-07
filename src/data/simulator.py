"""
대안 데이터 시뮬레이터.
telecom_payment_rate, utility_payment_rate, spending_consistency,
regular_payment_count, app_login_frequency 5개 변수를 생성한다.

방법 1: 공분산 행렬 기반 생성
- 공분산 행렬로 정의된 상관관계를 유지
- 공식: Y = r*X + sqrt(1-r^2)*Z
  (r: 목표 상관계수, X: 정규화된 Target, Z: 독립적인 노이즈)
- 결과: 생성된 대안 변수와 실제 Target 간 0.30~0.50 상관계수

thin_filer_ratio: 씬파일러 비율 조정 파라미터
bias_ratio: 성별/연령대별 승인율 편향 조정 파라미터
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
                               thin_filer_ratio: float = 0.3,
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
        thin_filer_ratio: 씬파일러 비율 (0-1)
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
    df_new = label_thin_filer(df_new, thin_filer_ratio)

    # ===== 단계 7: 상관계수 검증 =====
    # 생성된 대안 변수와 실제 Target의 상관계수 계산
    _validate_correlations(df_new, target_col)

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


def label_thin_filer(df, thin_filer_ratio: float = None):
    """
    씬파일러 판정 로직

    판정 기준:
    1. 금융 연체 관련 컬럼 2개 이상 결측
    2. 또는 신용카드 거래 이력 12개월 미만 (NumberOfOpenCreditLinesAndLoans < 2)

    Args:
        df: 입력 데이터프레임
        thin_filer_ratio: 씬파일러 비율 조정 (0-1, None이면 기준에 따라 자동 판정)
                         예: 0.1 = 10% 씬파일러

    Returns:
        is_thin_filer 컬럼이 추가된 데이터프레임
    """
    _setup_logger()

    df['is_thin_filer'] = 0

    # 기준 1: 금융 연체 관련 컬럼 2개 이상 결측
    financial_cols = [
        'NumberOfTime30-59DaysPastDueNotWorse',
        'NumberOfTimes90DaysLate',
        'NumberOfTime60-89DaysPastDueNotWorse'
    ]
    existing_cols = [col for col in financial_cols if col in df.columns]

    if existing_cols:
        missing_count = df[existing_cols].isna().sum(axis=1)
        criteria_1 = missing_count >= 2
    else:
        criteria_1 = pd.Series(False, index=df.index)

    # 기준 2: 신용카드 거래 이력 12개월 미만
    # (NumberOfOpenCreditLinesAndLoans < 2 로 판단)
    if 'NumberOfOpenCreditLinesAndLoans' in df.columns:
        criteria_2 = df['NumberOfOpenCreditLinesAndLoans'] < 2
    else:
        criteria_2 = pd.Series(False, index=df.index)

    # 두 기준 중 하나라도 만족하면 씬파일러
    df['is_thin_filer'] = (criteria_1 | criteria_2).astype(int)

    # thin_filer_ratio 파라미터로 비율 조정
    if thin_filer_ratio is not None:
        n = len(df)
        target_count = int(n * thin_filer_ratio)
        current_count = df['is_thin_filer'].sum()

        if current_count != target_count:
            # 현재 판정된 씬파일러 외에 추가로 설정해야 할 샘플 수
            diff = target_count - current_count

            if diff > 0:
                # 추가로 씬파일러로 표시할 샘플 선택
                non_thin_filer_indices = df[df['is_thin_filer'] == 0].index
                additional_indices = np.random.choice(
                    non_thin_filer_indices,
                    size=min(diff, len(non_thin_filer_indices)),
                    replace=False
                )
                df.loc[additional_indices, 'is_thin_filer'] = 1
            elif diff < 0:
                # 씬파일러에서 제거할 샘플 선택
                thin_filer_indices = df[df['is_thin_filer'] == 1].index
                remove_indices = np.random.choice(
                    thin_filer_indices,
                    size=min(-diff, len(thin_filer_indices)),
                    replace=False
                )
                df.loc[remove_indices, 'is_thin_filer'] = 0

    thin_filer_count = df['is_thin_filer'].sum()
    logger.info(f"씬파일러 판정 완료:")
    logger.info(f"   총 샘플: {len(df):,}")
    logger.info(f"   씬파일러: {thin_filer_count:,} ({thin_filer_count/len(df)*100:.2f}%)")
    logger.info(f"   일반: {len(df) - thin_filer_count:,} ({(1-thin_filer_count/len(df))*100:.2f}%)")
    if thin_filer_ratio is not None:
        logger.info(f"   (파라미터로 조정됨: {thin_filer_ratio*100:.1f}%)\n")
    else:
        logger.info(f"   (기준에 따라 자동 판정)\n")

    return df
