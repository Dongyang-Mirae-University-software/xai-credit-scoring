"""
대안 데이터 시뮬레이터 - 방법 3: Copula 기반 샘플링
변수 간 비선형 의존성을 모델링하여 생성
"""

import logging
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.linalg import cholesky

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


def generate_alternative_data_method3(df, target_col="SeriousDlqin2yrs",
                                      thin_filer_ratio: float = 0.3,
                                      bias_ratio: float = 0.1):
    """
    방법 3: Gaussian Copula를 사용한 대안 변수 생성

    변수 간 비선형 의존성을 모델링하여 생성

    Args:
        df: 기존 데이터프레임
        target_col: 타겟 변수명
        thin_filer_ratio: 씬파일러 비율
        bias_ratio: 편향 강도

    Returns:
        대안 변수가 추가된 데이터프레임
    """
    _setup_logger()

    df_new = df.copy()
    n = len(df_new)

    logger.info(f"대안 데이터 시뮬레이터 (방법 3: Gaussian Copula) 실행 중...")
    logger.info(f"   샘플 크기: {n:,}")
    logger.info(f"   목표 상관계수 범위: 0.30~0.50")
    logger.info(f"   생성 방법: Gaussian Copula")

    # ===== 단계 1: 실제 Target 표준화 =====
    target = df_new[target_col].values.astype(float)
    target_standardized = (target - target.mean()) / (target.std() + 1e-8)

    # ===== 단계 2: 공분산 행렬 정의 =====
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

    # 공분산 행렬 구성
    cov_matrix = np.eye(6)

    # 타겟과의 상관관계
    for idx, corr in enumerate(target_correlations.values(), start=1):
        cov_matrix[0, idx] = corr
        cov_matrix[idx, 0] = corr

    # 변수 간 상관관계
    for i in range(1, 6):
        for j in range(i+1, 6):
            cov_matrix[i, j] = np.random.uniform(0.20, 0.40)
            cov_matrix[j, i] = cov_matrix[i, j]

    # ===== 단계 3: Gaussian Copula 생성 =====
    # Step 1: 표준정규분포 샘플 생성
    try:
        L = cholesky(cov_matrix, lower=True)
        Z = np.random.standard_normal((n, 6))
        X = Z @ L.T  # 상관된 표준정규분포
    except np.linalg.LinAlgError:
        logger.warning("Cholesky 분해 실패")
        X = np.random.standard_normal((n, 6))

    # Step 2: 표준정규분포를 Uniform 분포로 변환 (CDF 적용)
    # Copula의 핵심: 마진 분포와 의존성 구조 분리
    U = norm.cdf(X)  # (n, 6) Uniform [0, 1]

    logger.info(f"   Gaussian Copula 생성 완료")

    # ===== 단계 4: 각 변수별 마진 분포로 변환 =====
    # 역함수를 사용하여 원래 분포로 변환
    r_values = list(target_correlations.values())

    # 평균과 표준편차
    means = np.array([0, 0.85, 0.80, 75, 8, 15])
    stds = np.array([1, 0.15, 0.15, 12, 4, 6])

    # 각 변수의 마진 분포로 변환
    samples = np.zeros_like(U)
    for i in range(6):
        samples[:, i] = norm.ppf(U[:, i]) * stds[i] + means[i]

    # ===== 단계 5: 공식으로 실제 Target과 연결 =====
    # Y = r*X + sqrt(1-r^2)*Z

    # 1. telecom_payment_rate
    r = r_values[0]
    Z_norm = (samples[:, 1] - samples[:, 1].mean()) / (samples[:, 1].std() + 1e-8)
    Y = r * target_standardized + np.sqrt(1 - r**2) * Z_norm
    df_new['telecom_payment_rate'] = np.clip(
        (Y - Y.min()) / (Y.max() - Y.min() + 1e-8), 0, 1
    )

    # 2. utility_payment_rate
    r = r_values[1]
    Z_norm = (samples[:, 2] - samples[:, 2].mean()) / (samples[:, 2].std() + 1e-8)
    Y = r * target_standardized + np.sqrt(1 - r**2) * Z_norm
    df_new['utility_payment_rate'] = np.clip(
        (Y - Y.min()) / (Y.max() - Y.min() + 1e-8), 0, 1
    )

    # 3. spending_consistency
    r = r_values[2]
    Z_norm = (samples[:, 3] - samples[:, 3].mean()) / (samples[:, 3].std() + 1e-8)
    Y = r * target_standardized + np.sqrt(1 - r**2) * Z_norm
    df_new['spending_consistency'] = np.clip(
        75 + 25 * ((Y - Y.min()) / (Y.max() - Y.min() + 1e-8)), 0, 100
    )

    # 4. regular_payment_count
    r = r_values[3]
    Z_norm = (samples[:, 4] - samples[:, 4].mean()) / (samples[:, 4].std() + 1e-8)
    Y = r * target_standardized + np.sqrt(1 - r**2) * Z_norm
    df_new['regular_payment_count'] = np.maximum(
        0, np.round(8 + 6 * ((Y - Y.min()) / (Y.max() - Y.min() + 1e-8))).astype(int)
    )

    # 5. app_login_frequency
    r = r_values[4]
    Z_norm = (samples[:, 5] - samples[:, 5].mean()) / (samples[:, 5].std() + 1e-8)
    Y = r * target_standardized + np.sqrt(1 - r**2) * Z_norm
    df_new['app_login_frequency'] = np.maximum(
        0, np.round(15 + 10 * ((Y - Y.min()) / (Y.max() - Y.min() + 1e-8))).astype(int)
    )

    # ===== 단계 6: 씬파일러 레이블 =====
    df_new = label_thin_filer(df_new)

    # ===== 단계 7: 상관계수 검증 =====
    _validate_correlations(df_new, target_col)

    logger.info(f"대안 데이터 생성 완료 (방법 3)\n")

    return df_new


def _validate_correlations(df, target_col):
    """상관계수 검증"""
    alt_vars = ['telecom_payment_rate', 'utility_payment_rate',
                'spending_consistency', 'regular_payment_count',
                'app_login_frequency']

    logger.info(f"\n{'='*75}")
    logger.info(f"생성된 대안 변수와 타겟(SeriousDlqin2yrs) 간 실제 상관계수")
    logger.info(f"{'='*75}")
    logger.info(f"{'변수명':<30} {'실제 상관계수':<18} {'범위 확인':<20}")
    logger.info("-" * 75)

    correlations = {}
    in_range_count = 0

    for var in alt_vars:
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


def label_thin_filer(df):
    """
    씬파일러 판정 로직

    판정 기준:
    1. 금융 연체 관련 컬럼 2개 이상 결측
    2. 또는 신용카드 거래 이력 12개월 미만 (NumberOfOpenCreditLinesAndLoans < 2)
    """
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
    if 'NumberOfOpenCreditLinesAndLoans' in df.columns:
        criteria_2 = df['NumberOfOpenCreditLinesAndLoans'] < 2
    else:
        criteria_2 = pd.Series(False, index=df.index)

    # 두 기준 중 하나라도 만족하면 씬파일러
    df['is_thin_filer'] = (criteria_1 | criteria_2).astype(int)

    thin_filer_count = df['is_thin_filer'].sum()
    logger.info(f"씬파일러 판정 완료:")
    logger.info(f"   총 샘플: {len(df):,}")
    logger.info(f"   씬파일러: {thin_filer_count:,} ({thin_filer_count/len(df)*100:.2f}%)")
    logger.info(f"   일반: {len(df) - thin_filer_count:,} ({(1-thin_filer_count/len(df))*100:.2f}%)\n")

    return df
