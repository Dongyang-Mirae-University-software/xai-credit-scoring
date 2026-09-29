"""
대안 데이터 시뮬레이터.
telecom_payment_rate, utility_payment_rate, spending_consistency,
regular_payment_count, app_login_frequency 5개 변수를 생성한다.

thin_filer_ratio: 씬파일러 비율 조정 파라미터
bias_ratio: 성별/연령대별 승인율 편향 조정 파라미터
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def label_thin_filer(df) -> pd.Series:
    """
    씬파일러 판정 함수

    기준: 금융 연체 관련 컬럼 2개 이상 결측
          또는 MonthlyIncome 결측/0

    Args:
        df (pd.DataFrame): 원본 데이터

    Returns:
        pd.Series: 씬파일러 여부 (1=씬파일러, 0=일반)
    """
    # 금융 연체 관련 컬럼
    delinquency_cols = [
        'NumberOfTime30-59DaysPastDueNotWorse',
        'NumberOfTimes90DaysLate',
        'NumberOfTime60-89DaysPastDueNotWorse'
    ]

    # 기준 1: 연체 컬럼 2개 이상 결측
    missing_count = df[delinquency_cols].isnull().sum(axis=1)
    is_thin_filer = (missing_count >= 2) | \
                    (df['MonthlyIncome'].isnull()) | \
                    (df['MonthlyIncome'] == 0)

    return is_thin_filer.astype(int)


def generate_alternative_data(df, target_col="SeriousDlqin2yrs",
                               thin_filer_ratio: float = 0.3,
                               bias_ratio: float = 0.1,
                               random_seed: int = 42) -> pd.DataFrame:
    """
    대안 데이터 생성 (통신비/공과금/소비 패턴)

    생성 변수:
    - telecom_payment_rate: 통신비 정상납부율 (0~1)
    - utility_payment_rate: 공과금 납부율 (0~1)
    - spending_consistency: 소비 일관성 점수 (0~100)
    - regular_payment_count: 정기결제 건수 (0~20)
    - app_login_frequency: 앱 로그인 빈도 (0~30)

    Args:
        df (pd.DataFrame): 원본 데이터
        target_col (str): 타겟 컬럼명
        thin_filer_ratio (float): 씬파일러 비율 (0~1)
        bias_ratio (float): 편향 강도 (0~1)
        random_seed (int): 재현성을 위한 시드

    Returns:
        pd.DataFrame: 대안 변수가 추가된 데이터프레임
    """
    np.random.seed(random_seed)
    df = df.copy()
    n = len(df)

    # 타겟 변수 추출
    target = df[target_col].values

    # 1. 씬파일러 레이블
    df['is_thin_filer'] = label_thin_filer(df)
    thin_filer_count = df['is_thin_filer'].sum()

    logger.info(f"✓ 대안 데이터 생성 중...")
    logger.info(f"  - 씬파일러: {thin_filer_count:,}명 ({100*thin_filer_count/n:.1f}%)")

    # 2. 대안 변수 생성 (타겟과 상관관계 유지)
    # 부도 가능성이 높을수록(target=1) 납부율이 낮아야 함 (음의 상관관계)

    noise_level = 0.15

    # telecom_payment_rate: 통신비 정상납부율
    df['telecom_payment_rate'] = np.clip(
        0.85 - 0.2 * target + np.random.normal(0, noise_level, n),
        0, 1
    )

    # utility_payment_rate: 공과금 납부율
    df['utility_payment_rate'] = np.clip(
        0.80 - 0.25 * target + np.random.normal(0, noise_level, n),
        0, 1
    )

    # spending_consistency: 소비 일관성 (0~100)
    df['spending_consistency'] = np.clip(
        60 - 20 * target + np.random.normal(0, 15, n),
        0, 100
    )

    # regular_payment_count: 정기결제 건수 (0~20)
    df['regular_payment_count'] = np.clip(
        8 - 4 * target + np.random.poisson(2, n),
        0, 20
    )

    # app_login_frequency: 앱 로그인 빈도 (월, 0~30)
    df['app_login_frequency'] = np.clip(
        12 - 6 * target + np.random.poisson(3, n),
        0, 30
    )

    # 3. 상관계수 검증
    alt_cols = ['telecom_payment_rate', 'utility_payment_rate',
                'spending_consistency', 'regular_payment_count',
                'app_login_frequency']

    logger.info(f"\n  대안 변수와 타겟의 상관계수:")
    corr_with_target = df[alt_cols + [target_col]].corr()[target_col][:-1]
    for col, corr in corr_with_target.items():
        logger.info(f"    - {col}: {corr:.4f}")

    logger.info(f"\n✓ 대안 데이터 생성 완료 (5개 변수)")

    return df


if __name__ == '__main__':
    import logging
    from pathlib import Path
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    from loader import load_gmsc

    # 데이터 로드
    df = load_gmsc('../../data/raw/kaggle_datasets/cs-training.csv')

    # 대안 데이터 생성
    df_alt = generate_alternative_data(df, target_col='SeriousDlqin2yrs')

    # 중간 결과 저장
    intermediate_dir = Path('../../data/intermediate')
    intermediate_dir.mkdir(exist_ok=True, parents=True)

    output_path = intermediate_dir / '02_simulated.csv'
    df_alt.to_csv(output_path, index=False)

    logger.info(f'\n✓ 대안 데이터 저장: {output_path}')
    print(f'\n대안 데이터 생성 완료: {df_alt.shape}')
    print(f'생성된 컬럼: {list(df_alt.columns[-6:])}')  # 마지막 6개 컬럼 (대안 5개 + is_thin_filer)
