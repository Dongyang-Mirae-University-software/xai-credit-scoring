"""
결측치 처리, 스케일링, 인코딩, Train/Validation/Test 분할.
분할은 Stratified Split만 사용한다.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def preprocess(df, target_col: str = 'SeriousDlqin2yrs') -> pd.DataFrame:
    """
    데이터 전처리 (결측치 처리, 정규화)

    단계:
    1. 결측치 처리 (중앙값/최빈값 대체)
    2. 이상치 제거 (IQR 방식)
    3. 정규화 (StandardScaler)

    Args:
        df (pd.DataFrame): 원본 데이터
        target_col (str): 타겟 컬럼명

    Returns:
        pd.DataFrame: 전처리된 데이터
    """
    df = df.copy()
    logger.info(f"✓ 전처리 시작")

    # 1단계: 결측치 처리
    logger.info(f"\n  1단계: 결측치 처리")

    # MonthlyIncome: 중앙값으로 대체
    if 'MonthlyIncome' in df.columns and df['MonthlyIncome'].isnull().sum() > 0:
        median_income = df['MonthlyIncome'].median()
        df['MonthlyIncome'].fillna(median_income, inplace=True)
        logger.info(f"    - MonthlyIncome: 중앙값({median_income:,.0f})로 대체")

    # NumberOfDependents: 최빈값으로 대체
    if 'NumberOfDependents' in df.columns and df['NumberOfDependents'].isnull().sum() > 0:
        mode_dependents = df['NumberOfDependents'].mode()[0]
        df['NumberOfDependents'].fillna(mode_dependents, inplace=True)
        logger.info(f"    - NumberOfDependents: 최빈값({mode_dependents})으로 대체")

    # 결측치 확인
    remaining_missing = df.isnull().sum().sum()
    logger.info(f"    - 결측치 남음: {remaining_missing}개")

    # 2단계: 이상치 제거 (IQR 방식)
    logger.info(f"\n  2단계: 이상치 제거")

    before_rows = len(df)

    if 'DebtRatio' in df.columns:
        Q1 = df['DebtRatio'].quantile(0.25)
        Q3 = df['DebtRatio'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df['DebtRatio'] < lower_bound) | (df['DebtRatio'] > upper_bound)].shape[0]
        df = df[(df['DebtRatio'] >= lower_bound) & (df['DebtRatio'] <= upper_bound)]
        logger.info(f"    - DebtRatio: {outliers}개 이상치 제거")

    after_rows = len(df)
    logger.info(f"    - 제거 후: {before_rows:,} → {after_rows:,}건")

    # 3단계: 정규화 (StandardScaler)
    logger.info(f"\n  3단계: 정규화")

    # 수치형 컬럼 선택 (타겟 제외)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    # StandardScaler 적용
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    logger.info(f"    - {len(numeric_cols)}개 수치형 변수 정규화 완료")

    # 정규화 검증
    means = df[numeric_cols].mean()
    stds = df[numeric_cols].std()
    logger.info(f"    - 평균: {means.mean():.6f} (≈0)")
    logger.info(f"    - 표준편차: {stds.mean():.6f} (≈1)")

    logger.info(f"\n✓ 전처리 완료 ({after_rows:,}건)")

    return df


def split_data(df, target_col: str = 'SeriousDlqin2yrs',
               train_ratio: float = 0.7,
               val_ratio: float = 0.15,
               test_ratio: float = 0.15,
               random_seed: int = 42):
    """
    Train/Validation/Test 분할 (Stratified Split)

    비율: 훈련 70% / 검증 15% / 테스트 15%

    Args:
        df (pd.DataFrame): 전처리된 데이터
        target_col (str): 타겟 컬럼명
        train_ratio (float): 훈련 비율
        val_ratio (float): 검증 비율
        test_ratio (float): 테스트 비율
        random_seed (int): 재현성을 위한 시드

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info(f"✓ 데이터 분할 시작")

    # 입력 변수와 타겟 분리
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # 1단계: 70:30 분할 (훈련:테스트)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        stratify=y,
        random_state=random_seed
    )

    logger.info(f"\n  1단계: 훈련 + 검증 : 테스트 = {train_ratio + val_ratio:.0%} : {test_ratio:.0%}")
    logger.info(f"    - 훈련+검증: {len(X_temp):,}건")
    logger.info(f"    - 테스트: {len(X_test):,}건")

    # 2단계: 훈련+검증을 다시 분할 (70:15 → 85:15)
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio_adjusted,
        stratify=y_temp,
        random_state=random_seed
    )

    logger.info(f"\n  2단계: 훈련 : 검증 = {train_ratio:.0%} : {val_ratio:.0%}")
    logger.info(f"    - 훈련: {len(X_train):,}건")
    logger.info(f"    - 검증: {len(X_val):,}건")

    # Stratification 확인
    logger.info(f"\n  Stratification 확인 (타겟 분포):")

    train_dist = y_train.value_counts(normalize=True).sort_index()
    val_dist = y_val.value_counts(normalize=True).sort_index()
    test_dist = y_test.value_counts(normalize=True).sort_index()

    for label in sorted(y.unique()):
        logger.info(f"    - 클래스 {label}:")
        logger.info(f"      훈련: {train_dist.get(label, 0):.1%}")
        logger.info(f"      검증: {val_dist.get(label, 0):.1%}")
        logger.info(f"      테스트: {test_dist.get(label, 0):.1%}")

    logger.info(f"\n✓ 분할 완료")
    logger.info(f"  - 총 데이터: {len(X):,}건")
    logger.info(f"  - 입력 변수: {X.shape[1]}개")

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == '__main__':
    from pathlib import Path
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 중간 데이터 로드 (simulator.py 결과)
    intermediate_dir = Path('../../data/intermediate')
    input_path = intermediate_dir / '02_simulated.csv'

    if not input_path.exists():
        print(f"❌ 파일 없음: {input_path}")
        print(f"먼저 python simulator.py를 실행하세요!")
        exit(1)

    df = pd.read_csv(input_path)
    logger.info(f"✓ 입력 파일 로드: {input_path}")

    # 전처리
    df_processed = preprocess(df, target_col='SeriousDlqin2yrs')

    # 분할
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df_processed,
        target_col='SeriousDlqin2yrs',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )

    # 최종 결과 저장 (main_step1.py와 동일한 위치)
    splits_dir = Path('../../data/splits')
    splits_dir.mkdir(exist_ok=True, parents=True)

    # 타겟 컬럼 추가해서 저장
    train_df = X_train.copy()
    train_df['SeriousDlqin2yrs'] = y_train.values
    train_df.to_csv(splits_dir / 'train_processed.csv', index=False)

    val_df = X_val.copy()
    val_df['SeriousDlqin2yrs'] = y_val.values
    val_df.to_csv(splits_dir / 'val_processed.csv', index=False)

    test_df = X_test.copy()
    test_df['SeriousDlqin2yrs'] = y_test.values
    test_df.to_csv(splits_dir / 'test_processed.csv', index=False)

    logger.info(f"\n✓ 최종 결과 저장:")
    logger.info(f"  - {splits_dir / 'train_processed.csv'}")
    logger.info(f"  - {splits_dir / 'val_processed.csv'}")
    logger.info(f"  - {splits_dir / 'test_processed.csv'}")
    print(f"\n전처리 완료!")
