"""
데이터 전처리: 결측치 처리, 스케일링, Train/Validation/Test 분할
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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


def preprocess(df, target_col='SeriousDlqin2yrs', drop_cols=None):
    """
    데이터 전처리: 결측치 처리, 범주형 인코딩, 수치형 스케일링

    Args:
        df: 입력 데이터프레임
        target_col: 타겟 컬럼명
        drop_cols: 제거할 컬럼 (기본값: ['Unnamed: 0'])

    Returns:
        전처리된 데이터프레임, 스케일러 객체
    """
    _setup_logger()

    df_processed = df.copy()

    # 고객 번호(원본 Unnamed: 0)는 분할 파일에 남기고, 모델 입력에서는 제외한다
    if 'Unnamed: 0' in df_processed.columns:
        df_processed = df_processed.rename(columns={'Unnamed: 0': 'customer_id'})
    if drop_cols is None:
        drop_cols = []

    logger.info(f"전처리 시작 (샘플: {len(df_processed):,})")

    # ===== 단계 1: 불필요한 컬럼 제거 =====
    cols_to_drop = [col for col in drop_cols if col in df_processed.columns]
    if cols_to_drop:
        df_processed = df_processed.drop(columns=cols_to_drop)
        logger.info(f"제거된 컬럼: {cols_to_drop}")

    # ===== 단계 2: 결측치 처리 =====
    logger.info(f"\n결측치 처리:")

    # 월소득 결측 행의 DebtRatio는 비율이 아니라 금액이므로, 소득 중앙값으로 나눠 비율로 환산한다
    if 'MonthlyIncome' in df_processed.columns and 'DebtRatio' in df_processed.columns:
        income_missing = df_processed['MonthlyIncome'].isnull()
        income_median = df_processed['MonthlyIncome'].median()
        df_processed.loc[income_missing, 'DebtRatio'] = df_processed.loc[income_missing, 'DebtRatio'] / income_median
        logger.info(f"  DebtRatio 보정: 월소득 결측 {income_missing.sum():,}행 금액 → 비율 환산 (월소득 중앙값 {income_median:,.0f})")

    # 결측치 현황
    missing_before = df_processed.isnull().sum()
    missing_cols = missing_before[missing_before > 0].index.tolist()

    for col in missing_cols:
        missing_count = df_processed[col].isnull().sum()
        missing_pct = (missing_count / len(df_processed)) * 100

        # 중앙값으로 대체
        median_val = df_processed[col].median()
        df_processed[col] = df_processed[col].fillna(median_val)

        logger.info(f"  {col}: {missing_count:,}개 ({missing_pct:.2f}%) → 중앙값({median_val:.2f})으로 대체")

    # ===== 단계 3: 범주형 변수 인코딩 =====
    logger.info(f"\n범주형 변수 인코딩:")

    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()

    # 제외할 범주형 컬럼 (이미 인코딩됨)
    exclude_categorical = [target_col, 'is_thin_filer']
    categorical_cols = [col for col in categorical_cols if col not in exclude_categorical]

    if categorical_cols:
        logger.info(f"  발견된 범주형 변수: {categorical_cols}")
        # One-Hot Encoding 적용 (drop_first=True로 다중공선성 방지)
        df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
        logger.info(f"  One-Hot Encoding 적용 완료")
    else:
        logger.info(f"  범주형 변수 없음")

    # ===== 단계 4: 수치형 변수 스케일링 =====
    logger.info(f"\n수치형 변수 스케일링 (StandardScaler):")

    # 타겟, 이진 변수, 인구통계 변수 제외한 특성 변수
    exclude_scaling = [target_col, 'is_thin_filer', 'gender', 'age_group', 'customer_id']
    feature_cols = [col for col in df_processed.columns
                    if col not in exclude_scaling]

    scaler = StandardScaler()
    df_processed[feature_cols] = scaler.fit_transform(df_processed[feature_cols])

    logger.info(f"  {len(feature_cols)}개 변수 스케일링 완료")
    logger.info(f"  (타겟/이진변수 제외)")

    logger.info(f"\n전처리 완료")
    logger.info(f"  최종 샘플: {len(df_processed):,}")
    logger.info(f"  최종 컬럼: {df_processed.shape[1]}")

    return df_processed, scaler


AGE_GROUP_LABELS = {0: '20-34', 1: '35-54', 2: '55+'}


def _to_split_csv_frame(frame):
    out = frame.copy()
    out['age_group'] = out['age_group'].map(AGE_GROUP_LABELS)
    return out


def split_data(df, target_col='SeriousDlqin2yrs',
               train_ratio: float = 0.7,
               val_ratio: float = 0.15,
               test_ratio: float = 0.15,
               random_seed: int = 42):
    """
    Stratified Train/Validation/Test 분할

    Args:
        df: 전처리된 데이터프레임
        target_col: 타겟 컬럼명
        train_ratio: 훈련 데이터 비율
        val_ratio: 검증 데이터 비율
        test_ratio: 테스트 데이터 비율
        random_seed: 난수 시드

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    _setup_logger()

    logger.info(f"\n데이터 분할 (Stratified Split):")
    logger.info(f"  Train : Val : Test = {train_ratio} : {val_ratio} : {test_ratio}")

    # 특성과 타겟 분리 (gender, age_group, is_thin_filer 제외)
    cols_to_drop = [target_col, 'gender', 'age_group', 'is_thin_filer', 'customer_id']
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[target_col]

    # 1단계: Train / (Val + Test) 분할
    test_val_ratio = val_ratio + test_ratio
    X_train, X_test_val, y_train, y_test_val = train_test_split(
        X, y,
        test_size=test_val_ratio,
        random_state=random_seed,
        stratify=y
    )

    # 2단계: (Val + Test) → Val / Test 분할
    val_test_ratio = test_ratio / test_val_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_test_val, y_test_val,
        test_size=val_test_ratio,
        random_state=random_seed,
        stratify=y_test_val
    )

    logger.info(f"  Train: {len(X_train):,} ({len(X_train)/len(df)*100:.1f}%)")
    logger.info(f"  Val  : {len(X_val):,} ({len(X_val)/len(df)*100:.1f}%)")
    logger.info(f"  Test : {len(X_test):,} ({len(X_test)/len(df)*100:.1f}%)")

    # 클래스 분포 확인
    logger.info(f"\n클래스 분포 (Target: {target_col}):")
    for data_type, y_data in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        if target_col == 'SeriousDlqin2yrs':
            pos_count = (y_data == 1).sum()
            neg_count = (y_data == 0).sum()
            logger.info(f"  {data_type}: Positive={pos_count:,} ({pos_count/len(y_data)*100:.2f}%), "
                       f"Negative={neg_count:,} ({neg_count/len(y_data)*100:.2f}%)")

    logger.info(f"\n데이터 분할 완료\n")

    # ===== Train/Val/Test 데이터 저장 =====
    import os
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'splits')
    os.makedirs(output_dir, exist_ok=True)

    # Train 데이터
    train_df = _to_split_csv_frame(df.loc[X_train.index].copy())
    train_path = os.path.join(output_dir, 'train_processed.csv')
    train_df.to_csv(train_path, index=False, encoding='utf-8')
    logger.info(f"Train 데이터 저장: {train_path} ({len(train_df):,} rows)")

    # Validation 데이터
    val_df = _to_split_csv_frame(df.loc[X_val.index].copy())
    val_path = os.path.join(output_dir, 'val_processed.csv')
    val_df.to_csv(val_path, index=False, encoding='utf-8')
    logger.info(f"Val 데이터 저장: {val_path} ({len(val_df):,} rows)")

    # Test 데이터
    test_df = _to_split_csv_frame(df.loc[X_test.index].copy())
    test_path = os.path.join(output_dir, 'test_processed.csv')
    test_df.to_csv(test_path, index=False, encoding='utf-8')
    logger.info(f"Test 데이터 저장: {test_path} ({len(test_df):,} rows)\n")

    return X_train, X_val, X_test, y_train, y_val, y_test
