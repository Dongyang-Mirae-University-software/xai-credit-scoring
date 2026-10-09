"""
Step 1: 데이터 생성 · 전처리 통합 테스트
이재희 담당 작업의 첫 단계를 실행하는 스크립트
"""

import logging
import os
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 경로 설정
PROJECT_ROOT = Path(__file__).parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw' / 'kaggle_datasets'
DATA_SPLITS = PROJECT_ROOT / 'data' / 'splits'
DATA_SPLITS.mkdir(exist_ok=True, parents=True)

# src import
from src.data.loader import load_gmsc
from src.data.simulator import generate_alternative_data
from src.data.preprocessor import preprocess, split_data


def main():
    """메인 실행 함수"""

    logger.info("=" * 60)
    logger.info("🚀 Step 1: 데이터 생성 · 전처리 시작")
    logger.info("=" * 60)

    # 1단계: 데이터 로드
    logger.info("\n[1/4] 데이터 로드")
    logger.info("-" * 60)
    gmsc_path = DATA_RAW / 'cs-training.csv'

    if not gmsc_path.exists():
        logger.error(f"✗ 파일 없음: {gmsc_path}")
        logger.info(f"  다운로드 경로: https://github.com/Dongyang-Mirae-University-software/xai-credit-scoring-datasets")
        return

    df = load_gmsc(str(gmsc_path))

    # 2단계: 대안 데이터 생성
    logger.info("\n[2/4] 대안 데이터 생성")
    logger.info("-" * 60)
    df = generate_alternative_data(
        df,
        target_col='SeriousDlqin2yrs',
        thin_filer_ratio=None,
        bias_ratio=0
    )

    # 3단계: 전처리 (결측치, 이상치, 정규화)
    logger.info("\n[3/4] 전처리 (결측치, 이상치, 정규화)")
    logger.info("-" * 60)
    df = preprocess(df, target_col='SeriousDlqin2yrs')

    # 4단계: 데이터 분할 (Train/Val/Test)
    logger.info("\n[4/4] 데이터 분할 (Train:Val:Test = 70:15:15)")
    logger.info("-" * 60)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df,
        target_col='SeriousDlqin2yrs',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )

    # 결과 저장
    logger.info("\n" + "=" * 60)
    logger.info("💾 결과 저장")
    logger.info("=" * 60)

    # Train 데이터
    train_df = X_train.copy()
    train_df['SeriousDlqin2yrs'] = y_train.values
    train_df.to_csv(DATA_SPLITS / 'train_processed.csv', index=False)
    logger.info(f"  ✓ train_processed.csv ({len(train_df):,}건)")

    # Validation 데이터
    val_df = X_val.copy()
    val_df['SeriousDlqin2yrs'] = y_val.values
    val_df.to_csv(DATA_SPLITS / 'val_processed.csv', index=False)
    logger.info(f"  ✓ val_processed.csv ({len(val_df):,}건)")

    # Test 데이터
    test_df = X_test.copy()
    test_df['SeriousDlqin2yrs'] = y_test.values
    test_df.to_csv(DATA_SPLITS / 'test_processed.csv', index=False)
    logger.info(f"  ✓ test_processed.csv ({len(test_df):,}건)")

    # 메타데이터 저장
    metadata = {
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'features': list(X_train.columns),
        'target': 'SeriousDlqin2yrs',
        'total_features': X_train.shape[1],
        'note': 'Stratified Split applied, StandardScaler normalized'
    }

    import json
    with open(DATA_SPLITS / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"  ✓ metadata.json")

    # 최종 요약
    logger.info("\n" + "=" * 60)
    logger.info("✅ Step 1 완료!")
    logger.info("=" * 60)
    logger.info(f"""
    저장 위치: {DATA_SPLITS}

    생성된 파일:
    - train_processed.csv: {len(X_train):,}건
    - val_processed.csv: {len(X_val):,}건
    - test_processed.csv: {len(X_test):,}건
    - metadata.json

    다음 단계: Step 2 - 모델 3중 묶음 학습
    """)

    logger.info("=" * 60)


if __name__ == '__main__':
    main()
