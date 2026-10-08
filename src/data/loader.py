"""
GMSC(Give Me Some Credit) 또는 German Credit 데이터 로드.
로드 시 컬럼명, 자료형, 결측치 비율을 로그로 출력한다.
"""

import logging
import pandas as pd
from pathlib import Path

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


def _print_data_info(df: pd.DataFrame, dataset_name: str):
    """데이터셋 정보 출력 (컬럼명, 자료형, 결측치 비율)"""
    _setup_logger()

    logger.info(f"\n{'='*80}")
    logger.info(f"📊 {dataset_name} 데이터 정보")
    logger.info(f"{'='*80}")
    logger.info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    logger.info(f"\n{'컬럼명':<30} {'자료형':<15} {'결측치 수':<12} {'결측 비율(%)':<12}")
    logger.info("-" * 80)

    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_rate = (missing_count / len(df)) * 100
        logger.info(
            f"{col:<30} {str(df[col].dtype):<15} {missing_count:<12} {missing_rate:>10.2f}%"
        )

    total_missing = df.isna().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    logger.info("-" * 80)
    logger.info(f"{'전체 결측치':<30} {'':<15} {total_missing:<12} {(total_missing/total_cells)*100:>10.2f}%")
    logger.info(f"{'='*80}\n")


def load_gmsc(train_path: str = "data/raw/cs-training.csv",
              test_path: str = "data/raw/cs-test.csv") -> tuple:
    """
    Give Me Some Credit 데이터셋 로드

    Args:
        train_path: 훈련 데이터 경로
        test_path: 테스트 데이터 경로

    Returns:
        (train_df, test_df) 튜플
    """
    _setup_logger()

    logger.info(f"📥 Give Me Some Credit 데이터 로드 중...")

    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        _print_data_info(train_df, "훈련 데이터셋 (cs-training.csv)")
        _print_data_info(test_df, "테스트 데이터셋 (cs-test.csv)")

        logger.info("✅ 데이터 로드 완료")
        return train_df, test_df

    except FileNotFoundError as e:
        logger.error(f"❌ 파일을 찾을 수 없습니다: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ 데이터 로드 중 오류 발생: {e}")
        raise


def load_german_credit():
    """
    German Credit 데이터셋 로드 (UCI ML Repository에서)

    Returns:
        (X, y) 튜플
    """
    _setup_logger()

    logger.info(f"📥 German Credit 데이터 로드 중...")

    try:
        from ucimlrepo import fetch_ucirepo

        german_credit = fetch_ucirepo(id=144)
        X = german_credit.data.features
        y = german_credit.data.targets

        df = pd.concat([X, y], axis=1)
        _print_data_info(df, "German Credit 데이터셋")

        logger.info("✅ German Credit 데이터 로드 완료")
        return X, y

    except ImportError:
        logger.error("❌ ucimlrepo 패키지가 설치되지 않았습니다. pip install ucimlrepo 실행")
        raise
    except Exception as e:
        logger.error(f"❌ German Credit 데이터 로드 중 오류 발생: {e}")
        raise
