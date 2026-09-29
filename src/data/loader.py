"""
GMSC(Give Me Some Credit) 또는 German Credit 데이터 로드.
로드 시 컬럼명, 자료형, 결측치 비율을 로그로 출력한다.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_gmsc(path: str) -> pd.DataFrame:
    """
    Give Me Some Credit 데이터셋 로드

    Args:
        path (str): CSV 파일 경로

    Returns:
        pd.DataFrame: 로드된 데이터프레임
    """
    try:
        df = pd.read_csv(path)

        # 로그 출력
        logger.info(f"✓ GMSC 데이터 로드 완료")
        logger.info(f"  - 크기: {df.shape[0]:,}건 × {df.shape[1]}개 컬럼")
        logger.info(f"  - 메모리: {df.memory_usage(deep=True).sum() / 1024**2:.2f}MB")

        # 컬럼 정보
        logger.info(f"\n  컬럼 정보:")
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing = df[col].isnull().sum()
            missing_pct = 100 * missing / len(df)
            logger.info(f"    - {col}: {dtype} (결측치: {missing:,}건, {missing_pct:.1f}%)")

        # 결측치 요약
        total_missing = df.isnull().sum().sum()
        logger.info(f"\n  결측치 총합: {total_missing:,}개 ({100*total_missing/(df.shape[0]*df.shape[1]):.2f}%)")

        # 타겟 변수 분포
        if 'SeriousDlqin2yrs' in df.columns:
            logger.info(f"\n  타겟 분포 (SeriousDlqin2yrs):")
            value_counts = df['SeriousDlqin2yrs'].value_counts(normalize=True)
            for val in sorted(value_counts.index):
                pct = 100 * value_counts[val]
                logger.info(f"    - {int(val)}: {value_counts[val]:.4f} ({pct:.1f}%)")

        return df

    except FileNotFoundError:
        logger.error(f"✗ 파일을 찾을 수 없습니다: {path}")
        raise
    except Exception as e:
        logger.error(f"✗ 데이터 로드 중 오류 발생: {str(e)}")
        raise


def load_german_credit() -> pd.DataFrame:
    """
    German Credit 데이터셋 로드 (UCI ML Repository)

    Returns:
        pd.DataFrame: 로드된 데이터프레임
    """
    try:
        from ucimlrepo import fetch_ucirepo

        logger.info("✓ German Credit 데이터 로드 중...")
        german_credit = fetch_ucirepo(id=144)

        # 특성과 타겟 결합
        X = german_credit.data.features
        y = german_credit.data.targets
        df = pd.concat([X, y], axis=1)

        # 로그 출력
        logger.info(f"✓ German Credit 데이터 로드 완료")
        logger.info(f"  - 크기: {df.shape[0]:,}건 × {df.shape[1]}개 컬럼")

        # 컬럼 정보
        logger.info(f"\n  컬럼 정보:")
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing = df[col].isnull().sum()
            logger.info(f"    - {col}: {dtype} (결측치: {missing}개)")

        return df

    except Exception as e:
        logger.error(f"✗ German Credit 로드 중 오류: {str(e)}")
        raise


if __name__ == '__main__':
    import logging
    from pathlib import Path
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Give Me Some Credit 데이터 로드
    df = load_gmsc('../../data/raw/kaggle_datasets/cs-training.csv')

    # 중간 결과 저장
    intermediate_dir = Path('../../data/intermediate')
    intermediate_dir.mkdir(exist_ok=True, parents=True)

    output_path = intermediate_dir / '01_loaded.csv'
    df.to_csv(output_path, index=False)

    logger.info(f'\n✓ 로드된 데이터 저장: {output_path}')
    print(f'\n데이터 로드 완료: {df.shape}')
