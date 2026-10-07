"""
9개 모델 학습 파이프라인 테스트
금융/대안/통합 데이터 × 3 알고리즘 = 9개 모델
각 모델에서 5-Fold 개별 점수 기록
"""

import os
import sys
import logging
from pathlib import Path

# 프로젝트 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import load_gmsc
from src.data.simulator import generate_alternative_data
from src.data.preprocessor import preprocess, split_data
from src.models.train import train_all_models_by_data_type

# MLflow 설정 (선택사항)
os.environ['MLFLOW_TRACKING_URI'] = 'http://mlflow.gosky.kr'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("="*80)
    logger.info("🚀 9개 모델 학습 파이프라인 테스트")
    logger.info("="*80)

    # 1. 데이터 로드
    logger.info("\n📥 Step 1: 데이터 로드")
    try:
        train_df, test_df = load_gmsc()
        logger.info(f"✅ 데이터 로드 완료: {train_df.shape[0]:,} 샘플, {train_df.shape[1]} 컬럼")
        df = train_df
    except Exception as e:
        logger.error(f"❌ 데이터 로드 실패: {e}")
        return

    # 2. 대안 데이터 생성
    logger.info("\n🔮 Step 2: 대안 데이터 생성")
    try:
        df = generate_alternative_data(df)
        logger.info(f"✅ 대안 데이터 추가 완료: {df.shape[1]} 컬럼")
    except Exception as e:
        logger.error(f"❌ 대안 데이터 시뮬레이션 실패: {e}")
        return

    # 3. 데이터 전처리
    logger.info("\n🧹 Step 3: 데이터 전처리")
    try:
        df, scaler = preprocess(df)
        result = split_data(df)
        X_train, X_val, X_test, y_train, y_val, y_test = result
        logger.info(f"✅ 데이터 전처리 완료")
        logger.info(f"   훈련: {X_train.shape[0]:,} 샘플, {X_train.shape[1]} 특성")
        logger.info(f"   검증: {X_val.shape[0]:,} 샘플")
        logger.info(f"   테스트: {X_test.shape[0]:,} 샘플")
        logger.info(f"   클래스 분포 (훈련): {(y_train == 0).sum():,} (0), {(y_train == 1).sum():,} (1)")
    except Exception as e:
        logger.error(f"❌ 데이터 전처리 실패: {e}")
        return

    # 4. 9개 모델 학습
    logger.info("\n🤖 Step 4: 9개 모델 학습 (금융/대안/통합 × 3 알고리즘)")
    try:
        results = train_all_models_by_data_type(X_train, y_train, models_dir='models', version='1.0')

        logger.info("\n" + "="*80)
        logger.info("✅ 9개 모델 학습 완료!")
        logger.info("="*80)

        # 결과 요약
        logger.info("\n📊 학습 결과 요약:")
        for data_type, algorithms in results.items():
            logger.info(f"\n  📁 {data_type.upper()}")
            for algo, metrics in algorithms.items():
                logger.info(f"    ✓ {algo}")

    except Exception as e:
        logger.error(f"❌ 모델 학습 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    logger.info("\n✨ 테스트 완료!")


if __name__ == '__main__':
    main()
