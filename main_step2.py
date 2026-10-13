"""
Step 2: 모델 학습 및 최종 모델 평가
step1이 만든 분할 파일(data/splits)로 9개 모델을 학습하고,
Val AUC로 최종 모델을 고른 뒤 Test에서 1회만 평가한다.
"""

import json
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / '.env')

from src.models.train import train_all_models_by_data_type, select_and_test_once

DATA_SPLITS = PROJECT_ROOT / 'data' / 'splits'
MODELS_DIR = PROJECT_ROOT / 'models'
TARGET = 'SeriousDlqin2yrs'
NON_FEATURE_COLS = [TARGET, 'customer_id', 'is_thin_filer', 'gender', 'age_group']

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_split(name):
    df = pd.read_csv(DATA_SPLITS / f'{name}_processed.csv')
    return df.drop(columns=NON_FEATURE_COLS), df[TARGET]


def main():
    metadata = json.loads((DATA_SPLITS / 'metadata.json').read_text(encoding='utf-8'))
    data_params = metadata.get('data_params', {})

    X_train, y_train = load_split('train')
    X_val, y_val = load_split('val')
    X_test, y_test = load_split('test')
    logger.info(f"분할 로드: train {len(X_train):,} / val {len(X_val):,} / test {len(X_test):,}")

    results = train_all_models_by_data_type(
        X_train, y_train, models_dir=str(MODELS_DIR), version='1.0', data_params=data_params
    )
    select_and_test_once(
        results, X_val, y_val, X_test, y_test, models_dir=str(MODELS_DIR), version='1.0',
        data_params=data_params,
    )
    logger.info("✅ Step 2 완료!")


if __name__ == '__main__':
    main()
