"""데이터 로딩 및 전처리 테스트"""
import pytest
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DATA_SPLITS = PROJECT_ROOT / 'data' / 'splits'


def test_train_data_exists():
    """훈련 데이터 파일 존재 확인"""
    train_path = DATA_SPLITS / 'train_processed.csv'
    assert train_path.exists(), f"훈련 데이터 파일 없음: {train_path}"


def test_val_data_exists():
    """검증 데이터 파일 존재 확인"""
    val_path = DATA_SPLITS / 'val_processed.csv'
    assert val_path.exists(), f"검증 데이터 파일 없음: {val_path}"


def test_test_data_exists():
    """테스트 데이터 파일 존재 확인"""
    test_path = DATA_SPLITS / 'test_processed.csv'
    assert test_path.exists(), f"테스트 데이터 파일 없음: {test_path}"


def test_train_data_shape():
    """훈련 데이터 형태 확인"""
    df = pd.read_csv(DATA_SPLITS / 'train_processed.csv')
    assert df.shape[0] > 50000, f"훈련 데이터 행 수가 너무 적음: {df.shape[0]}"
    assert df.shape[1] >= 16, f"훈련 데이터 컬럼 수가 부족함: {df.shape[1]}"


def test_test_data_split_ratio():
    """데이터 분할 비율 확인 (70:15:15)"""
    train = pd.read_csv(DATA_SPLITS / 'train_processed.csv')
    val = pd.read_csv(DATA_SPLITS / 'val_processed.csv')
    test = pd.read_csv(DATA_SPLITS / 'test_processed.csv')

    total = len(train) + len(val) + len(test)
    train_ratio = len(train) / total
    val_ratio = len(val) / total
    test_ratio = len(test) / total

    assert 0.68 < train_ratio < 0.72, f"훈련 비율 오류: {train_ratio:.2%}"
    assert 0.14 < val_ratio < 0.16, f"검증 비율 오류: {val_ratio:.2%}"
    assert 0.14 < test_ratio < 0.16, f"테스트 비율 오류: {test_ratio:.2%}"


def test_no_nan_values():
    """타겟 컬럼 NaN 값 확인"""
    train = pd.read_csv(DATA_SPLITS / 'train_processed.csv')
    assert train['SeriousDlqin2yrs'].isnull().sum() == 0, "타겟 컬럼에 NaN 있음"

    test = pd.read_csv(DATA_SPLITS / 'test_processed.csv')
    assert test['SeriousDlqin2yrs'].isnull().sum() == 0, "테스트 타겟 컬럼에 NaN 있음"


def test_target_column_exists():
    """타겟 컬럼 존재 확인"""
    train = pd.read_csv(DATA_SPLITS / 'train_processed.csv')
    assert 'SeriousDlqin2yrs' in train.columns, "타겟 컬럼 없음"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
