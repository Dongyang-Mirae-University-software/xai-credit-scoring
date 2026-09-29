"""모델 학습 및 성능 테스트"""
import pytest
from pathlib import Path
import json
import joblib

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'


def test_model_file_exists():
    """모델 파일 존재 확인"""
    model_path = MODELS_DIR / 'best_model_xgboost_v1.0.pkl'
    assert model_path.exists(), f"모델 파일 없음: {model_path}"


def test_model_metadata_exists():
    """메타데이터 파일 존재 확인"""
    metadata_path = MODELS_DIR / 'model_metadata.json'
    assert metadata_path.exists(), f"메타데이터 파일 없음: {metadata_path}"


def test_model_loadable():
    """모델 로드 가능 확인"""
    model_path = MODELS_DIR / 'best_model_xgboost_v1.0.pkl'
    try:
        model = joblib.load(model_path)
        assert model is not None, "모델이 비어있음"
    except Exception as e:
        pytest.fail(f"모델 로드 실패: {e}")


def test_metadata_content():
    """메타데이터 내용 확인"""
    metadata_path = MODELS_DIR / 'model_metadata.json'
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    required_keys = ['best_model', 'test_auc', 'test_metrics']
    for key in required_keys:
        assert key in metadata, f"메타데이터에 {key} 없음"


def test_auc_performance():
    """AUC 성능 목표 확인 (≥ 0.78)"""
    metadata_path = MODELS_DIR / 'model_metadata.json'
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    auc = metadata.get('test_auc', 0)
    assert auc >= 0.78, f"AUC가 목표값(0.78)보다 낮음: {auc:.4f}"


def test_ks_performance():
    """KS 성능 목표 확인 (≥ 0.28)"""
    metadata_path = MODELS_DIR / 'model_metadata.json'
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    ks = metadata.get('test_metrics', {}).get('ks', 0)
    assert ks >= 0.28, f"KS가 목표값(0.28)보다 낮음: {ks:.4f}"


def test_model_predictions_format():
    """모델 예측 형식 확인"""
    model_path = MODELS_DIR / 'best_model_xgboost_v1.0.pkl'
    model = joblib.load(model_path)

    import numpy as np
    dummy_input = np.random.randn(1, 16)

    # 예측
    prediction = model.predict(dummy_input)
    assert isinstance(prediction, np.ndarray), "예측이 배열이 아님"

    # 확률 예측
    proba = model.predict_proba(dummy_input)
    assert proba.shape == (1, 2), f"확률 형태 오류: {proba.shape}"
    assert np.all((proba >= 0) & (proba <= 1)), "확률이 0~1 범위를 벗어남"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
