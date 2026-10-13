"""모델 학습 및 성능 테스트

현재 저장소의 최종 모델(통합 데이터 LightGBM)과 설명 파일(model_metadata.json)을 기준으로 검사한다.
(산출물 정책 B안: 최종 모델 1개와 설명 파일만 저장소에 있음, models/README.md 참고)
"""
import pytest
from pathlib import Path
import json
import joblib

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
FINAL_MODEL = MODELS_DIR / 'lightgbm_is_unbalance_integrated_v1.0.joblib'
METADATA = MODELS_DIR / 'model_metadata.json'


def _load_metadata():
    with open(METADATA, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_model_file_exists():
    """최종 모델 파일 존재 확인"""
    assert FINAL_MODEL.exists(), f"최종 모델 파일 없음: {FINAL_MODEL}"


def test_model_metadata_exists():
    """메타데이터 파일 존재 확인"""
    assert METADATA.exists(), f"메타데이터 파일 없음: {METADATA}"


def test_model_loadable():
    """최종 모델 로드 가능 확인"""
    try:
        model = joblib.load(FINAL_MODEL)
        assert model is not None, "모델이 비어있음"
    except Exception as e:
        pytest.fail(f"모델 로드 실패: {e}")


def test_metadata_content():
    """메타데이터에 최종 선택 결과가 있는지 확인"""
    metadata = _load_metadata()
    assert 'selection' in metadata, "메타데이터에 selection 없음"
    for key in ['selected_model', 'test_auc', 'test_ks']:
        assert key in metadata['selection'], f"selection에 {key} 없음"


def test_auc_performance():
    """AUC 성능 목표 확인 (≥ 0.78)"""
    auc = _load_metadata()['selection']['test_auc']
    assert auc >= 0.78, f"AUC가 목표값(0.78)보다 낮음: {auc:.4f}"


def test_ks_performance():
    """KS 성능 목표 확인 (≥ 0.28)"""
    ks = _load_metadata()['selection']['test_ks']
    assert ks >= 0.28, f"KS가 목표값(0.28)보다 낮음: {ks:.4f}"


def test_model_predictions_format():
    """최종 모델 예측 형식 확인 (입력 변수 개수 기준)"""
    import numpy as np

    model = joblib.load(FINAL_MODEL)
    n_features = getattr(model, 'n_features_in_', None)
    if n_features is None:
        n_features = len(model.feature_names_in_)
    dummy_input = np.random.randn(1, n_features)

    prediction = model.predict(dummy_input)
    assert isinstance(prediction, np.ndarray), "예측이 배열이 아님"

    proba = model.predict_proba(dummy_input)
    assert proba.shape == (1, 2), f"확률 형태 오류: {proba.shape}"
    assert np.all((proba >= 0) & (proba <= 1)), "확률이 0~1 범위를 벗어남"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
